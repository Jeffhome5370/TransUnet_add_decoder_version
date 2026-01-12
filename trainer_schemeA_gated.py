# trainer.py (cleaned)
#test1
import argparse
import math
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import wandb

from utils import DiceLoss

# ----------------------------
# Global seed for dataloader workers
# ----------------------------
GLOBAL_WORKER_SEED = 1234

#===========================================================================
def compute_pred_stats(semantic_logits, label_ce):
    """
    semantic_logits: (B,C,H,W) logits (已套 bias 後、可 argmax)
    label_ce: (B,H,W) long
    """
    pred = semantic_logits.argmax(1)
    pred_fg_ratio = float((pred > 0).float().mean().item())
    gt_fg_ratio   = float((label_ce > 0).float().mean().item())
    pred_unique   = torch.unique(pred).detach().cpu().tolist()
    return pred, pred_fg_ratio, gt_fg_ratio, pred_unique

@torch.no_grad()
def fg_hist_dom(pred, num_classes):
    fg_pixels = pred[pred > 0]
    if fg_pixels.numel() == 0:
        return -1, 0.0
    hist = torch.bincount(fg_pixels, minlength=num_classes).float()
    fg_hist = hist[1:]
    fg_ratio = fg_hist / fg_hist.sum().clamp_min(1.0)
    dom_ratio = float(fg_ratio.max().item())
    dom_cls   = int(fg_ratio.argmax().item() + 1)
    return dom_cls, dom_ratio

def dice_softmax_loss(dice_loss_fn, semantic_logits, label_ce):
    # 你的 DiceLoss 可能吃 prob；這裡統一先 softmax
    prob = torch.softmax(semantic_logits, dim=1)
    return dice_loss_fn(prob, label_ce, softmax=False)



@torch.no_grad()
def log_margin_quantiles(semantic_logits):
    bg = semantic_logits[:, 0]
    fg = semantic_logits[:, 1:].amax(dim=1)
    d = (fg - bg).flatten()
    q01 = float(torch.quantile(d, 0.01).item())
    q05 = float(torch.quantile(d, 0.05).item())
    q10 = float(torch.quantile(d, 0.10).item())
    q50 = float(torch.quantile(d, 0.50).item())
    mean = float(d.mean().item())
    return q01, q05, q10, q50, mean

# -------------------------
# Phase schedule
# -------------------------
class PhaseScheduler:
    """
    Phase 1: 只求不塌陷（只留 fg_bg + fg_cls(+小dice)，所有正則=0）
    Phase 2: 加少量幾何正則（area/overlap 小權重）
    Phase 3: 再加 anti-collapse（cls_div / pix_div / q_div / pseudo）
    """
    def __init__(self, p1_epochs=2, p2_epochs=20, p3_dice_epochs=31, p3a_epochs=100, p3b_epochs=200,
                 p3_ready_window: int = 400,
                 p3_ready_min_samples: int = 200,
                 p3_ratio_med_thr: float = 2.0,
                 p3_ratio_p90_thr: float = 3.0,
                 p3_dom_med_thr: float = 0.85,
                 p3_min_classes_med: int = 5,
                 infl_ratio_thr: float = 3.0,
                 infl_min_gt: float = 0.01,
                 infl_boost: float = 1.0,
                 infl_max_scale: float = 4.0):
        self.p1_epochs = p1_epochs
        self.p2_epochs = p2_epochs
        self.p3_dice_epochs = p3_dice_epochs
        self.p3a_epochs = p3a_epochs
        self.p3b_epochs = p3b_epochs

        # ----- Phase3a -> Phase3b indicator-based gating -----
        # p3a_epochs is treated as the *upper bound* (max epoch index for phase=31).
        # If readiness criteria are met during phase=31, we will unlock phase=32 starting
        # from the *next epoch* (epoch boundary), to avoid mixing phases within the same epoch.
        self._p3b_start_epoch: Optional[int] = None
        self._p3_ready_window = int(p3_ready_window)
        self._p3_ready_min_samples = int(p3_ready_min_samples)
        self._p3_ratio_med_thr = float(p3_ratio_med_thr)
        self._p3_ratio_p90_thr = float(p3_ratio_p90_thr)
        self._p3_dom_med_thr = float(p3_dom_med_thr)
        self._p3_min_classes_med = int(p3_min_classes_med)

        self._p3_ratio_buf = deque(maxlen=self._p3_ready_window)
        self._p3_dom_buf = deque(maxlen=self._p3_ready_window)
        self._p3_ncls_buf = deque(maxlen=self._p3_ready_window)

        # ----- Inflation penalty auto-boost -----
        # When Pred_fg/GT_fg is far beyond threshold, we increase the effective weight of loss_infl.
        self.infl_ratio_thr = float(infl_ratio_thr)
        self.infl_min_gt = float(infl_min_gt)
        self.infl_boost = float(infl_boost)
        self.infl_max_scale = float(infl_max_scale)

    def phase(self, epoch_num):
        # epoch_num is 0-based
        if epoch_num < self.p1_epochs:
            return 1
        elif epoch_num < self.p2_epochs:
            return 2
        # Phase3: indicator-based gating
        # - If unlocked, phase=32 starts from _p3b_start_epoch
        # - Otherwise, stay in 31 until reaching the upper bound p3a_epochs
        if epoch_num < self.p3a_epochs:
            return 31
        else:
            return 32
        
    @staticmethod
    def _ramp(x: float) -> float:
        return float(max(0.0, min(1.0, x)))
    
    def weights(self, epoch_num):
        ph = self.phase(epoch_num)

        # ---- Base (always on) ----
        w = {
            "fb": 1.0,          # Stage1 fg/bg
            "cls": 1.0,         # Stage2 fg class CE
            "dice": 0.15,       # 小（避免主導）
            "cov": 0.0,         # 可選：先關，因為容易推 den 行為
            "area": 0.0,
            "ovlp": 0.0,
            "cls_div": 0.0,
            "pix_div": 0.0,
            "q_div": 0.0,
            "infl": 0.0,
            "pseudo": 0.0,
            "bgm":0.0,
            "recall":0.0
        }

        if ph == 1:
            # 輕微正則：只求先學會不塌
            w["dice"] = 0.10
            w["cls_div"] = 2e-3   # 原本 phase3=1e-2，這裡先 1/5
            w["pix_div"] = 0.0   # (方案A) Synapse: 移除 pix_div，避免錯誤 FG mask 扭曲分佈
            
        elif ph == 2:
            # 輕量幾何（先寬鬆、權重小）
            w["cls"] = 0.7
            w["dice"] = 0.15
            w["area"] = 1e-3
            w["ovlp"] = 2e-3
            w["cls_div"] = 2e-3
            w["pix_div"] = 0.0  # (方案A) disable pix_div
        elif ph == 31:
            # 最後才加分散/去塌陷
            w["cls"]     = 1.0
            w["dice"]    = 0.15
            w["area"]    = 1e-3
            w["ovlp"]    = 2e-3
            w["cls_div"] = 0.0
            w["pix_div"] = 0.0  # (方案A) disable pix_div
            w["q_div"]   = 1e-5
            w["infl"]    = 0.1 #（只在 inflation mode 啟用）
            w["bgm"] = 0.6
            
        else:
            
            w["cls"]  = 0.9      # 可以開始慢慢降，但別一口氣到 0.7
            w["dice"] = 0.20
            w["area"] = 1e-3
            w["ovlp"] = 2e-3
            w["bgm"] = 0.6

            # ramp span（幾個 epoch 內從 0 -> 目標值）
            ramp_epochs = 10.0
            p3b_start = self.p3b_epochs
            t = (epoch_num - p3b_start) / ramp_epochs
            r = self._ramp(t)
            w["cls_div"] = r * 2e-3  # (方案A) KL-to-prior, keep small
            w["pix_div"] = 0.0  # (方案A) disable pix_div
            w["q_div"]   = r * 2e-5  # (方案A) anti-collapse only, keep tiny
            w["infl"]    = 0.1      # (方案A) 前景膨脹抑制（只在 inflation mode 啟用）

        return ph, w

    def p3a_end_epoch(self) -> int:
        """Return the last epoch index that will be treated as phase=31."""
        if self._p3b_start_epoch is not None:
            return min(self.p3a_epochs - 1, self._p3b_start_epoch - 1)
        return self.p3a_epochs - 1
    def get_phase_epoch(self):
        return self.p1_epochs, self.p2_epochs, self.p3_dice_epochs, self.p3a_epochs, self.p3b_epochs
    '''
    def update_phase3_gate(
        self,
        *,
        epoch_num: int,
        pred_fg_ratio: float,
        gt_fg_ratio: float,
        dom_ratio: float,
        num_fg_classes: int,
    ) -> None:
        """
        Update rolling metrics during phase=31 and decide whether to unlock phase=32.
        Unlock happens at epoch boundary: if ready at epoch e, then phase=32 starts at e+1.
        """
        # Only track during phase=31 (Phase3a)
        if self.phase(epoch_num) != 31:
            return

        # Robust ratio: avoid exploding when GT is extremely tiny
        denom = max(float(gt_fg_ratio), 1e-12)
        ratio = float(pred_fg_ratio) / denom

        self._p3_ratio_buf.append(ratio)
        self._p3_dom_buf.append(float(dom_ratio))
        self._p3_ncls_buf.append(int(num_fg_classes))

        if len(self._p3_ratio_buf) < self._p3_ready_min_samples:
            return

        ratios = np.asarray(self._p3_ratio_buf, dtype=np.float64)
        doms   = np.asarray(self._p3_dom_buf, dtype=np.float64)
        ncls   = np.asarray(self._p3_ncls_buf, dtype=np.float64)

        ratio_med = float(np.median(ratios))
        ratio_p90 = float(np.quantile(ratios, 0.90))
        dom_med   = float(np.median(doms))
        ncls_med  = float(np.median(ncls))

        ready = (
            (ratio_med < self._p3_ratio_med_thr) and
            (ratio_p90 < self._p3_ratio_p90_thr) and
            (dom_med < self._p3_dom_med_thr) and
            (ncls_med >= float(self._p3_min_classes_med))
        )

        if ready:
            start_ep = int(epoch_num) + 1
            if self._p3b_start_epoch is None or start_ep < self._p3b_start_epoch:
                self._p3b_start_epoch = start_ep
                try:
                    logging.info(
                        f"[P3GATE] unlock phase3b: start_epoch={start_ep} "
                        f"ratio_med={ratio_med:.3f} ratio_p90={ratio_p90:.3f} "
                        f"dom_med={dom_med:.3f} ncls_med={ncls_med:.1f}"
                    )
                except Exception:
                    pass
    '''
    def infl_scale(self, *, pred_fg_ratio: float, gt_fg_ratio: float) -> float:
        """Auto-boost scale for inflation penalty weight when Pred_fg/GT_fg is extreme."""
        denom = max(float(gt_fg_ratio), float(self.infl_min_gt))
        ratio = float(pred_fg_ratio) / denom
        if ratio <= self.infl_ratio_thr:
            return 1.0
        scale = 1.0 + self.infl_boost * max(0.0, (ratio / self.infl_ratio_thr) - 1.0)
        return float(min(self.infl_max_scale, scale))


# -------------------------
# Controller (cooldown + extreme-only)
# -------------------------
class BgBiasController:
    """
    只在「極端全前景 / 極端全背景」才介入，並加 cooldown。
    用途：止血，不是主訓練方向盤。
    """
    def __init__(self,
             init_bias=0.0,
             max_bias=20.0,
             margin_q50_target=-0.05,
             gain=0.25,
             step_clip=0.30,
             ema_w=0.10):

        self.bias = None
        self.init_bias = init_bias
        self.max_bias = max_bias

        self.margin_q50_target = margin_q50_target
        self.gain = gain
        self.step_clip = step_clip
        self.ema_w = ema_w

        self.margin_q50_ema = None

    def _ensure(self, device):
        if self.bias is None:
            self.bias = torch.tensor(self.init_bias, device=device)

    def apply(self, semantic_logits_raw):
        self._ensure(semantic_logits_raw.device)
        bg0 = semantic_logits_raw[:, 0:1] + self.bias
        return torch.cat([bg0, semantic_logits_raw[:, 1:]], dim=1)

    @torch.no_grad()
    def step_from_logits(self,
                     semantic_logits_biased,
                     pred_fg_ratio=None,
                     phase=None):

    # -----------------------------
    # Phase-specific hyperparams
    # -----------------------------
        if phase == 2:
            gain = 0.05
            step_clip = 0.03
            ema_w = 0.03
            deadband = 0.08
            margin_q50_target = -0.10
        else:  # Phase3a / others
            gain = self.gain
            step_clip = self.step_clip
            ema_w = self.ema_w
            deadband = 0.05
            margin_q50_target = -0.05

        # -----------------------------
        # Compute margin distribution
        # margin = fg_max - bg
        # -----------------------------
        bg_logit = semantic_logits_biased[:, 0]
        fg_logit_max = semantic_logits_biased[:, 1:].amax(dim=1)
        margin_all = (fg_logit_max - bg_logit).flatten()

        margin_q50_now = torch.quantile(margin_all, 0.50)

        # -----------------------------
        # EMA on median margin
        # -----------------------------
        if self.margin_q50_ema is None:
            self.margin_q50_ema = margin_q50_now
        else:
            w = ema_w
            self.margin_q50_ema = (
                (1 - w) * self.margin_q50_ema + w * margin_q50_now
            )

        # ==========================================================
        # (A) 全背景保護：幾乎全 BG → 快速釋放 bias
        # ==========================================================
        if pred_fg_ratio is not None and pred_fg_ratio < 0.02:
            release = 0.03 if phase == 2 else 0.20
            self.bias = (self.bias - release).clamp(0.0, self.max_bias)
            return (
                -release,
                float(self.margin_q50_ema.item()),
                float(margin_q50_now.item())
            )

        # ==========================================================
        # (B) 翻正保險：只要 q50 > 0，立刻加壓
        # ==========================================================
        if float(margin_q50_now.item()) > 0.0:
            step = min(step_clip, 0.15)
            self.bias = (self.bias + step).clamp(0.0, self.max_bias)
            return (
                step,
                float(self.margin_q50_ema.item()),
                float(margin_q50_now.item())
            )

        # ==========================================================
        # (C) Deadband：接近目標不動
        # ==========================================================
        margin_err = float(self.margin_q50_ema.item() - margin_q50_target)
        if abs(margin_err) < deadband:
            self.bias = (self.bias * 0.999).clamp(0.0, self.max_bias)
            return (
                0.0,
                float(self.margin_q50_ema.item()),
                float(margin_q50_now.item())
            )

        # ==========================================================
        # (D) Proportional control (asymmetric)
        # ==========================================================
        step = margin_err * gain
        step = float(np.clip(step, -step_clip * 2.0, +step_clip))
        self.bias = (self.bias + step).clamp(0.0, self.max_bias)

        return (
            step,
            float(self.margin_q50_ema.item()),
            float(margin_q50_now.item())
        )

# -------------------------
# Core: build semantic logits from (class_logits, mask_probs)
# -------------------------
def build_semantic_logits(class_logits, mask_probs, den_clamp=0.2, T_cls=4.0, epoch_num=None):
    """
    class_logits: (B,Q,C) raw logits
    mask_probs  : (B,Q,H,W) sigmoid(mask)
    Return semantic_logits_raw: (B,C,H,W) (logits space)
    """
    B, Q, H, W = mask_probs.shape
    # safe den
    den_raw = mask_probs.sum(dim=1)              # (B,H,W)
    den = den_raw.clamp_min(den_clamp)           # (B,H,W)

    # log-sum-exp trick (stable mixture of query logits weighted by mask probs)
    log_mask = torch.log(mask_probs.clamp_min(1e-6))              # (B,Q,H,W)
    log_den  = torch.log(den.clamp_min(1e-6)).unsqueeze(1)        # (B,1,H,W)

    # temperature for class logits to prevent huge dominance early
    T_cls = 8.0 if epoch_num < 1 else 4.0
    cls = (class_logits / T_cls).clamp(-5.0, +5.0)                                   # (B,Q,C)
    semantic_logits_raw = torch.logsumexp(
        cls.unsqueeze(-1).unsqueeze(-1) + log_mask.unsqueeze(2),  # (B,Q,C,H,W)
        dim=1
    ) - log_den                                                  # (B,C,H,W)

    return semantic_logits_raw, den_raw

# -------------------------
# Stage1 fg/bg loss (keep as you already do)
# -------------------------
def stage1_fg_bg_loss(
    semantic_logits2, label_ce,
    target_ratio_mult=1.5,
    tau_ema_state=None,
    tau_mom=0.98,
    posw_cap=80.0,
    target_min=0.0,
    target_max=0.20,
    freeze_tau_if_gt0=True,
):
    bg_logit = semantic_logits2[:, 0:1]
    fg_logit = semantic_logits2[:, 1:].amax(1, True)
    diff = fg_logit - bg_logit

    with torch.no_grad():
        gt_is_fg = (label_ce > 0).float().unsqueeze(1)
        gt_cnt = int(gt_is_fg.sum().item())

        # --- GT = 0: 不要強迫畫 FG，也不要更新 tau（避免 drift）
        if gt_cnt == 0:
            target_ratio = torch.tensor(0.0, device=diff.device)
            pos_weight = torch.tensor(1.0, device=diff.device)

            if tau_ema_state is None:
                # 讓 fb_logit 幾乎全為負 => stage1_fg_ratio ~ 0
                tau_ema = diff.detach().max()
            else:
                tau_ema = tau_ema_state if freeze_tau_if_gt0 else diff.detach().max()

        else:
            gt_fg_ratio = gt_is_fg.mean().clamp(1e-4, 0.5)
            target_ratio = (gt_fg_ratio * target_ratio_mult).clamp(target_min, target_max)

            flat = diff.detach().flatten()
            N = flat.numel()
            k = int((1.0 - float(target_ratio.item())) * N)
            k = max(0, min(N - 1, k))
            tau_batch = flat.kthvalue(k + 1).values

            if tau_ema_state is None:
                tau_ema = tau_batch
            else:
                tau_ema = tau_mom * tau_ema_state + (1 - tau_mom) * tau_batch

            fg_frac = gt_is_fg.mean().clamp_min(1e-4)
            pos_weight = ((1.0 - fg_frac) / fg_frac).clamp(max=posw_cap)

    fb_logit = (diff - tau_ema).clamp(-12, 12)
    loss_fg_bg = F.binary_cross_entropy_with_logits(fb_logit, gt_is_fg, pos_weight=pos_weight)

    with torch.no_grad():
        stage1_fg_ratio = float((fb_logit > 0).float().mean().item())

    # 保留你原本的 safety：真的快全背景時放鬆 tau
    if stage1_fg_ratio < 0.01 and gt_cnt > 0:
        tau_ema = tau_ema - 0.5
        fb_logit = (diff - tau_ema).clamp(-12, 12)
        loss_fg_bg = F.binary_cross_entropy_with_logits(fb_logit, gt_is_fg, pos_weight=pos_weight)
        stage1_fg_ratio = float((fb_logit > 0).float().mean().item())

    return loss_fg_bg, fb_logit, stage1_fg_ratio, tau_ema


@torch.no_grad()
def build_energy_balanced_weights_from_usage(
    class_usage_ema: torch.Tensor,   # shape (num_classes,) includes bg at 0
    gt_counts_8: torch.Tensor,       # shape (8,) counts over y in 0..7 (fg classes)
    alpha: float = 0.7,
    w_min: float = 0.5,
    w_max: float = 4.0,
    eps: float = 1e-6,
    e_ref_floor: float = 0.1,       # 防止 e_ref 太小造成爆衝
) -> torch.Tensor:
    """
    方案A：用 prediction 的 class_usage_ema 做「反能量」權重
    - 只對 batch 內 GT 出現的類別做加權（gt_counts_8>0），其他類別=1
    - 權重越大 => 該類在 CE 裡越“貴” => 小器官更容易拿到梯度

    回傳 shape (8,) 的 w_energy
    """
    device = class_usage_ema.device
    w_energy = torch.ones(8, device=device)

    present = (gt_counts_8 > 0)
    if not present.any():
        return w_energy

    # 取出這個 batch 真的有 GT 的類別的「能量」
    # class_usage_ema index: 0=bg, 1..8=organs
    e = class_usage_ema[1:9].clamp_min(eps)           # (8,)
    e_present = e[present]

    # 參考能量：用 present 類別的 median，比固定值更穩
    e_ref = torch.median(e_present).clamp_min(e_ref_floor)

    # 反能量權重：能量越低 => weight 越大
    w_present = (e_ref / e_present).pow(alpha)

    # clamp + normalize（只在 present 子集合內 normalize，保持尺度穩）
    w_present = torch.clamp(w_present, w_min, w_max)
    w_present = w_present / w_present.mean().clamp_min(eps)

    w_energy[present] = w_present
    return w_energy

def stage2_energy_paremeter(epoch_num = 20, ph = 31, p2_epochs = 20):
    if ph == 31:
        if epoch_num - p2_epochs == 0:
            energy_alpha = 0.0
            energy_wmax  = 1.0
            enable_energy_reweight = False
        elif epoch_num - p2_epochs == 1:
            energy_alpha = 0.35
            energy_wmax  = 2.0
            enable_energy_reweight = True
        else:
            energy_alpha = 0.7
            energy_wmax  = 4.0
            enable_energy_reweight = True
    else:
        enable_energy_reweight = False
    return energy_alpha, energy_wmax, enable_energy_reweight

# -------------------------
# Stage2 fg class CE on GT fg (recommended early)
# -------------------------
def stage2_fg_cls_loss(
    semantic_logits2,
    label_ce,
    cls_count_ema=None,
    cls_mom=0.99,
    iter_num=None,
    class_usage_ema=None,          # <-- 新增：prediction energy EMA
    enable_energy_reweight=True,   # <-- 新增：開關
    energy_alpha=0.7,
    energy_wmin=0.5,
    energy_wmax=4.0,
):
    """
    只在 GT fg pixels 做 8-class CE (label-1)
    + (原本) GT-count EMA inverse-freq reweight
    + (方案A) prediction energy (class_usage_ema) 的反能量 reweight（只對 GT present 類）
    """
    device = semantic_logits2.device
    gt_fg = (label_ce > 0)
    if not gt_fg.any():
        return torch.tensor(0.0, device=device), cls_count_ema

    y = (label_ce[gt_fg] - 1).long()  # 0..7
    logits_fg = semantic_logits2[:, 1:].permute(0, 2, 3, 1)[gt_fg]  # (N,8)

    counts = torch.bincount(y, minlength=8).float().to(device)

    if cls_count_ema is None:
        cls_count_ema = torch.ones(8, device=device)

    # ---------------------------
    # (1) 原本：GT-count EMA inverse-freq
    # ---------------------------
    with torch.no_grad():
        cls_m = 0.95 if (iter_num is not None and iter_num < 2000) else 0.99
        cls_count_ema = cls_count_ema * cls_m + counts * (1 - cls_m)
        w_invfreq = (cls_count_ema.sum() / (cls_count_ema + 1.0)).clamp(1.0, 10.0)
        w_invfreq = w_invfreq / w_invfreq.mean().clamp_min(1e-6)

    w_final = w_invfreq

    # ---------------------------
    # (2) 方案A：prediction energy reweight（只對 GT present 類）
    # ---------------------------
    if enable_energy_reweight and (class_usage_ema is not None):
        w_energy = build_energy_balanced_weights_from_usage(
            class_usage_ema=class_usage_ema,
            gt_counts_8=counts,
            alpha=float(energy_alpha),
            w_min=float(energy_wmin),
            w_max=float(energy_wmax),
        )
        w_final = w_final * w_energy
        w_final = w_final / w_final.mean().clamp_min(1e-6)  # 保持整體尺度

    loss = F.cross_entropy(logits_fg, y, weight=w_final, label_smoothing=0.05)
    return loss, cls_count_ema


# -------------------------
# Regularizers (only enabled in later phases)
# -------------------------
def reg_area_pen(mask_probs):
    """
    area_pen = log(Q) - H(area distribution) >=0
    """
    Q = mask_probs.size(1)
    area = mask_probs.mean(dim=(2, 3))  # (B,Q)
    p = area / (area.sum(dim=1, keepdim=True) + 1e-6)
    p = p.clamp_min(1e-6)
    H = -(p * p.log()).sum(dim=1).mean()
    area_pen = math.log(Q) - H
    return area_pen

def reg_overlap_pen(mask_probs, den_raw, thr=2.0):
    # 門檻寬鬆，不要早期把 den 壓死
    return F.relu(den_raw - thr).pow(2).mean()

def reg_cls_div(class_logits, cls_count_ema, eps: float = 1e-3):
    """
    (方案A) GT-prior matching instead of uniform.
    batch-level KL(p_pred_fg || p_prior_fg)

    - p_pred_fg: FG class distribution averaged over (B,Q)
    - p_prior_fg: normalized EMA of GT counts (cls_count_ema, shape=(8,))
    """
    # Pred FG distribution from queries
    q = torch.softmax(class_logits, dim=-1)           # (B,Q,C)
    p = q[..., 1:].mean(dim=(0, 1))                   # (C-1,)
    p = (p + eps) / (p.sum() + eps * p.numel())

    # GT prior (EMA) over FG classes
    prior = cls_count_ema.detach().float().clamp_min(0.0)
    prior = prior + eps
    prior = prior / prior.sum().clamp_min(1e-6)

    return torch.sum(p * (p.log() - prior.log()))

def reg_pix_div(semantic_logits2, label_ce, fb_logit):
    """
    pixel-level KL(p || U) over predicted FG class distribution
    Priority:
      1) Stage1 predicted FG pixels (fb_logit > 0)
      2) GT FG pixels (fallback)
    """
    device = semantic_logits2.device

    # FG class probabilities (exclude background)
    prob = torch.softmax(semantic_logits2, dim=1)[:, 1:]  # (B,8,H,W)

    # 1) Stage1 predicted FG
    stage1_fg = (fb_logit.squeeze(1) > 0)   # (B,H,W)
    if stage1_fg.any():
        p_fg = prob.permute(0, 2, 3, 1)[stage1_fg].mean(dim=0)

    else:
        # 2) Fallback to GT FG
        gt_fg = (label_ce > 0)
        if not gt_fg.any():
            return torch.tensor(0.0, device=device)

        p_fg = prob.permute(0, 2, 3, 1)[gt_fg].mean(dim=0)

    # Normalize & KL to uniform
    p_fg = p_fg / p_fg.sum().clamp_min(1e-6)
    u = torch.full_like(p_fg, 1.0 / p_fg.numel())

    return torch.sum(p_fg * (p_fg.clamp_min(1e-6).log() - u.log()))

def reg_q_div(class_logits, margin=0.9):
    """
    讓 queries 的 FG class distribution 不要彼此太像（cosine sim）
    """
    probs = torch.softmax(class_logits, dim=-1)[..., 1:]  # (B,Q,8)
    q = probs.mean(dim=0)                                 # (Q,8)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sim = torch.matmul(qn, qn.t())                        # (Q,Q)
    eye = torch.eye(sim.size(0), device=sim.device)
    sim_off = sim * (1.0 - eye)
    return torch.relu(sim_off - margin).pow(2).mean()



def reg_fg_inflation(pred_fg_ratio: float, gt_fg_ratio: float, device, ratio_thr: float = 3.0, min_gt: float = 0.005, semantic_logits2 = None):
    """
    (方案A) 前景膨脹抑制：當 pred_fg_ratio 明顯大於 GT 時，施加懲罰。
    penalty = ReLU(pred_fg_ratio - ratio_thr * max(gt_fg_ratio, min_gt))^2
    """
    prob = torch.softmax(semantic_logits2, dim=1)
    pred_fg_soft = (1.0 - prob[:, 0]).mean()   # differentiable
    target = ratio_thr * max(gt_fg_ratio, min_gt)
    #excess = max(0.0, pred_fg_ratio - target)
    loss_infl = torch.relu(pred_fg_soft - target).pow(2)
    return loss_infl
# -------------------------
# Wandb image logger (every N iters)
# -------------------------
@torch.no_grad()
def wandb_log_image(wandb, image_batch, pred_mask, gt_mask, class_labels, caption):
    """
    image_batch: (B,1/3,H,W) tensor
    pred_mask/gt_mask: (H,W) numpy or tensor
    """
    img = image_batch[0, 0].detach().cpu().float().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    pm = pred_mask.detach().cpu().numpy() if torch.is_tensor(pred_mask) else pred_mask
    gm = gt_mask.detach().cpu().numpy() if torch.is_tensor(gt_mask) else gt_mask
    wandb.log({
        "train/pred_vs_gt": wandb.Image(
            img,
            masks={
                "predictions": {"mask_data": pm, "class_labels": class_labels},
                "ground_truth": {"mask_data": gm, "class_labels": class_labels},
            },
            caption=caption
        )
    })

# =========================================================
# ✅ Drop-in skeleton for your trainer loop core
# =========================================================
def training_step_core(args, model, dice_loss_fn, optimizer, writer, wandb,
                       class_labels,
                       iter_num,
                       phase_sched,
                       controller,
                       tau_ema_state,
                       cls_count_ema,
                       trainable_params,
                       epoch_num,
                       image_batch,
                       label_batch,
                       class_usage_ema,
                       class_usage_mom):
    """
    你在 trainer_synapse 裡每個 batch 呼叫一次即可。
    回傳更新後狀態：iter_num, tau_ema_state, cls_count_ema
    """
    model.train()
    image_batch = image_batch.cuda()
    label_batch = label_batch.cuda()

    # label -> (B,H,W)
    if label_batch.dim() == 4 and label_batch.size(1) == 1:
        label_ce = label_batch.squeeze(1).long()
    else:
        label_ce = label_batch.long()

    # -------------------------
    # forward (你的 model 回傳 class_logits, masks(list))
    # -------------------------
    class_logits, masks = model(image_batch)
    mask_logits = masks[-1]
    mask_logits = F.interpolate(mask_logits, size=(args.img_size, args.img_size),
                                mode="bilinear", align_corners=False)
    mask_probs = torch.sigmoid(mask_logits)  # (B,Q,H,W)
    
    semantic_logits_raw, den_raw = build_semantic_logits(
        class_logits, mask_probs, den_clamp=0.2, T_cls=4.0, epoch_num=epoch_num
    )

    # -------------------------
    # controller apply (safe)
    # -------------------------
    semantic_logits2 = controller.apply(semantic_logits_raw)

    pred_fg_soft, mean_margin = compute_pred_fg_soft_and_mean_margin(semantic_logits2)
    underseg_flag, underseg_ratio = is_underseg(gt_fg_ratio, pred_fg_ratio, mean_margin)


    # compute pred stats (after bias)
    pred, pred_fg_ratio, gt_fg_ratio, pred_unique = compute_pred_stats(semantic_logits2, label_ce)
    dom_cls, dom_ratio = fg_hist_dom(pred, args.num_classes)

    # semantic prediction
    #pred_cls = semantic_logits2.argmax(1) == pred           # (B,H,W)
    pred_fg_mask = (pred > 0)

    # 更新 class usage EMA
    class_usage_ema = update_class_usage_ema(
        class_usage_ema,
        pred,
        pred_fg_mask,
        mom=class_usage_mom
    )

    # 計算 num_fg_classes
    num_fg_classes, class_usage_snapshot = compute_num_fg_classes(
        class_usage_ema,
        min_ratio=0.005
    )

    # Phase3a -> Phase3b indicator-based gating (p3a_epochs as upper bound)

    # phase_sched.update_phase3_gate(
    #     epoch_num=epoch_num,
    #     pred_fg_ratio=pred_fg_ratio,
    #     gt_fg_ratio=gt_fg_ratio,
    #     dom_ratio=dom_ratio,
    #     num_fg_classes=num_fg_classes,
    # )




    # -------------------------
    # Phase weights
    # -------------------------
    ph, w = phase_sched.weights(epoch_num)
    
    # controller step (extreme-only + cooldown)
    delta, q50_ema, q50_now = controller.step_from_logits(semantic_logits2, pred_fg_ratio=pred_fg_ratio, phase=ph)

    # NOTE: 你要讓「當次 loss」用更新後 bias 嗎？
    # 這裡採「下一 iter 生效」更穩（避免同 iter 震盪）
    # 如果你想當次生效，把上面 apply/compute 移到 step 後再 apply 一次。

    

    # -------------------------
    # Stage1 loss (fg/bg)
    # -------------------------
    loss_fg_bg, fb_logit, stage1_fg_ratio, tau_ema_state = stage1_fg_bg_loss(
        semantic_logits2, label_ce,
        target_ratio_mult=1.5,
        tau_ema_state=tau_ema_state,
        tau_mom=0.98,
        posw_cap=80.0
    )

    # -------------------------
    # Stage2 loss (fg class on GT fg)
    # -------------------------
    _, p2_epochs, p3_dice_epochs, _, _ = phase_sched.get_phase_epoch()
    energy_alpha, energy_wmax, enable_energy_reweight = stage2_energy_paremeter(epoch_num=epoch_num, ph=ph, p2_epochs=p2_epochs)
    if epoch_num in(p2_epochs, p2_epochs+1, p2_epochs+2) and iter_num%300 == 0:
        logging.info(
            f"[P3A-warmup] Epochs={epoch_num} "
            f"enable={enable_energy_reweight} "
            f"alpha={energy_alpha:.2f} wmax={energy_wmax:.2f}"
        )
    loss_fg_cls, cls_count_ema = stage2_fg_cls_loss(
        semantic_logits2,
        label_ce,
        cls_count_ema=cls_count_ema,
        cls_mom=0.99,
        iter_num=iter_num,
        class_usage_ema=class_usage_ema,
        enable_energy_reweight=enable_energy_reweight,
        energy_alpha=energy_alpha,
        energy_wmin=0.5,
        energy_wmax=energy_wmax,
    )

    # -------------------------
    # Dice (small)
    # -------------------------
        if epoch_num < p3_dice_epochs:
            loss_dice = dice_softmax_loss(dice_loss_fn, semantic_logits2, label_ce)
        else:
            loss_dice = dice_per_class_gt_present(semantic_logits2, label_ce, num_classes=args.num_classes, min_pixels=20, class_boost={2: 1.5})
            w["dice"] = 0.25

    # -------------------------
    # Regularizers (phase-gated)
    # -------------------------
    div_mode = gate_diversity(
        ph=ph,
        gt_fg_ratio=gt_fg_ratio,
        pred_fg_ratio=pred_fg_ratio,
        stage1_fg_ratio=stage1_fg_ratio,
        num_fg_classes=num_fg_classes,
        dom_ratio=dom_ratio,
        q10_ema=float(q50_ema) if q50_ema is not None else 0.0
    )

    # (方案A) 3-state gating (only for ph==32)
    # div_mode: 0=normal, 1=collapse (enable cls_div/q_div), 2=inflation (enable infl only)
    if div_mode != 1:
        w["cls_div"] = 0.0
        w["pix_div"] = 0.0
        w["q_div"]   = 0.0
    

    # (方案A+) Auto-boost inflation penalty only when it is enabled (div_mode==2)
    if div_mode == 2 and w.get("infl", 0.0) > 0:
        w["infl"] = float(w["infl"]) * float(phase_sched.infl_scale(pred_fg_ratio=pred_fg_ratio, gt_fg_ratio=gt_fg_ratio))

    # --- under-seg guard: 只在 under-seg 才啟動 ---
    loss_recall_floor = torch.tensor(0.0, device=semantic_logits2.device)
    if underseg_flag and ph in (31, 32):
        # (1) 放鬆煞車：快速釋放一些 bg bias（避免 locked-in）
        with torch.no_grad():
            severity = min(1.0, max(0.0, (0.30 - underseg_ratio) / 0.30))  # ratio_us = Pred/GT
            delta = 0.05 + 0.10 * severity  # 0.05~0.15
            controller.bias = (controller.bias - delta).clamp(0.0, controller.max_bias)

        # (2) 下限補償（soft）：pred_fg_soft 至少要到 k * GT_fg
        k = 0.30
        target_floor = torch.tensor(k * gt_fg_ratio, device=semantic_logits2.device)
        pred_soft_t = torch.tensor(pred_fg_soft, device=semantic_logits2.device)
        loss_recall_floor = F.relu(target_floor - pred_soft_t).pow(2)

        # 這個權重建議很小，因為它是「只補下限」
        w.setdefault("recall", 0.0)
        w["recall"] = 0.05
    else:
        w.setdefault("recall", 0.0)

    loss_area = reg_area_pen(mask_probs) if w["area"] > 0 else torch.tensor(0.0, device=semantic_logits2.device)
    loss_ovlp = reg_overlap_pen(mask_probs, den_raw, thr=2.0) if w["ovlp"] > 0 else torch.tensor(0.0, device=semantic_logits2.device)
    loss_cls_div = reg_cls_div(class_logits, cls_count_ema) if w["cls_div"] > 0 else torch.tensor(0.0, device=semantic_logits2.device)
    loss_pix_div = torch.tensor(0.0, device=semantic_logits2.device)  # (方案A) pix_div removed
    loss_q_div = reg_q_div(class_logits, margin=0.9) if w["q_div"] > 0 else torch.tensor(0.0, device=semantic_logits2.device)
    loss_infl = reg_fg_inflation(pred_fg_ratio, gt_fg_ratio, semantic_logits2.device, semantic_logits2=semantic_logits2) if w["infl"] > 0 else torch.tensor(0.0, device=semantic_logits2.device)
    loss_mbg = loss_bg_margin_on_gt_bg(semantic_logits2, label_ce,  margin=0.5, power=2.0, use_softplus=False)

    # pseudo（Phase4 才開；這裡留鉤子，你可直接接你原本 pseudo 寫法）
    loss_pseudo = torch.tensor(0.0, device=semantic_logits2.device)

    # -------------------------
    # total loss
    # -------------------------
    loss = (
        w["fb"]   * loss_fg_bg +
        w["cls"]  * loss_fg_cls +
        w["dice"] * loss_dice +
        w["area"] * loss_area +
        w["ovlp"] * loss_ovlp +
        w["cls_div"] * loss_cls_div +
        w["pix_div"] * loss_pix_div +
        w["q_div"] * loss_q_div +
        w["infl"] * loss_infl +
        w["pseudo"] * loss_pseudo +
        w["bgm"] * loss_mbg +
        w["recall"] * loss_recall_floor
    )
    loss_dict = {
        "fb":       loss_fg_bg,
        "cls":      loss_fg_cls,
        "dice":     loss_dice,
        "area":     loss_area,
        "ovlp":     loss_ovlp,
        "cls_div":  loss_cls_div,
        "pix_div":  loss_pix_div,
        "q_div":    loss_q_div,
        "infl":     loss_infl,
        "pseudo":   loss_pseudo,
        "bgm":      loss_mbg,
        "recall":   loss_recall_floor,
    }

    

    # -------------------------
    # backward / step
    # -------------------------
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # grad norm debug
    if iter_num % 100 == 0:
        with torch.no_grad():
            norms = [p.grad.detach().norm(2) for p in trainable_params if p.grad is not None]
            total_norm = torch.norm(torch.stack(norms), 2).item() if norms else 0.0
        print("DEBUG grad_norm (before clip):", total_norm)

    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()

    # -------------------------
    # logs
    # -------------------------
    if iter_num % 100 == 0:
        q01, q05, q10, q50, m = log_margin_quantiles(semantic_logits2)
        '''
        print("===== DEBUG (Phase Scheduler + Safe Controller) =====")
        print(f"[iter] {iter_num}, [epoch] {epoch_num}")
        print(f"[phase] {ph} | controller_delta={delta:+.3f} | bias={float(controller.bias.item()):.3f} | q10_now={q10_now:+.3f} | q10_ema={q10_ema:+.3f}")
        print(f"GT_fg_ratio: {gt_fg_ratio:.4f} | Pred_fg_ratio(argmax): {pred_fg_ratio:.4f} | Stage1_fg_ratio: {stage1_fg_ratio:.4f}")
        print(f"pred_unique: {pred_unique[:20]}")
        print(f"[FG hist] dom_cls={dom_cls}, dom_ratio={dom_ratio:.3f}")
        print(f"[margin] q01={q01:+.3f} q05={q05:+.3f} q10={q10:+.3f} q50={q50:+.3f} mean={m:+.3f}")
        print(f"[loss] total={loss.item():.4f} | fg_bg={loss_fg_bg.item():.4f} | fg_cls={loss_fg_cls.item():.4f} | dice={loss_dice.item():.4f}")
        if ph >= 2:
            print(f"[reg] area={float(loss_area.item() if torch.is_tensor(loss_area) else loss_area):.6f} | ovlp={loss_ovlp.item():.6f}")
        if ph == 2:
            print(f"[div] cls_div={loss_cls_div.item():.6f} | pix_div={loss_pix_div.item():.6f}")
        if ph == 3:
            print(f"[div] cls_div={loss_cls_div.item():.6f} | pix_div={loss_pix_div.item():.6f} | q_div={loss_q_div.item():.6f}")
        if ph == 4:
            print(f"[div] cls_div={loss_cls_div.item():.6f} | pix_div={loss_pix_div.item():.6f} | q_div={loss_q_div.item():.6f} | pseudo={loss_pseudo.item():.6f}")
        print("=======check 每個 class 分配能量=======")
        print(f"[FG CLASS STATE] num_fg_classes={num_fg_classes}")
        for c in range(1, args.num_classes):
            print(f"  class {c}: ema_ratio={class_usage_snapshot[c]:.4f}")
        '''
        logging.info("===== DEBUG (Phase Scheduler + Safe Controller) =====")
        logging.info(f"[ITER] iter={iter_num} epoch={epoch_num}")

        logging.info(
            f"[PHASE] phase={ph} "
            f"controller_delta={delta:+.3f} "
            f"bias={float(controller.bias.item()):.3f} "
            f"q50_now={q50_now:+.3f} "
            f"q50_ema={q50_ema:+.3f}"
        )

        logging.info(
            f"[FG_RATIO] GT_fg={gt_fg_ratio:.4f} "
            f"Pred_fg(argmax)={pred_fg_ratio:.4f} "
            f"Stage1_fg={stage1_fg_ratio:.4f}"
        )

        logging.info(f"[PRED] pred_unique={pred_unique[:20]}")
        logging.info(f"[FG_HIST] dom_cls={dom_cls} dom_ratio={dom_ratio:.3f}")

        logging.info(
            f"[MARGIN] q01={q01:+.3f} q05={q05:+.3f} "
            f"q10={q10:+.3f} q50={q50:+.3f} mean={m:+.3f}"
        )

        logging.info(
            f"[LOSS] total={loss.item():.4f} "
            f"fg_bg={loss_fg_bg.item():.4f} "
            f"fg_cls={loss_fg_cls.item():.4f} "
            f"dice={loss_dice.item():.4f}"
        )

        if ph >= 2:
            logging.info(
                f"[REG] area={float(loss_area.item() if torch.is_tensor(loss_area) else loss_area):.6f} "
                f"ovlp={loss_ovlp.item():.6f}"
            )

        if ph == 2:
            logging.info(
                f"[DIV] cls_div={loss_cls_div.item():.6f} "
                f"pix_div={loss_pix_div.item():.6f}"
            )

        if ph in (31, 32):
            logging.info(
                f"[DIV] cls_div={loss_cls_div.item():.6f} "
                f"pix_div={loss_pix_div.item():.6f} "
                f"q_div={loss_q_div.item():.6f} "
                f"infl={loss_infl.item():.6f} "
                f"mbg={loss_mbg.item():.6f} "
                f"div_mode={int(div_mode)}"
                f"train/underseg_flag"={int(underseg_flag):.6f}"
                f"train/underseg_ratio"={float(underseg_ratio):.6f}"
                f"train/pred_fg_soft"={float(pred_fg_soft):.6f}"
                f"train/mean_margin"={float(mean_margin):.6f}"
                f"train/loss_recall_floor"={float(loss_recall_floor.item()):.6f}"
            )

        if ph == 4:
            logging.info(
                f"[DIV] cls_div={loss_cls_div.item():.6f} "
                f"pix_div={loss_pix_div.item():.6f} "
                f"q_div={loss_q_div.item():.6f} "
                f"pseudo={loss_pseudo.item():.6f}"
            )
        log_loss_contributions(w, loss_dict, loss)

        logging.info("======= CHECK FG CLASS ENERGY =======")
        logging.info(f"[FG_CLASS_STATE] num_fg_classes={num_fg_classes}")
        for c in range(1, args.num_classes):
            logging.info(f"[FG_CLASS] class={c} ema_ratio={class_usage_snapshot[c]:.4f}")

        with torch.no_grad():
        # present 類（這個 batch GT 有出現的 organ）
            gt_fg = (label_ce > 0)
            if gt_fg.any():
                y = (label_ce[gt_fg] - 1).long()
                device = "cuda"
                counts = torch.bincount(y, minlength=8).float().to(device)
                present = (counts > 0).nonzero().flatten().tolist()
                e = class_usage_ema[1:9].detach().cpu().tolist()
                logging.info(f"[A-WEIGHT] GT_present_cls={present} usage_e={[round(e[i],4) for i in present]}")

        
        
    # tensorboard
    writer.add_scalar("train/phase", ph, iter_num)
    writer.add_scalar("train/loss_total", loss.item(), iter_num)
    writer.add_scalar("train/loss_fg_bg", loss_fg_bg.item(), iter_num)
    writer.add_scalar("train/loss_fg_cls", loss_fg_cls.item(), iter_num)
    writer.add_scalar("train/loss_dice", loss_dice.item(), iter_num)
    writer.add_scalar("train/pred_fg_ratio", pred_fg_ratio, iter_num)
    writer.add_scalar("train/stage1_fg_ratio", stage1_fg_ratio, iter_num)
    writer.add_scalar("train/bg_bias", float(controller.bias.item()), iter_num)

    # wandb scalar
    wandb.log({
        "train/phase": ph,
        "train/loss_total": loss.item(),
        "train/loss_fg_bg": loss_fg_bg.item(),
        "train/loss_fg_cls": loss_fg_cls.item(),
        "train/loss_dice": loss_dice.item(),
        "train/pred_fg_ratio": pred_fg_ratio,
        "train/stage1_fg_ratio": stage1_fg_ratio,
        "train/bg_bias": float(controller.bias.item()),
        "train/controller_delta": delta,
    })

    # ✅ wandb image every 100 iters
    if iter_num % 100 == 0:
        wandb_log_image(
            wandb,
            image_batch,
            pred_mask=pred[0],
            gt_mask=label_ce[0],
            class_labels=class_labels,
            caption=f"Epoch {epoch_num} Iter {iter_num} | phase={ph} | pred_fg={pred_fg_ratio:.3f}"
        )

    iter_num += 1
    return iter_num, tau_ema_state, cls_count_ema, class_usage_ema

    #=======================================================================================

def worker_init_fn(worker_id: int):
    random.seed(GLOBAL_WORKER_SEED + worker_id)

def _to_label_ce(label_batch: torch.Tensor) -> torch.Tensor:
    # label_batch: (B,1,H,W) or (B,H,W)
    if label_batch.dim() == 4 and label_batch.size(1) == 1:
        label_ce = label_batch.squeeze(1)
    else:
        label_ce = label_batch
    return label_ce.long()

def _maybe_skip_slice(label_ce: torch.Tensor, bg_keep_prob: float, min_fg_ratio: float) -> bool:
    # return True => skip
    if (label_ce == 2).any():
        return False

    fg_cnt = (label_ce > 0).sum().item()
    if fg_cnt == 0:
        return (torch.rand(1, device=label_ce.device).item() > bg_keep_prob)

    fg_ratio = (label_ce > 0).float().mean().item()
    if fg_ratio < min_fg_ratio:
        return (torch.rand(1, device=label_ce.device).item() > 0.2)

    return False

@torch.no_grad()
def update_class_usage_ema(
    class_usage_ema,
    pred,          # (B,H,W) semantic argmax
    pred_fg_mask,      # (B,H,W) bool
    mom=0.95
):
    """
    使用 prediction（不是 GT）更新 class 使用率 EMA
    """
    fg_pixels = pred[pred_fg_mask]

    if fg_pixels.numel() == 0:
        return class_usage_ema

    counts = torch.bincount(
        fg_pixels,
        minlength=class_usage_ema.numel()
    ).float()

    counts[0] = 0.0  # 排除 background

    ratios = counts / counts.sum().clamp_min(1e-6)
    class_usage_ema = mom * class_usage_ema + (1 - mom) * ratios
    return class_usage_ema

@torch.no_grad()
def compute_num_fg_classes(
    class_usage_ema,
    min_ratio=0.005,      # 0.5%（保留 Gallbladder）
    max_ratio=0.90        # 防止單一器官壟斷
):
    """
    根據 EMA 判斷「穩定存在的前景類別數」
    """
    valid = (
        (class_usage_ema > min_ratio) &
        (class_usage_ema < max_ratio)
    )

    valid[0] = False  # background 不算
    num_fg_classes = int(valid.sum().item())

    return num_fg_classes, class_usage_ema.clone()

def gate_q_div(
    num_fg_classes,
    dom_ratio,
    q10_ema,
    min_classes=4,        # Liver + Kidney + Pancreas + Gallbladder
    max_dom_ratio=0.85,
    q10_range=(-0.3, 0.3)
):
    """
    判斷 Phase3 是否啟用 q_div
    """
    if num_fg_classes < min_classes:
        return False

    if dom_ratio > max_dom_ratio:
        return False

    if not (q10_range[0] <= q10_ema <= q10_range[1]):
        return False

    return True

def gate_diversity(
    *,
    ph: int,
    gt_fg_ratio: float,
    pred_fg_ratio: float,
    stage1_fg_ratio: float,
    num_fg_classes: int,
    dom_ratio: float,
    q10_ema: float,
    # thresholds (可微調)
    min_gt_fg: float = 0.005,
    inflation_ratio_thr: float = 3.0,
    inflation_pred_abs: float = 0.20,
    collapse_min_classes: int = 3,
    collapse_dom_thr: float = 0.90,
) -> int:
    """
    (方案A) Phase3b gating: return mode
      0 = normal (div off)
      1 = semantic-collapse (enable anti-collapse only: cls_div KL-to-prior + tiny q_div)
      2 = fg-inflation (div must be OFF; enable inflation penalty)

    Notes:
    - Phase3a(31)/Phase3b(32) 都會檢查「前景膨脹」(mode=2)。
    - 只有 Phase3b(32) 才會檢查「語義塌陷」(mode=1)。
    - 「前景膨脹」優先級最高：一票否決 div，避免把錯誤 FG 做多類別均勻化。
    """
    # Allow inflation detection in both Phase3a (31) and Phase3b (32).
    # Collapse (mode=1) remains Phase3b-only.
    if ph not in (31, 32):
        return 0

    # ---------- Mode 2: FG inflation (highest priority) ----------
    ratio = pred_fg_ratio / (gt_fg_ratio + 1e-6) if gt_fg_ratio > 0 else (1e9 if pred_fg_ratio > 0 else 0.0)
    if ratio >= 3.0:
        return 2

    # ---------- Mode 1: semantic collapse ----------
    # Only consider collapse when there is enough GT FG signal; otherwise skip.
    if gt_fg_ratio < min_gt_fg:
        return 0

    collapse = (num_fg_classes < collapse_min_classes) or (dom_ratio >= collapse_dom_thr)

    # stage1 too low => fg detection unstable; do not enable div (leave to controller/CE)
    if stage1_fg_ratio < 0.03:
        return 0

    return 1 if collapse else 0


@torch.no_grad()
def validate_synapse(
    model: torch.nn.Module,
    valloader,
    args,
    writer=None,
    epoch_num: int = 0,
    device: str = "cuda",
    do_wandb: bool = True,
):
    """
    Validation: slice-level, GT-only dice (你的原本邏輯不變)
    - 逐 slice 計算：每一類別若該 slice GT=0 則略過
    - 每 slice 的 dice = 有 GT 的類別 dice 平均
    - 最終 mean dice = 所有有 GT 的 slices 平均
    """
    model.eval()

    dice_per_class = {c: [] for c in range(1, args.num_classes)}
    dice_per_slice = []

    for _, val_batch in tqdm(enumerate(valloader), total=len(valloader),
                             desc=f"Validating Epoch {epoch_num}"):

        val_img = val_batch["image"].to(device)
        val_label = val_batch["label"].to(device)

        # --- normalize label to (B,H,W) long ---
        if val_label.dim() == 4 and val_label.size(1) == 1:
            val_label_ce = val_label.squeeze(1)
        elif val_label.dim() == 3:
            val_label_ce = val_label
        elif val_label.dim() == 5:
            val_label_ce = val_label.squeeze(2)
            if val_label_ce.dim() == 4 and val_label_ce.size(1) == 1:
                val_label_ce = val_label_ce.squeeze(1)
        else:
            val_label_ce = val_label
        val_label_ce = val_label_ce.long()

        # --- forward ---
        if args.add_decoder:
            # eval 回 (None, prob) 你原本的處理保留
            _, val_out = model(val_img)
            val_out = val_out + 1e-7
            val_out = val_out / (val_out.sum(dim=1, keepdim=True).clamp_min(1e-6))
        else:
            val_out = model(val_img)
            val_out = torch.softmax(val_out, dim=1)

        if val_out.dim() == 3:
            val_out = val_out.unsqueeze(0)

        pred = val_out.argmax(dim=1)  # (B,H,W)

        # --- dice ---
        dices_this_slice = []
        for cls in range(1, args.num_classes):
            gt = (val_label_ce == cls)
            if gt.sum() == 0:
                continue
            pd = (pred == cls)
            inter = (gt & pd).sum().float()
            dice = (2 * inter) / (gt.sum() + pd.sum() + 1e-5)

            dice_per_class[cls].append(float(dice.item()))
            dices_this_slice.append(float(dice.item()))

        if len(dices_this_slice) > 0:
            dice_per_slice.append(sum(dices_this_slice) / len(dices_this_slice))

    # --- aggregate ---
    class_mean = {}
    for cls, dices in dice_per_class.items():
        class_mean[cls] = (float(np.mean(dices)) if len(dices) > 0 else None)

    avg_val_dice = float(np.mean(dice_per_slice)) if len(dice_per_slice) > 0 else 0.0

    # --- logging / wandb / tensorboard ---
    print("===== Validation Per-class Dice =====")
    for cls in range(1, args.num_classes):
        m = class_mean[cls]
        if m is None:
            print(f"class {cls}: N/A (n=0)")
        else:
            print(f"class {cls}: {m:.4f} (n={len(dice_per_class[cls])})")

    print(f"===== Validation mean dice (slice-level, GT-only) = {avg_val_dice:.6f} =====")

    if do_wandb:
        wandb.log({
            "val/mean_dice_gt_only": avg_val_dice,
            **{f"val/dice_class_{cls}": (class_mean[cls] if class_mean[cls] is not None else 0.0)
               for cls in range(1, args.num_classes)},
            "epoch": epoch_num
        })

    if writer is not None:
        writer.add_scalar("info/val_dice", avg_val_dice, epoch_num)

    logging.info("Epoch %d : Validation Mean Dice: %f", epoch_num, avg_val_dice)

    return avg_val_dice, class_mean

def loss_bg_margin_on_gt_bg(
    semantic_logits: torch.Tensor,
    label_ce: torch.Tensor,
    margin: float = 0.5,
    power: float = 2.0,
    use_softplus: bool = False,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Penalize cases where any FG logit exceeds BG logit on GT=background pixels.

    Args:
        semantic_logits: (B, C, H, W) raw logits. C includes background at index 0.
        label_ce:        (B, H, W) integer labels, 0 = background.
        margin:          desired safety margin: enforce (fg_max - bg) <= -margin on GT-bg pixels.
                         i.e., penalize relu((fg_max - bg) + margin).
        power:           penalty power (2.0 => squared hinge).
        use_softplus:    if True, use softplus instead of ReLU for smoother gradients.
        reduction:       "mean" or "sum" (only applied over GT-bg pixels).
        eps:             numerical safety for empty-mask handling.

    Returns:
        A scalar tensor loss.
    """
    assert semantic_logits.dim() == 4, "semantic_logits must be (B,C,H,W)"
    assert label_ce.dim() == 3, "label_ce must be (B,H,W)"
    assert semantic_logits.size(0) == label_ce.size(0)
    assert semantic_logits.size(2) == label_ce.size(1)
    assert semantic_logits.size(3) == label_ce.size(2)
    assert semantic_logits.size(1) >= 2, "Need at least 2 classes (bg + fg)."

    # BG logit: (B,H,W)
    bg = semantic_logits[:, 0]

    # Max FG logit over classes 1..C-1: (B,H,W)
    fg_max = semantic_logits[:, 1:].amax(dim=1)

    # m > 0 means some FG is beating BG at that pixel
    m = fg_max - bg  # (B,H,W)

    # GT background mask
    gt_bg = (label_ce == 0)

    if gt_bg.any():
        # Penalize when FG wins BG (or is too close) on GT-bg pixels:
        # want m <= -margin  -> penalize (m + margin) > 0
        x = m[gt_bg] + float(margin)

        if use_softplus:
            # smooth hinge: softplus(x)
            pen = F.softplus(x)
        else:
            # hinge: relu(x)
            pen = F.relu(x)

        if power != 1.0:
            pen = pen.pow(float(power))

        if reduction == "mean":
            return pen.mean()
        elif reduction == "sum":
            return pen.sum()
        else:
            raise ValueError(f"Unsupported reduction={reduction}")
    else:
        # No GT-bg pixels in this batch (rare but possible); return 0 with correct device/dtype
        return semantic_logits.sum() * 0.0
    
def log_loss_contributions(
    w: dict,
    losses: dict,
    total_loss: torch.Tensor,
    prefix: str = "[LOSS_BREAKDOWN]",
    eps: float = 1e-12,
):
    """
    Print weighted loss contributions and percentage of total loss.

    Args:
        w:        dict of weights, e.g. w["cls"], w["dice"], ...
        losses:   dict of raw losses, keys must match w keys (subset ok)
        total_loss: scalar tensor (already summed with weights)
        prefix:   log prefix
        eps:      numerical stability
    """
    total_val = float(total_loss.detach().item())
    denom = abs(total_val) + eps

    logging.info(prefix)
    for k, raw in losses.items():
        weight = float(w.get(k, 0.0))
        raw_val = float(raw.detach().item()) if raw is not None else 0.0
        contrib = weight * raw_val
        ratio = 100.0 * contrib / denom
        logging.info(
            f"  {k:>8s}: "
            f"raw={raw_val:9.6f} | "
            f"w={weight:9.2e} | "
            f"w*raw={contrib:9.6f} | "
            f"{ratio:6.2f}%"
        )

    logging.info(f"  {'TOTAL':>8s}: {total_val:8.6f} (100.00%)")

def dice_per_class_gt_present(
    semantic_logits2,
    target,
    eps=1e-6,
    min_pixels=20,
    class_boost=None,   # e.g. {2: 1.5}
):
    probs = torch.softmax(semantic_logits2, dim=1)
    B, C, H, W = probs.shape
    device = probs.device

    losses = []

    for cls in range(1, C):
        gt_mask = (target == cls)
        if gt_mask.sum() < min_pixels:
            continue

        pred = probs[:, cls]
        gt = gt_mask.float()

        inter = (pred * gt).sum()
        denom = pred.sum() + gt.sum()

        dice = (2.0 * inter + eps) / (denom + eps)
        l = 1.0 - dice

        if class_boost is not None and cls in class_boost:
            l = l * float(class_boost[cls])

        losses.append(l)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def compute_pred_fg_soft_and_mean_margin(semantic_logits2):
    prob = torch.softmax(semantic_logits2, dim=1)
    pred_fg_soft = float((1.0 - prob[:, 0]).mean().item())

    bg = semantic_logits2[:, 0]
    fg = semantic_logits2[:, 1:].amax(dim=1)
    mean_margin = float((fg - bg).mean().item())
    return pred_fg_soft, mean_margin

def is_underseg(gt_fg_ratio, pred_fg_ratio, mean_margin,
                gt_min=0.03, ratio_thr=0.30, margin_thr=-0.40):
    if gt_fg_ratio < gt_min:
        return False, 999.0
    ratio = pred_fg_ratio / max(gt_fg_ratio, 1e-6)
    flag = (ratio < ratio_thr) and (mean_margin < margin_thr)
    return flag, ratio

'''



# ----------------------------
# Helpers
# ----------------------------




def _poly_lr(base_lr: float, iter_num: int, max_iterations: int, min_lr_ratio: float = 0.05, power: float = 0.9) -> float:
    min_lr = base_lr * min_lr_ratio
    t = min(1.0, max(0.0, iter_num / max_iterations))
    return min_lr + (base_lr - min_lr) * (1.0 - t) ** power

@torch.no_grad()
def _update_ema_scalar(old: Optional[torch.Tensor], new: torch.Tensor, mom: float) -> torch.Tensor:
    return new if old is None else (mom * old + (1.0 - mom) * new)

def _logsumexp_semantic_logits(
    class_logits: torch.Tensor,     # (B,Q,C)
    mask_probs: torch.Tensor,       # (B,Q,H,W)
    den_raw: torch.Tensor,          # (B,H,W) = sum_q mask_probs
    T_cls: float = 4.0,
) -> torch.Tensor:
    """
    semantic_logits_raw[c,h,w] = log( sum_q exp(class_logit[q,c]) * mask_prob[q,h,w] ) - log( sum_q mask_prob[q,h,w] )
    以 logsumexp 方式做，數值更穩。
    """
    # temperature on class logits to reduce extreme domination
    cls = class_logits / T_cls                                # (B,Q,C)
    log_mask = torch.log(mask_probs.clamp_min(1e-6))          # (B,Q,H,W)
    log_den  = torch.log(den_raw.clamp_min(1e-6)).unsqueeze(1)  # (B,1,H,W)

    # (B,Q,C,1,1) + (B,Q,1,H,W) -> (B,Q,C,H,W) log terms
    logits = torch.logsumexp(
        cls.unsqueeze(-1).unsqueeze(-1) + log_mask.unsqueeze(2),
        dim=1
    ) - log_den                                               # (B,C,H,W)
    return logits

@torch.no_grad()
def _bg_bias_controller(
    semantic_logits_raw: torch.Tensor,     # (B,C,H,W)
    bg_bias_ema: torch.Tensor,             # scalar tensor
    q10_ema: Optional[torch.Tensor],
    iter_num: int,
    q10_target: float = -0.2,              # want q10(fg-bg) <= -0.2
    q10_mom: float = 0.95,
    gain: float = 0.20,
    step_clip: float = 0.12,
    ema_w: float = 0.08,
    bias_min: float = 0.0,
    bias_max: float = 20.0,
    debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    """
    用 (fg_max - bg) 的分位數 q10 做回授控制，調整 bg bias，抑制全前景崩潰。
    回傳：semantic_logits_biased, new_bg_bias_ema, new_q10_ema, log_dict
    """
    # apply current bias
    bg0 = semantic_logits_raw[:, 0:1] + bg_bias_ema
    semantic_logits_biased = torch.cat([bg0, semantic_logits_raw[:, 1:]], dim=1)

    # margins
    bg = semantic_logits_biased[:, 0]               # (B,H,W)
    fg = semantic_logits_biased[:, 1:].amax(dim=1)  # (B,H,W)
    d  = (fg - bg).flatten()                        # (B*H*W,)

    m_q10 = torch.quantile(d, 0.10)
    q10_ema = m_q10 if q10_ema is None else (q10_mom * q10_ema + (1.0 - q10_mom) * m_q10)

    # want q10 <= target => err = q10 - target
    err = float(q10_ema.item() - q10_target)
    step_val = float(np.clip(err * gain, -step_clip, +step_clip))
    step = torch.tensor(step_val, device=semantic_logits_raw.device)

    new_bg_bias_ema = (bg_bias_ema + ema_w * step).clamp(bias_min, bias_max)

    # re-apply updated bias for downstream use
    bg0 = semantic_logits_raw[:, 0:1] + new_bg_bias_ema
    semantic_logits_biased = torch.cat([bg0, semantic_logits_raw[:, 1:]], dim=1)

    # fg ratio before/after (for logs)
    pre_fg  = float((semantic_logits_raw.argmax(1) > 0).float().mean().item())
    post_fg = float((semantic_logits_biased.argmax(1) > 0).float().mean().item())

    log_dict = {
        "bias/bg_bias_ema": float(new_bg_bias_ema.item()),
        "bias/q10_ema": float(q10_ema.item()),
        "bias/step": float(step.item()),
        "bias/fg_pre": pre_fg,
        "bias/fg_post": post_fg,
    }

    if debug:
        q01 = float(torch.quantile(d, 0.01).item())
        q05 = float(torch.quantile(d, 0.05).item())
        q50 = float(torch.quantile(d, 0.50).item())
        log_dict.update({"bias/q01": q01, "bias/q05": q05, "bias/q50": q50, "bias/d_mean": float(d.mean().item())})

    return semantic_logits_biased, new_bg_bias_ema, q10_ema, log_dict

@torch.no_grad()
def _calibrate_tau_from_target_ratio(
    diff: torch.Tensor,             # (B,1,H,W) fg_logit - bg_logit
    label_ce: torch.Tensor,         # (B,H,W)
    tau_ema: Optional[torch.Tensor],
    iter_num: int,
    target_mult: float = 1.5,
    target_min: float = 0.01,
    target_max: float = 0.12,
    tau_mom_base: float = 0.98,
    tau_mom_fast: float = 0.95,
    stage1_fg_ratio_ema: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    依 GT 前景比例決定 target_ratio，再用 diff 的 kthvalue 取 tau，
    使得 (diff > tau) 的比例大約落在 target_ratio。
    """
    gt_is_fg = (label_ce > 0).float().unsqueeze(1)
    gt_fg_ratio = gt_is_fg.mean().clamp(1e-4, 0.5)
    target_ratio = (gt_fg_ratio * target_mult).clamp(target_min, target_max)

    flat = diff.flatten()
    N = flat.numel()
    k = int((1.0 - float(target_ratio.item())) * N)
    k = max(0, min(N - 1, k))
    tau_batch = flat.kthvalue(k + 1).values

    if tau_ema is None:
        tau_ema = tau_batch
    else:
        # 如果 stage1 太寬鬆（>0.25）就讓 tau 更新更快一些
        mom = tau_mom_fast if (stage1_fg_ratio_ema is not None and stage1_fg_ratio_ema > 0.25 and iter_num % 10 == 0) else tau_mom_base
        tau_ema = mom * tau_ema + (1.0 - mom) * tau_batch

    return tau_ema, tau_batch, target_ratio, float(gt_fg_ratio.item())

def _compute_stage1_loss(
    semantic_logits: torch.Tensor,  # (B,C,H,W) (after bias)
    label_ce: torch.Tensor,         # (B,H,W)
    tau: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Stage1: fg vs bg
    使用 fb_logit = (fg_max - bg) - tau
    """
    gt_is_fg = (label_ce > 0).float().unsqueeze(1)
    bg_logit = semantic_logits[:, 0:1]
    fg_logit = semantic_logits[:, 1:].amax(1, keepdim=True)
    diff = fg_logit - bg_logit
    fb_logit = (diff - tau).clamp(-12, 12)

    with torch.no_grad():
        fg_frac = gt_is_fg.mean().clamp_min(1e-4)
        pos_weight = ((1.0 - fg_frac) / fg_frac).clamp(max=80.0)

    loss_fg_bg = F.binary_cross_entropy_with_logits(fb_logit, gt_is_fg, pos_weight=pos_weight)
    return loss_fg_bg, fb_logit, diff, float(pos_weight.item())

def _compute_stage2_fg_cls_loss(
    semantic_logits: torch.Tensor,  # (B,C,H,W) after bias
    label_ce: torch.Tensor,         # (B,H,W)
    cls_count_ema: torch.Tensor,    # (8,)
    cls_mom: float = 0.99,
    w_clip: Tuple[float, float] = (1.0, 10.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stage2: 只用 GT fg 像素做 8 類 CE（1..8）
    並做 fg-only reweight（用 EMA 的 inverse-freq）
    """
    gt_fg = (label_ce > 0)
    device = semantic_logits.device
    loss_fg_cls = torch.tensor(0.0, device=device)

    if not gt_fg.any():
        return loss_fg_cls, cls_count_ema

    fg_logits = semantic_logits[:, 1:]  # (B,8,H,W)
    y = (label_ce[gt_fg] - 1).long()    # (N_fg,) in 0..7
    logits_fg = fg_logits.permute(0, 2, 3, 1)[gt_fg]  # (N_fg, 8)

    counts = torch.bincount(y, minlength=8).float().to(device)  # (8,)
    with torch.no_grad():
        cls_count_ema = cls_count_ema * cls_mom + counts * (1.0 - cls_mom)
        w = (cls_count_ema.sum() / (cls_count_ema + 1.0)).clamp(w_clip[0], w_clip[1])
        w = w / w.mean().clamp_min(1e-6)

    loss_fg_cls = F.cross_entropy(logits_fg, y, weight=w)
    return loss_fg_cls, cls_count_ema

@torch.no_grad()
def _compute_pseudo_loss(
    semantic_logits: torch.Tensor,  # (B,C,H,W)
    fb_logit: torch.Tensor,         # (B,1,H,W) stage1 gate
    epoch_num: int,
    enable_pred_fg_cap: float = 0.60,
    conf_thr: float = 0.92,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    pseudo: 高信心 fg 像素，用 label smoothing 的 CE。
    """
    device = semantic_logits.device
    prob = torch.softmax(semantic_logits, dim=1)
    pred = prob.argmax(dim=1)
    pred_fg_ratio = float((pred > 0).float().mean().item())

    enable = (pred_fg_ratio < enable_pred_fg_cap and epoch_num > 1)
    loss_pseudo = torch.tensor(0.0, device=device)

    if not enable:
        return loss_pseudo, {"pseudo/enabled": 0.0, "pseudo/pred_fg_ratio": pred_fg_ratio, "pseudo/n": 0.0}

    p_fg = 1.0 - prob[:, 0]  # (B,H,W)
    pred_is_fg = (fb_logit.squeeze(1) > 0)
    pseudo_mask = (pred > 0) & pred_is_fg & (p_fg > conf_thr)
    if not pseudo_mask.any():
        return loss_pseudo, {"pseudo/enabled": 1.0, "pseudo/pred_fg_ratio": pred_fg_ratio, "pseudo/n": 0.0}

    pseudo_y = (pred[pseudo_mask] - 1).long()  # 0..7
    logits_sel = semantic_logits[:, 1:].permute(0, 2, 3, 1)[pseudo_mask]  # (N,8)
    loss_pseudo = F.cross_entropy(logits_sel, pseudo_y, label_smoothing=0.10)

    return loss_pseudo, {"pseudo/enabled": 1.0, "pseudo/pred_fg_ratio": pred_fg_ratio, "pseudo/n": float(pseudo_mask.sum().item())}

def _compute_regularizers(
    mask_probs: torch.Tensor,      # (B,Q,H,W)
    den_raw: torch.Tensor,         # (B,H,W)
    label_ce: torch.Tensor,        # (B,H,W)
    explore_mode: bool,
    overlap_thr: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    coverage: 只在 GT fg 上拉 den，避免全圖變 fg
    area_pen: log(Q) - entropy(area dist)
    overlap_pen: relu(den_raw - thr)^2
    """
    device = mask_probs.device
    gt_fg = (label_ce > 0)
    if gt_fg.any():
        den_fg = den_raw[gt_fg].mean()
    else:
        den_fg = den_raw.mean()

    t_cov = 1.0 if explore_mode else 0.8
    loss_cov = F.relu(t_cov - den_fg).pow(2)

    Q = mask_probs.size(1)
    area = mask_probs.mean(dim=(2, 3))                    # (B,Q)
    p = area / (area.sum(dim=1, keepdim=True) + 1e-6)
    p = p.clamp_min(1e-6)
    H = -(p * p.log()).sum(dim=1).mean()                  # entropy
    area_pen = torch.tensor(float(np.log(Q)), device=device) - H  # >=0

    overlap_pen = F.relu(den_raw - overlap_thr).pow(2).mean()

    logs = {
        "reg/den_fg": float(den_fg.item()),
        "reg/area_H": float(H.item()),
        "reg/N_eff": float(torch.exp(H).item()),
    }
    return loss_cov, area_pen, overlap_pen, logs

def _compute_diversity_losses(
    class_logits: torch.Tensor,    # (B,Q,C)
    semantic_prob: torch.Tensor,   # (B,C,H,W)
    label_ce: torch.Tensor,        # (B,H,W)
    q_sim_margin: float = 0.90,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    - loss_cls_div: batch-level class distribution vs uniform (avoid single-class collapse)
    - loss_pix_div: GT fg pixels 的 predicted fg distribution vs uniform
    - loss_q_div  : query-wise class distribution cosine similarity penalty
    """
    device = class_logits.device
    q_cls_prob = torch.softmax(class_logits, dim=-1)      # (B,Q,C)

    # cls_div: mean distribution over queries & batch, FG only
    p_cls = q_cls_prob[..., 1:].mean(dim=(0, 1))          # (8,)
    p_cls = p_cls / p_cls.sum().clamp_min(1e-6)
    u = torch.full_like(p_cls, 1.0 / p_cls.numel())
    loss_cls_div = torch.sum(p_cls * (p_cls.clamp_min(1e-6).log() - u.log()))  # KL(p||U)

    # pix_div: predicted fg distribution over GT fg pixels
    gt_fg = (label_ce > 0)
    loss_pix_div = torch.tensor(0.0, device=device)
    if gt_fg.any():
        pmap = semantic_prob[:, 1:]                                  # (B,8,H,W)
        p_fg = pmap.permute(0, 2, 3, 1)[gt_fg].mean(dim=0)           # (8,)
        p_fg = p_fg / p_fg.sum().clamp_min(1e-6)
        u2 = torch.full_like(p_fg, 1.0 / p_fg.numel())
        loss_pix_div = torch.sum(p_fg * (p_fg.clamp_min(1e-6).log() - u2.log()))  # KL

    # q_div: query distribution similarity
    q = q_cls_prob[..., 1:].mean(dim=0)                              # (Q,8)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    q_norm = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sim = torch.matmul(q_norm, q_norm.t())                           # (Q,Q)
    eye = torch.eye(sim.size(0), device=device)
    sim_off = sim * (1.0 - eye)
    loss_q_div = torch.relu(sim_off - q_sim_margin).pow(2).mean()

    return loss_cls_div, loss_pix_div, loss_q_div

@torch.no_grad()
def _update_explore_mode(
    fb_logit: torch.Tensor,                 # (B,1,H,W)
    stage1_fg_ratio_ema: Optional[float],
    explore_mode: bool,
    ema_alpha: float = 0.95,
    hi_off: float = 0.20,
    lo_on: float = 0.05,
) -> Tuple[float, bool, float]:
    pred_is_fg = (fb_logit > 0).float()
    stage1_fg_ratio = float(pred_is_fg.mean().item())
    stage1_fg_ratio_ema = stage1_fg_ratio if stage1_fg_ratio_ema is None else (ema_alpha * stage1_fg_ratio_ema + (1.0 - ema_alpha) * stage1_fg_ratio)

    if explore_mode:
        if stage1_fg_ratio_ema > hi_off:
            explore_mode = False
    else:
        if stage1_fg_ratio_ema < lo_on:
            explore_mode = True

    return stage1_fg_ratio, explore_mode, stage1_fg_ratio_ema
'''
def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    global GLOBAL_WORKER_SEED
    GLOBAL_WORKER_SEED = args.seed
    
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    class_labels = {
        0: "Background",
        1: "Aorta",
        2: "Gallbladder",
        3: "Kidney(L)",
        4: "Kidney(R)",
        5: "Liver",
        6: "Pancreas",
        7: "Spleen",
        8: "Stomach"
    }

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    #max_epoch = args.max_epochs
    device = "cuda"
    class_usage_ema = torch.zeros(num_classes, device=device)
    class_usage_mom = 0.95


    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train_split",
        transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    )
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val")

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    print("--------------------------------------------------------------")
    print(f"The length of train set is: {len(db_train)}")
    print(f"The length of val set is: {len(db_val)}")
    print("--------------------------------------------------------------")

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    dice_loss = DiceLoss(num_classes)

    # ----------------------------
    # optimizer (trainable params only) + class_head smaller lr
    # ----------------------------
    m = model.module if isinstance(model, nn.DataParallel) else model

    class_head_params = list(m.class_head.parameters())
    other_trainable_params = []
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("class_head."):
            continue
        other_trainable_params.append(p)

    optimizer = optim.AdamW(
        [
            {"params": other_trainable_params, "lr": base_lr},
            {"params": class_head_params, "lr": base_lr * 0.1},
        ],
        weight_decay=1e-4
    )
    # ----------------------------
    # Resume from Phase2 checkpoint (or any ckpt)
    resume_path = getattr(args, "resume", None)  # 你可以用 argparse 新增 --resume
    start_epoch = 0

    

    trainable_params = other_trainable_params + class_head_params

    print("Trainable param tensors (other):", len(other_trainable_params))
    print("Trainable param tensors (class_head):", len(class_head_params))
    print("Optimizer param groups:", len(optimizer.param_groups))

    # ----------------------------
    # knobs (keep your original behavior)
    # ----------------------------
    bg_keep_prob = 0.2
    min_fg_ratio = 0.005
    ema_alpha = 0.95

    # stage controllers / EMAs
    explore_mode = True
    stage1_fg_ratio_ema: Optional[float] = None
    bg_bias_ema: Optional[torch.Tensor] = None
    tau_ema: Optional[torch.Tensor] = None
    q10_ema: Optional[torch.Tensor] = None

    # class freq EMA (8 fg classes)
    cls_count_ema = torch.ones(8, device="cuda")
    cls_mom = 0.99

    # temperature
    T_cls = 4.0

    # loss weights (same as your current setup)
    lambda_fb   = 1.0
    lambda_dice = 0.2
    lambda_cov  = 1e-2
    lambda_area = 2e-3
    lambda_ovlp = 1e-2
    lambda_cls_div = 1e-2
    lambda_q_div = 5e-4
    lambda_pseudo = 0.05
    # pix_div schedule: first epoch smaller
    # (we keep your original behavior: epoch<1 => 0.02 else 0.08)

    #iter_num = 0
    best_performance = 0.0

    p1_epochs=2
    p2_epochs=20
    p3_dice_epochs=31
    p3a_epochs=100
    p3b_epochs=200

    phase_sched = PhaseScheduler(p1_epochs=p1_epochs, p2_epochs=p2_epochs, p3_dice_epochs=p3_dice_epochs, p3a_epochs=p3a_epochs, p3b_epochs=p3b_epochs)
    controller = BgBiasController(
        init_bias=0.0,
        max_bias=20.0,      # 先放大，因為你 margin 會到 +3，max_bias=2 根本不夠
        margin_q50_target=-0.1,    # 目標：讓 q10(fg-bg) 壓回負值
        gain=0.12,          # 回授增益
        step_clip=0.12,     # 單步最大調整（止血用，之後可再縮）
        ema_w=1.0
    )

    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cuda")

        # 1) model
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        logging.info(f"[RESUME] Loaded model from {resume_path}")
        logging.info(f"[RESUME] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

        # 2) optimizer（可選，但建議同時載入以保持動量/AdamW 狀態）
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                logging.info("[RESUME] Optimizer state loaded.")
            except Exception as e:
                logging.info(f"[RESUME] Optimizer load failed -> {e} (will continue with fresh optimizer)")

        # 3) iter / epoch
        iter_num = int(ckpt.get("iter", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1  # 從下一個 epoch 接著跑
        logging.info(f"[RESUME] iter_num={iter_num}, start_epoch={start_epoch}")

        # 4) training states
        if "class_usage_ema" in ckpt and ckpt["class_usage_ema"] is not None:
            class_usage_ema = ckpt["class_usage_ema"].to(device)
            logging.info("[RESUME] class_usage_ema restored.")

        if "tau_ema" in ckpt:  # tau_ema_state 可能是 tensor / dict / None
            tau_ema_state = ckpt["tau_ema"]
            # 如果是 tensor，確保在 cuda
            if isinstance(tau_ema_state, torch.Tensor):
                tau_ema_state = tau_ema_state.to(device)
            logging.info("[RESUME] tau_ema_state restored.")

        if "bg_bias" in ckpt and ckpt["bg_bias"] is not None:
            # 你的 ckpt 存的是 controller.bias（可能是 tensor）
            controller.bias = ckpt["bg_bias"].detach().to(device)
            logging.info(f"[RESUME] controller.bias restored: {float(controller.bias.item()):+.4f}")

    else:
        iter_num = 0
        start_epoch = 0
        tau_ema_state = None
        cls_count_ema = None
    max_epoch = args.max_epochs + start_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

    iterator = iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    if resume_path is not None:
        best_performance, _ = validate_synapse(
            model=model,
            valloader=valloader,
            args=args,
            writer=None,
            epoch_num=start_epoch - 1,
            device="cuda",
            do_wandb=False
        )
        logging.info(f"[RESUME] init best_performance = {best_performance:.6f}")

    for epoch_num in iterator:
        model.train()
        if epoch_num == p3_dice_epochs:
            print("============================")
            print("Change Dice Loss Function")
            print("============================")
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            label_ce = _to_label_ce(label_batch)

            # skip slices (same logic as before)
            if _maybe_skip_slice(label_ce, bg_keep_prob, min_fg_ratio):
                continue

            # ----------------------------
            # forward: model -> (class_logits, masks[-1]) -> mask_probs/den
            # ----------------------------
            iter_num, tau_ema_state, cls_count_ema, class_usage_ema= training_step_core(
                args, model, dice_loss, optimizer, writer, wandb,
                class_labels,
                iter_num,
                phase_sched,
                controller,
                tau_ema_state,
                cls_count_ema,
                trainable_params,
                epoch_num,
                image_batch, label_ce,
                class_usage_ema = class_usage_ema,
                class_usage_mom = class_usage_mom
            )

        if epoch_num == p1_epochs - 1:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),   # 可選
                "iter": iter_num,
                "epoch": epoch_num,

                # ===== training state =====
                "class_usage_ema": class_usage_ema,
                "tau_ema": tau_ema_state,
                "bg_bias": controller.bias,
            }
            save_mode_path = os.path.join(snapshot_path, 'phase1_epoch_' + str(epoch_num) + '.pth')
            torch.save(ckpt, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num == p2_epochs - 1:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),   # 可選
                "iter": iter_num,
                "epoch": epoch_num,

                # ===== training state =====
                "class_usage_ema": class_usage_ema,
                "tau_ema": tau_ema_state,
                "bg_bias": controller.bias,
            }
            save_mode_path = os.path.join(snapshot_path, 'phase2_epoch_' + str(epoch_num) + '.pth')
            torch.save(ckpt, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num == p3_dice_epochs - 1:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),   # 可選
                "iter": iter_num,
                "epoch": epoch_num,

                # ===== training state =====
                "class_usage_ema": class_usage_ema,
                "tau_ema": tau_ema_state,
                "bg_bias": controller.bias,
            }
            save_mode_path = os.path.join(snapshot_path, 'phase3_dice_epoch_' + str(epoch_num) + '.pth')
            torch.save(ckpt, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        
        if epoch_num == phase_sched.p3a_end_epoch():
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),   # 可選
                "iter": iter_num,
                "epoch": epoch_num,

                # ===== training state =====
                "class_usage_ema": class_usage_ema,
                "tau_ema": tau_ema_state,
                "bg_bias": controller.bias,
            }
            save_mode_path = os.path.join(snapshot_path, 'phase31_epoch_' + str(epoch_num) + '.pth')
            torch.save(ckpt, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num == p3b_epochs - 1:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),   # 可選
                "iter": iter_num,
                "epoch": epoch_num,

                # ===== training state =====
                "class_usage_ema": class_usage_ema,
                "tau_ema": tau_ema_state,
                "bg_bias": controller.bias,
            }
            save_mode_path = os.path.join(snapshot_path, 'phase32_epoch_' + str(epoch_num) + '.pth')
            torch.save(ckpt, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        # ================================
        # Validation Stage (保留你原本版本)
        # ================================

        if ((epoch_num + 1) % 5 == 0 and epoch_num != 0):
            avg_val_dice, class_mean = validate_synapse(
                model=model,
                valloader=valloader,
                args=args,
                writer=writer,
                epoch_num=epoch_num,
                device="cuda",
                do_wandb=True
            )

            # best model
            if avg_val_dice > best_performance:
                best_performance = avg_val_dice
                save_best_path = os.path.join(snapshot_path, "best_model.pth")

                if args.n_gpu > 1:
                    torch.save(model.module.state_dict(), save_best_path)
                else:
                    torch.save(model.state_dict(), save_best_path)

                logging.info(
                    "######## Saved new best model (Dice: {:.4f}) to {} ########".format(
                        best_performance, save_best_path
                    )
                )
                wandb.run.summary["best_val_dice"] = best_performance

        # periodic save
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

        if (epoch_num % 50 == 0):
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"


# ==========================================================
# 你的模型 TransUNet_TransformerDecoder 不需要為「保守版 + 止血版 loss」改動
# （因為這些改動完全發生在 trainer 的 loss pipeline）
# 你可以保持你原本的 class_logits / refined_masks 輸出。
# ==========================================================
