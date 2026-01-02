import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from utils import DiceLoss
from torchvision import transforms
import torch.nn.functional as F  # <--- 新增: 用於 Upsample

# --- 修改 1: 定義一個模組級別的全域變數 ---
GLOBAL_WORKER_SEED = 1234

# --- 修改 2: 將 worker_init_fn 移到外面，並使用全域變數 ---
def worker_init_fn(worker_id):
    # 使用全域變數加上 worker_id
    random.seed(GLOBAL_WORKER_SEED + worker_id)

def build_semantic_logits(model, images, img_size):
    out = model(images)

    # 兼容：有的 forward 回 (class_logits, masks)，有的回 (None, semantic)
    if isinstance(out, (tuple, list)):
        # 若最後一個就是 (B,C,H,W) 的 semantic（例如 eval 模式）
        if out[-1].dim() == 4 and out[-1].size(1) > 1:
            return out[-1]  # 當作 logits (或至少是 unnorm scores)
        class_logits, masks = out[0], out[1]
    else:
        # 原版 TransUNet 可能直接回 (B,C,H,W) logits
        return out

def check_health(
    *,
    iter_num: int,
    prob_sum_mean: float,
    gt_fg_ratio: float,
    pred_fg_ratio: float,
    pred_unique: list,
    s_min: float,
    s_max: float,
    logit_margin_mean: float,
    den_min: float,
    den_max: float,
    mask_mean: float,
    area_q_min: float,
    area_q_max: float,
    p_gt_fg_mean: float | None = None,
    p_bg_fg_mean: float | None = None,
    # 可依你任務調整的門檻（先給 Synapse/多器官/Q~20 的實務值）
    cfg: dict | None = None,
):
    """
    Print health status as OK/WARN/BAD based on debug metrics.
    Returns: (status_str, issues_list)
    """

    default_cfg = {
        # prob_sum_mean
        "prob_sum_ok": (0.999, 1.001),
        "prob_sum_warn": (0.995, 1.005),

        # pred_fg_ratio sanity (Synapse 常見 1%~8%，但 batch/slice 會波動)
        "pred_fg_bad_low": 0.002,   # 幾乎全背景
        "pred_fg_warn_low": 0.005,
        "pred_fg_warn_high": 0.20,  # 偏亂噴（保守）
        "pred_fg_bad_high": 0.30,   # 幾乎全前景

        # pred vs gt ratio (只做弱檢查；gt 可能很小)
        "pred_gt_warn_mult_low": 0.3,
        "pred_gt_warn_mult_high": 2.0,

        # semantic_logits range (避免 den 放大爆掉)
        "slogit_warn_abs": 12.0,
        "slogit_bad_abs": 20.0,

        # logit_margin
        "margin_warn_low": 0.05,  # 太平
        "margin_ok_low": 0.10,
        "margin_warn_high": 3.0,  # 太尖（常伴隨 den.min 太低）
        "margin_bad_high": 5.0,

        # mask_probs mean
        "mask_mean_warn_low": 0.05,
        "mask_mean_ok_low": 0.08,
        "mask_mean_ok_high": 0.20,
        "mask_mean_warn_high": 0.25,

        # den stability
        "den_min_ok": 0.15,
        "den_min_warn": 0.10,
        "den_min_bad": 0.05,
        "den_max_warn": 8.0,

        # area_per_q max/min
        "area_max_ok": 0.22,
        "area_max_warn": 0.28,
        "area_max_bad": 0.35,
        "area_min_warn": 0.005,  # 太多 query 死掉的徵兆

        # on-GT-fg probs (若有 GT fg 才有效)
        "p_bg_on_fg_ok": 0.50,
        "p_bg_on_fg_warn": 0.70,
        "p_bg_on_fg_bad": 0.85,
        "p_gt_on_fg_ok": 0.15,
        "p_gt_on_fg_warn": 0.08,
    }

    if cfg is not None:
        default_cfg.update(cfg)
    c = default_cfg

    issues = []  # (severity, message)
    def add(sev: str, msg: str):
        issues.append((sev, msg))

    # 1) prob_sum_mean
    lo_ok, hi_ok = c["prob_sum_ok"]
    lo_w, hi_w = c["prob_sum_warn"]
    if not (lo_w <= prob_sum_mean <= hi_w):
        add("BAD", f"prob_sum_mean={prob_sum_mean:.4f} (prob not normalized?)")
    elif not (lo_ok <= prob_sum_mean <= hi_ok):
        add("WARN", f"prob_sum_mean={prob_sum_mean:.4f} (slightly off 1.0)")

    # 2) pred_fg_ratio extremes
    if pred_fg_ratio < c["pred_fg_bad_low"]:
        add("BAD", f"pred_fg_ratio={pred_fg_ratio:.4f} (near-all background collapse)")
    elif pred_fg_ratio < c["pred_fg_warn_low"]:
        add("WARN", f"pred_fg_ratio={pred_fg_ratio:.4f} (very low foreground)")
    elif pred_fg_ratio > c["pred_fg_bad_high"]:
        add("BAD", f"pred_fg_ratio={pred_fg_ratio:.4f} (near-all foreground / noisy)")
    elif pred_fg_ratio > c["pred_fg_warn_high"]:
        add("WARN", f"pred_fg_ratio={pred_fg_ratio:.4f} (high foreground; check noise)")

    # pred vs gt (只做弱檢查：gt 可能接近 0)
    if gt_fg_ratio > 1e-6:
        mult = pred_fg_ratio / gt_fg_ratio
        if mult < c["pred_gt_warn_mult_low"] or mult > c["pred_gt_warn_mult_high"]:
            add("WARN", f"pred/gt fg ratio mult={mult:.2f} (gt={gt_fg_ratio:.4f}, pred={pred_fg_ratio:.4f})")

    # 3) semantic logits range (abs)
    abs_max = max(abs(s_min), abs(s_max))
    if abs_max > c["slogit_bad_abs"]:
        add("BAD", f"semantic_logits abs_max≈{abs_max:.2f} (likely den too small / scale blow-up)")
    elif abs_max > c["slogit_warn_abs"]:
        add("WARN", f"semantic_logits abs_max≈{abs_max:.2f} (scale a bit large)")

    # 4) margin
    if logit_margin_mean < c["margin_warn_low"]:
        add("WARN", f"logit_margin_mean={logit_margin_mean:.3f} (too flat / weak class separation)")
    if logit_margin_mean > c["margin_bad_high"]:
        add("BAD", f"logit_margin_mean={logit_margin_mean:.3f} (too sharp; often unstable)")
    elif logit_margin_mean > c["margin_warn_high"]:
        add("WARN", f"logit_margin_mean={logit_margin_mean:.3f} (very sharp; check den_min)")

    # 5) mask mean
    if mask_mean < c["mask_mean_warn_low"]:
        add("WARN", f"mask_probs.mean={mask_mean:.4f} (queries cover too little; risk all-BG)")
    elif mask_mean > c["mask_mean_warn_high"]:
        add("WARN", f"mask_probs.mean={mask_mean:.4f} (queries too large; risk all-FG)")

    # 6) den stability
    if den_min < c["den_min_bad"]:
        add("BAD", f"den.min={den_min:.4f} (VERY small => semantic_logits blown up)")
    elif den_min < c["den_min_warn"]:
        add("WARN", f"den.min={den_min:.4f} (small; consider clamp_min ~0.15-0.25)")
    elif den_min < c["den_min_ok"]:
        add("WARN", f"den.min={den_min:.4f} (borderline; watch stability)")

    if den_max > c["den_max_warn"]:
        add("WARN", f"den.max={den_max:.2f} (many queries overlap heavily)")

    # 7) area per query
    if area_q_max > c["area_max_bad"]:
        add("BAD", f"area_per_q.max={area_q_max:.4f} (query dominates; collapse likely)")
    elif area_q_max > c["area_max_warn"]:
        add("WARN", f"area_per_q.max={area_q_max:.4f} (one query too large)")
    elif area_q_max > c["area_max_ok"]:
        add("WARN", f"area_per_q.max={area_q_max:.4f} (slightly high; consider mild area cap)")

    if area_q_min < c["area_min_warn"]:
        add("WARN", f"area_per_q.min={area_q_min:.4f} (many queries might be dead)")

    # 8) on-GT-fg probs (if available)
    if p_bg_fg_mean is not None and p_gt_fg_mean is not None:
        if p_bg_fg_mean > c["p_bg_on_fg_bad"]:
            add("BAD", f"p(bg|GT_fg)={p_bg_fg_mean:.3f} (model insists BG on GT fg)")
        elif p_bg_fg_mean > c["p_bg_on_fg_warn"]:
            add("WARN", f"p(bg|GT_fg)={p_bg_fg_mean:.3f} (still BG-heavy on GT fg)")
        elif p_bg_fg_mean > c["p_bg_on_fg_ok"]:
            add("WARN", f"p(bg|GT_fg)={p_bg_fg_mean:.3f} (borderline)")

        if p_gt_fg_mean < c["p_gt_on_fg_warn"]:
            add("WARN", f"p(gt|GT_fg)={p_gt_fg_mean:.3f} (very low; learning weak)")
        elif p_gt_fg_mean < c["p_gt_on_fg_ok"]:
            add("WARN", f"p(gt|GT_fg)={p_gt_fg_mean:.3f} (low; should rise over time)")

    # Decide overall status
    status = "OK"
    if any(sev == "BAD" for sev, _ in issues):
        status = "BAD"
    elif any(sev == "WARN" for sev, _ in issues):
        status = "WARN"

    # Print summary
    print(f"[HEALTH] iter={iter_num} => {status}")
    if issues:
        # BAD first, then WARN
        for sev in ("BAD", "WARN"):
            for s, msg in issues:
                if s == sev:
                    print(f"  - {sev}: {msg}")
    else:
        print("  - OK: no issues detected")

    # Quick hints (optional)
    if status != "OK":
        hints = []
        if den_min < c["den_min_warn"]:
            hints.append("Consider den clamp_min (e.g., 0.15~0.25) in semantic_logits normalization.")
        if area_q_max > c["area_max_warn"]:
            hints.append("Consider mild area cap regularizer to prevent one query dominating.")
        if pred_fg_ratio < c["pred_fg_warn_low"]:
            hints.append("Foreground too low; dice weight or BG sampling/weights may be too BG-favoring.")
        if pred_fg_ratio > c["pred_fg_warn_high"]:
            hints.append("Foreground too high; check mask saturation or class weights causing over-FG.")
        if hints:
            print("  Hints:")
            for h in hints[:3]:
                print(f"   * {h}")

    return status, issues


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    # --- 修改 3: 在這裡更新全域種子，確保吃到 args.seed ---
    global GLOBAL_WORKER_SEED
    GLOBAL_WORKER_SEED = args.seed
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # -------------------------------------------------------
    # [新增] 定義正確的 Class Labels (用於 WandB 視覺化)
    # -------------------------------------------------------
    class_labels = {
        0: "Background", 
        1: "Aorta",        # 主動脈
        2: "Gallbladder",  # 膽囊
        3: "Kidney(L)",    # 左腎
        4: "Kidney(R)",    # 右腎
        5: "Liver",        # 肝臟
        6: "Pancreas",     # 胰臟
        7: "Spleen",       # 脾臟
        8: "Stomach"       # 胃
    }
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_epoch = args.max_epochs
    
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train_split",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val")

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    print("--------------------------------------------------------------")
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))
    print("--------------------------------------------------------------")

    best_performance = 0.0 # 用來記錄最佳 Dice
    iterator = tqdm(range(max_epoch), ncols=70)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    # weights = torch.tensor(
    #     [1.0] + [3.0] * (num_classes - 1),
    #     device='cuda'
    # )
    #ce_loss = CrossEntropyLoss(weight=weights)
    #ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # ------------------------------------------------------------------
    # 修改 1: 優化器只追蹤需要訓練的參數 (Decoder)
    # ------------------------------------------------------------------
    # 原本: optimizer = optim.SGD(model.parameters(), ...)
    # 修改後: 過濾 requires_grad=True 的參數
    #確認trainable params & optimizer 真的吃到它
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=0.0001)
    print("Trainable param tensors:", len(trainable_params))
    print("Trainable param elements:", sum(p.numel() for p in trainable_params))
    print("Optimizer param groups:", len(optimizer.param_groups))
    print("Params in group0:", len(optimizer.param_groups[0]["params"]))
    # ------------------------------------------------------------------

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    count = 0
    for epoch_num in iterator:
        model.train()
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            if label_batch.dim() == 4 and label_batch.size(1) == 1:
                label_ce = label_batch.squeeze(1)
            else:
                label_ce = label_batch
            
            bg_keep_prob = 0.2
            if (label_ce > 0).sum() == 0:
                if torch.rand(1, device=label_ce.device).item() > bg_keep_prob:
                    continue
            min_fg_ratio = 0.005  # 0.5% 起手（你可調 0.2%~1%）
            if (label_ce > 0).float().mean().item() < min_fg_ratio:
                if torch.rand(1, device=label_ce.device).item() > 0.2:
                    continue
            class_logits, masks = model(image_batch)     # class_logits: (B,Q,C), masks: list[(B,Q,h,w)]
            # 新模型的輸出是 (class_logits, refined_masks_list)
            # 我們取出最後一層的 Mask 預測
            # 形狀通常是 (B, num_queries, 14, 14)
            mask_logits = masks[-1]                      # (B,Q,h,w)
            
            
            # mask_logits -> mask_probs
            mask_logits = F.interpolate(masks[-1], size=(args.img_size, args.img_size),
                                        mode='bilinear', align_corners=False)
            mask_probs = torch.sigmoid(mask_logits)  # (B,Q,H,W)

            # 1) 用 logits 組 semantic_logits（不要先 softmax class_logits）
            # class_logits: (B,Q,C)
            semantic_logits = torch.einsum(
                "bqc,bqhw->bchw",
                class_logits,
                mask_probs
            )  # (B,C,H,W)

            den_raw = mask_probs.sum(dim=1)  # (B,H,W) 真的 coverage
            den = den_raw.clamp_min(0.2)    # 用於除法的安全版
            semantic_logits = semantic_logits / den.unsqueeze(1)
            #semantic_logits[:, 1:] += 0.2

            # ======================================================
            # Losses
            # ======================================================
            label_ce = label_ce.long()
            # ---------- Stage 1: FG vs BG ----------
            bg_logit = semantic_logits[:, 0:1]                      # (B,1,H,W)
            fg_logit = semantic_logits[:, 1:].logsumexp(1, True)    # (B,1,H,W)

            # GT 是否為前景
            gt_is_fg = (label_ce > 0).float().unsqueeze(1)          # (B,1,H,W)

            # binary logit：fg > bg
            fb_logit = (fg_logit - bg_logit)/2                          

            loss_fg_bg = F.binary_cross_entropy_with_logits(
                fb_logit,
                gt_is_fg
            )
           
            # ---------- Stage 2: FG class (only on GT FG pixels) ----------
            fg_mask = (label_ce > 0)             # (B,H,W)

            loss_fg_cls = torch.tensor(0.0, device=semantic_logits.device)

            if fg_mask.any():
                fg_logits = semantic_logits[:, 1:]                  # (B,C_fg,H,W)
                loss_fg_cls = F.cross_entropy(
                    fg_logits.permute(0,2,3,1)[fg_mask],             # (N_fg, C_fg)
                    (label_ce[fg_mask] - 1).long()                   # class index from 0
                )
            
            #--------------Step 3：Dice 只輔助前景（保留，但弱化--------------
            semantic_prob = torch.softmax(semantic_logits, dim=1)

            has_fg = fg_mask.any()
            loss_dice = dice_loss(semantic_prob, label_ce, softmax=False) if has_fg else \
                        torch.tensor(0.0, device=semantic_logits.device)
            
            #-------------------------------------------------------------
            # coverage regularization：希望 den_raw 不要太常 < 0.3
            loss_den = F.relu(0.3 - den_raw).pow(2).mean()
            
            #限制單一 query 面積過大
            area = mask_probs.mean(dim=(2,3))           # (B,Q)
            max_area = 0.22 
            area_pen = torch.relu(area - max_area).pow(2).mean()   # scalar
            overlap_pen = torch.relu(den_raw - 2.0).pow(2).mean()
            #--------------------------------------------------------------------

            lambda_fb   = 1.0    # 非常重要
            lambda_cls  = 1.0
            lambda_dice = 0.5

            loss = (
                lambda_fb  * loss_fg_bg +
                lambda_cls * loss_fg_cls +
                lambda_dice * loss_dice +
                1e-3 * loss_den +
                3e-3 * area_pen +
                3e-3 * overlap_pen
            )

            # ===== debug（建議改成看更有意義的東西）=====
            if iter_num % 100 == 0:
                with torch.no_grad():
                    # Stage-1 predicted FG ratio (should become reasonable, not necessarily 1.0)
                    pred_is_fg = (fb_logit > 0).float()                            # fg if fg_logit > bg_logit
                    pred_fg_ratio_stage1 = float(pred_is_fg.mean().item())

                    # Stage-2 final argmax (for monitoring only; don't drive heuristics)
                    pred = semantic_prob.argmax(dim=1)
                    pred_fg_ratio = float((pred > 0).float().mean().item())

                    gt_fg_ratio = float((label_ce > 0).float().mean().item())
                    pred_unique = torch.unique(pred).detach().cpu().tolist()[:20]

                    print("===== DEBUG (Two-stage) =====")
                    print(f"GT_fg_ratio: {gt_fg_ratio:.4f} | Pred_fg_ratio(argmax): {pred_fg_ratio:.4f} | Stage1_fg_ratio: {pred_fg_ratio_stage1:.4f}")
                    print(f"pred_unique: {pred_unique}")

                    print(f"[loss] fg_bg: {float(loss_fg_bg.item()):.4f} | fg_cls: {float(loss_fg_cls.item()):.4f} | dice: {float(loss_dice.item()):.4f}")
                    print(f"[reg] den: {float(loss_den.item()):.6f} | area: {float(area_pen.item()):.6f} | ovlp: {float(overlap_pen.item()):.6f}")

                    # Helpful health stats
                    print(f"[den_raw] mean/min/max: {float(den_raw.mean().item()):.4f} / {float(den_raw.min().item()):.4f} / {float(den_raw.max().item()):.4f}")
                    print(f"[mask_probs] mean/min/max: {float(mask_probs.mean().item()):.4f} / {float(mask_probs.min().item()):.4f} / {float(mask_probs.max().item()):.4f}")
                    print("=============================")
            # ======================================================
            optimizer.zero_grad()
            loss.backward()
            if iter_num % 100 == 0:
                total_norm = torch.norm(torch.stack([
                    p.grad.detach().norm(2)
                    for p in trainable_params
                    if p.grad is not None
                ]), 2).item()
                print("DEBUG grad_norm (before clip):", total_norm)
            #-----------------------------------------------------
            torch.nn.utils.clip_grad_norm_(
                trainable_params,   # 或 model.parameters()
                max_norm=1.0
            )
            optimizer.step()
            min_lr = base_lr * 0.05   # 例如保留 5% 的 base_lr
            lr_ = min_lr + (base_lr - min_lr) * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            # --- [WandB] 紀錄訓練數值 ---
            wandb.log({
                # ----- total -----
                "train/loss_total": loss.item(),
                "train/lr": lr_,
                "epoch": epoch_num,

                # ----- Stage 1: FG vs BG -----
                "train/loss_fg_bg": loss_fg_bg.item(),
                "train/stage1_fg_ratio": pred_fg_ratio_stage1,

                # ----- Stage 2: FG class -----
                "train/loss_fg_cls": loss_fg_cls.item(),

                # ----- Dice (aux) -----
                "train/loss_dice": loss_dice.item(),

                # ----- Regularization -----
                "train/loss_den": loss_den.item(),
                "train/loss_area": area_pen.item(),
                "train/loss_overlap": overlap_pen.item(),
            })
            writer.add_scalar("loss/total", loss.item(), iter_num)
            writer.add_scalar("loss/fg_bg", loss_fg_bg.item(), iter_num)
            writer.add_scalar("loss/fg_cls", loss_fg_cls.item(), iter_num)
            writer.add_scalar("loss/dice", loss_dice.item(), iter_num)
            writer.add_scalar("stat/stage1_fg_ratio", pred_fg_ratio_stage1, iter_num)
            writer.add_scalar("lr", lr_, iter_num)



            logging.info(
                f"iter {iter_num:6d} | "
                f"loss={loss.item():.4f} | "
                f"fg_bg={loss_fg_bg.item():.4f} | "
                f"fg_cls={loss_fg_cls.item():.4f} | "
                f"dice={loss_dice.item():.4f} | "
                f"stage1_fg_ratio={pred_fg_ratio_stage1:.4f}"
            )

            # --- [WandB] 視覺化圖片 (每 50 個 Iteration) ---
            if iter_num % 50 == 0:
                # 1. 準備原圖 (取 batch 第一張, 轉為 numpy, 正規化到 0-1)
                image_show = image_batch[0, 0, :, :].cpu().detach().numpy()
                image_show = (image_show - image_show.min()) / (image_show.max() - image_show.min() + 1e-8)

                # 2. 準備預測 Mask (Argmax 轉成 0-8 的整數)
                pred_mask = semantic_prob.argmax(dim=1)[0].detach().cpu().numpy()
                
                # 3. 準備 Ground Truth Mask
                gt_mask = label_ce[0].cpu().numpy()

                # 4. 上傳到 WandB (使用 class_labels)
                wandb.log({
                    "train/visualization": wandb.Image(
                        image_show,
                        masks={
                            "predictions": {
                                "mask_data": pred_mask,
                                "class_labels": class_labels
                            },
                            "ground_truth": {
                                "mask_data": gt_mask,
                                "class_labels": class_labels
                            }
                        },
                        caption=f"Epoch {epoch_num} Iter {iter_num}"
                    )
                })
                writer.add_image('train/Image', image_show[None, ...], iter_num)

        # ================================
        #       Validation Stage
        # ================================
        if (epoch_num % 1 == 0):
            model.eval()

            dice_per_class = {c: [] for c in range(1, args.num_classes)}
            dice_per_slice = []  # 每張 slice 的平均 dice（只平均該 slice 有 GT 的器官）

            with torch.no_grad():
                for _, val_batch in tqdm(enumerate(valloader), total=len(valloader),
                                        desc=f"Validating Epoch {epoch_num}"):

                    val_img = val_batch['image'].cuda()
                    val_label = val_batch['label'].cuda()

                    # ---- label 統一成 (B,H,W) 的 long ----
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
                    val_label_ce = val_label_ce.long()  # ✅ 確保 long

                    # ---- forward：拿到 (B,C,H,W) 機率 ----
                    if args.add_decoder:
                        _, val_out = model(val_img)  # 你模型 eval 會回 (None, prob)
                        val_out = val_out + 1e-7
                        val_out = val_out / (val_out.sum(dim=1, keepdim=True).clamp_min(1e-6))
                    else:
                        val_out = model(val_img)
                        # 如果原版輸出 logits，這裡要 softmax
                        val_out = torch.softmax(val_out, dim=1)

                    if val_out.dim() == 3:
                        val_out = val_out.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)

                    pred = val_out.argmax(dim=1)  # (B,H,W)

                    # ---- 計算 per-slice dice（只平均該 slice 有 GT 的 class）----
                    dices_this_slice = []
                    for cls in range(1, args.num_classes):
                        gt = (val_label_ce == cls)
                        if gt.sum() == 0:
                            continue
                        pd = (pred == cls)
                        inter = (gt & pd).sum().float()
                        dice = (2 * inter) / (gt.sum() + pd.sum() + 1e-5)

                        dice_per_class[cls].append(dice.item())
                        dices_this_slice.append(dice.item())

                    if len(dices_this_slice) > 0:
                        dice_per_slice.append(sum(dices_this_slice) / len(dices_this_slice))

            # ---- 匯總：per-class + overall mean（不灌水）----
            print("===== Validation Per-class Dice =====")
            class_mean = {}
            for cls, dices in dice_per_class.items():
                if len(dices) == 0:
                    print(f"class {cls}: N/A (n=0)")
                    class_mean[cls] = None
                else:
                    m = float(np.mean(dices))
                    print(f"class {cls}: {m:.4f} (n={len(dices)})")
                    class_mean[cls] = m

            avg_val_dice = float(np.mean(dice_per_slice)) if len(dice_per_slice) > 0 else 0.0
            print(f"===== Validation mean dice (slice-level, GT-only) = {avg_val_dice:.6f} =====")

            wandb.log({
                "val/mean_dice_gt_only": avg_val_dice,
                **{f"val/dice_class_{cls}": (m if m is not None else 0.0) for cls, m in class_mean.items()},
                "epoch": epoch_num
            })
            writer.add_scalar('info/val_dice', avg_val_dice, epoch_num)
            logging.info('Epoch %d : Validation Mean Dice: %f' % (epoch_num, avg_val_dice))
            
            # ================================
            #       Save Best Model
            # ================================
            if avg_val_dice > best_performance:
                best_performance = avg_val_dice
                save_best_path = os.path.join(snapshot_path, 'best_model.pth')
                count = 0
                if args.n_gpu > 1:
                    torch.save(model.module.state_dict(), save_best_path)
                else:
                    torch.save(model.state_dict(), save_best_path)
                
                logging.info("######## Saved new best model (Dice: {:.4f}) to {} ########".format(best_performance, save_best_path))
                wandb.run.summary["best_val_dice"] = best_performance
            else:
                count += 1
            if (count*5 == args.patience):
                logging.info(f"Early stopping triggered at epoch {epoch_num}")
                break

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