# trainer.py (完整可直接替換你目前版本的 trainer_synapse 訓練段落 + 相關輔助)
# - 保守版（穩）：Stage2 只用 GT fg，但加 fg-only class reweight
# - 止血版 loss 組合：移除全圖 loss_den → 改 GT 前景上的 coverage loss_cov；overlap 門檻提高；area_pen 改成 >=0
#
# 你原本的其他功能（wandb、writer、視覺化、驗證流程）我都保留；
# 只針對 forward->loss pipeline、explore_mode 的影響範圍、正則項作最少但關鍵的修改。

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
import torch.nn.functional as F

# ----------------------------
# Global seed for dataloader workers
# ----------------------------
GLOBAL_WORKER_SEED = 1234

def worker_init_fn(worker_id):
    random.seed(GLOBAL_WORKER_SEED + worker_id)

def focal_bce_with_logits(logits, targets, alpha=0.80, gamma=2.0, pos_weight=None):
    # logits/targets: (B,1,H,W)
    # 先用帶 pos_weight 的 BCE，再套 focal
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - pt).pow(gamma) * bce
    return loss.mean()

# ---- helper: q80 function ----
def q80(x_flat: torch.Tensor) -> torch.Tensor:
    n = x_flat.numel()
    k = int(0.80 * n)
    k = max(0, min(n - 1, k))
    return x_flat.kthvalue(k + 1).values  # 1-based

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    global GLOBAL_WORKER_SEED
    GLOBAL_WORKER_SEED = args.seed

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # ----------------------------
    # class labels for wandb vis
    # ----------------------------
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
    max_epoch = args.max_epochs

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
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))
    print("--------------------------------------------------------------")

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    dice_loss = DiceLoss(num_classes)

    # ----------------------------
    # optimizer only on trainable params
    # ----------------------------


    m = model.module if isinstance(model, nn.DataParallel) else model

    # 取出 class_head 參數
    class_head_params = list(m.class_head.parameters())

    # 其餘可訓練參數（排除 class_head）
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
        weight_decay=0.0001
    )
    trainable_params = other_trainable_params + class_head_params
    print("Trainable param tensors (other):", len(other_trainable_params))
    print("Trainable param tensors (class_head):", len(class_head_params))
    print("Optimizer param groups:", len(optimizer.param_groups))
    print("Params in group0(other):", len(optimizer.param_groups[0]["params"]))
    print("Params in group1(class_head):", len(optimizer.param_groups[1]["params"]))



    print("Trainable param tensors:", len(trainable_params))
    print("Trainable param elements:", sum(p.numel() for p in trainable_params))
    print("Optimizer param groups:", len(optimizer.param_groups))
    print("Params in group0:", len(optimizer.param_groups[0]["params"]))

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    # ----------------------------
    # explore_mode (只影響 lambda_cls；不再切換 Stage1 loss 規則)
    # ----------------------------
    explore_mode = True
    stage1_fg_ratio_ema = None
    bg_bias_ema = None
    ema_alpha = 0.95

    # ----------------------------
    # sampling knobs (保留你原本的 bg slice keep)
    # ----------------------------
    bg_keep_prob = 0.2
    min_fg_ratio = 0.005  # 0.5%

    

    # ----------------------------
    # Stage1 scaling (先固定，避免你現在 /3 亂飄)
    # ----------------------------
    fb_div = 2.0  # 你原本 /3，先保留；若仍敏感再做 scale-EMA
    tau_ema = None
    for epoch_num in iterator:
        model.train()

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # label -> (B,H,W) long
            if label_batch.dim() == 4 and label_batch.size(1) == 1:
                label_ce = label_batch.squeeze(1)
            else:
                label_ce = label_batch
            label_ce = label_ce.long()

            # ----------------------------
            # skip background-only slices (保留你原策略)
            # ----------------------------
            if (label_ce > 0).sum() == 0:
                if torch.rand(1, device=label_ce.device).item() > bg_keep_prob:
                    continue

            if (label_ce > 0).float().mean().item() < min_fg_ratio:
                if torch.rand(1, device=label_ce.device).item() > 0.2:
                    continue

            # ----------------------------
            # forward
            # ----------------------------
            class_logits, masks = model(image_batch)  # class_logits: (B,Q,C), masks: list[(B,Q,h,w)]
            mask_logits = masks[-1]
            mask_logits = F.interpolate(
                mask_logits,
                size=(args.img_size, args.img_size),
                mode='bilinear',
                align_corners=False
            )
            mask_probs = torch.sigmoid(mask_logits)  # (B,Q,H,W)

            # semantic logits = einsum(class_logits (logits), mask_probs)
            #semantic_logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_probs)

            den_raw = mask_probs.sum(dim=1)                 # (B,H,W)
            den = den_raw.clamp_min(0.2)                    # safe for division
            #semantic_logits = semantic_logits / den.unsqueeze(1)
            class_probs = torch.softmax(class_logits, dim=-1)                 # (B,Q,C) in [0,1], sum=1
            semantic_prob = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)  # (B,C,H,W) in [0,1]
            semantic_prob = semantic_prob / den.unsqueeze(1)
            semantic_prob = semantic_prob.clamp(1e-6, 1-1e-6)
            semantic_logits = torch.log(semantic_prob)    
            # ---- multi-class extreme-value correction ----
            # ----------------------------
            # Stage1 logits for FG vs BG (先算出 diff，controller 也用它)
            # ----------------------------
            bg_logit = semantic_logits[:, 0:1]                 # (B,1,H,W)
            fg_logit = semantic_logits[:, 1:].logsumexp(1, True)    # (B,1,H,W)
            diff = fg_logit - bg_logit                         # fg - bg, >0 表示 fg 贏

            # ----------------------------
            # BG bias controller (用 diff 做回授，不用 argmax fg%)
            # 目標：讓 diff 的高分位數 <= 0（代表大多數像素 BG 不輸）
            # ----------------------------

            # init EMA
            if bg_bias_ema is None:
                bg_bias_ema = torch.tensor(0.0, device=semantic_logits.device)

            # --- fg% BEFORE bias (log only) ---
            with torch.no_grad():
                fg_before = (semantic_logits.argmax(1) > 0).float().mean()

            # --- apply current bias (non-inplace) ---
            bg_mean_before = semantic_logits[:, 0].mean()
            bg0 = semantic_logits[:, 0:1] + bg_bias_ema
            semantic_logits = torch.cat([bg0, semantic_logits[:, 1:]], dim=1)
            bg_mean_after = semantic_logits[:, 0].mean()

            # --- fg% AFTER bias (feedback) ---
            with torch.no_grad():
                fg_after = (semantic_logits.argmax(1) > 0).float().mean()

            # --- controller update (use fg_after) ---
            target_fg = 0.12
            gain = 1.0                     # 0.6~1.5 可調；先用 1.0
            step = (fg_after - target_fg) * gain
            step = step.clamp(-0.10, 0.10) # 每步最多動 0.10，避免暴衝

            # faster when very wrong
            ema_w = 0.05 if fg_after.item() > 0.50 else 0.01

            # fg 太高 -> step>0 -> 增加 bg_bias
            # fg 太低 -> step<0 -> 減少 bg_bias
            bg_bias_ema = (bg_bias_ema + ema_w * step).clamp(0.0, 2.5)

            # --- logging ---
            if iter_num % 100 == 0:
                print(
                    f"[bias_fg] step={step.item():+.3f} bg_bias_ema={bg_bias_ema.item():.3f} "
                    f"bg_delta={(bg_mean_after - bg_mean_before).item():+.3f} "
                    f"fg% {fg_before.item():.4f}->{fg_after.item():.4f}"
                )
            # ----------------------------
            # Stage1: FG vs BG (加權 BCE；用 soft GT 提穩)
            # ----------------------------
            '''
            bg_logit = semantic_logits[:, 0:1]                      # (B,1,H,W)
            fg_logit = semantic_logits[:, 1:].amax(1, True)   # 取 max，和 argmax 競爭一致

            diff = fg_logit - bg_logit                              # (B,1,H,W) 連續分數，不要先除
            '''
            
            
            with torch.no_grad():
                gt_is_fg = (label_ce > 0).float().unsqueeze(1)      # HARD target
                gt_fg_ratio = gt_is_fg.mean().clamp(1e-4, 0.5) 
                target_ratio = (gt_fg_ratio * 2.0).clamp(0.005, 0.12)    
                # 取 diff 的分位數當閾值 tau，使得 pred_fg_ratio ≈ target_ratio
                # pred_is_fg = diff > tau
                flat = diff.detach().flatten()
                N = flat.numel()
                k = int((1.0 - float(target_ratio.item())) * N)
                k = max(0, min(N - 1, k))
                #tau = flat.kthvalue(k + 1).values   # kthvalue 是 1-based
                tau_batch = flat.kthvalue(k + 1).values
            if tau_ema is None:
                tau_ema = tau_batch
            else:
                if iter_num % 10 == 0:
                    tau_ema = tau_batch if tau_ema is None else 0.98 * tau_ema + 0.02 * tau_batch
                #tau_ema = 0.9 * tau_ema + 0.1 * tau_batch

            tau = tau_ema
            fb_logit = (diff - tau).clamp(-12, 12)
            # 把 tau 當成 bias：fb_logit = diff - tau
            # 注意：tau detached，所以不會反傳梯度造成奇怪震盪

            # pos_weight（你原本那套保留，但上限 80 比較合理）
            with torch.no_grad():
                fg_frac = gt_is_fg.mean().clamp_min(1e-4)
                pos_weight = ((1.0 - fg_frac) / fg_frac).clamp(max=80.0)

            loss_fg_bg = F.binary_cross_entropy_with_logits(
                fb_logit, gt_is_fg, pos_weight=pos_weight
            )
            '''
            if iter_num % 100 == 0:
                with torch.no_grad():
                    bg = semantic_logits[:, 0]                 # bias 後
                    fg = semantic_logits[:, 1:].amax(dim=1)    # bias 後
                    x1 = fg - bg

                    post_fg = (semantic_logits.argmax(1) > 0).float().mean()

                    print(f"[fg%] pre={pre_fg.item():.4f} post={post_fg.item():.4f}")
                    print(f"[bg] mean={bg.mean().item():.3f} min={bg.min().item():.3f} max={bg.max().item():.3f}")
                    print(f"[fg(max)] mean={fg.mean().item():.3f} min={fg.min().item():.3f} max={fg.max().item():.3f}")
                    print(f"[x1=fg-bg] mean={x1.mean().item():.3f} min={x1.min().item():.3f} max={x1.max().item():.3f}")
                    print(f"[controller] raw_fg={raw_fg_ratio.item():.4f} step={bg_bias_step.item():+.4f} bg_bias_ema={bg_bias_ema.item():.3f}")
            '''
            # ----------------------------
            # Stage2: FG class CE (保守版：只用 GT fg) + fg-only reweight
            # ----------------------------
            # ---- fuse Stage1 gate into semantic logits (log-prior) ----
            #p_fg = torch.sigmoid(fb_logit)            # (B,1,H,W)
            '''
            prior = torch.tanh((-diff) / 2.0)                      # in [-0.5, 0.5]
            with torch.no_grad():
                dm = diff.mean().abs().item()
            #beta = float(np.clip(dm, 0.5, 1.5))  # 不要乘 2，先小一點
            beta = 1
            fg_region = (diff > tau + 0.5).float()  # (B,1,H,W)
            s = 0.5  # 0.3~1.0 之間都可
            w = torch.exp(-(diff - tau ).clamp(min=0.0) / s)  # diff 越高，推力越小
            delta = beta * prior * fg_region * w
            delta = delta.clamp(-0.0, 0.15)
            
            

            semantic_logits2 = semantic_logits.clone()
            semantic_logits2[:, 0:1] = semantic_logits2[:, 0:1] + delta
            #semantic_logits2[:, 1: ] = semantic_logits2[:, 1: ] - delta 
            '''
            semantic_logits2 = semantic_logits  
            gt_fg = (label_ce > 0)            # (B,H,W) bool
            loss_fg_cls = torch.tensor(0.0, device=semantic_logits2.device)

            if gt_fg.any():
                fg_logits = semantic_logits2[:, 1:]  # (B,8,H,W)

                y = (label_ce[gt_fg] - 1).long()     # (N_fg,)
                # batch-level class counts
                counts = torch.bincount(y, minlength=8).float()  # (8,)
                w = (counts.sum() / (counts + 1.0)).clamp(1.0, 10.0)  # 反比權重，避免爆
                # normalize to keep scale stable
                w = w / w.mean().clamp_min(1e-6)

                loss_fg_cls = F.cross_entropy(
                    fg_logits.permute(0,2,3,1)[gt_fg],  # (N_fg,8)
                    y,
                    weight=w.to(fg_logits.device)
                )

            # --- Stage2 aux: high-confidence pseudo-FG (SOFT) ---
            with torch.no_grad():
                prob = torch.softmax(semantic_logits2, dim=1)  # (B,9,H,W)
                p_fg = 1.0 - prob[:, 0]                        # (B,H,W)
                pred = prob.argmax(dim=1)                      # (B,H,W)
                pred_fg_ratio = (pred > 0).float().mean().item()
                pred_is_fg = (fb_logit.squeeze(1) > 0)  # (B,H,W) Stage1 gate
                pseudo_mask = (pred > 0) & pred_is_fg & (p_fg > 0.92)      # 高信心前景
                pseudo_y = (pred[pseudo_mask] - 1).long()      # 0..7

            enable_pseudo = (pred_fg_ratio < 0.60 and epoch_num > 1)
            loss_pseudo = torch.tensor(0.0, device=semantic_logits2.device)
            if enable_pseudo and pseudo_mask.any():
                fg_logits = semantic_logits2[:, 1:]            # (B,8,H,W)
                logits_sel = fg_logits.permute(0,2,3,1)[pseudo_mask]  # (N,8)
                # soft CE 用 label smoothing 避免互打
                loss_pseudo = F.cross_entropy(logits_sel, pseudo_y, label_smoothing=0.10)

            #=============loss fg-cap===================================
            with torch.no_grad():
                gt_is_fg = (label_ce > 0).float().unsqueeze(1)      # HARD target
                gt_fg_ratio = gt_is_fg.mean().clamp(1e-4, 0.5)      # 避免 0/太大

                # 你可以讓 stage1 稍微「偏寬鬆」一點（避免全背景）
                # 例如 target_ratio = gt_fg_ratio * 1.5，最多不超過 0.30
                target_ratio = (gt_fg_ratio * 2.0).clamp(0.005, 0.12)#常看到 Stage1_fg_ratio < 0.10、FN_rate > 0.8，就再提高.clamp(0.10, 0.25)
                target_pred_fg = min(max(gt_fg_ratio * 3.0, 0.05), 0.30)   # 5%~30%
            
            #pred_fg_ratio_soft = (1.0 - prob[:,0]).mean()              # mean p_fg
            #loss_fgcap = F.relu(pred_fg_ratio_soft - target_pred_fg).pow(2)

            # ---- BG-only CE on GT background (small weight) ----
            gt_bg = (label_ce == 0)                 # (B,H,W)
            bg_l = semantic_logits2[:, 0]           # (B,H,W)
            fg_l = semantic_logits2[:, 1:].amax(dim=1)   # (B,H,W)
            margin = 0.5
            '''
            viol = margin + fg_l - bg_l             # (B,H,W)

            loss_fgcap = F.relu(viol).mean()
            loss_bg = F.relu(viol)[gt_bg].mean() if gt_bg.any() else torch.tensor(0.0, device=label_ce.device)
            '''
            
            # ----------------------------
            # Dice (輔助，弱化)
            # ----------------------------
            semantic_prob = torch.softmax(semantic_logits2, dim=1)
            loss_dice = dice_loss(semantic_prob, label_ce, softmax=False) if gt_fg.any() else \
                torch.tensor(0.0, device=semantic_logits2.device)

            # ----------------------------
            # Regularization (止血版)
            # 1) coverage: 只在 GT fg 上拉 den，避免全圖變前景
            # ----------------------------
            if gt_fg.any():
                den_fg = den_raw[gt_fg].mean()
            else:
                den_fg = den_raw.mean()

            t_cov = 1.0 if explore_mode else 0.8
            loss_cov = F.relu(t_cov - den_fg).pow(2)

            # ----------------------------
            # 2) area dispersion: area_pen = log(Q) - H >= 0
            # ----------------------------
            Q = mask_probs.size(1)
            area = mask_probs.mean(dim=(2, 3))  # (B,Q)
            p = area / (area.sum(dim=1, keepdim=True) + 1e-6)
            p = p.clamp_min(1e-6)
            H = -(p * p.log()).sum(dim=1).mean()
            area_pen = float(np.log(Q)) - H  # >= 0

            # ----------------------------
            # 3) overlap penalty: 門檻提高避免把 den 壓死
            # ----------------------------
            overlap_pen = F.relu(den_raw - 2.0).pow(2).mean()

            # ----------------------------
            # explore_mode：只影響 lambda_cls（不再切 Stage1 loss）
            # ----------------------------
            with torch.no_grad():
                pred_is_fg_stage1 = (fb_logit > 0).float()
                stage1_fg_ratio = float(pred_is_fg_stage1.mean().item())

                if stage1_fg_ratio_ema is None:
                    stage1_fg_ratio_ema = stage1_fg_ratio
                else:
                    stage1_fg_ratio_ema = ema_alpha * stage1_fg_ratio_ema + (1 - ema_alpha) * stage1_fg_ratio

                # hysteresis
                if explore_mode:
                    if stage1_fg_ratio_ema > 0.20:
                        explore_mode = False
                else:
                    if stage1_fg_ratio_ema < 0.05:
                        explore_mode = True

            lambda_cls = 0.2 if explore_mode else 1.0

            # ===== class anti-collapse regularizer (batch-level) =====
            # q_cls_prob: (B,Q,C)
            q_cls_prob = torch.softmax(class_logits, dim=-1)

            # 只看前景類別 1..C-1（不含 BG）
            p_cls = q_cls_prob[..., 1:].mean(dim=(0,1))              # (C-1,)
            p_cls = p_cls / p_cls.sum().clamp_min(1e-6)

            u = torch.full_like(p_cls, 1.0 / p_cls.numel())
            loss_cls_div = torch.sum(p_cls * (p_cls.clamp_min(1e-6).log() - u.log()))  # KL(p||U)
            
            #============================================================

            # ---- pixel-level anti-collapse on predicted FG distribution ----
            gt_fg = (label_ce > 0)

            
            # 用 GT fg pixels 統計「模型預測的前景類別分布」
            loss_pix_div = torch.tensor(0.0, device=semantic_prob.device)
            if gt_fg.any():
                pmap = semantic_prob[:, 1:]  # (B,8,H,W)
                p_fg = pmap.permute(0,2,3,1)[gt_fg].mean(dim=0)  # (8,)
                p_fg = p_fg / p_fg.sum().clamp_min(1e-6)
                u = torch.full_like(p_fg, 1.0 / p_fg.numel())
                loss_pix_div = torch.sum(p_fg * (p_fg.clamp_min(1e-6).log() - u.log()))  # KL(p||U)
            
            #============================================================

            with torch.no_grad():
                if pred_fg_ratio > 0.7:
                    with torch.no_grad():
                        pred_raw = semantic_logits.argmax(dim=1)
                        pred2    = semantic_logits2.argmax(dim=1)
                        print(f"[pred_fg_ratio] raw={(pred_raw>0).float().mean().item():.4f} | gated={(pred2>0).float().mean().item():.4f}")

            # ----------------------------
            # LOSS weights 
            # ----------------------------
            lambda_fb   = 1.0
            lambda_dice = 0.2        # 先弱化 dice，避免和 stage loss 互打
            lambda_cov  = 1e-2       # GT fg coverage
            lambda_area = 2e-3       # query area dispersion
            lambda_ovlp = 1e-2       # overlap penalty (門檻提高後再給小權重)
            lambda_cls_div = 1e-2
            lambda_bg = 0.01
            lambda_pix_div = 2e-2
            lambda_pseudo = 0.05
            lambda_fgcap = 0.05 
            
            loss = (
                lambda_fb   * loss_fg_bg +
                lambda_cls  * loss_fg_cls +
                lambda_dice * loss_dice +
                lambda_cov  * loss_cov +
                lambda_area * area_pen +
                lambda_ovlp * overlap_pen +
                lambda_cls_div * loss_cls_div +
                #lambda_bg * loss_bg +
                lambda_pix_div * loss_pix_div +
                lambda_pseudo * loss_pseudo
                #lambda_fgcap * loss_fgcap
            )

            # ----------------------------
            # debug print (你原本的我保留 + 補充 den_fg / N_eff)
            # ----------------------------
            if iter_num % 100 == 0:
                with torch.no_grad():
                    pred = semantic_prob.argmax(dim=1)
                    pred_fg_ratio = float((pred > 0).float().mean().item())
                    #gt_fg_ratio = float((label_ce > 0).float().mean().item())
                    pred_unique = torch.unique(pred).detach().cpu().tolist()[:20]

                    # FG hist / dom
                    fg_pixels = pred[pred > 0]
                    if fg_pixels.numel() > 0:
                        hist = torch.bincount(fg_pixels, minlength=num_classes).float()
                        fg_hist = hist[1:]
                        fg_class_ratio = fg_hist / fg_hist.sum().clamp_min(1.0)
                        fg_dom_ratio = float(fg_class_ratio.max().item())
                        fg_dom_cls = int(fg_class_ratio.argmax().item() + 1)
                    else:
                        fg_dom_ratio = 0.0
                        fg_dom_cls = -1

                    # den stats
                    den_mean = float(den_raw.mean().item())
                    den_min = float(den_raw.min().item())
                    den_max = float(den_raw.max().item())
                    den_fg_dbg = float(den_fg.item()) if torch.is_tensor(den_fg) else float(den_fg)

                    # N_eff
                    Hq = -(p * p.log()).sum(dim=1).mean()
                    N_eff = float(torch.exp(Hq).item())

                    # Stage1 TP/FP/FN
                    pred_is_fg = (fb_logit > 0).squeeze(1)
                    gt_is_fg = (label_ce > 0)
                    TP = (pred_is_fg & gt_is_fg).sum().item()
                    FP = (pred_is_fg & (~gt_is_fg)).sum().item()
                    FN = ((~pred_is_fg) & gt_is_fg).sum().item()
                    TN = ((~pred_is_fg) & (~gt_is_fg)).sum().item()

                    gt_fg_cnt = gt_is_fg.sum().item()
                    gt_bg_cnt = (~gt_is_fg).sum().item()
                    FP_rate = FP / max(gt_bg_cnt, 1)
                    FN_rate = FN / max(gt_fg_cnt, 1)

                    q_cls = torch.softmax(class_logits, dim=-1)          # (B,Q,C)
                    p_cls_dbg = q_cls[..., 1:].mean(dim=(0, 1))          # (C-1,)
                    p_cls_dbg = p_cls_dbg / p_cls_dbg.sum().clamp_min(1e-6)
                    
                    print("===== DEBUG (Conservative GT-fg Stage2 + Stop-bleeding losses) =====")
                    print("[class_head p_cls]", p_cls_dbg.detach().cpu().numpy().round(3))
                    print(f"GT_fg_ratio: {gt_fg_ratio:.4f} | Pred_fg_ratio(argmax): {pred_fg_ratio:.4f} | Stage1_fg_ratio: {stage1_fg_ratio:.4f}")
                    print(f"pred_unique: {pred_unique}")
                    print(f"[FG hist] dom_cls={fg_dom_cls}, dom_ratio={fg_dom_ratio:.3f}")
                    print(f"[stage1] FP_rate={FP_rate:.4f} | FN_rate={FN_rate:.4f} | pos_weight={float(pos_weight.item()):.2f}")
                    print(f"[loss] fg_bg={loss_fg_bg.item():.4f} | fg_cls={loss_fg_cls.item():.4f} | dice={loss_dice.item():.4f}")
                    print(f"[reg] cov={loss_cov.item():.6f} | area_pen={area_pen.item():.6f} | ovlp={overlap_pen.item():.6f}")
                    print(f"[den_raw] mean/min/max: {den_mean:.4f} / {den_min:.4f} / {den_max:.4f} | den_fg={den_fg_dbg:.4f}")
                    print(f"[query] N_eff={N_eff:.2f} (<=Q={Q}) | area_H={float(H.item()):.3f}")
                    print(f"[EMA] stage1_fg_ema={stage1_fg_ratio_ema:.3f} | explore_mode={explore_mode}")
                    print(f"[stage1 calib] gt_fg_ratio={gt_fg_ratio.item():.4f} target_ratio={target_ratio.item():.4f} tau={tau.item():.3f} diff_mean={diff.mean().item():.3f}")
                    print("=====================================================================")
                    print("All Loss elegant")
                    print(f"[fg_bg] loss_fg_bg={loss_fg_bg.item():.4f} | weighted={lambda_fb*loss_fg_bg.item():.6f}")
                    print(f"[fg_cls] loss_fg_cls={loss_fg_cls.item():.4f} | weighted={lambda_cls  * loss_fg_cls.item():.6f}")
                    print(f"[dice] loss_dice={loss_dice.item():.4f} | weighted={lambda_dice * loss_dice.item():.6f}")
                    print(f"[cov] loss_cov={loss_cov.item():.4f} | weighted={lambda_cov  * loss_cov.item():.6f}")
                    print(f"[area] area_pen={area_pen.item():.4f} | weighted={lambda_area * area_pen.item():.6f}")
                    print(f"[overlap_pen] overlap_pen={overlap_pen.item():.4f} | weighted={lambda_ovlp * overlap_pen.item():.6f}")
                    print(f"[cls_div] loss_cls_div={loss_cls_div.item():.4f} | weighted={lambda_cls_div*loss_cls_div.item():.6f}")
                    #print(f"[bg] loss_bg={loss_bg.item():.4f} | weighted={lambda_bg * loss_bg.item():.6f}")
                    print(f"[pix_div] loss_pix_div={loss_pix_div.item():.4f} | weighted={lambda_pix_div * loss_pix_div.item():.6f}")
                    print(f"[pseudo] loss_pseudo={loss_pseudo.item():.4f} | weighted={lambda_pseudo * loss_pseudo.item():.6f}")
                    #print(f"[fgcap] loss_fgcap ={loss_fgcap .item():.4f} | weighted={lambda_fgcap * loss_fgcap.item():.6f}")       
                    
            # ----------------------------
            # backward / step
            # ----------------------------
            optimizer.zero_grad()
            loss.backward()

            if iter_num % 100 == 0:
                with torch.no_grad():
                    norms = []
                    for p_ in trainable_params:
                        if p_.grad is not None:
                            norms.append(p_.grad.detach().norm(2))
                    if norms:
                        total_norm = torch.norm(torch.stack(norms), 2).item()
                    else:
                        total_norm = 0.0
                print("DEBUG grad_norm (before clip):", total_norm)

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # poly lr schedule (保留你原本)
            min_lr = base_lr * 0.05
            lr_main = min_lr + (base_lr - min_lr) * (1.0 - iter_num / max_iterations) ** 0.9

            optimizer.param_groups[0]["lr"] = lr_main          # other params
            optimizer.param_groups[1]["lr"] = lr_main * 0.1    # class_head

            iter_num += 1

            # ----------------------------
            # wandb logs (補上 cov/area/ovlp)
            # ----------------------------
            # 這裡的 FP_rate/FN_rate 只有在 iter%100 時有算，
            # 但你原本每 iter 都 log；為避免 undefined，這裡每 iter 也計一次簡版。
            with torch.no_grad():
                pred_is_fg = (fb_logit > 0).squeeze(1)
                gt_is_fg = (label_ce > 0)
                FP = (pred_is_fg & (~gt_is_fg)).sum().item()
                FN = ((~pred_is_fg) & gt_is_fg).sum().item()
                gt_fg_cnt = gt_is_fg.sum().item()
                gt_bg_cnt = (~gt_is_fg).sum().item()
                FP_rate = FP / max(gt_bg_cnt, 1)
                FN_rate = FN / max(gt_fg_cnt, 1)

                pred = semantic_prob.argmax(dim=1)
                fg_pixels = pred[pred > 0]
                if fg_pixels.numel() > 0:
                    hist = torch.bincount(fg_pixels, minlength=num_classes).float()
                    fg_hist = hist[1:]
                    fg_class_ratio = fg_hist / fg_hist.sum().clamp_min(1.0)
                    fg_dom_ratio = float(fg_class_ratio.max().item())
                    fg_dom_cls = int(fg_class_ratio.argmax().item() + 1)
                else:
                    fg_dom_ratio = 0.0
                    fg_dom_cls = -1

            wandb.log({
                "train/loss": loss.item(),
                "train/loss_fg_bg": loss_fg_bg.item(),
                "train/loss_fg_cls": loss_fg_cls.item() if gt_fg.any() else 0.0,
                "train/loss_dice": loss_dice.item(),

                "train/loss_cov": loss_cov.item(),
                "train/area_pen": area_pen.item(),
                "train/overlap_pen": overlap_pen.item(),

                "train/stage1_fg_ratio": float(pred_is_fg_stage1.mean().item()),
                "train/FP_rate": FP_rate,
                "train/FN_rate": FN_rate,

                "train/pred_fg_ratio": float((pred > 0).float().mean().item()),
                "train/fg_dom_ratio": fg_dom_ratio,
                "train/fg_dom_cls": fg_dom_cls,

                "train/pos_weight": float(pos_weight.item()),
                "train/explore_mode": int(explore_mode),
                "lr_main": lr_main,
            })

            writer.add_scalar("loss/total", loss.item(), iter_num)
            writer.add_scalar("loss/fg_bg", loss_fg_bg.item(), iter_num)
            writer.add_scalar("loss/fg_cls", loss_fg_cls.item(), iter_num)
            writer.add_scalar("loss/dice", loss_dice.item(), iter_num)
            writer.add_scalar("loss/cov", loss_cov.item(), iter_num)
            writer.add_scalar("loss/area_pen", area_pen.item(), iter_num)
            writer.add_scalar("loss/overlap_pen", overlap_pen.item(), iter_num)
            writer.add_scalar("stat/stage1_fg_ratio", stage1_fg_ratio, iter_num)
            writer.add_scalar("stat/explore_mode", int(explore_mode), iter_num)
            writer.add_scalar("stat/pos_weight", float(pos_weight.item()), iter_num)
            writer.add_scalar("lr_main", lr_main, iter_num)

            logging.info(
                f"iter {iter_num:5d} | "
                f"loss={loss.item():.4f} | "
                f"fg_bg={loss_fg_bg.item():.4f} | "
                f"fg_cls={loss_fg_cls.item() if gt_fg.any() else 0.0:.4f} | "
                f"dice={loss_dice.item():.4f} | "
                f"cov={loss_cov.item():.6f} | "
                f"area={area_pen.item():.6f} | "
                f"ovlp={overlap_pen.item():.6f} | "
                f"stage1_fg_ratio={stage1_fg_ratio:.4f} | "
                f"pos_weight={float(pos_weight.item()):.2f} | "
                f"FP_rate={FP_rate:.4f} | FN_rate={FN_rate:.4f}"
            )

            # ----------------------------
            # wandb image vis (保留你原本)
            # ----------------------------
            if iter_num % 50 == 0:
                image_show = image_batch[0, 0, :, :].cpu().detach().numpy()
                image_show = (image_show - image_show.min()) / (image_show.max() - image_show.min() + 1e-8)

                pred_mask = semantic_prob.argmax(dim=1)[0].detach().cpu().numpy()
                gt_mask = label_ce[0].cpu().numpy()

                wandb.log({
                    "train/visualization": wandb.Image(
                        image_show,
                        masks={
                            "predictions": {"mask_data": pred_mask, "class_labels": class_labels},
                            "ground_truth": {"mask_data": gt_mask, "class_labels": class_labels}
                        },
                        caption=f"Epoch {epoch_num} Iter {iter_num}"
                    )
                })
                writer.add_image('train/Image', image_show[None, ...], iter_num)

        # ================================
        # Validation Stage (保留你原本版本)
        # ================================
        if (epoch_num % 1 == 0):
            model.eval()

            dice_per_class = {c: [] for c in range(1, args.num_classes)}
            dice_per_slice = []

            with torch.no_grad():
                for _, val_batch in tqdm(enumerate(valloader), total=len(valloader),
                                         desc=f"Validating Epoch {epoch_num}"):

                    val_img = val_batch['image'].cuda()
                    val_label = val_batch['label'].cuda()

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

                    if args.add_decoder:
                        _, val_out = model(val_img)  # eval 回 (None, prob)
                        val_out = val_out + 1e-7
                        val_out = val_out / (val_out.sum(dim=1, keepdim=True).clamp_min(1e-6))
                    else:
                        val_out = model(val_img)
                        val_out = torch.softmax(val_out, dim=1)

                    if val_out.dim() == 3:
                        val_out = val_out.unsqueeze(0)

                    pred = val_out.argmax(dim=1)

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

            if avg_val_dice > best_performance:
                best_performance = avg_val_dice
                save_best_path = os.path.join(snapshot_path, 'best_model.pth')

                if args.n_gpu > 1:
                    torch.save(model.module.state_dict(), save_best_path)
                else:
                    torch.save(model.state_dict(), save_best_path)

                logging.info("######## Saved new best model (Dice: {:.4f}) to {} ########".format(best_performance, save_best_path))
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
