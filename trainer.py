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
            semantic_logits[:, 1:] += 0.2
            
            # ======================================================
            # Losses
            # ======================================================

            # coverage regularization：希望 den_raw 不要太常 < 0.3
            loss_den = F.relu(0.3 - den_raw).pow(2).mean()
            
            #限制單一 query 面積過大
            area = mask_probs.mean(dim=(2,3))           # (B,Q)
            max_area = 0.22 
            area_pen = torch.relu(area - max_area).pow(2).mean()   # scalar
            

            # ---- Cross Entropy（保守穩定權重）----
            label_ce = label_ce.long()

            weights = torch.tensor([0.3, 1.5, 3.0, 1.0, 1.2, 0.35, 1.8, 0.9, 0.9],
                                device=semantic_logits.device)

            # per-pixel CE (no reduction)
            semantic_prob = torch.softmax(semantic_logits, dim=1)
            ce_map = F.cross_entropy(semantic_logits, label_ce, weight=weights, reduction="none")  # (B,H,W)
            ce_flat = ce_map.flatten()
            fg = (label_ce > 0)
            bg = ~fg
            #CE 對所有 GT 前景像素都算，但對 GT 背景像素，只挑「最像前景的那一小撮」來算。
            p_fg_any = semantic_prob[:, 1:].sum(1)
            num_fg = int(fg.sum().item())
            # --- FG indices ---
            gt_fg = (label_ce > 0)
            gt_bg = (label_ce == 0)
            fg_idx = gt_fg.flatten().nonzero(as_tuple=False).squeeze(1)  # (N_fg,)

            # --- BG hard negatives: pick highest p_fg_any on GT_bg ---
            bg_idx = gt_bg.flatten().nonzero(as_tuple=False).squeeze(1)  # (N_bg,)

            if bg_idx.numel() > 0:
                pfg_flat = p_fg_any.flatten()[bg_idx]                    # (N_bg,)
                num_bg = min(bg_idx.numel(), 8192)                       # 固定上限，不綁 N_fg
                if num_bg > 0:
                    topk = torch.topk(pfg_flat, k=num_bg, largest=True).indices
                    bg_sel = bg_idx[topk]
                else:
                    bg_sel = bg_idx
            else:
                bg_sel = bg_idx  # empty

            # --- CE loss combine ---
            if fg_idx.numel() > 0:
                loss_ce = ce_flat[fg_idx].mean() + ce_flat[bg_sel].mean()
            else:
                # 沒前景：只用 hard-negative bg
                loss_ce = ce_flat[bg_sel].mean()

            area_per_q = mask_probs.mean(dim=(2, 3))
            overlap_pen = torch.relu(den_raw - 2.0).pow(2).mean()
            # ---- Dice ----
            #semantic_prob = torch.softmax(semantic_logits, dim=1)
            has_fg = (label_ce > 0).any()

            loss_dice = dice_loss(semantic_prob, label_ce, softmax=False) if has_fg else torch.tensor(0.0, device=semantic_logits.device)
            #GT 說是背景的像素上，你的前景機率越高，我就越罰你。
            gt_bg = (label_ce == 0)
                
            p_bg = semantic_prob[:, 0]  # (B,H,W)
            loss_fp = (-torch.log(p_bg.clamp_min(1e-6))[gt_bg]).mean()

            # ---- Total Loss ----
            has_fg = (label_ce > 0).any()

            loss_dice = dice_loss(semantic_prob, label_ce, softmax=False) if has_fg else torch.tensor(0.0, device=semantic_logits.device)
            
            # ----fix Loss ratio----
            pred_fg_ratio = (semantic_prob.argmax(1) > 0).float().mean().item() #模型目前「把多少比例的像素預測成前景（非背景）」。

            prob = semantic_prob                   # (B,C,H,W)
            bg = prob[:, 0]                        # (B,H,W)
            gt_fg = (label_ce > 0)

            if gt_fg.any():
                p_bg_on_fg = bg[gt_fg].mean().item()
            else:
                p_bg_on_fg = 1.0

            gt_fg = (label_ce > 0)
            gt_fg_ratio = gt_fg.float().mean().item()
            ratio_mult = pred_fg_ratio / (gt_fg_ratio + 1e-8)
            
            # 最高優先：argmax 已經全背景 → 不能再讓 Dice 主導
            if pred_fg_ratio == 0.0:
                lambda_dice = 0.5 
            # 亂噴前景就壓 Dice（不分 gt_fg_ratio）
            elif ratio_mult > 3.0:
                lambda_dice = 0.2

            else:
                # (2) GT 很少：Dice 只做輕微扶正
                if gt_fg_ratio < 0.01:
                    if pred_fg_ratio < 0.002 or p_bg_on_fg > 0.90:
                        lambda_dice = 0.5   # 甚至可以 0.3~0.5
                    else:
                        lambda_dice = 0.2   # 更保守，避免噪音擴散

                # (3) GT 足夠：才讓 Dice 去救全背景偷懶
                else:
                    if pred_fg_ratio < 0.002 or p_bg_on_fg > 0.80:
                        lambda_dice = 4.0
                    elif pred_fg_ratio < 0.01 or p_bg_on_fg > 0.60:
                        lambda_dice = 3.0
                    else:
                        lambda_dice = 2.0
            
            lambda_fp = 0.1
            if pred_fg_ratio == 0.0:
                lambda_fp_eff = 0.0
            else:
                lambda_fp_eff = lambda_fp

            loss = loss_ce + lambda_dice * loss_dice
            loss = loss + 1e-3 * loss_den
            loss = loss + 3e-3 * area_pen
            loss = loss + 3e-3 * overlap_pen
            loss = loss + lambda_fp_eff * loss_fp

            w_ce      = float(loss_ce.item())
            w_dice    = float((lambda_dice * loss_dice).item())
            w_den     = float((1e-3 * loss_den).item())
            w_area    = float((3e-3 * area_pen).item())
            w_overlap = float((3e-3 * overlap_pen).item())
            w_fp      = float((lambda_fp_eff * loss_fp).item())

            w_total = w_ce + w_dice + w_den + w_area + w_overlap + w_fp
            # ===== debug（建議改成看更有意義的東西）=====
            if iter_num % 100 == 0:
                with torch.no_grad():
                    # -------- 基本張量 --------
                    prob = semantic_prob                   # (B,C,H,W)
                    pred = prob.argmax(dim=1)              # (B,H,W)
                    bg = prob[:, 0]                        # (B,H,W)
                    fg = prob[:, 1:].max(dim=1).values     # (B,H,W)

                    # -------- GT / 前景統計 --------
                    gt_fg = (label_ce > 0)
                    gt_fg_ratio = gt_fg.float().mean().item()
                    pred_fg_ratio = (pred > 0).float().mean().item()

                    # -------- 最重要：logits 尺度（找「softmax太平」根因）--------
                    # 注意：這裡用 semantic_logits（你前面一定有）
                    s_logit = semantic_logits
                    s_mean = float(s_logit.mean().item())
                    s_min  = float(s_logit.min().item())
                    s_max  = float(s_logit.max().item())
                    # 每像素 top1-top2 margin：越大代表分類越「有把握」
                    top2 = torch.topk(s_logit, k=2, dim=1).values  # (B,2,H,W)
                    logit_margin = (top2[:, 0] - top2[:, 1])
                    logit_margin_mean = float(logit_margin.mean().item())
                    logit_margin_min  = float(logit_margin.min().item())
                    logit_margin_max  = float(logit_margin.max().item())

                    

                    # 每個 query 的平均面積分布（看是不是都趨近 0）
                    area_per_q = mask_probs.mean(dim=(2, 3))  # (B,Q)
                    area_q_mean = float(area_per_q.mean().item())
                    area_q_max  = float(area_per_q.max().item())
                    area_q_min  = float(area_per_q.min().item())

                    # -------- GT 上到底有沒有在學（超關鍵）--------
                    # GT 前景像素上：p(gt) 應該慢慢變大、p(bg) 應該慢慢變小
                    if gt_fg.any():
                        p_gt = prob.gather(1, label_ce.unsqueeze(1)).squeeze(1)   # (B,H,W)
                        p_gt_fg_mean = float(p_gt[gt_fg].mean().item())
                        p_bg_fg_mean = float(bg[gt_fg].mean().item())
                    else:
                        p_gt_fg_mean = float("nan")
                        p_bg_fg_mean = float("nan")

                    # -------- 你原本的幾個指標（保留但放後面）--------
                    prob_sum_mean = float(prob.sum(dim=1).mean().item())  # 應≈1
                    pred_unique = torch.unique(pred).detach().cpu().tolist()[:20]

                    print("===== DEBUG =====")
                    print(f"prob_sum_mean: {prob_sum_mean:.4f} (should be ~1)")
                    print(f"GT_fg_ratio: {gt_fg_ratio:.4f} | Pred_fg_ratio: {pred_fg_ratio:.4f}")
                    print(f"pred_unique: {pred_unique}")

                    print(f"[semantic_logits] mean/min/max: {s_mean:.4f} / {s_min:.4f} / {s_max:.4f}")
                    print(f"[logit_margin top1-top2] mean/min/max: {logit_margin_mean:.4f} / {logit_margin_min:.4f} / {logit_margin_max:.4f}")

                    print(f"[bg prob] mean/min/max: {float(bg.mean()):.4f} / {float(bg.min()):.4f} / {float(bg.max()):.4f}")
                    print(f"[fg prob(max over classes)] mean/min/max: {float(fg.mean()):.4f} / {float(fg.min()):.4f} / {float(fg.max()):.4f}")

                    print(f"[mask_logits] mean/min/max: {float(mask_logits.mean()):.4f} / {float(mask_logits.min()):.4f} / {float(mask_logits.max()):.4f}")
                    print(f"[mask_probs]  mean/min/max: {float(mask_probs.mean()):.4f} / {float(mask_probs.min()):.4f} / {float(mask_probs.max()):.4f}")
                    print(f"[den_raw] mean/min/max: {den_raw.mean():.4f} / {den_raw.min():.4f} / {den_raw.max():.4f}")
                    print(f"[den_clamped] mean/min/max: {den.mean():.4f} / {den.min():.4f} / {den.max():.4f}")
                    # den_raw: (B,H,W)
                    thr_list = [2, 3, 4, 6]
                    for t in thr_list:
                        ratio = (den_raw > t).float().mean().item()
                        print(f"pixels with den > {t}: {ratio:.4f}")
                    print(f"[area_per_q] mean/min/max: {area_q_mean:.4f} / {area_q_min:.4f} / {area_q_max:.4f}")

                    print(f"loss_ce: {float(loss_ce.item()):.4f}")
                    print(f"loss_dice: {float(loss_dice.item()):.4f}" if has_fg else "loss_dice: 0.0 (no fg)")
                    if gt_fg.any():
                        print(f"[on GT fg pixels] mean p(gt): {p_gt_fg_mean:.4f} | mean p(bg): {p_bg_fg_mean:.4f}")
                    else:
                        print("[on GT fg pixels] N/A (no fg in GT)")
                    print("[LOSS MIX]")
                    print(f"  CE       : {w_ce:.4f} ({w_ce/w_total:.1%})")
                    print(f"  Dice*w   : {w_dice:.4f} ({w_dice/w_total:.1%})  lambda={lambda_dice:.2f}")
                    print(f"  den*1e-3 : {w_den:.4f} ({w_den/w_total:.1%})")
                    print(f"  area*3e-3: {w_area:.4f} ({w_area/w_total:.1%})")
                    print(f"  ovlp*3e-3: {w_overlap:.4f} ({w_overlap/w_total:.1%})")
                    print(f"  fp*{lambda_fp_eff:.4f}: {w_fp:.4f} ({w_fp/w_total:.1%})")
                    fp_mean = float(p_fg_any[gt_bg].mean().item())
                    fp_q95  = float(torch.quantile(p_fg_any[gt_bg], 0.95).item())
                    print(f"[FP on GT_bg] mean={fp_mean:.4f}, q95={fp_q95:.4f}")
                    print("=================")
            if iter_num % 100 == 0:
                check_health(
                    iter_num=iter_num,
                    prob_sum_mean=prob_sum_mean,
                    gt_fg_ratio=gt_fg_ratio,
                    pred_fg_ratio=pred_fg_ratio,
                    pred_unique=pred_unique,
                    s_min=s_min,
                    s_max=s_max,
                    logit_margin_mean=logit_margin_mean,
                    den_min=den.min().item(),
                    den_max=den.max().item(),
                    mask_mean=float(mask_probs.mean().item()),
                    area_q_min=area_q_min,
                    area_q_max=area_q_max,
                    p_gt_fg_mean=(None if not gt_fg.any() else p_gt_fg_mean),
                    p_bg_fg_mean=(None if not gt_fg.any() else p_bg_fg_mean),
                )                    
            # ==========================================
            optimizer.zero_grad()
            loss.backward()
            #-----------check grad_norm---------------------    -----
            if iter_num % 50 == 0:
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
                "train/loss": loss.item(),
                "train/loss_ce": loss_ce.item(),
                "train/loss_dice": loss_dice.item(),
                "train/lr": lr_,
                "epoch": epoch_num,
                "train/pred_fg_ratio": pred_fg_ratio,
                "train/p_bg_on_fg": p_bg_on_fg,
                "train/lambda_dice": lambda_dice,
                "train/loss_den": loss_den.item(),
                "train/loss_area": area_pen.item(),
                "train/lambda_dice": lambda_dice,
            })
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/pred_fg_ratio', pred_fg_ratio, iter_num)
            writer.add_scalar('info/p_bg_on_fg', p_bg_on_fg, iter_num)
            writer.add_scalar('info/lambda_dice', lambda_dice, iter_num)



            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
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