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
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # ------------------------------------------------------------------
    # 修改 1: 優化器只追蹤需要訓練的參數 (Decoder)
    # ------------------------------------------------------------------
    # 原本: optimizer = optim.SGD(model.parameters(), ...)
    # 修改後: 過濾 requires_grad=True 的參數
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(trainable_params, lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # ------------------------------------------------------------------

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            model_outputs = model(image_batch)
            # 新模型的輸出是 (class_logits, refined_masks_list)
            # 我們取出最後一層的 Mask 預測
            # 形狀通常是 (B, num_queries, 14, 14)
            _, refined_masks = model_outputs
            outputs = refined_masks[-1] 
            
            # 執行上採樣 (Upsampling) 到圖片原始大小 (例如 224x224)
            # 因為 Transformer Decoder 在 Patch Level 運作
            outputs = F.interpolate(outputs, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
            
            # 重要假設：
            # 這裡假設 num_queries == num_classes。
            # 如果你的 Query 數量 (20) 與 類別數量 (9) 不同，標準 Dice/CE Loss 會報錯。
            # 在這種簡單實作下，請在 train.py 確保 --num_queries 等於 --num_classes
            # ------------------------------------------------------------------
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            # --- [WandB] 紀錄訓練數值 ---
            wandb.log({
                "train/loss": loss.item(),
                "train/loss_ce": loss_ce.item(),
                "train/loss_dice": loss_dice.item(),
                "train/lr": lr_,
                "epoch": epoch_num
            })
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            # --- [WandB] 視覺化圖片 (每 50 個 Iteration) ---
            if iter_num % 50 == 0:
                # 1. 準備原圖 (取 batch 第一張, 轉為 numpy, 正規化到 0-1)
                image_show = image_batch[0, 0, :, :].cpu().detach().numpy()
                image_show = (image_show - image_show.min()) / (image_show.max() - image_show.min() + 1e-8)

                # 2. 準備預測 Mask (Argmax 轉成 0-8 的整數)
                pred_mask = torch.argmax(torch.softmax(outputs, dim=1), dim=1)[0].cpu().detach().numpy()
                
                # 3. 準備 Ground Truth Mask
                gt_mask = label_batch[0].cpu().detach().numpy()

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
        if (epoch_num % 5 == 0): 
            model.eval()
            val_dice_sum = 0.0
            num_val_batches = 0
            
            # 使用 tqdm 顯示驗證進度
            with torch.no_grad():
                for i_val, val_batch in tqdm(enumerate(valloader), total=len(valloader), desc=f"Validating Epoch {epoch_num}"):
                    val_img, val_label = val_batch['image'].cuda(), val_batch['label'].cuda()
                    
                    # --- Validation Forward ---
                    if args.add_decoder:
                        _, val_masks = model(val_img)
                        val_out = val_masks[-1]
                        val_out = F.interpolate(val_out, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
                    else:
                        val_out = model(val_img)
                    
                    # --- 計算 Dice Score ---
                    # 這裡為了效率，直接利用 DiceLoss 計算 (Dice = 1 - DiceLoss)
                    # 這樣可以快速得到 Batch 平均 Dice，不用做複雜的 Metric 計算
                    v_loss_dice = dice_loss(val_out, val_label, softmax=True)
                    val_dice = 1.0 - v_loss_dice.item()
                    
                    val_dice_sum += val_dice
                    num_val_batches += 1
            
            # 計算平均 Dice
            avg_val_dice = val_dice_sum / num_val_batches if num_val_batches > 0 else 0.0
            wandb.log({
                "val/mean_dice": avg_val_dice,
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
                
                if args.n_gpu > 1:
                    torch.save(model.module.state_dict(), save_best_path)
                else:
                    torch.save(model.state_dict(), save_best_path)
                
                logging.info("######## Saved new best model (Dice: {:.4f}) to {} ########".format(best_performance, save_best_path))
                wandb.run.summary["best_val_dice"] = best_performance

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"