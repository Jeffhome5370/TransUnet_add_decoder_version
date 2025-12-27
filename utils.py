import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        if target.shape == inputs.shape:
            pass # 什麼都不做，直接用
        else:
            # 只有當形狀不一樣時 (例如 target 是單層索引)，才做 One-Hot
            if target.dim() == 4 and target.shape[1] == 1:
                target = target.squeeze(1)
            target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        target = target.float()
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # 1. 移除 Batch 維度: (1, D, H, W) -> (D, H, W)
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    # 2. 安全檢查：如果有時候數據是 (1, D, H, W) 導致 squeeze 後還是 (1, D, H, W)，再 squeeze 一次
    if len(image.shape) == 4:
        image = image.squeeze(0)
    if len(label.shape) == 4:
        label = label.squeeze(0)

    prediction = np.zeros_like(label)
    
    # 3. 逐層切片預測
    net.eval()
    with torch.no_grad():
        for ind in range(image.shape[0]):
            slice = image[ind, :, :] # 取出單張切片 (H, W)
            x, y = slice.shape[0], slice.shape[1]
            
            # 如果尺寸不合，進行縮放
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
                
            # 4. [關鍵修正] 轉為 Tensor 並確保是 4D: (1, 1, H, W)
            input = torch.from_numpy(slice).float().cuda()
            
            # 如果是 2D (H, W)，加兩個維度 -> (1, 1, H, W)
            if len(input.shape) == 2:
                input = input.unsqueeze(0).unsqueeze(0)
            # 如果是 3D (1, H, W)，加一個維度 -> (1, 1, H, W)
            elif len(input.shape) == 3:
                input = input.unsqueeze(0)
                
            # 再次確認維度 (防止 5D 錯誤)
            if len(input.shape) > 4:
                input = input.view(1, 1, input.shape[-2], input.shape[-1])

            
            raw_outputs = net(input)

            # 你的模型推論回傳 (None, semantic_segmentation)
            if isinstance(raw_outputs, (list, tuple)):
                outputs = raw_outputs[-1]   # 取 semantic_segmentation: (B, C, H, W)
            else:
                outputs = raw_outputs

            # 安全檢查：確保是 (B,C,H,W)
            if (not torch.is_tensor(outputs)) or outputs.dim() != 4:
                raise RuntimeError(f"Expected outputs to be 4D tensor (B,C,H,W), but got {type(outputs)}")

            if ind == 0:
                print("DEBUG outputs shape:", tuple(outputs.shape))
                print("DEBUG outputs min/max:", float(outputs.min().item()), float(outputs.max().item()))
                print("DEBUG outputs abs-mean:", float(outputs.abs().mean().item()))
                bg = outputs[:, 0].mean().item()
                mx = outputs.max(dim=1).values.mean().item()
                print("DEBUG bg_score_mean:", bg, "DEBUG max_score_mean:", mx)

            out = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

            if ind == 0:
                print("DEBUG pred unique:", np.unique(out)[:20], "fg:", float((out > 0).mean()))

            # resize 回原圖大小（如果有縮放）
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out

            prediction[ind] = pred

    # 計算評估指標
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
        
    return metric_list