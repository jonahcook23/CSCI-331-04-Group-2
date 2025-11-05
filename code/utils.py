# utils.py
import torch
import numpy as np

def masks_to_iou(pred_masks, gt_masks, threshold=0.5):
    """
    pred_masks: tensor [N, H, W] (float logits or probs)
    gt_masks: tensor [M, H, W] (binary)
    compute pairwise IoU matrix NxM
    """
    pred_bin = (pred_masks > threshold).float()
    if pred_bin.dim() == 2:
        pred_bin = pred_bin.unsqueeze(0)
    if gt_masks.dim() == 2:
        gt_masks = gt_masks.unsqueeze(0)
    N = pred_bin.shape[0]
    M = gt_masks.shape[0]
    iou = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        p = pred_bin[i].cpu().numpy()
        for j in range(M):
            g = gt_masks[j].cpu().numpy()
            inter = (p & g).sum()
            union = (p | g).sum()
            if union == 0:
                iou[i,j] = 0.0
            else:
                iou[i,j] = inter / union
    return iou

def simple_map_evaluation(model, data_loader, device, iou_thresh=0.5):
    """
    A simple mask-based map-like metric:
    For each GT instance, check if any predicted mask has IoU>threshold -> TP
    Compute precision-like score = TP / (TP + FP)
    This is a lightweight measure for model selection.
    """
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            # single image per batch assumed for val loader
            out = outputs[0]
            pred_masks = out.get("masks")  # [N,1,H,W]
            if pred_masks is not None and pred_masks.shape[0] > 0:
                pred_masks = pred_masks.squeeze(1)  # [N,H,W]
            else:
                pred_masks = torch.zeros((0,1,1), dtype=torch.uint8)
            gt_masks = targets[0]['masks'].to(device).float()
            if gt_masks.dim() == 3:
                pass
            # if both empty
            if pred_masks.shape[0] == 0 and gt_masks.shape[0] == 0:
                continue
            if pred_masks.shape[0] == 0:
                total_fn += gt_masks.shape[0]
                continue
            if gt_masks.shape[0] == 0:
                total_fp += pred_masks.shape[0]
                continue
            iou_mat = masks_to_iou(pred_masks, gt_masks, threshold=iou_thresh)
            # greedy matching
            matched_gt = set()
            matched_pred = set()
            for i in range(iou_mat.shape[0]):
                j = np.argmax(iou_mat[i])
                if iou_mat[i,j] >= iou_thresh:
                    matched_pred.add(i)
                    matched_gt.add(j)
            tp = len(matched_gt)
            fp = pred_masks.shape[0] - tp
            fn = gt_masks.shape[0] - tp
            total_tp += tp
            total_fp += fp
            total_fn += fn
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    # returns a single scalar for simple model selection
    return f1
