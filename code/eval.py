# eval.py
"""
Evaluates saved models on a dataset and compute IoU, precision, recall, F1.
Supports Mask R-CNN (instance masks).
Usage (incomplete, need to modify):
python eval.py --model_type maskrcnn --model_path outputs/maskrcnn_fold0/best_maskrcnn.pth --data_dir data/archive --fold 0

"""
import argparse
import os
import torch
from dataset import MaskRCNNDataset
from torch.utils.data import DataLoader
#from models.unet import UNet
import numpy as np
from PIL import Image
from utils import masks_to_iou

def eval_maskrcnn(model_path, data_dir, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import torchvision
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    images_dir = os.path.join(data_dir, "images")
    inst_dir = os.path.join(data_dir, "inst_masks")
    fold_dir = os.path.join(data_dir, f"fold_{fold}")
    val_list = os.path.join(fold_dir, "val.txt")
    with open(val_list) as f:
        names = [l.strip() for l in f if l.strip()]

    from utils import masks_to_iou
    ious = []
    for n in names:
        img_p = os.path.join(images_dir, n)
        inst_p = os.path.join(inst_dir, os.path.splitext(n)[0] + ".png")
        img = Image.open(img_p).convert("RGB")
        import torchvision.transforms as T
        x = T.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(list(x))[0]
        pred_masks = out.get("masks")  # [N,1,H,W]
        if pred_masks is None or pred_masks.shape[0] == 0:
            continue
        pred_masks = pred_masks.squeeze(1).cpu()
        gt_mask = np.array(Image.open(inst_p).convert("L"))
        object_ids = np.unique(gt_mask)
        object_ids = object_ids[object_ids != 0]
        gt_masks = []
        for v in object_ids:
            gt_masks.append((gt_mask == v).astype(np.uint8))
        gt_masks = torch.as_tensor(np.stack(gt_masks, axis=0))
        # compute pairwise IoU, average best per GT
        iou_mat = masks_to_iou(pred_masks, gt_masks, threshold=0.5)
        best = iou_mat.max(axis=0) if iou_mat.size else np.array([])
        if best.size:
            ious.append(best.mean())
    print("Mean instance IoU (over images with instances):", np.mean(ious))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["maskrcnn"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    if args.model_type == "maskrcnn":
        eval_maskrcnn(args.model_path, args.data_dir, args.fold)
