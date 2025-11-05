"""
Model: Torchvision for trainning Mask R-CNN
Usage(still modifying):
python train_maskrcnn.py --data_dir data/archive --fold 0 --epochs 10 --batch_size 4 --save_dir outputs/maskrcnn_fold0
python train_maskrcnn.py --data_dir data/archive --fold 0 --epochs 10 --batch_size 2 --lr 0.005 --save_dir outputs/maskrcnn_fold0

"""
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import MaskRCNNDataset
import argparse
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils # needed refernce for helper function
from tqdm import tqdm

def get_model_instance_segmentation(number_of_classes):
    #loads an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)
    #retrieves the number of input features for classifier
    input_features = model.roi_heads.box_predictor.cls_score.input_features
    #replace box predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, number_of_classes)
    #mask predictor
    input_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(input_features_mask, hidden_layer, number_of_classes)
    return model

def collate_function(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    losses = 0.0
    pbar = tqdm(data_loader)
    for images, targets in pbar:
        images = list(img.to(device)for img in images)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses_batch = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses_batch.backward()
        optimizer.step()
        losses += losses_batch.item()
        pbar.set_description(f"loss {losses_batch.item():.4f}")
    return losses / len(data_loader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = os.path.join(args.data_dir, "images")
    inst_dir = os.path.join(args.data_dir, "inst_masks")
    fold_dir = os.path.join(args.data_dir, f"fold_{args.fold}")
    train_list = os.path.join(fold_dir, "train.txt")
    val_list = os.path.join(fold_dir, "val.txt")

    def read_list(lst):
        with open(lst) as f:
            return [l.strip() for l in f if l.strip()]
    train_names = read_list(train_list)
    val_names = read_list(val_list)
    train_img_paths = [os.path.join(images_dir, n) for n in train_names]
    train_inst_paths = [os.path.join(inst_dir, os.path.splitext(n)[0]+".png") for n in train_names]
    val_img_paths = [os.path.join(images_dir, n) for n in val_names]
    val_inst_paths = [os.path.join(inst_dir, os.path.splitext(n)[0]+".png") for n in val_names]

    train_ds = MaskRCNNDataset(train_img_paths, train_inst_paths)
    val_ds = MaskRCNNDataset(val_img_paths, val_inst_paths)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_function, num_workers=2)

    num_classes = 2  # background + sign
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    best_map = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch} train_loss: {train_loss:.4f}")
        # Optionally runs a small evaluation pass computing IoU on masks for val set
        map_score = utils.simple_map_evaluation(model, val_loader, device)
        print(f"Validation simple MAP (mask IoU threshold): {map_score:.4f}")
        if map_score > best_map:
            best_map = map_score
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_maskrcnn.pth"))
    print("Done. best_map:", best_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--save_dir", default="outputs/maskrcnn")
    args = parser.parse_args()
    main(args)