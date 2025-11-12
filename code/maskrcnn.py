import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json


# File Configurations: Setting up our paths for training and validating the data, outputs, and model storage.

DATA_DIR = os.path.join('..', 'archive', 'car') 
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
OUTPUT_DIR = os.path.join('..', 'outputs')
MODEL_DIR = os.path.join('..', 'models')

#THis will ensure if folders exist or create one if not
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#Detects GPU or fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Defining our Dataset
class TrafficSignDataset(Dataset):
    #Reads all image and flask filenames in folder
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted([f for f in os.listdir(os.path.join(root, 'images')) if f.endswith('.jpg') or f.endswith('.png')])
        self.masks = sorted([f for f in os.listdir(os.path.join(root, 'masks')) if f.endswith('.png')])

#For a given index: it will load the image and mask, find unique objects in the mask (obj_ids), convert to binary masks per object, compute bounding boxes, and create a dictionary holding the coordinates of each object (boxes) , class labels, masks, and other variables before flipping or converting to tensor. 
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        mask_path = os.path.join(self.root, 'masks', self.masks[idx])

        img = Image.open(img_path).convert('RGB')
        mask = np.array(Image.open(mask_path))

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # remove background id = 0

        masks = mask == obj_ids[:, None, None]
        boxes = []
        for m in masks:
            pos = np.where(m)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target #Pytorch expects these variables to be returned to create each sample

    def __len__(self):
        return len(self.imgs)

# Transformations: Coverts images to PyTorch tensors
def get_transform(train):
    tfs = [T.ToTensor()]
    if train:
        tfs.append(T.RandomHorizontalFlip(0.5)) #Data Augmentation
    return T.Compose(tfs) #combines multiple transformations 

# Model Setup Implementation: Loads pretrained Mask RCNN, also it replaces the class. and mask. head to match the 2 num_classes
def get_instance_segmentation_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# Metrics for Evaluation: This will loop over the predictions and ground truth to calculate binary classification metrics 
def compute_metrics(preds, targets):
    y_true, y_pred = [], []
    for t, p in zip(targets, preds):
        y_true.append(t["masks"].sum().item() > 0) #Checks if any mask exists 
        y_pred.append(any([score > 0.5 for score in p["scores"]])) #prediction confidence 

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

# Training and Evaluation Loop
def train_model():
    num_classes = 2  # background + traffic sign
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    dataset = TrafficSignDataset(TRAIN_DIR, get_transform(train=True))
    dataset_val = TrafficSignDataset(VAL_DIR, get_transform(train=False))

    #collate_function: tarhets and images are returned as lists
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_function=lambda x: tuple(zip(*x)))
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2, collate_function=lambda x: tuple(zip(*x)))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) #learning rate reduced every 3 epochs

    num_epochs = 5
    print("Starting training...")
    #moves images/targets to GPU, calculates losses from model, and handles updating weights 
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for imgs, targets in data_loader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(data_loader):.4f}")

    print("Training complete. Evaluating...")
    model.eval()
    preds, targets_list = [], []
    with torch.no_grad():
        for imgs, targets in data_loader_val:
            imgs = list(img.to(device) for img in imgs)
            outputs = model(imgs)
            preds.extend(outputs)
            targets_list.extend(targets)

    precision, recall, f1 = compute_metrics(preds, targets_list)
    metrics = {"precision": precision, "recall": recall, "f1": f1}

    with open(os.path.join(OUTPUT_DIR, 'maskrcnn_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'maskrcnn_model.pth')) #saves trained weights 
    print("Saved model and metrics:", metrics)


if __name__ == '__main__':
    train_model() #only runs if script directly executes with no issues 
