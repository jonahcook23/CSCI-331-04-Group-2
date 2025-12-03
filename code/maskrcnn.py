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
from tqdm import tqdm  #Import for progress bar and batch tracking


def collate_fn(batch):
    """
    Since the batch is a list of tuples (img, target), this function zips 
    the images and targets together into two separate lists.
    This is required by PyTorch's object detection models.
    """
    return tuple(zip(*batch))


#Sets up paths for training and validating the data, outputs, and model storage.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'archive', 'car') 
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

#This will ensure if folders exist or create one if not
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#Detects GPU or fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#Defining the Dataset
class TrafficSignDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        #Define paths to image and mask folders
        image_folder = os.path.join(root, 'images')
        mask_folder = os.path.join(root, 'masks')
        
        #Check if directories exist
        if not os.path.isdir(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        if not os.path.isdir(mask_folder):
            raise FileNotFoundError(f"Mask folder not found: {mask_folder}") 
        
        #Collect image and mask filenames and ensure they are paired
        img_files = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))}
        mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_folder) if f.endswith('.png')}
        
        #Only use files that have both an image and a corresponding mask
        valid_filenames = sorted(list(img_files.intersection(mask_files)))
        
        self.items = []
        for name in valid_filenames:
            #Check for actual extension used in the image folder
            img_ext = '.jpg' if os.path.exists(os.path.join(image_folder, name + '.jpg')) else '.png'
            self.items.append({
                'img_path': os.path.join(image_folder, name + img_ext),
                'mask_path': os.path.join(mask_folder, name + '.png')
            })
            
        if not self.items:
            print(f"Warning: No paired images and masks found in {root}.")


    def __getitem__(self, idx):
        item = self.items[idx]
        
        #Use the stored full paths
        img = Image.open(item['img_path']).convert('RGB')
        
    
        #This prevents the ValueError related to broadcasting (N, H, W, 3 vs N, 1, 1).
        mask = np.array(Image.open(item['mask_path']).convert('L')) 
        
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # remove background id = 0

        #Handle the case where a mask file exists but contains no objects
        if len(obj_ids) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            #Operation for mask shape (H, W)
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

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.items)

#Transformations: Coverts images to PyTorch tensors
def get_transform(train):
    tfs = [T.ToTensor()]
    if train:
        tfs.append(T.RandomHorizontalFlip(0.5)) #Data Augmentation
    return T.Compose(tfs) #combines multiple transformations 

#Loads pretrained Mask RCNN, also it replaces the class. and mask. head to match the 2 num_classes
def get_instance_segmentation_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

#Metrics for Evaluation: This will loop over the predictions and ground truth to calculate binary classification metrics 
def compute_metrics(preds, targets):
    y_true, y_pred = [], []
    for t, p in zip(targets, preds):
        gt_objects_exist = t["boxes"].numel() > 0
        y_true.append(gt_objects_exist) 
        
        pred_objects_exist = any([score > 0.5 for score in p["scores"]])
        y_pred.append(pred_objects_exist) 

    if not y_true:
        return 0.0, 0.0, 0.0
        
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

#Training and Evaluation Loop
def train_model():
    num_classes = 2  # background + traffic sign
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    # Uses dynamically calculated paths
    dataset = TrafficSignDataset(TRAIN_DIR, get_transform(train=True))
    dataset_val = TrafficSignDataset(VAL_DIR, get_transform(train=False))

    if not dataset or not dataset.items:
        print(f"Error: Training dataset found no paired items at {TRAIN_DIR}. Check file paths and structure.")
        return
    if not dataset_val or not dataset_val.items:
        print(f"Warning: Validation dataset found no paired items at {VAL_DIR}. Continuing with training, but evaluation will be skipped.")

    # Using the globally defined collate_fn
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 

    num_epochs = 5
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Use tqdm for progress bar and batch loss tracking 
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        
        for imgs, targets in pbar:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            current_loss = losses.item() #Getsthe loss for the current batch

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += current_loss
            
            #Updates the progress bar with the current batch loss
            pbar.set_postfix({"Batch Loss": f"{current_loss:.4f}"})

        lr_scheduler.step()
        #Final average loss is printed after the progress bar completes
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss/len(data_loader):.4f}")

    print("Training complete. Evaluating...")
    
    if dataset_val and dataset_val.items:
        model.eval()
        preds, targets_list = [], []
        #Uses tqdm for validation progress as well
        pbar_val = tqdm(data_loader_val, desc="Validation", unit="batch")
        
        with torch.no_grad():
            for imgs, targets in pbar_val:
                imgs = list(img.to(device) for img in imgs)
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets] 
                
                outputs = model(imgs)
                preds.extend(outputs)
                targets_list.extend(targets_cpu)

        precision, recall, f1 = compute_metrics(preds, targets_list)
        metrics = {"precision": precision, "recall": recall, "f1": f1}

        with open(os.path.join(OUTPUT_DIR, 'maskrcnn_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'maskrcnn_model.pth')) 
        print("Saved model and metrics:", metrics)
    else:
        print("Evaluation skipped due to empty validation dataset.")
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'maskrcnn_model.pth'))
        print("Saved model weights only.")


if __name__ == '__main__':
    train_model()
    print(torch.__version__)
    print(torch.rand(2, 2))