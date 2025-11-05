import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

#Returns the image and it's target dictionary with boxes, labels, and masks.
class MaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, instance_mask_paths, transforms=None):
        self.image_paths = image_paths
        self.instance_mask_paths = instance_mask_paths
        self.transforms = transforms
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        imq = np.array(Image.open(self.image_paths[idx]).convert("RBG"))
        instance_mask = np.array(Image.open(self.instance_mask_paths[idx]).convert("L"))
        # Every instance has it's distinct pixel value
        object_ids = np.unique(instance_mask)
        object_ids = object_ids[object_ids != 0]
        masks = [(instance_mask == var).astype(np.unit8) for var in object_ids]
        boxes = []
        for m in masks:
            ys,xs = np.where(m)
            if ys.size ==0:
                boxes.append([0,0,0,0])
            else:
                xmin = np.min(xs)
                xmax = np.max(xs)
                ymin = np.min(ys)
                ymax = np.max(ys)
                boxes.append([xmin,ymin,xmax,ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64) #traffic sign as a single class
        masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
        iscrowd = torch.zeros((len(boxes), ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms: #updating masks after augmentation is nontrivial
            augment = self.transforms(image=image, masks=[m for m in masks.numpy()])
            image = augment['image']

            image = torch.as_tensor(image.transpose(2,0,1)).float() / 255.0
            return image, target 