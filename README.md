# a traffic sign detection and classification system

Title of the Project:  Traffic Sign Detection System

Abstract (a brief paragraph of your project): 
For our project, we will be comparing U-Net and Mask R-CNN (COCO-style instance annotations). Mask R-CNN will be used for object level instance segmentation and bounding boxes 
Our contents:
requirements.txt
data_prep file: This will prepare our datasets, builds appropriate masks, and create any neccessary folds. 
dataset file: Includes most dataset utility functions.
train_maskrcnn file: This python file will handles Mask R CNN training using primarily torchvision
eval file: Handles evaluations (IoU, precision, recall, F1) for both model ofutputs.
maskrcnn_training.ipynb: our runnable jupyter notebook 

List of developers: Jonah Cook, Miguel De Vera, Bishop Oparaugo

Major contributions: 

Running project guides (steps, commands, etc.):

Download Kaggle dataset and place raw images and annotations into a consistent layout:

data/archive/.../.../images/*.jpg
data/archive/.../.../labels/*.txt   

Prepare processed data & folds:

Train U-Net:

Train Mask R-CNN:

Evaluate:


