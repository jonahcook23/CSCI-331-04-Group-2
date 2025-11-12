# a traffic sign detection and classification system

Title of the Project:  Traffic Sign Detection System

Abstract (a brief paragraph of your project): 


For our project:

Python libraries:-
    a) torch & torchvision: Core Pytorch libraries used for deep learning and computer vision
    b) maskrcnn_resnet50_fpn: Pretrained Mask R-CNNN model for instance segmentation. It also has a ResNet backnone
    c) Predictor classes (FastRCNN and MaskRCNN): help replace the last layer to match our dataset classes 
    d) transforms: Preprocessess images
    e) Dataset & DataLoader: Tools for handling images and masks in batches
    f) PIL and numpy: Manipulate and Load images (data)
    g) sklearn.metrics: Calculates precision, recall, and F1 score evaluation metrics criteria
    h) json: Saves data to a new file 
We will be comparing U-Net and Mask R-CNN (COCO-style instance annotations). Mask R-CNN will be used for object level instance segmentation and bounding boxes 


List of developers: Jonah Cook, Miguel De Vera, Bishop Oparaugo

Major contributions: 

Running project guides (steps, commands, etc.):



Project Sturucture
archive/
|____car/
    |___test/
    |   |___images/
    |   |___labels/
    |   |___masks/
    |___train/
    |   |___images/
    |   |___labels/
    |   |___masks/
    |___valid/
    |   |___images/
    |   |___labels/
    |   |___masks/

With images containing a '.jpg'file and corressponding masks with its binary mask files for data segmentation.

Be sure to download all dependencies from our 'archive/requirements.txt' file
"pip install -r requirements.txt"


Train U-Net:

Train Mask R-CNN:
For this segmentation, we utilized PyTorch's built-in 'maskrcnn_resnet50_fpn' to help in modelling for traffic sign detection.
The model uses transfer learning with pretrained COCO (Common Objects in Context) weights.
Running steps:
After cloning the github repository to your local repository
run these operations via terminal or bash:
cd code
python train_maskrcnn.py

This will load the data from 'archive/car/'
Train Mask R-CNN for 5 epochs
Evaluate using the IoU, precision, recall, and F1 score 
Save and return all the results into the created 'maskrcnn_model.pth' and 'maskrcnn_metrics.json' output files 

Example of results
json {
    "precision": 0.92,
    "recall": 0.89,
    "f1": 0.90
}

Evaluate: Focuses on testing performace and validation of the set
    Precision: How many detected signs are correct
    Recall: How many true signs were detected
    F1: Harmonic mean of precision and recall.

