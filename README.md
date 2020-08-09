# Deep-Distance-Map-Regression-for-Tumor-Segmentation
# Three-stage-Curriculum-Training-for-Tumor-Segmentation

## 0. Introduction
This repository contains Pytorch code for the paper entitled with"Deep Distance Map Regression Network with Shape-aware Loss for Imbalanced Medical Image Segmentation" . This paper was initially described in .... 
## 1. Getting Started
Clone the repo: https://github.com/Huiyu-Li/Deep-Distance-Map-Regression-for-Tumor-Segmentation.git
#### Requirements
~~~
python>=3.6
torch>=0.4.0
torchvision
csv
pandas
json
scipy
SimpleITK
medpy
numpy
time
shutil
sys
os
~~~
## 2. Data Prepare
   You need to have downloaded at least the LiTS 2017 training dataset.
   First, you are supposed to make a dataset directory.
   Second, you may need to preprocess the data by  https://github.com/Huiyu-Li/Preprocess-of-CT-data
   Third, change the file path in the **hyperparameters** part in the Main.py
#### prepare for ground Truth Distance map
Preprocess_DistanceMap.py
There are many different kinds of distance maps. You can use by your purpose.
## 3. Usage
### To train the MapNet( the light-weighted regression network):
main function>new_Gt_MapNet.py
### To train the whole segmentaion framework:
Threre are many different kinds of main funtions, where "Sig" means sign norn inverse distance map usage, 's' means norm inverse distance map with sigmoid activation function.
main function>new_Gt_MapDicecascadeL1.py
