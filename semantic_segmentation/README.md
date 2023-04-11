# Time Resolved Laser Energy Absorptance Prediction using the Modular Appraoch
# This code focuses on the semantic segmentation model training 

## Data
The segmentation dataset, including x-ray images and annotated masks, are openly available using this [Google Drive link](https://drive.google.com/file/d/1scn1lq92aqQWnjsfhLXDHEULpccxze93/view?usp=sharing).

For full data description, please read the the Table 4 of the paper.

## Requirements
The following packages are required to run the training program:

| Package  | Version  |
| :------------ |:---------------|
| CUDA                  | 11.0    |
| Python                | 3.8.5   |
| Pytorch |
| torchvision |
| imagecodecs |

## To run the model training code
1. Set up AWS, Google Colab, or any other platform that has GPU support to train deep learning models
2. Install all necessary packages
3. Train the model
    - To train the model in Jupyter Notebook, open the `model_train_segmentation.ipynb` file, and run each block sequentially. 
    - To run the model using `run.py` file, you can use the following command:
     `python run.py unet --pretrain False --split_num 1`
    - In both methods, rememeber to change data path to where you store your training data.


