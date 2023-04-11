# Time Resolved Laser Energy Absorptance Prediction using the End-to-End Approach

## Data
A fragment of the raw absorptance data is available in the following links:
1. [A-AMB2022-01 Benchmark Challenge Problems](https://www.nist.gov/ambench/amb2022-01-benchmark-challenge-problems)
2. [Asynchronous AM Bench 2022 Challenge Data: Real-time, simultaneous absorptance and high-speed Xray imaging](https://data.nist.gov/od/id/mds2-2525)

To get the full dataset for the absorptance prediction, please send your request to either of the following people
* Runbo Jiang (rubyjiang2017@gmail.com or runboj@andrew.cmu.edu)
* Brian Simonds (brian.simonds@nist.gov)

For full data description, please read the the Table 2 and Table 3 of the paper.

<!-- <p align="center">
<img src="docs/absorptance_dataset.png" height="300">
<br>
<b>Laser energy absorptance dataset description</b>
</p> -->

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
    - To train the model in Jupyter Notebook, open the **model_train_aws.ipynb** file, and run each block sequentially. 
    - To run the model using **run.py** file, you can use the following command: `python run.py convnext --pretrain True --split_num 1`
    - In both methods, rememeber to change file path to where you store the data.


## Acknowledgement
This work is implemented using [ResNet50](https://github.com/KaimingHe/deep-residual-networks) and [ConvNeXt_tiny](https://github.com/facebookresearch/ConvNeXt). The model interpretation deployed the [CAM](https://github.com/jacobgil/pytorch-grad-cam). 

