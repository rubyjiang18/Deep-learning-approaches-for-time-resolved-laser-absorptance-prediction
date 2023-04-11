# Code release for model training

### Folder "end_to_end_approach"
You can find the code to train ResNet-50 and ConvNeXt-T for laser energy absorptance prediction. See README.md file in this folder for detailed documentataions.

### Folder "semantic_segmentation"
You can find code to train segmentation models to segmentent keyholes from the x-ray images. The geometric features can be easily extracted from the keyhole segments. See README.md file in this folder for detailed documentataions.

### Trained models
The top-performing trained models for these two practice (ConvNeXt-T and UNet)can be found in this [Google Drive link](https://drive.google.com/file/d/1hA5GGtu0Nk-4lgavteZWqTZZ2DW1Po1u/view?usp=sharing). You can load these weights for your down-stream tasks. In the above link, you can find each model has 5 splits trained seperately. **ConvNeXt-T** was trained using the absorptance dataset without powder layer, and the weights were initially initiated from the ImageNet pretrained weights. Average smooth L1 loss on test absorptance data of the ConvNeXt-T over the 5 splits is **2.35&plusmn;0.35**. Note that the prediction range for absorptance is [0,100]. **Unet** was trained from scratch using the segmentation dataset. The mIoU on test data is **90.4&plusmn;0.6**.

To find the data used in these models, check out the [offical website for keyhole absorptance prediction](https://rubyjiang18.github.io/keyholeofficial/).

