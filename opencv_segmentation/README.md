# Keyhole Segmentition using the more traditional Thresholding Method

The package we use to achieve the segmentation is the opencv2 package. See this paper [Time-Resolved Geometric Feature Tracking Elucidates Laser-Induced Keyhole Dynamics](https://link.springer.com/article/10.1007/s40192-021-00241-4) for detailed information.

As shown in the above paprt, the threshold method works very well for good quality Ti64 synchrotron x-ray images. However, based on my personal experience applying it to AA6061, IN718, and images with lower contrasts, it cannot generate the desired segmentation masks I needed to extract the accurate geometric feature. This is a main motivation for the Deep Learning method, which generalize well given enough training data, i.e., ray images and human annotated masks.

