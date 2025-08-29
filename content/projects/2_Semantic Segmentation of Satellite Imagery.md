---
title: "Semantic Segmentation of Satellite Imagery"
header:
  teaser: assets/images/semantic_seg_sat_imgs_1.png
sidebar:
  - title: "Languages and technologies"
    image: assets/images/semantic_seg_sat_imgs_1.png
    image_alt: "logo"
    text: "Python, Pytorch"
  - title: "Keywords"
    text: "U-Net, ResNet, U-Net++, Semantic Segmentation, Satellite Imagery"
---

The ability to automatically classify the class at the pixel level of satellite imagery has a wide range of applications including monitoring, managing, and detecting changes of land cover. In this work, we used different versions of the ResNet model for classification tasks, and U-Net for the segmentation tasks. We explored the effect of three different losses, namely, Focal, Dice, and Cross entropy together with the mentioned models to find out the ideal configuration.


We trained and tested our models on three datasets: LandCoverNet, Crop Type In Ghana, and Landcover.ai. Due to challenges such as cloud cover and low-resolution images, we decided to mostly focus and tune our models using the Landcover.ai dataset. The mean IoU scores for different models on the Landcover.ai datasets we obtained are 0.884 for U-Net with Resnet34, 0.871 for U-Net with ResNet18, 0.887 for U-Net with ResNet101, and finally 0.886 for U-Net++ with ResNet34. We also observe that Focal loss performs significantly better than Dice loss when we have a skewed dataset on our hands.

[Paper](https://roldanjrgl.github.io/files/semantic_segmentation_of_satellite_imagery.pdf) -- 
[Repository](https://github.com/roldanjrgl/semantic_segmentation_for_satellite_imagery)
