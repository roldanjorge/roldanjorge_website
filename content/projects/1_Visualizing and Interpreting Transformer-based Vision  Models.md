---
title: "Visualizing and Interpreting Transformer-based Vision  Models"
header:
  teaser: assets/images/project_1_image.png
sidebar:
  - title: "Languages and technologies"
    image:  assets/images/project_1_image.png
    image_alt: "logo"
    text: "Python, Pytorch"
  - title: "Keywords"
    text: "Interpretability, Visualization, Shapley values, Vision Transformer, Masked Auto-encoder"
---

Transformer-based vision models are increasingly popular and we need better ways to interpret and visualize their predictions. Previous works have been limited to visualizing attention maps; we apply a Shapley-value based method (FastSHAP) to Vision Transformers and Masked Au- toencoders, comparing the results to a classical ResNet. We find that choosing ResNet as the surrogate model for FastSHAP lets us successfully interpret and visualize transformer-based vision models. We observe that the estimated Shapley values of ResNet and ViT trained on CIFAR-10 are qualitatively different, even though the models’ predictions are mostly consistent. Keywords: Interpretability, Visualization, Shapley values, Vision Transformer, Masked Auto-encoder

[Paper](https://roldanjrgl.github.io/files/visualizing_and_interpreting_transformer_based_vision_models.pdf)

[//]: # ( TODO: Update project repository - include proper credit to original FastSHAP authors)
[//]: # (# [Github]&#40;https://github.com/roldanjrgl/ml_project&#41;)
