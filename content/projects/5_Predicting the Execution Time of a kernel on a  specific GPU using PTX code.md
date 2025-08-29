---
title: "Predicting the Execution Time of a kernel on a  specific GPU using PTX code"
header:
  teaser: /assets/images/sumeet-singh-CuGWtbkd5-k-unsplash.jpg
sidebar:
  - title: "Languages and technologies"
    image: /assets/images/sumeet-singh-CuGWtbkd5-k-unsplash.jpg
    image_alt: "logo"
    text: "Python, Scikit-learn"
  - title: "Keywords"
    text: "CUDA, PTX, Machine learning, multi-threaded applications"
---


During the last couple of decades, there has been an exponential growth in the amount of time and energy required to run workloads on high-performance computing systems, which nowadays rely heavily upon GPUs. In order to reduce the resources required by these systems, one clear approach is to avoid inefficient applications by using prediction models that could inform developers of the approximate execution time. In this work, we have trained models based on ensemble learning techniques such as Random Forest and Gradient Boosted Decision trees, as well as the deep learning architecture TabNet to predict the execution time of a specific kernel on a specific GPU architecture. We used data obtained using the CUDA-Flux profiler from the PTX code as input features. The best performing model in terms of the number of predictions with an error in the range of (0-10%) is CatBoost with 91.6%, Random Forests with 85.4%, and TabNet with 76.6%.

[Paper](https://roldanjrgl.github.io/files/predicting_the_execution_time_of_a_kernel_on_a_specific_gpu_using_ptx_code.pdf)
