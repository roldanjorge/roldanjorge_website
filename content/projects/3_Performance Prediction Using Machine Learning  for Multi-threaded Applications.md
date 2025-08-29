---
title: "Performance Prediction Using Machine Learning  for Multi-threaded Applications"
header:
  teaser: assets/images/project_performance_pred_ml_multi_app_fig1.png
sidebar:
  - title: "Languages and technologies"
    image: assets/images/project_performance_pred_ml_multi_app_fig1.png
    image_alt: "logo"
    text: "Python, Scikit-learn"
  - title: "Keywords"
    text: "Machine learning, multi-threaded applications, performance prediction"
---
Multi-core processors, in 2022 are the default choice for a variety of computing needs from small Internet-of-Things devices to large super-computing clusters.  While developing applications and algorithms are primarily done with the needs of a single-threaded or single-core processor at the forefront, running these applications or algorithms in a multi-threaded setting is increasingly important with the limits of the processor manufacturing nearing the edge of quantum physics. 

Running the programs in a multi-threaded setting poses a set of challenges and prediction of its performance is increasingly complex with the various parameters involved when moving from a single-threaded execution environment to a multi-threaded environment. Thus having a simulation and prediction model for performance gain without actually running it on the often-expensive hardware is of growing importance.

In this paper, we propose a learning based approach for predicting the performance of standard multi-threaded benchmarks. We attempt to achieve this by collecting various underlying parameters of the application execution and under various input parameters. We then evaluate various machine learning models and to estimate the expected performance.
To achieve this, we use industry standard, publicly available benchmark suites, Parsec 3.0 and Splash 3.0 and use a standard profiler called perf, which is included in the Linux kernel. We found that the best performing model to predict the speedup to be the K-Neighbors regressor with an average mean absolute error of 0.8.

[Paper](https://roldanjrgl.github.io/files/performance_prediction_using_machine_learning_for_multi_threaded_applications.pdf) -- 
[Repository](https://github.com/roldanjrgl/performance_prediction_using_ml_for_multi_threaded_apps)
