# Virtual Satellite

This project was inspired by the research presented in https://arxiv.org/abs/2210.06280 and its practical application in the 
[Kaggle Wild Blueberry Yield Prediction](https://www.kaggle.com/competitions/playground-series-s3e14) competition, 
in the [link](https://www.kaggle.com/code/inversion/make-synthetic-blueberry-yield-data).

The primary objective of this work is to adapt the mentioned technique for addressing challenges in satellite imagery, including tasks such as: generating Cloud-Free Satellite Images, creating continuous Virtual Satellite imagery, and more.

This project utilized the LLM distilled gpt-2 version from huggingface because it more faster than base gpt-2, and Sophia's 2nd order optimizer was implemented for fast training and cost reduction.

The data folder contains Landsat-8 harmonized imagery because it's atmospherically corrected, contains cloud masks, and all imagery coregistered for time series usability for training.
