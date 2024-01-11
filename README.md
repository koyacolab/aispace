# Virtual Satellite

This project was inspired by the research presented in https://arxiv.org/abs/2210.06280 and its practical application in the 
[Kaggle Wild Blueberry Yield Prediction](https://www.kaggle.com/competitions/playground-series-s3e14) competition, 
in the [link](https://www.kaggle.com/code/inversion/make-synthetic-blueberry-yield-data).

The primary objective of this work is to adapt the mentioned technique for addressing challenges in satellite imagery, including tasks such as: generating Cloud-Free Satellite Images, creating continuous Virtual Satellite imagery, and more.

This project utilized the LLM Distilled GPT-2 version from [Hugging Face](https://huggingface.co/distilgpt2) because it more faster than the base GPT-2 and more suitable for fast R&D purposes. Additionally, [Sophia's 2nd order optimizer](https://arxiv.org/abs/2305.14342) was implemented for Hugging Face Trainer for rapid training and cost reduction.

The 'data' folder contains Landsat-8 harmonized imagery, chosen for its atmospheric correction, inclusion of cloud masks, and co-registration of all imagery for time-series usability during training.

******************************************************************************************

Example of Landsat-8 harmonized time series images, original and masked images. The masked Area of Interest (AOI) is excluded from the training data and will be generated by GPT-2 after the training process.

<img width="657" alt="Landsat-8" src="https://github.com/koyacolab/aispace/assets/115004547/dcc1853c-8655-4b5d-ab28-0b10dd50fd2c">

******************************************************************************************

Results, 

50x50 Landsat-8 patch:

<img width="632" alt="50x50" src="https://github.com/koyacolab/aispace/assets/115004547/fb1f6389-effe-43b2-9914-c042217d1826">

*****************************************************************************************

100x100 Landsat-8 patch:

<img width="632" alt="100x100" src="https://github.com/koyacolab/aispace/assets/115004547/01f2e598-9c25-478d-b50c-7d2f3382b073">

*****************************************************************************************

Conclusions and next steps:

Autoregressive Language Models such as GPT-2 utilize causal from the left to the right. Autoregressive approaches are preferable for generating long sequences (for example, entire documents), but since such causal models only condition on previous tokens, they cannot be applied to text-infilling tasks and cannot profit from MLM pre-training. 

