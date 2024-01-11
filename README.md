# Virtual Satellite

This project was inspired by the research presented in https://arxiv.org/abs/2210.06280 and its practical application in the 
[Kaggle Wild Blueberry Yield Prediction](https://www.kaggle.com/competitions/playground-series-s3e14) competition, 
in the [link](https://www.kaggle.com/code/inversion/make-synthetic-blueberry-yield-data).

The primary objective of this work is to adapt the mentioned technique for addressing challenges in satellite imagery, including tasks such as: generating Cloud-Free Satellite Images, creating continuous Virtual Satellite imagery, and more.

This project utilized the LLM Distilled GPT-2 version from [Hugging Face](https://huggingface.co/distilgpt2) because it more faster than the base GPT-2 and more suitable for fast R&D purposes. Additionally, [Sophia's 2nd order optimizer](https://arxiv.org/abs/2305.14342) was implemented for Hugging Face Trainer for rapid training and cost reduction.

The 'data' folder contains Landsat-8 harmonized imagery, chosen for its atmospheric correction, inclusion of cloud masks, and co-registration of all imagery for time-series usability during training.

******************************************************************************************

Example of Landsat-8 harmonized time series images, original and masked images. The masked Area of Interest (AOI) is excluded from the training dataset and will be generated by GPT-2 after the training process.

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

I. Autoregressive Language Models such as GPT-2 utilize causals from the left to the right. Autoregressive approaches are preferable for generating long sequences (for example, entire documents), but since such causal models only condition on previous tokens, they are worse applicable to be applied to text-infilling tasks and not profitable from MLM pre-training. In this project, this requires random permutations of tokens, which leads to increased training time as the dataset size increases. However, MLM approaches fail to generate longer sequences due to their independence assumption. To unify both worlds and retain the benefits of autoregressive modeling in combination with a bidirectional context, several methods will be provided in the next steps.

II. Representing every number as a single token is suboptimal due to a lack of generalization to new numbers and the sparsity of the provided tokens. Due to the inherent structure of numbers, learning the embeddings of numerical tokens in a purely data-driven way is ineffective. Moreover, since the GPT-2 is trained with cross-entropy loss, no notion of similarity between numerical tokens is conveyed. As a remedy, the simple inductive bias about the semantic proximity of numerical tokens, similar to positional encodings will be provided in the next steps. 

III. Today this POC project utilizes only optical data from the Landsat-8 satellite, the imagery frequency of which suffers due to the cloudiness and the sparse revisit time. In the future, this will be supplemented by Sentinel-1 Synthetic Aperture Radar images.

