# ISIC 2024 - Skin Cancer Detection with 3D-TBP

This repository is a quick overview of my solution for the ISIC 2024 Kaggle competition. 
The goal of the competition was to create machine learning models that would predict the probability of a skin lesion being malignant. 
For this, an image dataset of different lesions with additional tabular data was provided for training the model. For more information and details, see the 
[competition page](https://www.kaggle.com/competitions/isic-2024-challenge/overview).

## Approach overview

As every image in the dataset was provided additional tabular data, then the most prevalent approach in the competition was to use an ensemble of different gradient boosting machines. My final solution used a publicly shared [notebook](https://www.kaggle.com/code/vyacheslavbolotin/isic-2024-2-image-lines?scriptVersionId=193804105) as the starting point.

My addition to the existing architecture was to train a deep learning model, which would predict the probability of the image being malignant and provide its output as another feature for the gradient boosting models.

This is done in the notebook `FinalSubmission/skin-cancer-model-training-w-additional-data.ipynb`, where
the EfficientNetB0 model is used as the backbone. Additionally, in order to get more positive sample, the training data from the previous iterations of the competitions was used (years 2018-2020). 
Important step in the training process was also to use random weighted sampling, since the two classes were heavily unbalanced (ca 400000 vs 6000).

Finally, the notebook `FinalSubmission/isic-2024-nn.ipynb` combined the previously trained network predictions with the tabular data and produced the final results using gradient boosting models (final rank 225/2739).
All the notebooks in this repository are copied from Kaggle, while the original submission notebook can be found [here](https://www.kaggle.com/code/oskarkuuse/isic-2024-2-image-lines-nn-copy/notebook).
