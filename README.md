# Vision Transformer Experiments on Tiny-ImageNet

This project contains code for training different Vision Transformer (ViT) variants on the Tiny-ImageNet dataset.

## Dataset Setup

To run the code, you will first need to download the Tiny-ImageNet dataset:

- Download from [Kaggle - Tiny ImageNet Dataset](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet).
- This will give you a file named `archive.zip`.
- Create a folder called `data/` and unzip `archive.zip` inside it.


Once the dataset is ready, you can run the training scripts!

---

## How to Train Models

### 1. Train a Base ViT
Run the following command:
```bash
python vit_off.py
```

### 2. Run/Train a ViT with Hierarchical Locality Enforced Attention-Masking
Run the following command to run inference:
```bash
python vit_window_train.py
```
Run the following command to run training:
```bash
python window_inference.py
```

Note:
This script is set up to train a model with 6 transformer layers, which was the best-performing setup.
If you want to adjust to a different number of layers (other than 6 or 8), you may need to modify Transformer_local.py accordingly.

### 3. Training a ViT with early convolutions
Run the following command:
```bash
to be implemented
```

### 4. Training a ViT with channelwise splits
Run the following command:
```bash
to be implemented
```





