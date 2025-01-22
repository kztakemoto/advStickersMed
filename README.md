# advStickersMed
This repository contains code for assessing the vulnerability of medical mobile applications to physical camera-based adversarial attacks.

## Terms of Use
This project is MIT licensed. If you use this code in your research, please cite our paper:

Oda J and Takemoto K (2025) Mobile applications for skin cancer detection are vulnerable to physical camera-based adversarial attacks.

## Requirements

* Python 3.11
* PyTorch (v2.3.0)

Install dependencies:
```
pip install -r requirements.txt
```

## Usage
### Download Skin Lesion Image Dataset
Download the dataset from [ISIC 2018 Challenge.](https://challenge.isic-archive.com/data/#2018)

Direct links:
* [Skin Lesion Images](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip)
* [Ground Truth Labels](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip)

### Format the Dataset
```
python split_train_test.py
```

### Train Surrogate Models
Train a ResNet-18 model:
```
python train_surrogate_models.py --net res18
```

Available model architectures:

* VGG-16: ``--net vgg16``
* MobileNetV2 ``--net mobilenetv2``
* EfficientNet-B1: ``--net efficientnet_b1``
* DenseNe-121: ``--net densenet121``
* ViT-Small-16: ``--net vit_small_16``
* MobileNetV3: ``--net mobilenetv3``
* ResNet-50: ``--net res50``
* ViT-Base-16: ``--net vit_base_16``
* DenseNet-201: ``--net densenet201``
* VGG-19: ``--net vgg19``

### Generate Adversarial Camera Stickers
Generate a adversarial sticker with 25 dots using ResNet-18 as the surrogate model:
```
python generate_advStickers.py --nb_dots 25 --net res18
```

To specify the surrogate model, use the above arguments.

### Evaluete Transfer-Based Attacks
Compute the attack success rate of 25-dot stickers generated using ResNet-18 (surrogate model) against VGG-16 (target model):
```
python compute_ASR.py --nb_dots 25 --sticker_net res18 --net vgg16
```
