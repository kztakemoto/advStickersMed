# advStickersMed
This repository contains code for assessing the vulnerability of medical mobile applications to physical camera-based adversarial attacks.

## Terms of Use
This project is MIT licensed. If you use this code in your research, please cite our paper:

Oda J and Takemoto K (2025) **Mobile applications for skin cancer detection are vulnerable to physical camera-based adversarial attacks.** PREPRINT (Version 1) available at Research Square. https://doi.org/10.21203/rs.3.rs-5934018/v1

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

Skin Lesion Images
```
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
```

Ground Truth Labels
```
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip
```

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

* ResNet-50: ``--net res50``
* VGG-16: ``--net vgg16``
* VGG-19: ``--net vgg19``
* MobileNetV2 ``--net mobilenetv2``
* MobileNetV3: ``--net mobilenetv3``
* EfficientNet-B1: ``--net efficientnet_b1``
* DenseNet-121: ``--net densenet121``
* DenseNet-201: ``--net densenet201``
* ViT-Small-16: ``--net vit_small_16``
* ViT-Base-16: ``--net vit_base_16``

### Generate Adversarial Camera Stickers
Generate an adversarial sticker with 25 dots using ResNet-18 as the surrogate model:
```
python generate_advStickers.py --nb_dots 25 --net res18
```

To specify the surrogate model, use the above arguments.

### Evaluete Transfer-Based Attacks
Compute the attack success rate of 25-dot stickers generated using ResNet-18 (surrogate model) against VGG-16 (target model):
```
python compute_ASR.py --nb_dots 25 --sticker_net res18 --net vgg16
```
