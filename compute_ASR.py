import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import numpy as np
import pickle
import argparse
import os

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--net', default='res18')
parser.add_argument('--dataset', default="melanoma")
parser.add_argument('--sticker_net', default='vgg16', type=str)
parser.add_argument('--target_label', help='Target label index for given image. A negative index indicates non-targeted attacks', default=-1, type=int)
parser.add_argument('--radius_ratio', help='ratio of dot radius on image size', default=0.1, type=float)
parser.add_argument('--size', default='224', type=int)
parser.add_argument('--nb_dots', default='25', type=int)
parser.add_argument('--alpha', help='degree of opacity', default=0.2, type=float)
parser.add_argument('--beta', help='shape of the influence region of the color dots', default=0.6, type=float)
parser.add_argument('--result_dir', default="results")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.Resize((args.size, args.size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Prepare dataset
if args.dataset == 'melanoma':
    data_dir = 'ISIC2018_dataset/'
else:
    raise ValueError("invalid dataset")

testset = torchvision.datasets.ImageFolder(root = data_dir + 'test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
class_index = testset.classes
nb_classes = len(class_index)

if args.net=='res50':
    net = timm.create_model('resnet50', pretrained=True, num_classes=nb_classes)
elif args.net=='res18':
    net = timm.create_model('resnet18', pretrained=True, num_classes=nb_classes)
elif args.net=='vgg16':
    net = timm.create_model('vgg16', pretrained=True, num_classes=nb_classes)
elif args.net=='vgg19':
    net = timm.create_model('vgg19', pretrained=True, num_classes=nb_classes)
elif args.net=='densenet121':
    net = timm.create_model('densenet121', pretrained=True, num_classes=nb_classes)
elif args.net=='densenet201':
    net = timm.create_model('densenet201', pretrained=True, num_classes=nb_classes)
elif args.net=='efficientnet_b1':
    net = timm.create_model('efficientnet_b1', pretrained=True, num_classes=nb_classes)
elif args.net=='mobilenetv2':
    net = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=nb_classes)
elif args.net=='mobilenetv3':
    net = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=nb_classes)
elif args.net=='vit_base_16':
    net = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=nb_classes)
elif args.net=='vit_small_16':
    net = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=nb_classes)
else:
    raise ValueError("invalid network model")

# load stickers
with open(f"{args.result_dir}/para/stickers_params_{args.dataset}_{args.sticker_net}_target{args.target_label}_nb_dots{args.nb_dots}_radiusR{args.radius_ratio}_beta{args.beta}_alpha{args.alpha}_bs32_lr0.001_epoch30_seed123.plk", 'rb') as f:
    stickers = pickle.load(f)

#####################################
class ImageDot(nn.Module):
    """
    Class to treat an image with translucent color dots.
    forward method creates a blended image of base and color dots.
    Center positions and colors are hard-coded.
    """
    def __init__(self, stickers):
        super(ImageDot, self).__init__()
        self.means = [0.5, 0.5, 0.5] # need to change according to the transform
        self.stds = [0.5, 0.5, 0.5] # need to change according to the transform
        self.alpha = args.alpha
        self.radius = float(args.size) * args.radius_ratio
        self.beta = args.beta
        self.center = nn.Parameter(torch.tensor(stickers[0].tolist()), requires_grad=False)
        self.color = nn.Parameter(torch.tensor(stickers[1].tolist()), requires_grad=False)

    def forward(self, x):
        _, _, height, width = x.shape
        blended = x
        for idx in range(self.center.shape[0]):
            mask = self._create_circle_mask(height, width,
                                            self.center[idx] * width, self.beta)
            normalized_color = self._normalize_color(self.color[idx],
                                                     self.means, self.stds)
            blended = self._create_blended_img(blended, mask, normalized_color)
        return blended

    def _normalize_color(self, color, means, stds):
        return list(map(lambda x, m, s: (x - m) / s, color, means, stds))

    def _create_circle_mask(self, height, width, center, beta):
        hv, wv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)],  indexing='ij')
        hv, wv = hv.type(torch.FloatTensor).to(device), wv.type(torch.FloatTensor).to(device)
        d = ((hv - center[0]) ** 2 + (wv - center[1]) ** 2) / self.radius ** 2
        return torch.exp(- d ** beta + 1e-10)

    def _create_blended_img(self, base, mask, color):
        alpha_tile = self.alpha * mask.expand(3, mask.shape[0], mask.shape[1])
        color_tile = torch.zeros_like(base).to(device)
        for c in range(3):
            color_tile[:, c, :, :] = color[c]
        return (1. - alpha_tile) * base + alpha_tile * color_tile

class AttackModel(nn.Module):
    """
    Class to create an adversarial example.
    forward method returns the prediction result of the perturbated image.
    """
    def __init__(self, net, stickers):
        super(AttackModel, self).__init__()
        self.image_dot = ImageDot(stickers).to(device)
        self.base_model = net.to(device).eval()

    def forward(self, x):
        x = self.image_dot(x)
        return self.base_model(x)

#####################################

# Load checkpoint
checkpoint = torch.load('checkpoint_medical/{}-{}-bs256-adam-lr5e-05-randaug2-14ckpt.t7'.format(args.dataset, args.net))

# For Multi-GPU
if 'cuda' in device:
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

net.load_state_dict(checkpoint['model'])

# wrap the DNN model with adversarial stickers
model_adv = AttackModel(net, stickers)

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model_adv(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Test ASR
if args.target_label < 0:
    asr = np.mean(np.array(all_predictions) != np.array(all_labels))
else:
    asr = np.mean(np.array(all_predictions) == args.target_label)

print("===========================")
print(f"Dataset: {args.dataset}")
print(f"Surrogate: {args.sticker_net}, Target: {args.net}")
print(f"ASR: {asr:.4%}")

# print(f"{args.dataset}\t{args.target_label}\t{args.sticker_net}\t{args.net}\t{args.nb_dots}\t{args.alpha}\t{args.beta}\t{asr*100.0:.4f}")

# confusion matrix
cm = confusion_matrix(np.array(all_labels), np.array(all_predictions))

# row normalization
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

# save CSV file
df_cm = pd.DataFrame(
    cm, 
    index=class_index, 
    columns=class_index
)
os.makedirs(f"{args.result_dir}/conf_mtx/", exist_ok=True)
csv_filename = f"{args.result_dir}/conf_mtx/confusion_matrix_{args.dataset}_nb_dots{args.nb_dots}_alpha{args.alpha}_beta{args.beta}_{args.sticker_net}_to_{args.net}_target{args.target_label}.csv"
df_cm.to_csv(csv_filename)

df_cm_norm = pd.DataFrame(
    cm_normalized, 
    index=class_index, 
    columns=class_index
)

# display the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(
    df_cm_norm, 
    annot=True, 
    fmt='.1f',
    cmap='Blues',
    xticklabels=class_index,
    yticklabels=class_index,
    cbar=False,
)
plt.title(f'Confusion Matrix ({args.dataset})\n{args.sticker_net} to {args.net} (Target: {args.target_label})\n#dots:{args.nb_dots}, alpha:{args.alpha}, beta:{args.beta}')
plt.xlabel('Predicted')
plt.ylabel('True')

# save figure
plt.savefig(f"{args.result_dir}/conf_mtx/confusion_matrix_{args.dataset}_nb_dots{args.nb_dots}_alpha{args.alpha}_beta{args.beta}_{args.sticker_net}_to_{args.net}_target{args.target_label}.png")
