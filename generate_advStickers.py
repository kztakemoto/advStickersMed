# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy

import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as data

import os
import argparse

import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchsampler import ImbalancedDatasetSampler
import timm


# parsers
parser = argparse.ArgumentParser(description='adversarial attcks on medical DNN models')
parser.add_argument('--net', default='res18')
parser.add_argument('--bs', default='32', type=int) #default512
parser.add_argument('--size', default='224', type=int)
parser.add_argument('--nb_dots', help='Number of dots', default=25, type=int)
parser.add_argument('--radius_ratio', help='ratio of dot radius on image size', default=0.1, type=float)
parser.add_argument('--alpha', help='degree of opacity', default=0.2, type=float)
parser.add_argument('--beta', help='shape of the influence region of the color dots', default=0.6, type=float)
parser.add_argument('--target_label', help='Target label index for given image. A negative index indicates non-targeted attacks', default=-1, type=int)
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--epoch', help='Number of training epochs', default=30, type=int)
parser.add_argument('--num_ops', type=int, default='2')
parser.add_argument('--magnitude', type=int, default='14')
parser.add_argument('--dataset', default="melanoma")
parser.add_argument('--result_dir', default="results")
parser.add_argument("--seed", type=int, default=123, help="random seed")
args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bs = int(args.bs)
size = int(args.size)

# Data
print('==> Preparing data..')

# Load medical dataset
transform_train = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.RandAugment(num_ops=args.num_ops, magnitude=args.magnitude),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# Prepare dataset
if args.dataset == 'melanoma':
    print('melanoma used')
    data_dir = 'ISIC2018_dataset/'
else:
    raise ValueError("invalid dataset")

trainset = torchvision.datasets.ImageFolder(root = data_dir + 'train', transform=transform_train)
nb_classes = len(trainset.classes)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, sampler=ImbalancedDatasetSampler(trainset), num_workers=8)

testset = torchvision.datasets.ImageFolder(root = data_dir + 'test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
class_index = testset.classes
nb_classes = len(class_index)

print('target_label is {}'.format(args.target_label))
print('nb_dots is {}'.format(args.nb_dots))
print('beta is {}'.format(args.beta))
print('alpha is {}'.format(args.alpha))

# Model factory..
print('==> Building model..')
print('{} used'.format(args.net))
# net = VGG('VGG19')
print('==> Building model..')
if args.net=='res50':
    print('resnet50 used')
    net = timm.create_model('resnet50', pretrained=True, num_classes=nb_classes)
elif args.net=='res18':
    print('resnet18 used')
    net = timm.create_model('resnet18', pretrained=True, num_classes=nb_classes)
elif args.net=='vgg16':
    print('vgg16 used')
    net = timm.create_model('vgg16', pretrained=True, num_classes=nb_classes)
elif args.net=='vgg19':
    print('vgg19 used')
    net = timm.create_model('vgg19', pretrained=True, num_classes=nb_classes)
elif args.net=='densenet121':
    print('densenet121 used')
    net = timm.create_model('densenet121', pretrained=True, num_classes=nb_classes)
elif args.net=='densenet201':
    print('densenet201 used')
    net = timm.create_model('densenet201', pretrained=True, num_classes=nb_classes)
elif args.net=='mobilenetv2':
    print('mobilenetv2 used')
    net = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=nb_classes)
elif args.net=='mobilenetv3':
    print('mobilenetv3 used')
    net = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=nb_classes)
elif args.net=='efficientnet_b1':
    print('efficientnet_b1 used')
    net = timm.create_model('efficientnet_b1', pretrained=True, num_classes=nb_classes)
elif args.net=='vit_base_16':
    print('vit base 16 used')
    net = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=nb_classes)
elif args.net=='vit_small_16':
    print('vit small 16 used')
    net = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=nb_classes)
else:
    raise ValueError("invalid network model")

# For Multi-GPU
print('device', device)
if 'cuda' in device:
    print("using data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

# Load checkpoint.
print('==> Loading checkpoint..')
checkpoint = torch.load('../checkpoint_medical/{}-{}-bs256-adam-lr5e-05-randaug2-14ckpt.t7'.format(args.dataset, args.net))

net.load_state_dict(checkpoint['model'])

class ImageDot(nn.Module):
    """
    Class to treat an image with translucent color dots.
    forward method creates a blended image of base and color dots.
    Center positions and colors are hard-coded.
    """
    def __init__(self):
        super(ImageDot, self).__init__()
        self.means = [0.5, 0.5, 0.5] # need to change according to the transform
        self.stds = [0.5, 0.5, 0.5] # need to change according to the transform
        self.alpha = args.alpha
        self.beta = args.beta
        self.center = nn.Parameter(torch.tensor(np.random.random((args.nb_dots, 2)).tolist()), requires_grad=True)
        self.color = nn.Parameter(torch.tensor(np.random.random((args.nb_dots, 3)).tolist()), requires_grad=True)
        self.radius = float(args.size) * args.radius_ratio

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
    def __init__(self, net):
        super(AttackModel, self).__init__()
        self.image_dot = ImageDot().to(device)
        self.base_model = net.to(device).eval()
        self._freeze_pretrained_model()

    def _freeze_pretrained_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.image_dot(x)
        return self.base_model(x)


def compute_loss(pred: torch.Tensor, true_labels: torch.Tensor, target_label: int = -1) -> torch.Tensor:
    """
    Compute the adversarial loss for either non-targeted or targeted attacks.
    
    Args:
    pred (torch.Tensor): The model's predictions (logits).
    true_labels (torch.Tensor): The true labels of the inputs.
    target_label (int): The target label for targeted attacks. Use -1 for non-targeted attacks.
    
    Returns:
    torch.Tensor: The computed adversarial loss.
    """
    batch_size, num_classes = pred.shape
    
    if target_label < 0:  # Non-targeted attack
        # l_non-targeted = -log(H(k(x̂), 1_cx))
        cross_entropy = F.cross_entropy(pred, true_labels)
        loss = -torch.log(cross_entropy + 1e-10)
    
    else:  # Targeted attack
        # l_targeted = log(H(k(x̂), 1_t))
        target_labels = torch.full_like(true_labels, target_label)
        cross_entropy = F.cross_entropy(pred, target_labels)
        loss = torch.log(cross_entropy + 1e-10)
    
    return loss


# base file name
base_filename = f"{args.dataset}_{args.net}_target{args.target_label}_nb_dots{args.nb_dots}_radiusR{args.radius_ratio}_beta{args.beta}_alpha{args.alpha}_bs{args.bs}_lr{args.lr}_epoch{args.epoch}_seed{args.seed}"

# wrap the DNN model with adversarial stickers
model = AttackModel(net)

min_loss = float('inf')
max_asr_train = -float('inf')
max_asr_test = -float('inf')
best_model_params = None
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

# open log file
os.makedirs(f'{args.result_dir}/log', exist_ok=True)
log_file = open(f'{args.result_dir}/log/log_{base_filename}.txt', 'w')

for epoch in range(args.epoch):
    # train
    total_loss = 0.0
    asr_train = 0
    for images, labels in tqdm(trainloader, desc=f"Training (Epoch {epoch + 1})", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # adam
        optimizer.zero_grad()
        pred = model(images)
        loss = compute_loss(pred, labels, args.target_label)

        loss.backward()
        optimizer.step()

        for param in model.parameters():
            if param.requires_grad:
                param.data.clamp_(min=0.0, max=1.0)

        total_loss += float(len(labels)) * loss.item()
        
        _, predicted = pred.max(1)
 
        if args.target_label >= 0:
            asr_train += (predicted == torch.tensor([args.target_label] * len(labels)).to(device)).sum()
        else:
            asr_train += (predicted != labels).sum()
    
    # evaluation (compute ASR on test data)
    asr_test = 0
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc=f"Testing (Epoch {epoch + 1})", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = pred.max(1)

            if args.target_label >= 0:
                asr_test += (predicted == torch.tensor([args.target_label] * len(labels)).to(device)).sum()
            else:
                asr_test += (predicted != labels).sum()

    scheduler.step()

    # save the best model
    if total_loss < min_loss:
        min_loss = total_loss
        max_asr_train = asr_train / len(trainset)
        max_asr_test = asr_test / len(testset)
        best_model_params = copy.deepcopy(model.state_dict())

    print(f"epoch: {epoch + 1}, ave loss (on train): {total_loss / len(trainset):.4f}, asr (on train): {asr_train / len(trainset):.4%}, asr (on test): {asr_test / len(testset):.4%}")

    log_file.write(f"epoch: {epoch + 1}, Loss (on train): {total_loss / len(trainset):.4f}, ASR on train: {asr_train / len(trainset):.4%}, ASR on test: {asr_test / len(testset):.4%}\n")

log_file.write(f"----------\nMinimal loss (on train): {min_loss:.4f}, ASR on train: {max_asr_train:.4%}, ASR on test: {max_asr_test:.4%}\n")
# close log file
log_file.close()

# regenerate the best model (with the smallest loss)
model.load_state_dict(best_model_params)

# get the parameters for adversarial stickers
adv_stickers_parameters = []
for elem in model.parameters():
    if elem.requires_grad == True:
        adv_stickers_parameters.append(elem.cpu().detach().numpy())

# save the adversarial stickers parameters
os.makedirs(f'{args.result_dir}/para', exist_ok=True)
with open(f'{args.result_dir}/para/stickers_params_{base_filename}.plk', 'wb') as f:
    pickle.dump(adv_stickers_parameters, f)


# display sample images
def prepare_image(img):
    img = img.squeeze().cpu().detach().numpy()
    img = img.transpose(1, 2, 0)
    img = 0.5 * img + 0.5
    return img

plt.figure(figsize=(15, 6))

# original clean image
plt.subplot(1, 3, 1)
image, _ = testset[0]
image = prepare_image(image)
plt.imshow(image)
plt.title("Clean")
plt.axis('off')

# Adversarial Stickers
plt.subplot(1, 3, 2)
doted_img = model.image_dot(torch.ones(3, 224, 224).unsqueeze(0).to(device))
doted_img = prepare_image(doted_img)
plt.imshow(doted_img)
plt.title("Stickers")
plt.axis('off')

# Image with Adversarial Stickers
plt.subplot(1, 3, 3)
image, _ = testset[0]
doted_img = model.image_dot(image.unsqueeze(0).to(device))
doted_img = prepare_image(doted_img)
plt.imshow(doted_img)
plt.title("Adversarial")
plt.axis('off')

plt.tight_layout()
os.makedirs(f'{args.result_dir}/figs', exist_ok=True)
plt.savefig(f'{args.result_dir}/figs/fig_{base_filename}.png')
plt.close()
