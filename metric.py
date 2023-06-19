# imports
import argparse
import matplotlib.pyplot as plt
import matplotlib
import joblib
import cv2
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import pretrainedmodels
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
matplotlib.style.use('ggplot')
'''SEED Everything'''
def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True # keep True if all the input have same size.
SEED=42
seed_everything(SEED=SEED)
'''SEED Everything'''


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', type=int, default=10, help='training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--model_pth', type=str, default="outputs/models_server/resnet18_epochs10_lr_0.0001_bs_64_dr_0.6.pth",
                    help='loaded model saved path')


args = parser.parse_args()





# # device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For M2 chip
print(f"PyTorch version: {torch.__version__}")
# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")
# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)


print(f"Using device: {device}")

epochs = args.epochs
BATCH_SIZE = args.bs
lr = args.lr
dropout_ratio = args.dropout_ratio

# For Imagenet + Hands data preparation

# label & img preparation
dataset_path = "data/onehand10k_256x256_dataset/test/imgs"
dataset_path = "data/archive/n01532829"
dataset_path = "data/archive/n01558993"

img_pths, labels = [], []

# Get hands data
for img_pth in list(paths.list_images(dataset_path))[:600]:
    img_pths.append(img_pth)
    labels.append('hand')

print("Hand data setup!")



# data = data
labels = np.array(labels)
# one hot encode
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(f"Total number of classes: {len(lb.classes_)}")

# define transforms
train_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
val_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])


print(f"img_pths examples: {len(img_pths)}\n")


# custom dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transforms=None):
        self.image_paths = image_paths
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.image_paths))

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = cv2.imread(image_path)
        data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

evaluated_data = ImageDataset(img_pths, labels, val_transform)


# dataloaders
loader = DataLoader(evaluated_data, batch_size=BATCH_SIZE, shuffle=True)

# the resnet18 model
class ResNet18(nn.Module):
    def __init__(self, pretrained):
        super(ResNet18, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet18'](pretrained=None)

        # change the classification layer
        self.l0 = nn.Linear(512, 101)
        self.dropout = nn.Dropout2d(dropout_ratio)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0



model = ResNet18(pretrained=True).to(device)


if args.model_pth is not None:
    if os.path.exists(args.model_pth):
        print(">> Find model path!")
    model.load_state_dict(torch.load(args.model_pth, map_location=device))
    print(">> Model restored!")



def evaluate(model, dataloader):
    likelihoods = torch.zeros((1, 101)).to(device)
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(evaluated_data)/dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            likelihoods = torch.vstack([likelihoods, outputs])
            # print(target.shape, torch.max(target.to(torch.int32), 1)[1].shape)
            # if i == 0:
            #     print(torch.max(target.to(torch.int32), 1)[1])
    likelihood = torch.mean(likelihoods[1:, :], axis=0)
    return likelihood




likelihood = evaluate(model, loader)
likelihood = torch.exp(likelihood) / torch.sum(torch.exp(likelihood))# torch.log(torch.exp(likelihood) / torch.sum(torch.exp(likelihood)))
print(likelihood)

#%%
