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


args = parser.parse_args()





# device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # For M2 chip
# print(f"PyTorch version: {torch.__version__}")
# # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
# print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
# print(f"Is MPS available? {torch.backends.mps.is_available()}")
# # Set the device
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = torch.device(device)


print(f"Using device: {device}")

epochs = args.epochs
BATCH_SIZE = args.bs
lr = args.lr
dropout_ratio = args.dropout_ratio

# For Imagenet + Hands data preparation

# label & img preparation
# dataset_path = "/Volumes/disk_2t/datasets/mini_imagenet_and_hands"
dataset_path = "/var/datasets/miniimagenet"

img_pths, labels = [], []

# Get hands data
for img_pth in list(paths.list_images(os.path.join(dataset_path, '102')))[:600]:
    img_pths.append(img_pth)
    labels.append('hand')

    # image = cv2.imread(img_pth)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # data.append(image)
    # labels.append('hand')
print("Hand setup!")

# For imagenet_mini
for i, label in enumerate(list(os.listdir(os.path.join(dataset_path, 'archive')))):
    # print(i, " - ", label)
    img_pths_label = list(paths.list_images(os.path.join(dataset_path, 'archive', label)))
    img_pths += img_pths_label
    for j, img_pth in enumerate(img_pths_label):
        # print("- ", j)
        # img_pths.append(img_pth)
        labels.append(label)

print("Mini Imagenet setup!")


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

# divide the data into train, validation, and test set
(X, x_val, Y, y_val) = train_test_split(img_pths, labels,
                                         test_size=0.2,
                                         stratify=labels,
                                         random_state=42)
(x_train, x_test, y_train, y_test) = train_test_split(X, Y,
                                                      test_size=0.25,
                                                      random_state=42)
print(f"x_train examples: {len(x_train)}\nx_test examples: {len(x_test)}\nx_val examples: {len(x_val)}")


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

train_data = ImageDataset(x_train, y_train, train_transform)
val_data = ImageDataset(x_val, y_val, val_transform)
test_data = ImageDataset(x_test, y_test, val_transform)

# dataloaders
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# the resnet18 model
class ResNet18(nn.Module):
    def __init__(self, pretrained):
        super(ResNet18, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet18'](pretrained=None)

        # change the classification layer
        self.l0 = nn.Linear(512, len(lb.classes_))
        self.dropout = nn.Dropout2d(dropout_ratio)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0

# the resnet34 model
class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)

        # change the classification layer
        self.l0 = nn.Linear(512, len(lb.classes_))
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
# model = ResNet34(pretrained=True).to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss function
criterion = nn.CrossEntropyLoss()

# training function
def fit(model, dataloader):
    print('Training')
    model.train()
    running_loss = 0.0
    running_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, torch.max(target.to(torch.int32), 1)[1])
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == torch.max(target.to(torch.int32), 1)[1]).sum().item()
        loss.backward()
        optimizer.step()

    loss = running_loss/len(dataloader.dataset)
    accuracy = 100. * running_correct/len(dataloader.dataset)

    print(f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}")

    return loss, accuracy

#validation function
def validate(model, dataloader):
    print('Validating')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, torch.max(target.to(torch.int32), 1)[1])

            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == torch.max(target.to(torch.int32), 1)[1]).sum().item()

        loss = running_loss/len(dataloader.dataset)
        accuracy = 100. * running_correct/len(dataloader.dataset)
        print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f}')

        return loss, accuracy

def test(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, target = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == torch.max(target.to(torch.int32), 1)[1]).sum().item()
    return correct, total



train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []
print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples...")
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = fit(model, trainloader)
    val_epoch_loss, val_epoch_accuracy = validate(model, valloader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
end = time.time()
print((end-start)/60, 'minutes')

suffix = "lr_{}_bs_{}_dr_{}".format(args.lr, args.bs, args.dropout_ratio)


torch.save(model.state_dict(), f"outputs/models/resnet18_epochs{epochs}_{suffix}.pth")
# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'outputs/plots/accuracy_{suffix}.png')
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'outputs/plots/loss_{suffix}.png')

# save the accuracy and loss lists as pickled files
print('Pickling accuracy and loss lists...')
joblib.dump(train_accuracy, 'outputs/models/train_accuracy_{suffix}.pkl')
joblib.dump(train_loss, 'outputs/models/train_loss_{suffix}.pkl')
joblib.dump(val_accuracy, 'outputs/models/val_accuracy_{suffix}.pkl')
joblib.dump(val_loss, 'outputs/models/val_loss_{suffix}.pkl')

correct, total = test(model, testloader)
print('Accuracy of the network on test images: %0.3f %%' % (100 * correct / total))
print('train.py finished running')