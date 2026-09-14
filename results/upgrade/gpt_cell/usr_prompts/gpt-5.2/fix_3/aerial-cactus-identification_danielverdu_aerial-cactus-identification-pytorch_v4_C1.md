# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.986

# 6. Current score

0.77275

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.77275) has done: 'Diagnosis: The crash happens because `train_path` and `test_path` are set to `../kaggle/temp/...`, but in this environment the extracted zip contents live under `/kaggle/...` (and may include an extra `aerial-cactus-identification/` folder). `_resolve_extracted_dir()` currently generates incorrect candidates like `../kaggle/temp/train//train` and doesn’t consider the dataset subdirectory, so it fails to find the real extracted folders and `os.listdir()` raises `FileNotFoundError`.  
Patch summary: Update `_resolve_extracted_dir()` to deterministically search a small set of plausible extraction roots (including both `../kaggle/temp/` and `/kaggle/temp/`, plus the optional `aerial-cactus-identification/` subfolder) and to properly normalize paths without duplicating `/train` or `/test`. Keep the rest of the cell’s behavior and outputs the same.  
Updated cells: Only cell 4 is changed.  
Compatibility notes for cell k+1: `train_path` and `test_path` remain string directory paths ending with `/`, so downstream dataset loading in later cells continue to work unchanged.  
Assumptions: The zip extraction in cell 3 succeeded and produced `train/` and `test/` directories either directly under the temp folder or under `aerial-cactus-identification/` within it.'

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import cv2
from sklearn.model_selection import train_test_split
from itertools import product
from PIL import Image
import glob
import zipfile

import torch
import torchvision
from torchvision import models,transforms,datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,ConcatDataset


## === cell 2
data_dir = '../input/aerial-cactus-identification/'
out_dir = './'

train_path = '../kaggle/temp/train/'
test_path = '../kaggle/temp/test/'


## === cell 3
with zipfile.ZipFile(data_dir + "train.zip","r") as z:
    z.extractall("../kaggle/temp/")
    
with zipfile.ZipFile(data_dir + "test.zip","r") as z:
    z.extractall("../kaggle/temp/")


## === cell 4
def _resolve_extracted_dir(p):
    p = (p or "").replace("\\", "/")
    p_norm = p.rstrip("/") + "/"

    def _norm_dir(d):
        d = (d or "").replace("\\", "/")
        d = d.rstrip("/") + "/"
        return d

    roots = []
    roots.append(p_norm)
    if p_norm.startswith("../kaggle/"):
        roots.append("/kaggle/" + p_norm[len("../kaggle/") :])
    if p_norm.startswith("/kaggle/"):
        roots.append("../kaggle/" + p_norm[len("/kaggle/") :])

    parent = os.path.dirname(p_norm.rstrip("/")) + "/"
    if parent and parent != p_norm:
        roots.append(_norm_dir(parent))
        if parent.startswith("../kaggle/"):
            roots.append(_norm_dir("/kaggle/" + parent[len("../kaggle/") :]))
        if parent.startswith("/kaggle/"):
            roots.append(_norm_dir("../kaggle/" + parent[len("/kaggle/") :]))

    split = None
    if "/train/" in p_norm:
        split = "train"
    elif "/test/" in p_norm:
        split = "test"

    candidates = []
    for r in roots:
        r = _norm_dir(r)
        candidates.append(r)
        candidates.append(r + "aerial-cactus-identification/")
        if split is not None:
            candidates.append(r + split + "/")
            candidates.append(r + "aerial-cactus-identification/" + split + "/")

    if split is None:
        for r in roots:
            r = _norm_dir(r)
            candidates.append(r + "train/")
            candidates.append(r + "test/")
            candidates.append(r + "aerial-cactus-identification/train/")
            candidates.append(r + "aerial-cactus-identification/test/")

    seen = set()
    for c in candidates:
        c = _norm_dir(c)
        if c in seen:
            continue
        seen.add(c)
        if os.path.isdir(c):
            return c

    return p_norm


train_path = _resolve_extracted_dir(train_path)
test_path = _resolve_extracted_dir(test_path)

print("Num train samples:{0}".format(len(os.listdir(train_path))))
print("Num test samples:{0}".format(len(os.listdir(test_path))))


## === cell 5
labels = pd.read_csv(data_dir + 'train.csv')
sub = pd.read_csv(data_dir + 'sample_submission.csv')


## === cell 6
labels.head()


## === cell 7
labels.info()


## === cell 8
num_cactus = labels[labels['has_cactus']==1]['id'].count()
num_no_cactus = labels[labels['has_cactus']==0]['id'].count()


## === cell 9
tags = 'Cactus', 'No cactus'
sizes = [num_cactus, num_no_cactus]
explode = (0, 0.1)  # "explode" the 2nd slice

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=tags, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Number of images with/without cactus")
plt.show()


## === cell 10
fig,ax = plt.subplots(1,5,figsize=(15,3))

for i, idx in enumerate(labels['id'][-5:]):
    path = os.path.join(train_path,idx)
    ax[i].imshow(cv2.imread(path)) # [...,[2,1,0]]


## === cell 11
num_epochs = 20
num_classes = 2
batch_size = 32
learning_rate = 0.002

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


## === cell 12
train, val = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)
train.shape, val.shape, labels.shape


## === cell 13
class MyDataset(Dataset):
    def __init__(self, df_data, data_dir = './', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name,label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


## === cell 14
trans_train = transforms.Compose([transforms.ToPILImage(),
                                  transforms.ToTensor()])

trans_valid = transforms.Compose([transforms.ToPILImage(),
                                  transforms.ToTensor()])

dataset_train = MyDataset(df_data=train, data_dir=train_path, transform=trans_train)
dataset_valid = MyDataset(df_data=val, data_dir=train_path, transform=trans_valid)

loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True)
loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size, shuffle=False)


## === cell 15
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(4)
        self.fc = nn.Linear(128, 1)
        self.out = nn.Sigmoid()
   
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)))) # first convolutional layer then batchnorm, then activation then pooling layer.
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.avg(x)
        x = x.view(-1, 128) # !!!
        x = self.fc(x)
        x = self.out(x)
        return x


## === cell 16
model = SimpleCNN().to(device)


## === cell 17
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))


## === cell 18
total_step = len(loader_train)
best_loss = 1000

for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(loader_train):
        targets = targets.float()
        
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs[:,0], targets)
        
        if loss < 0.0001:
            break
        
        
        loss.backward()
        optimizer.step()
        
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


## === cell 19
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    
    for images, targets in loader_valid:
        predicted = []
        targets = targets.float()
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        
        for out in outputs.data:
            if out < 0.5:
                predicted.append(0)
            else:
                predicted.append(1)
                
        predicted = torch.FloatTensor(predicted).to(device)

        total += targets.size(0)
        correct += (predicted == targets).sum()
          
    print('Test Accuracy of the model on the 1750 validation images: {} %'.format(100 * correct / total))


## === cell 20
dataset_test = MyDataset(df_data=sub, data_dir=test_path, transform=trans_valid)
loader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)


## === cell 21
model.eval()

preds = []
for test_img, test_target in loader_test:
    test_target = test_target.float()
    test_img, test_target = test_img.to(device), test_target.to(device)
    test_out = model(test_img)

    if test_out < 0.5:
        preds.append(0)
    else:
        preds.append(1)

sub['has_cactus'] = preds
sub.to_csv('submission.csv', index=False)

sub.head()


## === cell 22
sub[sub['has_cactus']==0]['has_cactus'].count()
