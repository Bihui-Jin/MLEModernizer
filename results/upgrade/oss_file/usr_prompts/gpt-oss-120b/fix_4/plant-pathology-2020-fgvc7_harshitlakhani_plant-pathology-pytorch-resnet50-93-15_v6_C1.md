# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
        input/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_0', 'Train_1088', 'Train_1098', 'Train_1097']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]

-> input/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> input/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']

-> input/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_0', 'Train_1088', 'Train_1098', 'Train_1097']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]

-> working/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> (stopped after 10 files for performance)

# 5. Target score

0.91943

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import OrderedDict

import cv2
import torch
from torch import optim
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torchvision import models, transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
root = "/kaggle/input/plant-pathology-2020-fgvc7/"
train = pd.read_csv(os.path.join(root, "train.csv"))
test = pd.read_csv(os.path.join(root, "test.csv"))
submission = pd.read_csv(os.path.join(root, "sample_submission.csv"))
images = os.path.join(root, "images")




## === cell 2
def ShowImages(images, typ):
    fig = figure(figsize=(16, 12))
    number_of_images = len(images)
    for i in range(number_of_images):
        a = fig.add_subplot(1, number_of_images, i + 1)
        a.set_title(typ, fontsize=10)
        image = imread(os.path.join(root, "images", images[i]))
        imshow(image)
        axis("off")


col = ["healthy", "multiple_diseases", "rust", "scab"]
print("Row's are in order of", col)

for column in col:
    images_sample = (
        train[train[column].apply(lambda x: x == 1)]["image_id"].sample(4).values
    ) + ".jpg"
    ShowImages(images_sample, column)




## === cell 3
def get_path(image):
    return os.path.join(root, "images", image + ".jpg")


train_data = train.copy()
train_data["image_path"] = train_data["image_id"].apply(get_path)
train_labels = train.loc[:, "healthy":"scab"]

test_data = test.copy()
test_data["image_path"] = test_data["image_id"].apply(get_path)
test_paths = test_data["image_path"]

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_data["image_path"],
    train_labels,
    test_size=0.2,
    random_state=23,
)

train_paths.reset_index(drop=True, inplace=True)
train_labels.reset_index(drop=True, inplace=True)
valid_paths.reset_index(drop=True, inplace=True)
valid_labels.reset_index(drop=True, inplace=True)



## === cell 4
mytransform = {
    "train": A.Compose(
        [
            A.Resize(height=256, width=256, p=1.0),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ]
    ),
    "validation": A.Compose(
        [
            A.Resize(height=256, width=256, p=1.0),
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ]
    ),
}




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/368269167.py in <cell line: 0>()
      3         [
      4             A.Resize(height=256, width=256, p=1.0),
----> 5             A.Flip(p=0.5),
      6             A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
      7             A.Normalize(p=1.0),

AttributeError: module 'albumentations' has no attribute 'Flip'

## === cell 5
class ImageDataset(Data.Dataset):
    def __init__(self, images_path, labels=None, test=False, transform=None):
        self.images_path = images_path
        self.test = test
        if not self.test:
            self.labels = labels
        self.images_transform = transform

    def __getitem__(self, index):
        if not self.test:
            label = torch.tensor(self.labels.iloc[index, :].values, dtype=torch.float32)
        img_path = (
            self.images_path.iloc[index]
            if isinstance(self.images_path, pd.Series)
            else self.images_path[index]
        )
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transformed = self.images_transform(image=image)
        if not self.test:
            return image_transformed["image"], label
        return image_transformed["image"]

    def __len__(self):
        return self.images_path.shape[0]




## === cell 6
def train_function(model, loader):
    running_loss = 0.0
    total_elements = 0
    correct_elements = 0
    progress = tqdm(loader, desc="Training")
    for _, (images, labels) in enumerate(progress):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.shape[0]

        probs = torch.sigmoid(predictions)
        preds_binary = (probs > 0.5).float()
        correct_elements += (preds_binary == labels).float().sum().item()
        total_elements += labels.numel()
    accuracy = correct_elements / total_elements if total_elements > 0 else 0.0
    return running_loss / TRAIN_SIZE, accuracy




## === cell 7
def valid_function(model, loader):
    running_loss = 0.0
    total_elements = 0
    correct_elements = 0
    progress = tqdm(loader, desc="Validation")
    for _, (images, labels) in enumerate(progress):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            model.eval()
            predictions = model(images)
        loss = loss_function(predictions, labels)
        running_loss += loss.item() * labels.shape[0]

        probs = torch.sigmoid(predictions)
        preds_binary = (probs > 0.5).float()
        correct_elements += (preds_binary == labels).float().sum().item()
        total_elements += labels.numel()
    accuracy = correct_elements / total_elements if total_elements > 0 else 0.0
    return running_loss / VALID_SIZE, accuracy, None




## === cell 8
BATCH_SIZE = 64
NUM_EPOCHS = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SIZE = train_labels.shape[0]
VALID_SIZE = valid_labels.shape[0]
learning_rate = 5e-5



## === cell 9
train_images = ImageDataset(
    images_path=train_paths, labels=train_labels, transform=mytransform["train"]
)
train_loader = Data.DataLoader(train_images, shuffle=True, batch_size=BATCH_SIZE)

valid_images = ImageDataset(
    images_path=valid_paths, labels=valid_labels, transform=mytransform["validation"]
)
valid_loader = Data.DataLoader(valid_images, shuffle=False, batch_size=BATCH_SIZE)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2554579712.py in <cell line: 0>()
      1 train_images = ImageDataset(
----> 2     images_path=train_paths, labels=train_labels, transform=mytransform["train"]
      3 )
      4 train_loader = Data.DataLoader(train_images, shuffle=True, batch_size=BATCH_SIZE)
      5 

NameError: name 'mytransform' is not defined

## === cell 10
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False  # freeze pretrained layers

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, 4, bias=True),
)

model.to(device)
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



## === cell 11
train_loss = []
valid_loss = []
train_acc = []
val_acc = []

for epoch in range(NUM_EPOCHS):
    tl, ta = train_function(model, loader=train_loader)
    vl, va, _ = valid_function(model, loader=valid_loader)
    train_loss.append(tl)
    valid_loss.append(vl)
    train_acc.append(ta)
    val_acc.append(va)
    printstr = (
        f"Epoch: {epoch}, Train loss: {tl:.4f}, Val loss: {vl:.4f}, "
        f"Train acc: {ta:.4f}, Val acc: {va:.4f}"
    )
    tqdm.write(printstr)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1563281697.py in <cell line: 0>()
      5 
      6 for epoch in range(NUM_EPOCHS):
----> 7     tl, ta = train_function(model, loader=train_loader)
      8     vl, va, _ = valid_function(model, loader=valid_loader)
      9     train_loss.append(tl)

NameError: name 'train_loader' is not defined

## === cell 12
import seaborn as sns

plt.figure()
plt.ylim(0, 1.5)
sns.lineplot(x=list(range(len(train_loss))), y=train_loss, label="Train")
sns.lineplot(x=list(range(len(valid_loss))), y=valid_loss, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()



## === cell 13
plt.figure()
sns.lineplot(x=list(range(len(train_acc))), y=train_acc, label="Train")
sns.lineplot(x=list(range(len(val_acc))), y=val_acc, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()




## === cell 14
def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def test_function(model, loader):
    preds_for_output = np.zeros((1, 4))
    progress = tqdm(loader, desc="Testing")
    with torch.no_grad():
        for _, images in enumerate(progress):
            images = images.to(device)
            model.eval()
            predictions = model(images)
            preds_for_output = np.concatenate(
                (preds_for_output, sigmoid(predictions.cpu().detach().numpy())),
                axis=0,
            )
    preds_for_output = np.delete(preds_for_output, 0, 0)
    return preds_for_output




## === cell 15
test_images = ImageDataset(
    images_path=test_paths, test=True, transform=mytransform["validation"]
)
test_loader = Data.DataLoader(test_images, shuffle=False, batch_size=BATCH_SIZE)

predictions = test_function(model, test_loader)

submission[["healthy", "multiple_diseases", "rust", "scab"]] = predictions
submission.to_csv("submission_2.csv", index=False)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2771227449.py in <cell line: 0>()
      1 test_images = ImageDataset(
----> 2     images_path=test_paths, test=True, transform=mytransform["validation"]
      3 )
      4 test_loader = Data.DataLoader(test_images, shuffle=False, batch_size=BATCH_SIZE)
      5 

NameError: name 'mytransform' is not defined
