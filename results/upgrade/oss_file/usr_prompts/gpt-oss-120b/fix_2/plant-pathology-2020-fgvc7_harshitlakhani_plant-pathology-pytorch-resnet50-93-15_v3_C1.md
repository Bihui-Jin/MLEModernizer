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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.87137

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
images_dir = os.path.join(root, "images")



## === cell 2
diseases = dict()
for column in ["healthy", "multiple_diseases", "rust", "scab"]:
    counts = pd.DataFrame(train[column].value_counts())
    diseases[column] = counts.iloc[1, 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 7))
ax1.bar(diseases.keys(), diseases.values(), color=["#6666ff"])
ax1.set_title("Bar Chart", fontsize=18)

ax2.pie(
    diseases.values(),
    labels=diseases.keys(),
    colors=["#6666ff", "#4da6ff", "#1ac6ff", "#c44dff"],
    autopct="%1.1f%%",
)
ax2.set_title("Pie Chart", fontsize=18)
ax2.axis("equal")
plt.show()




## === cell 3
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
print("Rows are in order of", col)

for column in col:
    images = (
        train[train[column].apply(lambda x: x == 1)]["image_id"].sample(4).values
    ) + ".jpg"
    ShowImages(images, column)




## === cell 4
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
    stratify=train_labels,
)

train_paths = train_paths.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
valid_paths = valid_paths.reset_index(drop=True)
valid_labels = valid_labels.reset_index(drop=True)



## === cell 5
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


class ImageDataset(Data.Dataset):
    def __init__(self, images_path, labels=None, test=False, transform=None):
        self.images_path = images_path
        self.test = test
        if not self.test:
            self.labels = labels
        self.images_transform = transform

    def __getitem__(self, index):
        image = cv2.imread(
            self.images_path.iloc[index]
            if isinstance(self.images_path, pd.Series)
            else self.images_path[index]
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transformed = self.images_transform(image=image)

        if not self.test:
            label = torch.tensor(np.argmax(self.labels.iloc[index].values))
            return image_transformed["image"], label
        return image_transformed["image"]

    def __len__(self):
        return len(self.images_path)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1615779727.py in <cell line: 0>()
      3         [
      4             A.Resize(height=256, width=256, p=1.0),
----> 5             A.Flip(p=0.5),
      6             A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
      7             A.Normalize(p=1.0),

AttributeError: module 'albumentations' has no attribute 'Flip'

## === cell 6
def train_function(model, loader):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    progress = tqdm(loader, desc="Training")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return running_loss / TRAIN_SIZE, accuracy




## === cell 7
def valid_function(model, loader):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    progress = tqdm(loader, desc="Validation")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        loss = loss_function(predictions, labels)

        running_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return running_loss / VALID_SIZE, accuracy, conf_matrix




## === cell 8
BATCH_SIZE = 64
NUM_EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SIZE = len(train_labels)
VALID_SIZE = len(valid_labels)
learning_rate = 5e-5

train_dataset = ImageDataset(
    images_path=train_paths, labels=train_labels, transform=mytransform["train"]
)
train_loader = Data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

valid_dataset = ImageDataset(
    images_path=valid_paths, labels=valid_labels, transform=mytransform["validation"]
)
valid_loader = Data.DataLoader(valid_dataset, shuffle=False, batch_size=BATCH_SIZE)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2879808018.py in <cell line: 0>()
      6 learning_rate = 5e-5
      7 
----> 8 train_dataset = ImageDataset(
      9     images_path=train_paths, labels=train_labels, transform=mytransform["train"]
     10 )

NameError: name 'ImageDataset' is not defined

## === cell 9
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Sequential(
    OrderedDict(
        [
            ("fc1", nn.Linear(num_features, 1024, bias=True)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(1024, 4, bias=True)),
        ]
    )
)

model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



## === cell 10
train_loss = []
valid_loss = []
train_acc = []
val_acc = []



## === cell 11
for epoch in range(NUM_EPOCHS):
    tl, ta = train_function(model, train_loader)
    vl, va, conf_mat = valid_function(model, valid_loader)

    train_loss.append(tl)
    valid_loss.append(vl)
    train_acc.append(ta)
    val_acc.append(va)

    print_str = (
        f"Epoch: {epoch}, Train loss: {tl:.4f}, Val loss: {vl:.4f}, "
        f"Train acc: {ta:.4f}, Val acc: {va:.4f}"
    )
    tqdm.write(print_str)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2968001536.py in <cell line: 0>()
      1 for epoch in range(NUM_EPOCHS):
----> 2     tl, ta = train_function(model, train_loader)
      3     vl, va, conf_mat = valid_function(model, valid_loader)
      4 
      5     train_loss.append(tl)

NameError: name 'train_loader' is not defined

## === cell 12
plt.figure()
plt.ylim(0, 1.5)
sns.lineplot(x=range(len(train_loss)), y=train_loss, label="Train")
sns.lineplot(x=range(len(valid_loss)), y=valid_loss, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/332361328.py in <cell line: 0>()
      1 plt.figure()
      2 plt.ylim(0, 1.5)
----> 3 sns.lineplot(x=range(len(train_loss)), y=train_loss, label="Train")
      4 sns.lineplot(x=range(len(valid_loss)), y=valid_loss, label="Val")
      5 plt.xlabel("Epoch")

NameError: name 'sns' is not defined

## === cell 13
plt.figure()
sns.lineplot(x=range(len(train_acc)), y=train_acc, label="Train")
sns.lineplot(x=range(len(val_acc)), y=val_acc, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1302666367.py in <cell line: 0>()
      1 plt.figure()
----> 2 sns.lineplot(x=range(len(train_acc)), y=train_acc, label="Train")
      3 sns.lineplot(x=range(len(val_acc)), y=val_acc, label="Val")
      4 plt.xlabel("Epoch")
      5 plt.ylabel("Accuracy")

NameError: name 'sns' is not defined

## === cell 14
def softmax_probs(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def test_function(model, loader):
    model.eval()
    preds = []

    progress = tqdm(loader, desc="Testing")
    with torch.no_grad():
        for images in progress:
            images = images.to(device)
            logits = model(images)
            probs = softmax_probs(logits.cpu().numpy())
            preds.append(probs)

    preds = np.vstack(preds)
    return preds




## === cell 15
test_dataset = ImageDataset(
    images_path=test_paths, test=True, transform=mytransform["validation"]
)
test_loader = Data.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

test_preds = test_function(model, test_loader)

submission_df = pd.DataFrame(
    test_preds, columns=["healthy", "multiple_diseases", "rust", "scab"]
)
submission_df.insert(0, "image_id", test["image_id"])
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3489535418.py in <cell line: 0>()
      1 # Prepare test loader
----> 2 test_dataset = ImageDataset(
      3     images_path=test_paths, test=True, transform=mytransform["validation"]
      4 )
      5 test_loader = Data.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

NameError: name 'ImageDataset' is not defined
