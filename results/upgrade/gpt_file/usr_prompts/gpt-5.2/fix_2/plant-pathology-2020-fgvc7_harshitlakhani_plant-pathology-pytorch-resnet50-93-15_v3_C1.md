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

torch.manual_seed(23)
np.random.seed(23)



## === cell 1
root = "/kaggle/input/plant-pathology-2020-fgvc7/"
train = pd.read_csv(os.path.join(root, "train.csv"))
test = pd.read_csv(os.path.join(root, "test.csv"))
submission = pd.read_csv(os.path.join(root, "sample_submission.csv"))
images = os.path.join(root, "images")

print(train.shape, test.shape, submission.shape)
print("Submission columns:", submission.columns.tolist())



## === cell 2
diseases = dict()

for column in ["healthy", "multiple_diseases", "rust", "scab"]:
    counts = pd.DataFrame(train[column].value_counts())
    diseases[column] = int(counts.loc[1, column]) if 1 in counts.index else 0

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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'healthy'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3937487664.py in <cell line: 0>()
      4     counts = pd.DataFrame(train[column].value_counts())
      5     # keep original intent (count positives) but guard if only one class appears
----> 6     diseases[column] = int(counts.loc[1, column]) if 1 in counts.index else 0
      7 
      8 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 7))

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1181             key = tuple(com.apply_if_callable(x, self.obj) for x in key)
   1182             if self._is_scalar_access(key):
-> 1183                 return self.obj._get_value(*key, takeable=self._takeable)
   1184             return self._getitem_tuple(key)
   1185         else:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _get_value(self, index, col, takeable)
   4212             return series._values[index]
   4213 
-> 4214         series = self._get_item_cache(col)
   4215         engine = self.index._engine
   4216 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _get_item_cache(self, item)
   4636             #  pending resolution of GH#33047
   4637 
-> 4638             loc = self.columns.get_loc(item)
   4639             res = self._ixs(loc, axis=1)
   4640 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'healthy'

## === cell 3
def ShowImages(images_list, typ):
    fig = figure(figsize=(16, 12))
    number_of_images = len(images_list)
    for i in range(number_of_images):
        a = fig.add_subplot(1, number_of_images, i + 1)
        a.set_title(typ, fontsize=10)
        image = imread(os.path.join(root, "images", images_list[i]))
        imshow(image)
        axis("off")


col = ["healthy", "multiple_diseases", "rust", "scab"]
print("Row's are in order of", col)

for column in col:
    img_ids = train[train[column].apply(lambda x: x == 1)]["image_id"]
    if len(img_ids) >= 4:
        images_list = (img_ids.sample(4, random_state=23).values) + ".jpg"
        ShowImages(images_list, column)




## === cell 4
def get_path(image):
    return os.path.join(root, "images", image + ".jpg")


train_data = train.copy()
train_data["image_path"] = train_data["image_id"].apply(get_path)
train_labels = train.loc[:, "healthy":"scab"]

test_data = test.copy()
test_data["image_path"] = test_data["image_id"].apply(get_path)
test_paths = test_data["image_path"]

try:
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_data["image_path"],
        train_labels,
        test_size=0.2,
        random_state=23,
        stratify=train_labels,
    )
except Exception as e:
    print("Stratified split failed; falling back to random split. Reason:", repr(e))
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_data["image_path"],
        train_labels,
        test_size=0.2,
        random_state=23,
        shuffle=True,
    )

train_paths.reset_index(drop=True, inplace=True)
train_labels.reset_index(drop=True, inplace=True)
valid_paths.reset_index(drop=True, inplace=True)
valid_labels.reset_index(drop=True, inplace=True)



## === cell 5
mytransform = {
    "train": A.Compose(
        [
            A.RandomResizedCrop(size=(256, 256), p=1.0),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ]
    ),
    "validation": A.Compose(
        [
            A.RandomResizedCrop(size=(256, 256), p=1.0),
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ]
    ),
}


class ImageDataset(Data.Dataset):
    def __init__(self, images_path, labels=None, test=False, transform=None):
        self.images_path = images_path
        self.test = test
        if self.test is False:
            self.labels = labels
        self.images_transform = transform

    def __getitem__(self, index):
        if self.test is False:
            labels = torch.tensor(
                int(np.argmax(self.labels.iloc[index, :])), dtype=torch.long
            )

        image = cv2.imread(self.images_path[index])
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.images_path[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transformed = self.images_transform(image=image)

        if self.test is False:
            return image_transformed["image"], labels
        return image_transformed["image"]

    def __len__(self):
        return self.images_path.shape[0]




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/835034012.py in <cell line: 0>()
      4         [
      5             A.RandomResizedCrop(size=(256, 256), p=1.0),
----> 6             A.Flip(p=0.5),
      7             A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
      8             A.Normalize(p=1.0),

AttributeError: module 'albumentations' has no attribute 'Flip'

## === cell 6
def train_function(model, loader):
    running_loss = 0.0
    preds_for_acc = []
    labels_for_acc = []

    progress = tqdm(loader, desc="Training")
    for _, (images_batch, labels_batch) in enumerate(progress):
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
        model.train()

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images_batch)
        loss = loss_function(predictions, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels_batch.shape[0]
        labels_for_acc = np.concatenate(
            (labels_for_acc, labels_batch.detach().cpu().numpy()), axis=0
        )
        preds_for_acc = np.concatenate(
            (preds_for_acc, np.argmax(predictions.detach().cpu().numpy(), axis=1)),
            axis=0,
        )

    accuracy = accuracy_score(labels_for_acc, preds_for_acc)
    return running_loss / TRAIN_SIZE, accuracy




## === cell 7
def valid_function(model, loader):
    running_loss = 0.0
    preds_for_acc = []
    labels_for_acc = []

    progress = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for _, (images_batch, labels_batch) in enumerate(progress):
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(
                device
            )
            model.eval()

            predictions = model(images_batch)
            loss = loss_function(predictions, labels_batch)

            running_loss += loss.item() * labels_batch.shape[0]
            labels_for_acc = np.concatenate(
                (labels_for_acc, labels_batch.detach().cpu().numpy()), axis=0
            )
            preds_for_acc = np.concatenate(
                (preds_for_acc, np.argmax(predictions.detach().cpu().numpy(), axis=1)),
                axis=0,
            )

    accuracy = accuracy_score(labels_for_acc, preds_for_acc)
    conf_matrix = confusion_matrix(labels_for_acc, preds_for_acc)
    return running_loss / VALID_SIZE, accuracy, conf_matrix




## === cell 8
BATCH_SIZE = 64
NUM_EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SIZE = train_labels.shape[0]
VALID_SIZE = valid_labels.shape[0]
learning_rate = 5e-5

print("device:", device, "TRAIN_SIZE:", TRAIN_SIZE, "VALID_SIZE:", VALID_SIZE)



## === cell 9
train_images = ImageDataset(
    images_path=train_paths, labels=train_labels, transform=mytransform["train"]
)
train_loader = Data.DataLoader(
    train_images, shuffle=True, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True
)

valid_images = ImageDataset(
    images_path=valid_paths, labels=valid_labels, transform=mytransform["validation"]
)
valid_loader = Data.DataLoader(
    valid_images, shuffle=False, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3430066562.py in <cell line: 0>()
----> 1 train_images = ImageDataset(
      2     images_path=train_paths, labels=train_labels, transform=mytransform["train"]
      3 )
      4 train_loader = Data.DataLoader(
      5     train_images, shuffle=True, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True

NameError: name 'ImageDataset' is not defined

## === cell 10
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

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

model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



## === cell 11
train_loss = []
valid_loss = []
train_acc = []
val_acc = []



## === cell 12
for epoch in range(NUM_EPOCHS):
    tl, ta = train_function(model, loader=train_loader)
    vl, va, conf_mat = valid_function(model, loader=valid_loader)
    train_loss.append(tl)
    valid_loss.append(vl)
    train_acc.append(ta)
    val_acc.append(va)

    printstr = (
        "Epoch: "
        + str(epoch)
        + ", Train loss: "
        + str(tl)
        + ", Val loss: "
        + str(vl)
        + ", Train acc: "
        + str(ta)
        + ", Val acc: "
        + str(va)
    )
    tqdm.write(printstr)

print("Last confusion matrix:\n", conf_mat)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3294033925.py in <cell line: 0>()
      1 for epoch in range(NUM_EPOCHS):
----> 2     tl, ta = train_function(model, loader=train_loader)
      3     vl, va, conf_mat = valid_function(model, loader=valid_loader)
      4     train_loss.append(tl)
      5     valid_loss.append(vl)

NameError: name 'train_loader' is not defined

## === cell 13
import seaborn as sns

plt.figure()
plt.ylim(0, 1.5)
sns.lineplot(x=list(range(len(train_loss))), y=train_loss)
sns.lineplot(x=list(range(len(valid_loss))), y=valid_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Val"])
plt.show()



## === cell 14
plt.figure()
sns.lineplot(x=list(range(len(train_acc))), y=train_acc)
sns.lineplot(x=list(range(len(val_acc))), y=val_acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Val"])
plt.show()




## === cell 15
def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def test_function(model, loader):
    preds_for_output = np.zeros((0, 4), dtype=np.float32)

    progress = tqdm(loader, desc="Testing")
    with torch.no_grad():
        for _, images_batch in enumerate(progress):
            images_batch = images_batch.to(device)
            model.eval()
            predictions = model(images_batch)
            preds_for_output = np.concatenate(
                (
                    preds_for_output,
                    sigmoid(predictions.detach().cpu().numpy()).astype(np.float32),
                ),
                axis=0,
            )
    return preds_for_output




## === cell 16
test_images = ImageDataset(
    images_path=test_paths.reset_index(drop=True),
    test=True,
    transform=mytransform["validation"],
)
test_loader = Data.DataLoader(
    test_images, shuffle=False, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True
)

test_preds = test_function(model, loader=test_loader)
print("test_preds shape:", test_preds.shape)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/786384348.py in <cell line: 0>()
----> 1 test_images = ImageDataset(
      2     images_path=test_paths.reset_index(drop=True),
      3     test=True,
      4     transform=mytransform["validation"],
      5 )

NameError: name 'ImageDataset' is not defined

## === cell 17
sub = submission.copy()
target_cols = [c for c in sub.columns if c != "image_id"]

if test_preds.shape[1] != len(target_cols):
    raise ValueError(
        f"Prediction shape {test_preds.shape} doesn't match target columns {target_cols}"
    )

sub["image_id"] = test["image_id"].values
sub[target_cols] = test_preds

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3557014191.py in <cell line: 0>()
      3 target_cols = [c for c in sub.columns if c != "image_id"]
      4 
----> 5 if test_preds.shape[1] != len(target_cols):
      6     raise ValueError(
      7         f"Prediction shape {test_preds.shape} doesn't match target columns {target_cols}"

NameError: name 'test_preds' is not defined
