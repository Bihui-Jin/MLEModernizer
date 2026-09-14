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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.0507706255666364

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fixed the Albumentations API usage, removed the nonexistent model checkpoint load, and ensured all variables are defined before they’re used so the notebook runs end‑to‑end and writes a proper `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'I fixed the Albumentations transform call (used `RandomCrop` with required arguments), corrected the dataset class to inherit from `torch.utils.data.Dataset`, added missing imports (`datetime`), made the prediction collection store NumPy arrays, and ensured all variables are defined before they are used. These changes let the notebook run end‑to‑end and produce a proper `submission.csv` while keeping the original model and training logic unchanged, so the score stays close to the current value (still above the target).'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import albumentations as albu
import matplotlib.pyplot as plt
import json
import seaborn as sns
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import datetime  # needed for logging in train_model
import torch
from torch.utils.data import Dataset, DataLoader




## === cell 1
BASE_DIR = "../input/cassava-leaf-disease-classification/"
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "train_images")
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))




## === cell 2
train_df.head()




## === cell 3
print("Count of training images {0}".format(len(os.listdir(TRAIN_IMAGES_DIR))))




## === cell 4
with open(f"{BASE_DIR}/label_num_to_disease_map.json", "r") as f:
    name_mapping = json.load(f)
name_mapping = {int(k): v for k, v in name_mapping.items()}
train_df["class_id"] = train_df["label"].map(name_mapping)




## === cell 5
name_mapping




## === cell 6
sns.countplot(y=train_df["label"].map(name_mapping), orient="v")
plt.title("Target Distribution")
plt.show()




## === cell 7
def visualize_images(image_ids, labels):
    plt.figure(figsize=(16, 12))
    for ind, (image_id, label) in enumerate(zip(image_ids, labels)):
        plt.subplot(3, 3, ind + 1)
        image = cv2.imread(os.path.join(TRAIN_IMAGES_DIR, image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title(f"Class: {label}", fontsize=12)
        plt.axis("off")
    plt.show()


def plot_augmentation(image_id, transform):
    plt.figure(figsize=(16, 4))
    img = cv2.imread(os.path.join(TRAIN_IMAGES_DIR, image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    plt.subplot(1, 3, 3)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    plt.show()


def visualize(images, transform):
    fig = plt.figure(figsize=(32, 16))
    for i, im in enumerate(images):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        plt.imshow(im)
    for i, im in enumerate(images):
        ax = fig.add_subplot(2, 5, i + 6, xticks=[], yticks=[])
        plt.imshow(transform(image=im)["image"])




## === cell 8
tm_df = train_df.sample(9)
image_ids = tm_df["image_id"].values
labels = tm_df["class_id"].values
visualize_images(image_ids, labels)




## === cell 9
train_df[train_df.label == 2]




## === cell 10
tmp_df = train_df[train_df["label"] == 0]
print(f"Total train images for class 0: {tmp_df.shape[0]}")
tmp_df = tmp_df.sample(9)
visualize_images(tmp_df["image_id"].values, tmp_df["class_id"].values)




## === cell 11
tmp_df = train_df[train_df["label"] == 1]
print(f"Total train images for class 1: {tmp_df.shape[0]}")
tmp_df = tmp_df.sample(9)
visualize_images(tmp_df["image_id"].values, tmp_df["class_id"].values)




## === cell 12
tmp_df = train_df[train_df["label"] == 2]
print(f"Total train images for class 2: {tmp_df.shape[0]}")
tmp_df = tmp_df.sample(9)
visualize_images(tmp_df["image_id"].values, tmp_df["class_id"].values)




## === cell 13
tmp_df = train_df[train_df["label"] == 3]
print(f"Total train images for class 3: {tmp_df.shape[0]}")
tmp_df = tmp_df.sample(9)
visualize_images(tmp_df["image_id"].values, tmp_df["class_id"].values)




## === cell 14
tmp_df = train_df[train_df["label"] == 4]
print(f"Total train images for class 4: {tmp_df.shape[0]}")
tmp_df = tmp_df.sample(9)
visualize_images(tmp_df["image_id"].values, tmp_df["class_id"].values)




## === cell 15
train_augs = albu.Compose(
    [
        albu.RandomCrop(height=256, width=256, p=1.0),
        albu.Transpose(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(p=0.5),
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

valid_augs = albu.Compose(
    [
        albu.Resize(height=256, width=256, p=1.0),
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)




## === cell 16
from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    train_df, test_size=0.1, random_state=42, stratify=train_df.label.values
)
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)




## === cell 17
class CassavaDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None
    ):
        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]["image_id"])
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if self.transforms:
            x = self.transforms(image=x)["image"]
        if self.train:
            y = self.df.iloc[index]["label"]
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.df)




## === cell 18
train_dataset = CassavaDataset(
    df=train, imfolder=TRAIN_IMAGES_DIR, train=True, transforms=train_augs
)

valid_dataset = CassavaDataset(
    df=valid, imfolder=TRAIN_IMAGES_DIR, train=True, transforms=valid_augs
)




## === cell 19
def plot_image(img_dict):
    image_tensor = img_dict[0]
    target = img_dict[1]
    print(target)
    plt.figure(figsize=(10, 10))
    image = image_tensor.permute(1, 2, 0)
    plt.imshow(image)
    plt.show()




## === cell 20
plot_image(train_dataset[5])




## === cell 21
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    num_workers=4,
    shuffle=False,
)




## === cell 22
def train_model(n_epochs, model, criterion, optimizer, train_loader, device):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            outputs = model(batch)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{datetime.datetime.now()} Epoch {epoch} ,Training Loss {loss_train/len(train_loader):.4f}"
            )


def validate(model, train_loader, valid_loader, device):
    model.eval()
    for name, loader in [("train", train_loader), ("valid", valid_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print(f"Accuracy {name} {correct/total:.2f}")
    model.train()




## === cell 23
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 5)

optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()




## === cell 24
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)




## === cell 25
validate(model, train_loader, valid_loader, device)




## === cell 26
test_df = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
test_image_path = os.path.join(BASE_DIR, "test_images/")

test_aug = albu.Compose(
    [
        albu.CenterCrop(height=256, width=256, p=1.0),
        albu.Resize(height=256, width=256, p=1.0),
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

test_dataset = CassavaDataset(
    df=test_df, imfolder=test_image_path, train=False, transforms=test_aug
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    num_workers=4,
    shuffle=False,
)




## === cell 27
model.eval()
predictions = []
for imgs in test_loader:
    imgs = imgs.to(device)
    with torch.no_grad():
        outputs = model(imgs)
        _, predicted = torch.max(outputs, dim=1)
        predictions.append(predicted.cpu().numpy())




## === cell 28
test_df["label"] = np.concatenate(predictions)




## === cell 29
test_df.to_csv("submission.csv", index=False)
