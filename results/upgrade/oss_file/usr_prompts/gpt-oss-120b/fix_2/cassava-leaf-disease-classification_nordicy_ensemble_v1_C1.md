# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True



## === cell 3
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"



## === cell 4
test_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)



## === cell 5
resnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(
        self, dataframe, image_dir, transform_resnet=None, transform_efficientnet=None
    ):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform_resnet = transform_resnet
        self.transform_efficientnet = transform_efficientnet

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # Image ID
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_resnet = (
            self.transform_resnet(image=image)["image"]
            if self.transform_resnet
            else None
        )
        img_efficient = (
            self.transform_efficientnet(image=image)["image"]
            if self.transform_efficientnet
            else None
        )

        return img_resnet, img_efficient, img_name




## === cell 7
test_dataset = CassavaTestDataset(
    test_df,
    test_image_dir,
    transform_resnet=resnet_transforms,
    transform_efficientnet=efficientnet_transforms,
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)




## === cell 8
class CassavaTrainDataset(Dataset):
    def __init__(
        self, dataframe, image_dir, transform_resnet=None, transform_efficientnet=None
    ):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform_resnet = transform_resnet
        self.transform_efficientnet = transform_efficientnet

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_resnet = (
            self.transform_resnet(image=image)["image"]
            if self.transform_resnet
            else None
        )
        img_efficient = (
            self.transform_efficientnet(image=image)["image"]
            if self.transform_efficientnet
            else None
        )

        return img_resnet, img_efficient, torch.tensor(label, dtype=torch.long)




## === cell 9
train_df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
train_df, val_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)

train_dataset = CassavaTrainDataset(
    train_df,
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    transform_resnet=resnet_transforms,
    transform_efficientnet=efficientnet_transforms,
)
val_dataset = CassavaTrainDataset(
    val_df,
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    transform_resnet=resnet_transforms,
    transform_efficientnet=efficientnet_transforms,
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)



## === cell 10
resnet_model = models.resnet50(pretrained=True)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_ckpt_path = (
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth"
)
if os.path.exists(resnet_ckpt_path):
    try:
        resnet_model.load_state_dict(torch.load(resnet_ckpt_path, map_location=device))
        print("Loaded ResNet checkpoint.")
    except Exception as e:
        print(f"Failed to load ResNet checkpoint: {e}")
else:
    print("ResNet checkpoint not found – using ImageNet weights.")

resnet_model = resnet_model.to(device)
resnet_model.train()  # we will fine‑tune it lightly
for param in resnet_model.parameters():
    param.requires_grad = True  # allow fine‑tuning

efficientnet_model = models.efficientnet_v2_s(weights="DEFAULT")
num_features_efficientnet = efficientnet_model.classifier[1].in_features
efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)

eff_ckpt_path = "/kaggle/input/eff-t/pytorch/default/1/Eff.pth"
if os.path.exists(eff_ckpt_path):
    try:
        efficientnet_model.load_state_dict(
            torch.load(eff_ckpt_path, map_location=device)
        )
        print("Loaded EfficientNet checkpoint.")
    except Exception as e:
        print(f"Failed to load EfficientNet checkpoint: {e}")
else:
    print("EfficientNet checkpoint not found – using ImageNet weights.")

efficientnet_model = efficientnet_model.to(device)
efficientnet_model.train()
for param in efficientnet_model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    list(resnet_model.parameters()) + list(efficientnet_model.parameters()), lr=1e-4
)



## === cell 11
EPOCHS = 2
for epoch in range(EPOCHS):
    resnet_model.train()
    efficientnet_model.train()
    running_loss = 0.0
    for imgs_resnet, imgs_efficient, labels in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"
    ):
        imgs_resnet, imgs_efficient, labels = (
            imgs_resnet.to(device),
            imgs_efficient.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()
        out_resnet = resnet_model(imgs_resnet)
        out_eff = efficientnet_model(imgs_efficient)

        loss_resnet = criterion(out_resnet, labels)
        loss_eff = criterion(out_eff, labels)
        loss = (loss_resnet + loss_eff) / 2.0
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

    resnet_model.eval()
    efficientnet_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs_resnet, imgs_efficient, labels in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"
        ):
            imgs_resnet, imgs_efficient, labels = (
                imgs_resnet.to(device),
                imgs_efficient.to(device),
                labels.to(device),
            )
            probs_resnet = F.softmax(resnet_model(imgs_resnet), dim=1)
            probs_eff = F.softmax(efficientnet_model(imgs_efficient), dim=1)
            combined = (probs_resnet + probs_eff) / 2
            preds = combined.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1} validation accuracy: {val_acc:.4f}")

resnet_model.eval()
efficientnet_model.eval()



## === cell 12
ensemble_predictions = []
image_names = []

with torch.no_grad():
    for images_resnet, images_efficientnet, img_names in tqdm(
        test_loader, desc="Generating predictions"
    ):
        images_resnet = images_resnet.to(device)
        images_efficientnet = images_efficientnet.to(device)

        probs_resnet = F.softmax(resnet_model(images_resnet), dim=1)
        probs_efficientnet = F.softmax(efficientnet_model(images_efficientnet), dim=1)

        combined_probs = (probs_resnet + probs_efficientnet) / 2
        preds = combined_probs.argmax(dim=1).cpu().numpy()

        ensemble_predictions.extend(preds)
        image_names.extend(img_names)



## === cell 13
submission_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
