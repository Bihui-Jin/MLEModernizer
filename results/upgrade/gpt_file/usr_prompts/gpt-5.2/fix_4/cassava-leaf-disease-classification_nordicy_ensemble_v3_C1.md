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
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass



## === cell 2
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_image_dir = f"{DATA_DIR}/train_images"
test_image_dir = f"{DATA_DIR}/test_images"

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
train_df.head(), test_df.head()



## === cell 3
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




## === cell 4
class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, "image_id"]
        label = int(self.dataframe.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, torch.tensor(label, dtype=torch.long)


class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, rgb_cache=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.rgb_cache = rgb_cache  # dict: image_id -> np.ndarray RGB

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]

        if self.rgb_cache is not None and img_name in self.rgb_cache:
            image = self.rgb_cache[img_name]
        else:
            img_path = os.path.join(self.image_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 6
tr_df, va_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

batch_size = 32


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

num_workers = min(4, os.cpu_count() or 1)

loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)

train_loader_resnet = DataLoader(
    CassavaTrainDataset(tr_df, train_image_dir, transform=resnet_transforms),
    shuffle=True,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)
val_loader_resnet = DataLoader(
    CassavaTrainDataset(va_df, train_image_dir, transform=resnet_transforms),
    shuffle=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)

train_loader_eff = DataLoader(
    CassavaTrainDataset(tr_df, train_image_dir, transform=efficientnet_transforms),
    shuffle=True,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)
val_loader_eff = DataLoader(
    CassavaTrainDataset(va_df, train_image_dir, transform=efficientnet_transforms),
    shuffle=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)



## === cell 7
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
resnet_model = resnet_model.to(device)

efficientnet_model_1 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.DEFAULT
)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)
efficientnet_model_1 = efficientnet_model_1.to(device)

efficientnet_model_7 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.DEFAULT
)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_7 = efficientnet_model_7.to(device)

efficientnet_model_8 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.DEFAULT
)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_8 = efficientnet_model_8.to(device)

if device.type == "cuda":
    resnet_model = resnet_model.to(memory_format=torch.channels_last)
    efficientnet_model_1 = efficientnet_model_1.to(memory_format=torch.channels_last)
    efficientnet_model_7 = efficientnet_model_7.to(memory_format=torch.channels_last)
    efficientnet_model_8 = efficientnet_model_8.to(memory_format=torch.channels_last)




## === cell 8
def train_one_model(model, train_loader, val_loader, epochs, lr, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            if device.type == "cuda":
                images = images.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
                labels = labels.to(device, non_blocking=True)
            else:
                images = images.to(device)
                labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        vcorrect = 0
        vtotal = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if device.type == "cuda":
                    images = images.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                    labels = labels.to(device, non_blocking=True)
                else:
                    images = images.to(device)
                    labels = labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)
                vcorrect += (preds == labels).sum().item()
                vtotal += labels.size(0)
        val_acc = vcorrect / max(vtotal, 1)

        print(
            f"epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}"
        )

    return model




## === cell 9
resnet_model = train_one_model(
    resnet_model, train_loader_resnet, val_loader_resnet, epochs=1, lr=1e-4
)
efficientnet_model_1 = train_one_model(
    efficientnet_model_1, train_loader_eff, val_loader_eff, epochs=1, lr=1e-4
)
efficientnet_model_7 = train_one_model(
    efficientnet_model_7, train_loader_eff, val_loader_eff, epochs=1, lr=1e-4
)
efficientnet_model_8 = train_one_model(
    efficientnet_model_8, train_loader_eff, val_loader_eff, epochs=1, lr=1e-4
)

resnet_model.eval()
efficientnet_model_1.eval()
efficientnet_model_7.eval()
efficientnet_model_8.eval()



## === cell 10
test_image_ids = test_df["image_id"].tolist()
test_rgb_cache = {}
for img_name in test_image_ids:
    img_path = os.path.join(test_image_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    test_rgb_cache[img_name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

test_dataset_resnet = CassavaTestDataset(
    test_df, test_image_dir, transform=resnet_transforms, rgb_cache=test_rgb_cache
)
test_dataset_eff = CassavaTestDataset(
    test_df, test_image_dir, transform=efficientnet_transforms, rgb_cache=test_rgb_cache
)

test_loader_resnet = DataLoader(
    test_dataset_resnet,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
test_loader_eff = DataLoader(
    test_dataset_eff,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)



## === cell 11
ensemble_predictions = []
image_names = []

resnet_probs_list = []

with torch.no_grad():
    for images, img_names in test_loader_resnet:
        if device.type == "cuda":
            images = images.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
        else:
            images = images.to(device)

        outputs = resnet_model(images)
        probs = F.softmax(outputs, dim=1).detach().cpu()
        resnet_probs_list.append(probs)
        image_names.extend(list(img_names))

resnet_probs_all = torch.cat(resnet_probs_list, dim=0)  # CPU tensor [N,5]

ensemble_predictions = []
image_names_eff = []
offset = 0

with torch.no_grad():
    for images, img_names in test_loader_eff:
        bs = images.size(0)
        if device.type == "cuda":
            images = images.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
        else:
            images = images.to(device)

        outputs_efficientnet1 = efficientnet_model_1(images)
        probs_efficientnet1 = F.softmax(outputs_efficientnet1, dim=1)

        outputs_efficientnet7 = efficientnet_model_7(images)
        probs_efficientnet7 = F.softmax(outputs_efficientnet7, dim=1)

        outputs_efficientnet8 = efficientnet_model_8(images)
        probs_efficientnet8 = F.softmax(outputs_efficientnet8, dim=1)

        eff_avg = (
            probs_efficientnet1 + probs_efficientnet7 + probs_efficientnet8
        ) / 3.0

        batch_resnet_probs = resnet_probs_all[offset : offset + bs].to(eff_avg.device)
        offset += bs

        combined_probs = (0.7 * eff_avg) + (0.3 * batch_resnet_probs)
        preds = combined_probs.argmax(dim=1).detach().cpu().numpy()

        ensemble_predictions.extend(preds.tolist())
        image_names_eff.extend(list(img_names))

pred_map = dict(zip(image_names_eff, ensemble_predictions))
submission_df = test_df.copy()
submission_df["label"] = submission_df["image_id"].map(pred_map).astype(int)

if submission_df["label"].isna().any():
    missing = (
        submission_df.loc[submission_df["label"].isna(), "image_id"].head(10).tolist()
    )
    raise RuntimeError(
        f"Missing predictions for some images (showing up to 10): {missing}"
    )

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())
