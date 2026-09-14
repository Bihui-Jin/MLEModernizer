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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 5. Code solution

## === cell 0
import os
import json
import random
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    Image.MAX_IMAGE_PIXELS = None
except Exception:
    pass
try:
    Image.Image.load = Image.Image.load  # no-op; keeps lint quiet
except Exception:
    pass




## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

USE_CHANNELS_LAST = device.type == "cuda"
print("channels_last:", USE_CHANNELS_LAST)




## === cell 2
base_path = "/kaggle/input/cassava-leaf-disease-classification/"
train_path = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_path = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

with open(base_path + "label_num_to_disease_map.json") as f:
    mapping = json.loads(f.read())
    mapping = {int(k): v for k, v in mapping.items()}

mapping




## === cell 3
train_data = pd.read_csv(base_path + "train.csv")
train_data.head()




## === cell 4
mapping_count = (
    train_data["label"]
    .value_counts(sort=False)
    .reindex([0, 1, 2, 3, 4], fill_value=0)
    .reset_index()
)
mapping_count.columns = ["label", "image_count"]
total_img = int(mapping_count["image_count"].sum())
mapping_count




## === cell 5
for i in range(5):
    image_path = train_path + train_data.image_id[i]
    with Image.open(image_path) as img:
        print(f"Image: {train_data.image_id[i]} | Dimensions: {img.size}")




## === cell 6
img_height, img_width = 800, 600  # keep original core preprocessing
batch_size = 32

image_dir = train_path

data_transforms = transforms.Compose(
    [
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

CACHE_DIR = None


class ImageDataset(Dataset):
    def __init__(
        self, csv_data, image_dir, transform=None, cache_dir: Path | None = None
    ):
        self.image_paths = csv_data["image_id"].values
        self.has_labels = "label" in csv_data.columns
        self.labels = csv_data["label"].values if self.has_labels else None
        self.image_dir = image_dir
        self.transform = transform
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.image_paths)

    def _cache_path(self, image_id: str) -> Path:
        return self.cache_dir / (image_id + ".pt")

    def __getitem__(self, idx):
        image_id = self.image_paths[idx]
        label = int(self.labels[idx]) if self.has_labels else -1  # dummy label for test

        if self.cache_dir is not None:
            cpath = self._cache_path(image_id)
            if cpath.exists():
                image = torch.load(cpath, map_location="cpu", weights_only=False)
                return image, label

        image_path = os.path.join(self.image_dir, image_id)
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)

        if self.cache_dir is not None:
            tmp = str(cpath) + ".tmp"
            torch.save(image, tmp)
            os.replace(tmp, cpath)

        return image, label


img_dataset = ImageDataset(
    train_data, image_dir, transform=data_transforms, cache_dir=CACHE_DIR
)


def split_dataset(dataset, train_size=0.8):
    train_length = int(len(dataset) * train_size)
    val_length = len(dataset) - train_length
    g = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        dataset, [train_length, val_length], generator=g
    )
    return train_dataset, val_dataset


total_images = len(img_dataset)
train_dataset, val_dataset = split_dataset(img_dataset, train_size=0.8)

print("computing class weights")
class_weights = [mapping_count["image_count"].iloc[i] / total_img for i in range(5)]

labels_all = train_data["label"].to_numpy(dtype=np.int64, copy=False)
sample_weights_all = np.asarray(
    [class_weights[int(l)] for l in labels_all], dtype=np.float64
)

train_indices = train_dataset.indices
train_sample_weights = sample_weights_all[
    np.asarray(train_indices, dtype=np.int64)
].tolist()
print("created sample weights")

sampler = WeightedRandomSampler(
    weights=train_sample_weights,
    num_samples=len(train_sample_weights),
    replacement=True,
)
print(
    "created sampler (not used due to original shuffle=True; kept to preserve code intent)"
)

cpu_cnt = multiprocessing.cpu_count()
num_workers = min(4, max(2, cpu_cnt // 2))
print("numworkers:", num_workers)

pin = torch.cuda.is_available()
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    prefetch_factor=2,
    persistent_workers=(num_workers > 0),
    pin_memory=pin,
)
print("created train loader")

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    prefetch_factor=2,
    persistent_workers=(num_workers > 0),
    pin_memory=pin,
)
print("created valid loader")

print(f"Total images: {total_images}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")




## === cell 7
def _warm_cache(dataset: Dataset, batch_size: int, num_workers: int):
    return 0


print("Cache warming skipped (cache disabled) to avoid timeout.")




## === cell 8
from torchvision.models import ResNet50_Weights

base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

for param in base_model.parameters():
    param.requires_grad = False

num_features = base_model.fc.in_features
base_model.fc = nn.Sequential(
    nn.Linear(num_features, 384),
    nn.ReLU(),
    nn.Linear(384, 5),
)

base_model = base_model.to(device)
if USE_CHANNELS_LAST:
    base_model = base_model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.fc.parameters(), lr=0.001)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            if USE_CHANNELS_LAST:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if USE_CHANNELS_LAST:
                    inputs = inputs.contiguous(memory_format=torch.channels_last)
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/kaggle/working/ResModel.pt")
            print("Model saved with Validation Accuracy: {:.4f}".format(best_val_acc))

    print("Training complete. Best Validation Accuracy: {:.4f}".format(best_val_acc))


train_model(base_model, train_loader, val_loader, criterion, optimizer, num_epochs=10)




## === cell 9
ckpt_path = "/kaggle/working/ResModel.pt"
if os.path.exists(ckpt_path):
    base_model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)
    )
base_model.eval()

test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

TEST_CACHE_DIR = None
test_dataset = ImageDataset(
    test_df, test_path, transform=data_transforms, cache_dir=TEST_CACHE_DIR
)

print("Test cache warming skipped (cache disabled) to avoid timeout.")

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    prefetch_factor=2,
    persistent_workers=(num_workers > 0),
    pin_memory=torch.cuda.is_available(),
)

predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        if USE_CHANNELS_LAST:
            images = images.contiguous(memory_format=torch.channels_last)
        images = images.to(device, non_blocking=True)
        outputs = base_model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

predictions = np.array(predictions)




## === cell 10
print(len(predictions), "predictions")




## === cell 11
test_df["label"] = predictions.astype(int)
test_df.to_csv("submission.csv", index=False)
print("Submission file created at /kaggle/working/submission.csv")
print(test_df.head())
