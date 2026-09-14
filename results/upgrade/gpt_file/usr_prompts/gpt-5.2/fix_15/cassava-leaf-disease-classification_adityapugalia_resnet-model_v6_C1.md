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
import io
import json
import random
import multiprocessing
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.float16  # standard CUDA AMP dtype

from torchvision.io import read_file, decode_jpeg
import torchvision.transforms.functional as TF

try:
    import torchvision

    torchvision.set_image_backend(
        "image"
    )  # prefers torchvision C++ image backend if available
except Exception:
    pass

torch.set_num_threads(1)


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)




## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

USE_CHANNELS_LAST = device.type == "cuda"
print("channels_last:", USE_CHANNELS_LAST)

if device.type == "cuda" and USE_AMP:
    from torch.amp import autocast, GradScaler

    scaler = GradScaler()
else:
    autocast = None
    scaler = None



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
print("Skipping sample image dimension prints to avoid unnecessary I/O.")



## === cell 6
img_height, img_width = 800, 600  # keep original core preprocessing
batch_size = 32

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _collate_decode_resize_norm(batch):
    paths, labels = zip(*batch)

    imgs = []
    for p in paths:
        img_bytes = read_file(p)
        img = decode_jpeg(img_bytes, device="cpu")  # uint8 CHW
        imgs.append(img)

    x = torch.stack(imgs, dim=0)  # B,C,H,W uint8
    x = TF.resize(x, [img_height, img_width])  # uint8 BCHW
    x = x.to(dtype=torch.float32).div_(255.0)
    x.sub_(MEAN).div_(STD)

    y = torch.as_tensor(labels, dtype=torch.int64)
    return x, y


class OnTheFlyImageDataset(Dataset):
    def __init__(self, csv_data: pd.DataFrame, image_dir: str, has_labels: bool = True):
        self.image_ids = csv_data["image_id"].astype(str).to_numpy()
        self.has_labels = has_labels and ("label" in csv_data.columns)
        self.labels = (
            csv_data["label"].to_numpy(dtype=np.int64) if self.has_labels else None
        )
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.image_dir, image_id)
        y = int(self.labels[idx]) if self.has_labels else -1
        return path, y


class TFRecordCassavaDataset(Dataset):
    def __init__(self, csv_data: pd.DataFrame, tfrec_dir: str, has_labels: bool):
        image_dir = train_path if has_labels else test_path
        self._ds = OnTheFlyImageDataset(
            csv_data, image_dir=image_dir, has_labels=has_labels
        )

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


_TFREC_TRAIN_DIR = os.path.join(base_path, "train_tfrecords")
_TFREC_TEST_DIR = os.path.join(base_path, "test_tfrecords")

full_dataset = TFRecordCassavaDataset(train_data, _TFREC_TRAIN_DIR, has_labels=True)


def split_dataset(dataset, train_size=0.8):
    train_length = int(len(dataset) * train_size)
    val_length = len(dataset) - train_length
    g = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        dataset, [train_length, val_length], generator=g
    )
    return train_dataset, val_dataset


total_images = len(full_dataset)
train_dataset, val_dataset = split_dataset(full_dataset, train_size=0.8)

print(
    "Skipping unused class-weight sampler computation (DataLoader uses shuffle=True)."
)

cpu_cnt = multiprocessing.cpu_count()
num_workers = min(4, max(2, cpu_cnt // 2))
print("numworkers:", num_workers)

pin = torch.cuda.is_available()
g_dl = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    prefetch_factor=(
        2 if num_workers > 0 else None
    ),  # slightly lower to reduce memory pressure
    persistent_workers=(num_workers > 0),
    pin_memory=pin,
    worker_init_fn=seed_worker,
    generator=g_dl,
    collate_fn=_collate_decode_resize_norm,  # batched decode/resize/norm
)
print("created train loader")

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    prefetch_factor=2 if num_workers > 0 else None,
    persistent_workers=(num_workers > 0),
    pin_memory=pin,
    worker_init_fn=seed_worker,
    generator=g_dl,
    collate_fn=_collate_decode_resize_norm,  # batched decode/resize/norm
)
print("created valid loader")

print(f"Total images: {total_images}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")




## === cell 7
def _warm_cache(dataset: Dataset, batch_size: int, num_workers: int):
    return 0


print("Cache warming skipped (on-the-fly decoding uses DataLoader worker prefetching).")



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

print("torch.compile disabled to avoid compile-time overhead within 600s budget")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.fc.parameters(), lr=0.001)


class CUDAPrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device, channels_last: bool):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        if self.device.type != "cuda":
            yield from self.loader
            return

        it = iter(self.loader)

        def _preload():
            try:
                inputs, labels = next(it)
            except StopIteration:
                return None
            with torch.cuda.stream(self.stream):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if self.channels_last:
                    inputs = inputs.contiguous(memory_format=torch.channels_last)
            return inputs, labels

        next_batch = _preload()
        while next_batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            inputs, labels = next_batch
            next_batch = _preload()
            yield inputs, labels


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_acc = 0.0

    train_len = len(train_dataset)
    val_len = len(val_dataset)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        running_corrects = torch.zeros((), device=device, dtype=torch.long)

        train_iter = CUDAPrefetcher(
            train_loader, device=device, channels_last=USE_CHANNELS_LAST
        )

        for inputs, labels in train_iter:
            if device.type != "cuda":
                inputs = inputs.to(device)
                labels = labels.to(device)
                if USE_CHANNELS_LAST:
                    inputs = inputs.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda" and USE_AMP:
                with autocast(device_type="cuda", dtype=AMP_DTYPE):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum()

        epoch_loss = running_loss / train_len
        epoch_acc = (running_corrects.double() / train_len).item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )

        model.eval()
        val_loss = 0.0
        val_corrects = torch.zeros((), device=device, dtype=torch.long)

        val_iter = CUDAPrefetcher(
            val_loader, device=device, channels_last=USE_CHANNELS_LAST
        )

        with torch.no_grad():
            for inputs, labels in val_iter:
                if device.type != "cuda":
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if USE_CHANNELS_LAST:
                        inputs = inputs.contiguous(memory_format=torch.channels_last)

                if device.type == "cuda" and USE_AMP:
                    with autocast(device_type="cuda", dtype=AMP_DTYPE):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (preds == labels).sum()

        val_loss = val_loss / val_len
        val_acc = (val_corrects.double() / val_len).item()
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

test_dataset = TFRecordCassavaDataset(test_df, _TFREC_TEST_DIR, has_labels=False)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    prefetch_factor=2 if num_workers > 0 else None,
    persistent_workers=(num_workers > 0),
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=seed_worker,
    generator=g_dl,
    collate_fn=_collate_decode_resize_norm,  # same batched pipeline for test
)

predictions = []
test_iter = CUDAPrefetcher(test_loader, device=device, channels_last=USE_CHANNELS_LAST)

with torch.no_grad():
    for images, _ in test_iter:
        if device.type != "cuda":
            images = images.to(device)
            if USE_CHANNELS_LAST:
                images = images.contiguous(memory_format=torch.channels_last)

        if device.type == "cuda" and USE_AMP:
            with autocast(device_type="cuda", dtype=AMP_DTYPE):
                outputs = base_model(images)
        else:
            outputs = base_model(images)

        preds = outputs.argmax(dim=1)
        predictions.extend(preds.detach().cpu().numpy())

predictions = np.array(predictions)



## === cell 10
print(len(predictions), "predictions")



## === cell 11
test_df["label"] = predictions.astype(int)
test_df.to_csv("submission.csv", index=False)
print("Submission file created at /kaggle/working/submission.csv")
print(test_df.head())
