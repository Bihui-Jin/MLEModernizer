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

2.7

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
scikit-image==0.25.2
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
from __future__ import print_function, division

import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2

import albumentations as A
import torchvision

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed_all(SEED)

DATA_ROOT = "../input/cassava-leaf-disease-classification/"
TEST_DIR = os.path.join(DATA_ROOT, "test_images/")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images/")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

folder_name = "effnetmodelb43"
model_full_name = "efficientnet-b4-e10"
model_name = "efficientnet-b4"
WEIGHTS_PATH = "../input/" + folder_name + "/" + model_full_name + ".pt"

if not os.path.isdir(TEST_DIR):
    alt = "/kaggle/input/cassava-leaf-disease-classification/test_images"
    if os.path.isdir(alt):
        TEST_DIR = alt
if not os.path.isdir(TRAIN_DIR):
    alt = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    if os.path.isdir(alt):
        TRAIN_DIR = alt
if not os.path.exists(SAMPLE_SUB_PATH):
    alt = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    if os.path.exists(alt):
        SAMPLE_SUB_PATH = alt
if not os.path.exists(TRAIN_CSV_PATH):
    alt = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    if os.path.exists(alt):
        TRAIN_CSV_PATH = alt


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

from torchvision.io import read_image, ImageReadMode


def fast_imread_rgb(path):
    try:
        t = read_image(path, mode=ImageReadMode.RGB)
        img = t.permute(1, 2, 0).contiguous().cpu().numpy()
        return img
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("Failed to read image: %s" % path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img




## === cell 1
class EfficientNet(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    @classmethod
    def from_name(cls, name, num_classes=5, use_pretrained=True):
        if name != "efficientnet-b4":
            raise ValueError(
                "Only efficientnet-b4 is supported in this environment fix."
            )
        weights = (
            torchvision.models.EfficientNet_B4_Weights.DEFAULT
            if use_pretrained
            else None
        )
        model = torchvision.models.efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return cls(model)


def _clean_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]
        cleaned[nk] = v
    return cleaned


def _try_load_external_checkpoint(model, weights_path, device):
    """
    Optional: try to load the originally-referenced checkpoint if it exists.
    Keeps core logic but avoids hard failure when file is missing.
    """
    if weights_path is None or (not os.path.exists(weights_path)):
        return False, "Checkpoint not found; using torchvision pretrained weights."
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = _clean_state_dict_keys(ckpt)
    try:
        model.backbone.load_state_dict(state_dict, strict=False)
        return True, "Loaded checkpoint into model.backbone (strict=False)."
    except Exception:
        try:
            model.load_state_dict(state_dict, strict=False)
            return True, "Loaded checkpoint into model (strict=False)."
        except Exception as e:
            return (
                False,
                "Failed to load checkpoint; using torchvision pretrained weights. Error: %s"
                % str(e),
            )




## === cell 2
class ToTensor(object):
    def __call__(self, image, force_apply=True):
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        output = np.ascontiguousarray(image.transpose((2, 0, 1)))
        return torch.from_numpy(output)


class AlbumentationsWithTensor(object):
    def __init__(self, a_transform):
        self.a_transform = a_transform
        self.to_tensor = ToTensor()

    def __call__(self, image):
        out = self.a_transform(image=image)
        img = out["image"]
        return self.to_tensor(img)


class TrainDataset(Dataset):
    def __init__(self, root_dir, df, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_ids = df["image_id"].to_numpy()
        self.labels = df["label"].to_numpy(dtype=np.int64)

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        label = int(self.labels[idx])
        img_path = os.path.join(self.root_dir, img_name)

        image = fast_imread_rgb(img_path)

        if self.transform:
            image = self.transform(image=image)

        return image.float(), label


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None, images=None):
        self.root_dir = root_dir
        self.transform = transform
        if images is None:
            self.images = sorted(os.listdir(root_dir))
        else:
            self.images = list(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = fast_imread_rgb(img_path)

        if self.transform:
            image = self.transform(image=image)

        return img_name, image.float()




## === cell 3
IMG_SIZE = 512

train_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=IMG_SIZE, p=1.0),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

valid_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=IMG_SIZE, p=1.0),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

test_transform = valid_transform

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["image_id"].tolist()

train_df = pd.read_csv(TRAIN_CSV_PATH)

from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
tr_idx, va_idx = next(splitter.split(train_df["image_id"], train_df["label"]))
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

train_ds = TrainDataset(
    TRAIN_DIR, tr_df, transform=AlbumentationsWithTensor(train_transform)
)
valid_ds = TrainDataset(
    TRAIN_DIR, va_df, transform=AlbumentationsWithTensor(valid_transform)
)
test_ds = TestDataset(
    TEST_DIR, transform=AlbumentationsWithTensor(test_transform), images=test_ids
)

_common_loader_kwargs = dict(
    num_workers=4,
    pin_memory=use_cuda,
    persistent_workers=True,
    prefetch_factor=2,
    worker_init_fn=seed_worker,
    generator=g,
)

train_loader = DataLoader(
    train_ds, batch_size=16, shuffle=True, **_common_loader_kwargs
)
valid_loader = DataLoader(
    valid_ds, batch_size=32, shuffle=False, **_common_loader_kwargs
)
testloader = DataLoader(test_ds, batch_size=32, shuffle=False, **_common_loader_kwargs)




## === cell 4
model = EfficientNet.from_name(model_name, num_classes=5, use_pretrained=True).to(
    device
)

if use_cuda:
    model = model.to(memory_format=torch.channels_last)

loaded, msg = _try_load_external_checkpoint(model, WEIGHTS_PATH, device)
print(msg)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass


def correct_count_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).sum().item()


EPOCHS = 3  # keep identical core training plan

best_state = None
best_val_acc = -1.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    n_train = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        if use_cuda:
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        train_loss += loss.item() * bs
        train_correct += correct_count_from_logits(logits.detach(), yb)
        n_train += bs

    model.eval()
    val_loss = 0.0
    val_correct = 0
    n_val = 0
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device, non_blocking=True)
            if use_cuda:
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True).long()
            logits = model(xb)
            loss = criterion(logits, yb)

            bs = xb.size(0)
            val_loss += loss.item() * bs
            val_correct += correct_count_from_logits(logits, yb)
            n_val += bs

    train_loss /= max(1, n_train)
    train_acc = float(train_correct) / float(max(1, n_train))
    val_loss /= max(1, n_val)
    val_acc = float(val_correct) / float(max(1, n_val))

    print(
        "Epoch %d/%d | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f"
        % (epoch + 1, EPOCHS, train_loss, train_acc, val_loss, val_acc)
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

if best_state is not None:
    model.load_state_dict(best_state)
print("Best val acc: %.4f" % best_val_acc)




## === cell 5
model.eval()

names = []
predicted = []
with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device, non_blocking=True)
        if use_cuda:
            images_batch = images_batch.contiguous(memory_format=torch.channels_last)
        output = model(images_batch)
        output = torch.max(output, 1)[1].cpu().numpy()
        names.extend(list(names_batch))
        predicted.extend(output.tolist())

pred_map = dict(zip(names, predicted))
final_pred = [int(pred_map[iid]) for iid in test_ids]

result = pd.DataFrame({"image_id": test_ids, "label": final_pred})
result.to_csv("submission.csv", index=False)

print(result.head())
print("Wrote submission.csv with {} rows".format(len(result)))
assert len(result) == len(
    sample_sub
), "Submission row count mismatch vs sample_submission.csv"
assert list(result.columns) == ["image_id", "label"], "Submission columns mismatch"
