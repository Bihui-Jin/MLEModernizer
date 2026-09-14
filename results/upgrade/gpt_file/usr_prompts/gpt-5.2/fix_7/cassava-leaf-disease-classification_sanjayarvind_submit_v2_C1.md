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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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
import random
import numpy as np
import pandas as pd

import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    cv2.setNumThreads(0)
except Exception:
    pass


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "../input/cassava-leaf-disease-classification"

SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at: {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test_images dir at: {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at: {TRAIN_CSV_PATH}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train_images dir at: {TRAIN_IMG_DIR}"

sample_sub_df = pd.read_csv(SAMPLE_SUB_PATH)
train_df_full = pd.read_csv(TRAIN_CSV_PATH)




## === cell 1
def get_test_transform(image_size, do_hflip=False):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    tfms = [
        A.Resize(600, 600),
        A.CenterCrop(image_size, image_size),
    ]
    if do_hflip:
        tfms.append(A.HorizontalFlip(p=1.0))
    tfms += [
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ]
    return A.Compose(tfms)


def get_train_transform(image_size):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    try:
        rrc = A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=1.0,
        )
    except TypeError:
        rrc = A.RandomResizedCrop(
            image_size,
            image_size,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=1.0,
        )

    tfms = [
        A.Resize(600, 600),
        rrc,
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ]
    return A.Compose(tfms)




## === cell 2
class CassavaLeafDataset(Dataset):
    def __init__(self, root_dir, transforms, sample_csv_path=None, sample_df=None):
        self.root_dir = root_dir
        self.transform = transforms

        if sample_df is None:
            df = pd.read_csv(sample_csv_path)
        else:
            df = sample_df

        if "image_id" not in df.columns:
            raise ValueError("sample_submission.csv must contain 'image_id' column")

        self.image_ids = df["image_id"].to_numpy()

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_ids[idx])

        try:
            image = cv2.imread(
                img_name,
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR_RGB,
            )
        except Exception:
            image = cv2.imread(img_name, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_name}")

        if image.shape[-1] == 3 and image[..., 0].mean() != image[..., 2].mean():
            pass
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)["image"]
        return image, 0


class CassavaTrainDataset(Dataset):
    def __init__(
        self, root_dir, transforms, train_csv_path=None, indices=None, train_df=None
    ):
        self.root_dir = root_dir
        self.transform = transforms

        if train_df is None:
            df = pd.read_csv(train_csv_path)
        else:
            df = train_df
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        if "image_id" not in df.columns or "label" not in df.columns:
            raise ValueError("train.csv must contain 'image_id' and 'label' columns")

        self.image_ids = df["image_id"].to_numpy()
        self.labels = df["label"].to_numpy(dtype=np.int64)

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_ids[idx])

        try:
            image = cv2.imread(
                img_name,
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR_RGB,
            )
        except Exception:
            image = cv2.imread(img_name, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_name}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        label = int(self.labels[idx])
        return image, label




## === cell 3
IMAGE_SIZE = 528
test_transforms = get_test_transform(IMAGE_SIZE, do_hflip=False)
test_transforms_hf = get_test_transform(IMAGE_SIZE, do_hflip=True)

test_data = CassavaLeafDataset(TEST_IMG_DIR, test_transforms, sample_df=sample_sub_df)
test_data_hf = CassavaLeafDataset(
    TEST_IMG_DIR, test_transforms_hf, sample_df=sample_sub_df
)

_cpu = os.cpu_count() or 2
_num_workers = min(8, _cpu)

test_loader = DataLoader(
    test_data,
    batch_size=16,
    shuffle=False,
    pin_memory=True,
    num_workers=_num_workers,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
test_loader_hf = DataLoader(
    test_data_hf,
    batch_size=16,
    shuffle=False,
    pin_memory=True,
    num_workers=_num_workers,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)

print("Test size:", len(test_data))



## === cell 4
weights_path_candidates = [
    "../input/cassava-models/cassava-resnext.pth",
    "/kaggle/input/cassava-models/cassava-resnext.pth",
]

custom_loaded = False
for wp in weights_path_candidates:
    if os.path.exists(wp):
        model = models.resnext50_32x4d(weights=None)
        model.fc = nn.Linear(in_features=2048, out_features=5, bias=True)
        state = torch.load(wp, map_location="cpu")
        model.load_state_dict(state)
        custom_loaded = True
        print(f"Loaded custom cassava weights: {wp}")
        break

if not custom_loaded:
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=5, bias=True)
    print(
        "Custom weights not found -> using ImageNet pretrained backbone + new 5-class FC head."
    )

model.to(device)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile not enabled:", repr(e))




## === cell 5
def make_split_indices(n, val_frac=0.1, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = int(n * val_frac)
    val_idx = idx[:val_n]
    trn_idx = idx[val_n:]
    return trn_idx, val_idx


trn_idx, val_idx = make_split_indices(len(train_df_full), val_frac=0.1, seed=SEED)

train_tfms = get_train_transform(IMAGE_SIZE)
val_tfms = get_test_transform(IMAGE_SIZE, do_hflip=False)

train_ds = CassavaTrainDataset(
    TRAIN_IMG_DIR, train_tfms, indices=trn_idx, train_df=train_df_full
)
val_ds = CassavaTrainDataset(
    TRAIN_IMG_DIR, val_tfms, indices=val_idx, train_df=train_df_full
)

TRAIN_BS = 32 if device.type == "cuda" else 16

train_loader = DataLoader(
    train_ds,
    batch_size=TRAIN_BS,
    shuffle=True,
    pin_memory=True,
    num_workers=_num_workers,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
val_loader = DataLoader(
    val_ds,
    batch_size=32 if device.type == "cuda" else 16,
    shuffle=False,
    pin_memory=True,
    num_workers=_num_workers,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)

criterion = nn.CrossEntropyLoss()

for p in model.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

EPOCHS = 2
best_state = None
best_val_acc = -1.0

for epoch in range(EPOCHS):
    model.train()
    tr_losses = []

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.detach().item())

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == targets).sum().cpu().item())
            total += int(targets.numel())

    mean_tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
    mean_val_acc = (correct / total) if total else float("nan")
    print(
        f"Epoch {epoch+1}/{EPOCHS} - train_loss={mean_tr_loss:.4f} val_acc={mean_val_acc:.4f}"
    )

    if mean_val_acc > best_val_acc:
        best_val_acc = mean_val_acc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)
print("Best val acc:", best_val_acc)

model.eval()




## === cell 6
def predict_probs(loader, n_items):
    out = torch.empty((n_items, 5), dtype=torch.float32)
    offset = 0
    with torch.no_grad():
        for images, _ in loader:
            bs = images.size(0)
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            out[offset : offset + bs].copy_(probs.detach().cpu())
            offset += bs
    return out


probs_0 = predict_probs(test_loader, len(test_data))
probs_1 = predict_probs(test_loader_hf, len(test_data_hf))
probs = 0.5 * (probs_0 + probs_1)

preds = torch.argmax(probs, dim=1).tolist()

print("Preds:", len(preds), "Expected:", len(test_data))
assert len(preds) == len(test_data), "Prediction count does not match test set size."



## === cell 7
sub = sample_sub_df.copy()
sub["label"] = preds
sub_path = "./submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub.head())
