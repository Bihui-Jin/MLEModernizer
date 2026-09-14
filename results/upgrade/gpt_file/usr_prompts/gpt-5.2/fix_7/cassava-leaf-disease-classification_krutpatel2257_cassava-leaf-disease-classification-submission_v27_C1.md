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
import random
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from torchvision.io import read_image, ImageReadMode



## === cell 1
config = {
    "DATA": {
        "IMAGES": "train_images",
        "LABELS": "train.csv",
        "SUB_IMAGES": "test_images",
        "SUB_LABELS": "sample_submission.csv",
        "SUB_OUTPUT": "submission.csv",
    },
    "DEVICE": "cuda",
    "NUM_GPU": torch.cuda.device_count(),
    "TRAIN_BATCH_SIZE": 32,
    "VAL_BATCH_SIZE": 16,
    "CLASSES": 5,
    "CV_FOLDS": 5,
    "NUM_EPOCHS": 15,
    "MODEL_PATH": "model.pth",
    "SGD": {"LR": 0.0005, "MOMENTUM": 0.9, "WEIGHT_DECAY": 0.001},
    "COS_ANN_LR": {"ETA_MIN": 0.00001},
    "MODEL_TYPE": "RESNET_50",
}



## === cell 2
model_path = "../input/en-b4-tta-calr-clahe-v2-12-14/model(16).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def seed_worker(worker_id: int):
    worker_seed = (seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)



## === cell 3
sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
        A.Transpose(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.CLAHE(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(p=0.5),
    ],
    p=1.0,
)



## === cell 4
if config["MODEL_TYPE"] == "RESNET_50":
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = nn.Linear(2048, config["CLASSES"])
else:
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = nn.Linear(2048, config["CLASSES"])

model.to(device)


def _load_checkpoint_if_available(model, ckpt_path, device):
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        return False

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "model_state_dict" in state and isinstance(
            state["model_state_dict"], dict
        ):
            state = state["model_state_dict"]

    if not isinstance(state, dict):
        return False

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    state = new_state

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"Checkpoint strict load failed: {e}")
        return False
    return True


loaded = _load_checkpoint_if_available(model, model_path, device)
print(f"Loaded external checkpoint: {loaded} (path: {model_path})")

if torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled")
    except Exception as e:
        print(f"torch.compile not available/enabled: {e}")




## === cell 5
def _read_rgb_uint8_hwc(img_path: str) -> np.ndarray:
    t = read_image(img_path, mode=ImageReadMode.RGB)  # CHW uint8
    return np.asarray(t.permute(1, 2, 0).numpy())


def _albumentations_output_to_chw_float_tensor(img: np.ndarray) -> torch.Tensor:
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    chw = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(chw)).float()


class CassavaDataset(Dataset):
    def __init__(self, df, images_dir, aug=None, has_labels=True):
        self.images_dir = images_dir
        self.aug = aug
        self.has_labels = has_labels

        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].to_numpy()
        self.labels = df["label"].to_numpy(dtype=np.int64) if has_labels else None

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, image_id)
        img = _read_rgb_uint8_hwc(img_path)

        if self.aug is not None:
            img = self.aug(image=img)["image"]

        x = _albumentations_output_to_chw_float_tensor(img)

        if self.has_labels:
            y = int(self.labels[idx])
            return x, y
        return x, image_id


train_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        if self.stream is None:
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)
        try:
            next_input, next_target = next(it)
        except StopIteration:
            return

        while True:
            with torch.cuda.stream(self.stream):
                next_input = next_input.to(self.device, non_blocking=True)
                next_target = next_target.to(self.device, non_blocking=True)

            torch.cuda.current_stream().wait_stream(self.stream)
            input_, target_ = next_input, next_target

            try:
                next_input, next_target = next(it)
            except StopIteration:
                yield input_, target_
                break

            yield input_, target_


if not loaded:
    df = pd.read_csv(train_csv_path)

    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_idx, va_idx = next(sss.split(df["image_id"], df["label"]))
    df_tr, df_va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()

    train_ds = CassavaDataset(df_tr, train_images_path, aug=train_aug, has_labels=True)
    val_ds = CassavaDataset(df_va, train_images_path, aug=train_aug, has_labels=True)

    cpu_cnt = os.cpu_count() or 2
    nw = min(8, max(2, cpu_cnt // 2))

    train_loader = DataLoader(
        train_ds,
        batch_size=config["TRAIN_BATCH_SIZE"],
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["VAL_BATCH_SIZE"],
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )

    train_iter = CUDAPrefetcher(train_loader, device)
    val_iter = CUDAPrefetcher(val_loader, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["SGD"]["LR"],
        momentum=config["SGD"]["MOMENTUM"],
        weight_decay=config["SGD"]["WEIGHT_DECAY"],
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["NUM_EPOCHS"],
        eta_min=config["COS_ANN_LR"]["ETA_MIN"],
    )

    best_acc = -1.0
    best_state = None

    for epoch in range(config["NUM_EPOCHS"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_iter:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

        scheduler.step()

        model.eval()
        v_correct = 0
        v_total = 0
        with torch.inference_mode():
            for xb, yb in val_iter:
                logits = model(xb)
                pred = logits.argmax(dim=1)
                v_correct += (pred == yb).sum().item()
                v_total += xb.size(0)

        train_acc = correct / max(1, total)
        val_acc = v_correct / max(1, v_total)
        print(
            f"Epoch {epoch+1}/{config['NUM_EPOCHS']} | loss {running_loss/max(1,total):.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, config["MODEL_PATH"])
        print(
            f"Saved trained model to: {config['MODEL_PATH']} (best val_acc={best_acc:.4f})"
        )

model.eval()




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, images_dir, aug):
        self.image_ids = np.asarray(image_ids)
        self.images_dir = images_dir
        self.aug = aug

    def __len__(self):
        return int(self.image_ids.shape[0])

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, str(image_id))
        img = _read_rgb_uint8_hwc(img_path)
        img = self.aug(image=img)["image"]
        x = _albumentations_output_to_chw_float_tensor(img)
        return x, str(image_id)


sample_sub = pd.read_csv(sample_sub_path)
tta_count = 5

test_bs = 64  # batching only; does not change predictions
cpu_cnt = os.cpu_count() or 2
nw = min(8, max(2, cpu_cnt // 2))

test_ds = CassavaTestDataset(sample_sub["image_id"].values, test_images_path, sub_aug)

test_loader = DataLoader(
    test_ds,
    batch_size=test_bs,
    shuffle=False,
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    persistent_workers=(nw > 0),
    prefetch_factor=4 if nw > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)

num_test = len(test_ds)
image_ids_ordered = sample_sub["image_id"].tolist()

logits_accum = torch.zeros(
    (num_test, config["CLASSES"]),
    dtype=torch.float32,
    pin_memory=torch.cuda.is_available(),
)

_prev_bench = torch.backends.cudnn.benchmark
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

with torch.inference_mode():
    for _ in range(tta_count):
        offset = 0
        for xb, batch_ids in test_loader:
            xb = xb.to(device, non_blocking=torch.cuda.is_available())
            out = model(xb)
            bs = out.size(0)
            logits_accum[offset : offset + bs].add_(
                out.detach().to("cpu", non_blocking=torch.cuda.is_available()),
                alpha=1.0,
            )
            offset += bs
        assert offset == num_test, f"Inference offset mismatch: {offset} != {num_test}"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = _prev_bench

logits_mean = logits_accum / float(tta_count)
pred_labels = logits_mean.argmax(dim=1).numpy().astype(int)

sub_df = pd.DataFrame({"image_id": image_ids_ordered, "label": pred_labels})
sub_df.to_csv(config["DATA"]["SUB_OUTPUT"], index=False)
print(sub_df.head())
print(f"Wrote submission to: {config['DATA']['SUB_OUTPUT']} (rows={len(sub_df)})")
