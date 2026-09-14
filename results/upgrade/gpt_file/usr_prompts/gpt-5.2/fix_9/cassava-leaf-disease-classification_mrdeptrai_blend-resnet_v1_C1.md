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

3.14

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
timm==1.0.19
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
import os, gc, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

SEED = 42
N_FOLDS = 4
NUM_CLASSES = 5
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"

CKPT_DIR = "/kaggle/input/resnet"  # resnet_fold{i}_best.pth (may not exist)
IMG_SIZE = 384

EPOCHS_FALLBACK = 2
LR_FALLBACK = 1e-3


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

try:
    cv2.setNumThreads(max(1, (os.cpu_count() or 4)))
except Exception:
    pass

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
labels = train_df.label.values

valid_aug = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])

train_aug = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ]
)


def _read_rgb_uint8(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class CachedNumpyAugDataset(Dataset):
    """Applies Albumentations on cached uint8 RGB images; augmentation semantics unchanged."""

    def __init__(self, imgs_uint8, labels=None, aug=None, is_test=False, indices=None):
        self.imgs = imgs_uint8
        self.labels = labels
        self.aug = aug
        self.is_test = is_test
        self.indices = indices  # optional index mapping

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.imgs)

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        img = self.imgs[real_idx]
        img = self.aug(image=img)["image"] if self.aug is not None else img
        if self.is_test:
            return img
        return img, int(self.labels[real_idx])


class ValidLazyCachedDataset(Dataset):
    def __init__(
        self, imgs_uint8_nhwc, y=None, indices=None, is_test=False, out_size=384
    ):
        self.imgs = imgs_uint8_nhwc
        self.y = y
        self.indices = np.asarray(indices) if indices is not None else None
        self.is_test = is_test
        self.out_size = int(out_size)

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
            3, 1, 1
        )
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
            3, 1, 1
        )

        self._cache = {}

    def __len__(self):
        return int(len(self.indices) if self.indices is not None else len(self.imgs))

    def _preprocess_one(self, real_idx: int) -> torch.Tensor:
        t = self._cache.get(real_idx, None)
        if t is not None:
            return t
        im = self.imgs[real_idx]
        im_rs = cv2.resize(
            im, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR
        )
        t = (
            torch.from_numpy(im_rs)
            .permute(2, 0, 1)
            .contiguous()
            .to(torch.float32)
            .div_(255.0)
        )
        t = (t - self.mean) / self.std
        self._cache[real_idx] = t
        return t

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx]) if self.indices is not None else int(idx)
        img = self._preprocess_one(real_idx)
        if self.is_test:
            return img
        return img, int(self.y[real_idx])


def _cache_images_to_uint8_nhwc(img_paths, desc):
    n = len(img_paths)
    cached = None
    for i, p in enumerate(tqdm(img_paths, desc=desc, leave=False)):
        im = _read_rgb_uint8(p)
        if cached is None:
            h, w, c = im.shape
            cached = np.empty((n, h, w, c), dtype=np.uint8)
        cached[i] = im
    return cached


def _resize_uint8_nhwc_to_chw_uint8_torch(imgs_nhwc_uint8, out_size):
    n = imgs_nhwc_uint8.shape[0]
    out = torch.empty((n, 3, out_size, out_size), dtype=torch.uint8)
    for i in tqdm(range(n), desc=f"resize->{out_size} torch.uint8", leave=False):
        im = imgs_nhwc_uint8[i]
        im_rs = cv2.resize(im, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        out[i] = torch.from_numpy(im_rs).permute(2, 0, 1).contiguous()
    return out


class ValidPreprocessedUint8Dataset(Dataset):
    def __init__(self, x_uint8_chw, y=None, is_test=False):
        self.x = x_uint8_chw
        self.y = y
        self.is_test = is_test
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
            3, 1, 1
        )
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
            3, 1, 1
        )

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        img = self.x[idx].to(torch.float32).div_(255.0)
        img = (img - self.mean) / self.std
        if self.is_test:
            return img
        return img, int(self.y[idx])


def _clean_state_dict(sd):
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_sd[nk] = v
    return new_sd


def try_load_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        return False
    state = torch.load(ckpt_path, map_location=device)
    state = _clean_state_dict(state)
    model.load_state_dict(state, strict=True)
    return True


def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loader(ds, batch_size, shuffle, drop_last=False):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )
    if NUM_WORKERS > 0:
        kwargs.update(
            dict(
                persistent_workers=True, prefetch_factor=4, worker_init_fn=_seed_worker
            )
        )
    return DataLoader(ds, **kwargs)


train_img_paths = (TRAIN_IMG_DIR + "/" + train_df["image_id"].values).tolist()
test_img_paths = (TEST_IMG_DIR + "/" + test_df["image_id"].values).tolist()

train_imgs_uint8 = _cache_images_to_uint8_nhwc(
    train_img_paths, desc="cache train uint8 (NHWC)"
)
test_imgs_uint8 = _cache_images_to_uint8_nhwc(
    test_img_paths, desc="cache test uint8 (NHWC)"
)

test_x_uint8 = _resize_uint8_nhwc_to_chw_uint8_torch(test_imgs_uint8, IMG_SIZE)
test_ds = ValidPreprocessedUint8Dataset(test_x_uint8, y=None, is_test=True)
test_loader = make_loader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)

gc.collect()



## === cell 1
oof_pred_t = torch.zeros((len(train_df), NUM_CLASSES), dtype=torch.float32)
test_pred_t = torch.zeros((len(test_df), NUM_CLASSES), dtype=torch.float32)

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df.label)):
    print(f"\n=== FOLD {fold} ===")

    model = timm.create_model(
        "resnet50d", pretrained=False, num_classes=NUM_CLASSES
    ).to(DEVICE)

    ckpt_path = f"{CKPT_DIR}/resnet_fold{fold}_best.pth"
    loaded = try_load_checkpoint(model, ckpt_path, DEVICE)

    if not loaded:
        print(
            f"⚠️ Checkpoint not found: {ckpt_path} -> training fallback for {EPOCHS_FALLBACK} epochs"
        )

        tr_labels_full = train_df["label"].values
        val_labels = tr_labels_full[val_idx]

        tr_ds = CachedNumpyAugDataset(
            train_imgs_uint8,
            labels=tr_labels_full,
            aug=train_aug,
            is_test=False,
            indices=tr_idx,
        )

        val_ds = ValidLazyCachedDataset(
            train_imgs_uint8,
            y=tr_labels_full,
            indices=val_idx,
            is_test=False,
            out_size=IMG_SIZE,
        )

        tr_loader = make_loader(
            tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
        val_loader = make_loader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FALLBACK)

        best_acc = -1.0
        best_state = None

        for epoch in range(EPOCHS_FALLBACK):
            model.train()
            running_loss = 0.0
            for imgs, y in tqdm(
                tr_loader, desc=f"train e{epoch+1}/{EPOCHS_FALLBACK}", leave=False
            ):
                imgs = imgs.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(imgs)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            val_preds = []
            val_true = []
            with torch.inference_mode():
                for imgs, y in tqdm(
                    val_loader, desc=f"valid e{epoch+1}/{EPOCHS_FALLBACK}", leave=False
                ):
                    imgs = imgs.to(DEVICE, non_blocking=True)
                    logits = model(imgs)
                    val_preds.append(logits.argmax(dim=1).cpu().numpy())
                    val_true.append(y.numpy())
            va = accuracy_score(np.concatenate(val_true), np.concatenate(val_preds))
            print(
                f"epoch {epoch+1}: val_acc={va:.5f}, train_loss={running_loss/max(1,len(tr_loader)):.5f}"
            )

            if va > best_acc:
                best_acc = va
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        del tr_ds, val_ds, tr_loader, val_loader, best_state
        gc.collect()

    model.eval()

    val_ds = ValidLazyCachedDataset(
        train_imgs_uint8,
        y=train_df["label"].values,
        indices=val_idx,
        is_test=False,
        out_size=IMG_SIZE,
    )
    val_loader = make_loader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    with torch.inference_mode():
        offset = 0
        for imgs, _ in tqdm(val_loader, desc="oof", leave=False):
            bs = imgs.shape[0]
            imgs = imgs.to(DEVICE, non_blocking=True)
            probs = torch.softmax(model(imgs), dim=1).detach().cpu()
            batch_indices = val_idx[offset : offset + bs]
            oof_pred_t[batch_indices] = probs
            offset += bs

    with torch.inference_mode():
        for b, imgs in enumerate(tqdm(test_loader, desc="test", leave=False)):
            imgs = imgs.to(DEVICE, non_blocking=True)
            probs = torch.softmax(model(imgs), dim=1).detach().cpu()
            start = b * BATCH_SIZE
            end = start + probs.shape[0]
            test_pred_t[start:end] += probs

    del model, val_ds, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

test_pred_t /= N_FOLDS

oof_pred = oof_pred_t.numpy()
test_pred = test_pred_t.numpy()



## === cell 2
oof_label = oof_pred.argmax(axis=1)
acc = accuracy_score(labels, oof_label)
print("\n🔥 FINAL OOF ACC:", acc)

sub = test_df[["image_id"]].copy()
sub["label"] = test_pred.argmax(axis=1).astype(int)
sub.to_csv("submission.csv", index=False)
print("✅ Saved submission.csv with shape:", sub.shape)
print(sub.head())
