# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

try:
    from PIL import features as _pil_features

    _ = _pil_features.check("jpg")
except Exception:
    pass

try:
    from torchvision.io import read_image as _tv_read_image

    _TV_READ_IMAGE_OK = True
except Exception:
    _TV_READ_IMAGE_OK = False




## === cell 1
BASE_DIR = "/kaggle/input/paddy-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 2
classes = sorted(train_df["label"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
num_classes = len(classes)


def build_train_image_id_to_path_from_df(df: pd.DataFrame, img_root: str) -> dict:
    return {
        img_id: os.path.join(img_root, lbl, img_id)
        for img_id, lbl in zip(df["image_id"].values, df["label"].values)
    }


train_id_to_path = build_train_image_id_to_path_from_df(train_df, TRAIN_IMG_DIR)


def _load_rgb_pil(path: str) -> Image.Image:
    if _TV_READ_IMAGE_OK:
        t = _tv_read_image(path)  # uint8, [C,H,W]
        if t.ndim == 3 and t.size(0) == 1:
            t = t.expand(3, -1, -1)
        return transforms.functional.to_pil_image(t)
    with Image.open(path) as im:
        return im.convert("RGB")


class PaddyTrainDataset(Dataset):
    def __init__(self, df, img_root, transform=None, id_to_path=None):
        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].tolist()
        self.img_root = img_root
        self.transform = transform
        self.id_to_path = id_to_path or {}

    def __len__(self):
        return len(self.image_ids)

    def _resolve_path(self, image_id, label):
        p = self.id_to_path.get(image_id)
        if p is not None:
            return p
        return os.path.join(self.img_root, label, image_id)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        img_path = self._resolve_path(image_id, label)
        img = _load_rgb_pil(img_path)
        if self.transform is not None:
            img = self.transform(img)
        y = class_to_idx[label]
        return img, y


class PaddyTestDataset(Dataset):
    def __init__(self, image_ids, img_dir, transform=None):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id)
        img = _load_rgb_pil(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, image_id


train_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

test_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)




## === cell 3
def stratified_split(df, val_frac=0.1, seed=42):
    rng = np.random.RandomState(seed)
    val_indices = []
    for lbl, g in df.groupby("label"):
        idxs = g.index.values
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac))
        val_indices.extend(idxs[:n_val].tolist())
    val_mask = df.index.isin(val_indices)
    return df.loc[~val_mask].copy(), df.loc[val_mask].copy()


tr_df, va_df = stratified_split(train_df, val_frac=0.1, seed=seed)
len(tr_df), len(va_df), num_classes




## === cell 4
batch_size = 32

cpu_cnt = os.cpu_count() or 2
num_workers = min(8, max(2, cpu_cnt))
pin_memory = torch.cuda.is_available()

train_ds = PaddyTrainDataset(
    tr_df, TRAIN_IMG_DIR, transform=train_tfms, id_to_path=train_id_to_path
)
val_ds = PaddyTrainDataset(
    va_df, TRAIN_IMG_DIR, transform=test_tfms, id_to_path=train_id_to_path
)


def _seed_worker(worker_id):
    worker_seed = (seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)




## === cell 5
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


epochs = 3  # unchanged
for ep in range(1, epochs + 1):
    model.train()
    running = 0.0
    seen = 0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running += loss.item() * y.size(0)
        seen += y.size(0)

    tr_loss = running / max(1, seen)
    va_loss, va_acc = evaluate(val_loader)
    print(
        f"Epoch {ep}/{epochs} - train_loss: {tr_loss:.4f} - val_loss: {va_loss:.4f} - val_acc: {va_acc:.4f}"
    )




## === cell 6
test_ids = sample_df["image_id"].tolist()
test_ds = PaddyTestDataset(test_ids, TEST_IMG_DIR, transform=test_tfms)
test_loader = DataLoader(
    test_ds,
    batch_size=128,  # --- Speed: larger batch for pure inference; preserves predictions semantics.
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

model.eval()
pred_labels = []
pred_ids = []
with torch.inference_mode():
    for x, image_ids in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy().tolist()
        pred_ids.extend(list(image_ids))
        pred_labels.extend([idx_to_class[i] for i in pred])

sub = pd.DataFrame({"image_id": pred_ids, "label": pred_labels})

default_label = train_df["label"].mode().iloc[0]
sub = sample_df[["image_id"]].merge(sub, on="image_id", how="left")
sub["label"] = sub["label"].fillna(default_label)

print("Submission shape:", sub.shape)
sub.head()




## === cell 7
out_path = "model_submission_v10.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(pd.read_csv(out_path).head())
