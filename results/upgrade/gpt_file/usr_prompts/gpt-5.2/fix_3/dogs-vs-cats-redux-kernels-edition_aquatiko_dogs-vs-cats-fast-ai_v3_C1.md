# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os
import re
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image

BASE_CANDIDATES = [
    "../input/dogs-vs-cats-redux-kernels-edition",
    "../input/dogs-vs-cats-redux-kernels-edition/dogs-vs-cats-redux-kernels-edition",
    "../input",
]
BASE_PATH = None
for p in BASE_CANDIDATES:
    if os.path.exists(p):
        BASE_PATH = p
        break
if BASE_PATH is None:
    raise FileNotFoundError("Could not find expected input directory under ../input")

print("Using BASE_PATH:", BASE_PATH)
print("Top-level:", os.listdir(BASE_PATH)[:20])

OUT_PATH = "./"
SUB_PATH = os.path.join(OUT_PATH, "submission.csv")

sz = 224

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
train_dir = os.path.join(BASE_PATH, "train")
test_dir = os.path.join(BASE_PATH, "test")

train_cat_dir = os.path.join(train_dir, "cat")
train_dog_dir = os.path.join(train_dir, "dog")

test_unknown_candidates = [
    os.path.join(test_dir, "unknown"),
    os.path.join(test_dir, "test", "unknown"),
    os.path.join(test_dir, "test", "test", "unknown"),
]
test_unknown_dir = None
for p in test_unknown_candidates:
    if os.path.exists(p):
        test_unknown_dir = p
        break

if not (os.path.isdir(train_cat_dir) and os.path.isdir(train_dog_dir)):
    raise FileNotFoundError(
        f"Train folders not found at {train_cat_dir} and {train_dog_dir}"
    )
if test_unknown_dir is None:
    raise FileNotFoundError(
        f"Test unknown folder not found. Tried: {test_unknown_candidates}"
    )

print("train_cat_dir:", train_cat_dir, "n=", len(os.listdir(train_cat_dir)))
print("train_dog_dir:", train_dog_dir, "n=", len(os.listdir(train_dog_dir)))
print("test_unknown_dir:", test_unknown_dir, "n=", len(os.listdir(test_unknown_dir)))



## === cell 2
cat_files = sorted(
    [
        os.path.join(train_cat_dir, f)
        for f in os.listdir(train_cat_dir)
        if f.lower().endswith(".jpg")
    ]
)
dog_files = sorted(
    [
        os.path.join(train_dog_dir, f)
        for f in os.listdir(train_dog_dir)
        if f.lower().endswith(".jpg")
    ]
)

train_files = np.array(cat_files + dog_files)
labels = np.array([0] * len(cat_files) + [1] * len(dog_files), dtype=np.int64)

print("Train files:", len(train_files), "labels:", labels.shape)
print("Example:", train_files[0], labels[0], "|", train_files[-1], labels[-1])



## === cell 3
import torchvision
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose(
    [
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

valid_tfms = transforms.Compose(
    [
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

try:
    from torchvision.io import read_image, ImageReadMode

    _HAS_TV_READ_IMAGE = True
except Exception:
    _HAS_TV_READ_IMAGE = False


class CatsDogsDataset(Dataset):
    def __init__(self, files, labels=None, transform=None, return_id=False):
        self.files = list(files)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = transform
        self.return_id = return_id

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]

        if _HAS_TV_READ_IMAGE:
            img = read_image(fp, mode=ImageReadMode.RGB)
            try:
                if self.transform is not None:
                    img = self.transform(img)
            except Exception:
                img = Image.open(fp).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
        else:
            img = Image.open(fp).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        if self.labels is None:
            if self.return_id:
                base = os.path.basename(fp)
                _id = int(os.path.splitext(base)[0])
                return img, _id
            return img

        y = int(self.labels[idx])
        return img, y


n = len(train_files)
idxs = np.arange(n)
rng = np.random.RandomState(seed)
rng.shuffle(idxs)
valid_sz = int(0.2 * n)
valid_idxs = idxs[:valid_sz]
train_idxs = idxs[valid_sz:]

tr_files, tr_y = train_files[train_idxs], labels[train_idxs]
va_files, va_y = train_files[valid_idxs], labels[valid_idxs]

print("Train/Valid sizes:", len(tr_files), len(va_files))



## === cell 4
batch_size = 32  # safe for GPU/CPU in 600s window

train_ds = CatsDogsDataset(tr_files, tr_y, transform=train_tfms)
valid_ds = CatsDogsDataset(va_files, va_y, transform=valid_tfms)


def _seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

num_workers = min(8, (os.cpu_count() or 2))
pin = torch.cuda.is_available()

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

print(model.__class__.__name__, "ready.")




## === cell 5
def run_epoch(model, dl, train=True):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0

    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        if train:
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total += bs

    return total_loss / max(total, 1), total_correct / max(total, 1)


epochs = 2
for ep in range(epochs):
    tr_loss, tr_acc = run_epoch(model, train_dl, train=True)
    va_loss, va_acc = run_epoch(model, valid_dl, train=False)
    print(
        f"Epoch {ep+1}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | valid loss {va_loss:.4f} acc {va_acc:.4f}"
    )



## === cell 6
test_files = sorted(
    [
        os.path.join(test_unknown_dir, f)
        for f in os.listdir(test_unknown_dir)
        if f.lower().endswith(".jpg")
    ]
)

test_ds = CatsDogsDataset(test_files, labels=None, transform=valid_tfms, return_id=True)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

print("Test files:", len(test_files))
print("Example test:", test_files[0])



## === cell 7
model.eval()

all_ids_chunks = []
all_probs_chunks = []

with torch.no_grad():
    for xb, ids in test_dl:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs_dog = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_ids_chunks.append(ids.numpy().astype(np.int64, copy=False))
        all_probs_chunks.append(probs_dog.astype(np.float64, copy=False))

all_ids = np.concatenate(all_ids_chunks, axis=0)
all_probs_dog = np.concatenate(all_probs_chunks, axis=0)

print(
    "Preds:",
    all_probs_dog.shape,
    "IDs:",
    all_ids.shape,
    "prob range:",
    (all_probs_dog.min(), all_probs_dog.max()),
)



## === cell 8
sample_sub_candidates = [
    os.path.join(BASE_PATH, "sample_submission.csv"),
    os.path.join("../input", "sample_submission.csv"),
]
sample_path = None
for p in sample_sub_candidates:
    if os.path.exists(p):
        sample_path = p
        break
if sample_path is None:
    raise FileNotFoundError(
        f"sample_submission.csv not found in {sample_sub_candidates}"
    )

sample = pd.read_csv(sample_path)
if not set(["id", "label"]).issubset(sample.columns):
    raise ValueError(f"Unexpected sample_submission columns: {sample.columns.tolist()}")

pred_df = pd.DataFrame({"id": all_ids, "label": all_probs_dog})
sub = sample[["id"]].merge(pred_df, on="id", how="left")

sub["label"] = sub["label"].fillna(0.5).astype(float)

eps = 1e-6
sub["label"] = sub["label"].clip(eps, 1 - eps)

print(sub.head())
print("Submission rows:", len(sub), "missing:", sub["label"].isna().sum())

sub.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH, "size bytes:", os.path.getsize(SUB_PATH))
print("Done.")
