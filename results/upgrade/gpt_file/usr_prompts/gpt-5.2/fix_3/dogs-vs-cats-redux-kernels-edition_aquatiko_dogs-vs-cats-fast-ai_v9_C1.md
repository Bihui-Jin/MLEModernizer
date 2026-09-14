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
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import models, transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
PATH = "../input/dogs-vs-cats-redux-kernels-edition/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
os.makedirs(TMP_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

sz = 224

train_dir = os.path.join(PATH, "train")  # has subfolders cat/ dog/
test_dir = os.path.join(PATH, "test")  # has nested test/unknown/*.jpg in this dump

print("PATH exists:", os.path.exists(PATH))
print("train_dir:", train_dir, "exists:", os.path.exists(train_dir))
print("test_dir:", test_dir, "exists:", os.path.exists(test_dir))



## === cell 2
cat_dir = os.path.join(train_dir, "cat")
dog_dir = os.path.join(train_dir, "dog")

cat_files = [
    os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(".jpg")
]
dog_files = [
    os.path.join(dog_dir, f) for f in os.listdir(dog_dir) if f.lower().endswith(".jpg")
]

fnames = np.array(sorted(cat_files + dog_files))
labels = np.array(
    [0 if os.path.basename(p).startswith("cat") else 1 for p in fnames], dtype=np.int64
)

print("n_train:", len(fnames), "label mean (dogs):", labels.mean())
print("example:", fnames[-2], labels[-2])



## === cell 3
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_tfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(sz, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

valid_tfms = transforms.Compose(
    [
        transforms.Resize(int(sz * 1.14)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        normalize,
    ]
)


class CatsDogsDataset(Dataset):
    def __init__(self, paths, labels=None, tfms=None):
        self.paths = list(paths)
        self.labels = None if labels is None else np.array(labels, dtype=np.int64)
        self.tfms = tfms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.tfms is not None:
            img = self.tfms(img)
        if self.labels is None:
            return img, os.path.basename(p)
        return img, int(self.labels[idx])




## === cell 4
arch = "resnet34"

model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

model



## === cell 5
n = len(fnames)
idx = np.arange(n)
np.random.shuffle(idx)

valid_pct = 0.1
n_valid = int(n * valid_pct)

valid_idx = idx[:n_valid]
train_idx = idx[n_valid:]

train_paths = fnames[train_idx]
train_labels = labels[train_idx]
valid_paths = fnames[valid_idx]
valid_labels = labels[valid_idx]

train_ds = CatsDogsDataset(train_paths, train_labels, tfms=train_tfms)
valid_ds = CatsDogsDataset(valid_paths, valid_labels, tfms=valid_tfms)

batch_size = 32 if torch.cuda.is_available() else 16

_num_workers = min(4, os.cpu_count() or 2)
train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

len(train_ds), len(valid_ds), batch_size




## === cell 6
def run_one_epoch(model, dl, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(xb)
            loss = criterion(out, yb)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total += xb.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


start = time.time()
epochs = 2
for ep in range(epochs):
    tr_loss, tr_acc = run_one_epoch(model, train_dl, train=True)
    va_loss, va_acc = run_one_epoch(model, valid_dl, train=False)
    print(
        f"epoch {ep+1}/{epochs} - train loss {tr_loss:.4f} acc {tr_acc:.4f} | valid loss {va_loss:.4f} acc {va_acc:.4f}"
    )
print("train_time_sec:", round(time.time() - start, 2))

torch.save(model.state_dict(), os.path.join(MODEL_PATH, "model1.pth"))




## === cell 7
def find_test_images(root):
    jpgs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".jpg"):
                jpgs.append(os.path.join(dirpath, f))
    ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in jpgs]
    order = np.argsort(ids)
    return [jpgs[i] for i in order]


test_images = find_test_images(test_dir)
print("n_test:", len(test_images))
print("first:", test_images[0] if test_images else None)
print("last :", test_images[-1] if test_images else None)

test_ds = CatsDogsDataset(test_images, labels=None, tfms=valid_tfms)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)



## === cell 8
model.eval()
all_ids = []
all_probs = []

softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for xb, names in test_dl:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        probs = softmax(out)[:, 1].detach().cpu().numpy()  # dog prob

        ids = [int(os.path.splitext(nm)[0]) for nm in names]
        all_ids.extend(ids)
        all_probs.extend(probs.astype(np.float64).tolist())

len(all_ids), (min(all_ids) if all_ids else None), (max(all_ids) if all_ids else None)



## === cell 9
ans = pd.DataFrame({"id": all_ids, "label": all_probs})
ans = ans.sort_values("id").reset_index(drop=True)

ans["label"] = ans["label"].clip(1e-6, 1 - 1e-6)

sub_path = "submission.csv"
ans.to_csv(sub_path, index=False)

print(ans.head())
print(ans.tail())
print("Wrote:", sub_path, "rows:", len(ans))
