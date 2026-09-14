# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

16.89656

# 6. Current score

0.69315

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'I fix the root path detection so `TRAIN_DIR` points to the actual ImageFolder root containing `cat/` and `dog/`, avoiding the current “class train” error caused by mistakenly selecting the nested `train/train` directory. Then I make the downstream cells robust by ensuring `BATCH_SIZE`, loaders, and test file discovery are defined even if a candidate directory is empty, and I harden the submission creation so it works even when prediction arrays are still Python lists. These changes are execution/stability fixes (score-neutral except that they enable training/inference to run at all) and preserve your model/training core logic.'
- What this solution (achieved 0.69315) has done: 'I fix the root-cause dataset path issue by tightening `TRAIN_DIR` discovery to avoid selecting the nested `train/train` directory that contains a `train` subfolder (which makes ImageFolder think there’s a “train” class with no images). Then I make the pipeline resilient by defining `BATCH_SIZE` and loaders only after the dataset is successfully created, and ensuring the test directory selection actually points to the folder that contains `.jpg` files. Finally, I keep your model/training/prediction logic unchanged, but ensure the submission is always aligned to `sample_submission.csv` ids and written as `submission.csv` with correct columns.'
- What this solution (achieved 0.69315) has done: 'I fix the root-path discovery so it can find the actual `ImageFolder` root even when the provided `train/` directory also contains an extra nested `train/` subfolder. Specifically, I change the `TRAIN_DIR` selection to accept a directory that has `cat/` and `dog/` with images, regardless of additional subfolders, instead of rejecting it outright. This unblocks `datasets.ImageFolder(TRAIN_DIR, ...)`, which then allows the rest of your existing training/inference/submission logic to run unchanged. I keep the rest of the pipeline the same, only adding a small safety fallback to pick the `cat/`+`dog/` root if the first pass still fails.'
- What this solution (achieved 0.69315) has done: 'I fix the dataset root selection so `TRAIN_DIR` always points to the ImageFolder directory that directly contains `cat/` and `dog/` (and not a nested folder that also contains a `train/` subfolder, which makes ImageFolder think there’s a bogus “train” class). Then I make `full_ds`, `BATCH_SIZE`, loaders, and `dog_idx` reliably defined by creating the dataset once and reusing it for the train/valid splits, preventing the downstream `NameError`s. Finally, I keep your model/training/inference logic the same but ensure the test directory truly contains `.jpg` files and the submission is always written as `submission.csv` with `id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.69315) has done: 'I fix the root-cause `ImageFolder` error by ensuring `TRAIN_DIR` points to the directory that directly contains `cat/` and `dog/` (not a parent that also contains an extra `train/` folder). Then I make downstream cells robust by (a) building `full_ds` once and reusing its indices for both train/valid splits, and (b) defining `BATCH_SIZE` immediately after the dataset is created so later cells never see `NameError`. Finally, I keep the model/training/inference logic intact, but ensure the `dog_idx` used for submission is taken from the same dataset/class mapping and the submission is always aligned to `sample_submission.csv` ids and written to `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import time
import random
import numpy as np
import pandas as pd

BASE_PATH_CANDIDATES = [
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/data/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/data/dogs-vs-cats-redux-kernels-edition/dogs-vs-cats-redux-kernels-edition",
]
BASE_PATH = next((p for p in BASE_PATH_CANDIDATES if os.path.isdir(p)), None)
if BASE_PATH is None:
    raise FileNotFoundError(f"Could not find BASE_PATH. Tried: {BASE_PATH_CANDIDATES}")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _has_images_in_dir(p):
    if not os.path.isdir(p):
        return False
    for fn in os.listdir(p):
        if fn.lower().endswith(IMG_EXTS):
            return True
    return False


def _is_imagefolder_root(p):
    """
    Fix: select the torchvision.datasets.ImageFolder root that directly contains
    class folders cat/ and dog/ with images.
    """
    if not os.path.isdir(p):
        return False
    cat_dir = os.path.join(p, "cat")
    dog_dir = os.path.join(p, "dog")
    return _has_images_in_dir(cat_dir) and _has_images_in_dir(dog_dir)


def _find_train_root(base_path):
    """
    Fix: Prefer the *actual* ImageFolder root (has cat/ and dog/). Avoid picking a directory
    that contains a nested 'train/' folder which would be treated as a bogus class 'train'.
    """
    candidates = [
        os.path.join(base_path, "train"),
        os.path.join(base_path, "train", "train"),
    ]
    for p in candidates:
        if _is_imagefolder_root(p):
            return p

    for root, dirs, files in os.walk(base_path):
        rel = os.path.relpath(root, base_path)
        if rel.count(os.sep) > 8:
            continue
        if _is_imagefolder_root(root):
            return root
    return None


TRAIN_DIR = _find_train_root(BASE_PATH)
if TRAIN_DIR is None:
    raise FileNotFoundError(
        f"Could not find a valid TRAIN_DIR containing 'cat' and 'dog' subfolders with images under BASE_PATH={BASE_PATH}"
    )

TEST_DIR_CANDIDATES = [
    os.path.join(BASE_PATH, "test", "test", "unknown"),
    os.path.join(BASE_PATH, "test", "unknown"),
    os.path.join(BASE_PATH, "test", "test"),
    os.path.join(BASE_PATH, "test"),
]
TEST_DIR = next((p for p in TEST_DIR_CANDIDATES if _has_images_in_dir(p)), None)
if TEST_DIR is None:
    TEST_DIR = next((p for p in TEST_DIR_CANDIDATES if os.path.isdir(p)), None)
if TEST_DIR is None:
    raise FileNotFoundError(
        f"Could not find a valid TEST_DIR. Tried: {TEST_DIR_CANDIDATES}"
    )

SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
if not os.path.exists(SAMPLE_SUB_PATH):
    alt = "/kaggle/input/sample_submission.csv"
    if os.path.exists(alt):
        SAMPLE_SUB_PATH = alt
    else:
        raise FileNotFoundError(f"Missing sample_submission.csv at: {SAMPLE_SUB_PATH}")

print("BASE_PATH:", BASE_PATH, "| exists:", os.path.exists(BASE_PATH))
print("TRAIN_DIR:", TRAIN_DIR, "| exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR :", TEST_DIR, "| exists:", os.path.exists(TEST_DIR))
print(
    "Sample submission:", SAMPLE_SUB_PATH, "| exists:", os.path.exists(SAMPLE_SUB_PATH)
)
print(
    "Train subfolders:", sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
)
print(
    "Num test images (if directly available):",
    (
        len([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
        if os.path.isdir(TEST_DIR)
        else 0
    ),
)




## === cell 1
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


seed_everything(42)



## === cell 2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 3
IMG_SIZE = 224
train_tfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_tfms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

full_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)

class_to_idx = full_ds.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
print("class_to_idx:", class_to_idx, " (dog should map to 1 probability output later)")

dog_idx = class_to_idx.get("dog", None)
cat_idx = class_to_idx.get("cat", None)
assert (
    dog_idx is not None and cat_idx is not None
), "Expected 'cat' and 'dog' folders under TRAIN_DIR."
print("dog_idx:", dog_idx, "| cat_idx:", cat_idx, "| total train imgs:", len(full_ds))

BATCH_SIZE = 64 if torch.cuda.is_available() else 32



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2882039253.py in <cell line: 0>()
     18 
     19 # Fix: create ImageFolder only from verified TRAIN_DIR root with cat/ and dog/
---> 20 full_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
     21 
     22 class_to_idx = full_ds.class_to_idx

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, transform, target_transform, loader, is_valid_file, allow_empty)
    326         allow_empty: bool = False,
    327     ):
--> 328         super().__init__(
    329             root,
    330             loader,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, loader, extensions, transform, target_transform, is_valid_file, allow_empty)
    148         super().__init__(root, transform=transform, target_transform=target_transform)
    149         classes, class_to_idx = self.find_classes(self.root)
--> 150         samples = self.make_dataset(
    151             self.root,
    152             class_to_idx=class_to_idx,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
    201             # is potentially overridden and thus could have a different logic.
    202             raise ValueError("The class_to_idx parameter cannot be None.")
--> 203         return make_dataset(
    204             directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty
    205         )

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
    102         if extensions is not None:
    103             msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
--> 104         raise FileNotFoundError(msg)
    105 
    106     return instances

FileNotFoundError: Found no valid file for the classes train. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

## === cell 4
n = len(full_ds)
indices = np.arange(n)
rng = np.random.RandomState(42)
rng.shuffle(indices)
valid_frac = 0.1
n_valid = int(n * valid_frac)
valid_idx = indices[:n_valid]
train_idx = indices[n_valid:]

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
valid_ds = datasets.ImageFolder(TRAIN_DIR, transform=valid_tfms)

train_subset = Subset(train_ds, train_idx.tolist())
valid_subset = Subset(valid_ds, valid_idx.tolist())

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
valid_loader = DataLoader(
    valid_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(train_subset), len(valid_subset)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2898256612.py in <cell line: 0>()
      1 # Fix: reuse full_ds length for split indices; create train/valid datasets with respective transforms
----> 2 n = len(full_ds)
      3 indices = np.arange(n)
      4 rng = np.random.RandomState(42)
      5 rng.shuffle(indices)

NameError: name 'full_ds' is not defined

## === cell 5
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device, non_blocking=torch.cuda.is_available())
            y = y.to(device, non_blocking=torch.cuda.is_available())
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
    return total_loss / total, correct / total




## === cell 6
start = time.time()
EPOCHS = 2

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(valid_loader, train=False)
    print(
        f"Epoch {epoch}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | valid loss {va_loss:.4f} acc {va_acc:.4f}"
    )

print("Training time (s):", round(time.time() - start, 2))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/511387294.py in <cell line: 0>()
      3 
      4 for epoch in range(1, EPOCHS + 1):
----> 5     tr_loss, tr_acc = run_epoch(train_loader, train=True)
      6     va_loss, va_acc = run_epoch(valid_loader, train=False)
      7     print(

NameError: name 'train_loader' is not defined

## === cell 7
test_files = (
    [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
    if os.path.isdir(TEST_DIR)
    else []
)

if len(test_files) == 0:
    sample_tmp = pd.read_csv(SAMPLE_SUB_PATH)
    test_files = [f"{int(i)}.jpg" for i in sample_tmp["id"].astype(int).tolist()]

test_files = sorted(test_files, key=lambda x: int(os.path.splitext(x)[0]))

from PIL import Image


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, files, transform=None):
        self.root = root
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        path = os.path.join(self.root, fn)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        img_id = int(os.path.splitext(fn)[0])
        return img, img_id


test_ds = TestDataset(TEST_DIR, test_files, transform=valid_tfms)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(test_ds), (test_files[:3] if len(test_files) else [])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1908189172.py in <cell line: 0>()
     37 test_loader = DataLoader(
     38     test_ds,
---> 39     batch_size=BATCH_SIZE,
     40     shuffle=False,
     41     num_workers=2,

NameError: name 'BATCH_SIZE' is not defined

## === cell 8
model.eval()
all_ids = []
all_probs = []
with torch.no_grad():
    for x, img_ids in test_loader:
        x = x.to(device, non_blocking=torch.cuda.is_available())
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, dog_idx].detach().cpu().numpy()
        all_probs.append(probs)
        all_ids.append(img_ids.detach().cpu().numpy())

all_probs = (
    np.concatenate(all_probs, axis=0)
    if len(all_probs)
    else np.array([], dtype=np.float32)
)
all_ids = (
    np.concatenate(all_ids, axis=0) if len(all_ids) else np.array([], dtype=np.int64)
)

all_probs = np.clip(all_probs, 1e-6, 1 - 1e-6)

len(all_ids), len(all_probs), all_ids[:5], all_probs[:5]



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3998945450.py in <cell line: 0>()
      3 all_probs = []
      4 with torch.no_grad():
----> 5     for x, img_ids in test_loader:
      6         x = x.to(device, non_blocking=torch.cuda.is_available())
      7         logits = model(x)

NameError: name 'test_loader' is not defined

## === cell 9
sample = pd.read_csv(SAMPLE_SUB_PATH)

all_ids = np.asarray(all_ids)
all_probs = np.asarray(all_probs)

sub = pd.DataFrame({"id": all_ids.astype(int), "label": all_probs.astype(float)})
sub = sub.sort_values("id").reset_index(drop=True)

sample_ids = sample["id"].astype(int).values
sub = sub.set_index("id").reindex(sample_ids)
sub["label"] = sub["label"].astype(float).fillna(0.5)
sub = sub.reset_index()

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Submission shape:", sub.shape)
print(sub.head())
print(sub.describe())
