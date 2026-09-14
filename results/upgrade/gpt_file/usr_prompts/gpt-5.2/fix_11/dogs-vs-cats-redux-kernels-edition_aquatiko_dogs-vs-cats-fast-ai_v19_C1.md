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
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision
from torchvision import transforms, models

try:
    Image.MAX_IMAGE_PIXELS = None
except Exception:
    pass

from torchvision.io import read_image, ImageReadMode

try:
    torchvision.set_image_backend("accimage")  # if available, faster decoding
except Exception:
    pass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("device:", device)




## === cell 1
PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"

TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
os.makedirs(TMP_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

sz = 224

train_dir = os.path.join(PATH, "train")
test_dir = os.path.join(PATH, "test")

print("PATH exists:", os.path.exists(PATH))
print("train_dir:", train_dir, "exists:", os.path.exists(train_dir))
print("test_dir:", test_dir, "exists:", os.path.exists(test_dir))

assert os.path.exists(train_dir), f"Train directory not found: {train_dir}"
assert os.path.exists(test_dir), f"Test directory not found: {test_dir}"




## === cell 2
cat_dir = os.path.join(train_dir, "cat")
dog_dir = os.path.join(train_dir, "dog")

assert os.path.exists(cat_dir) and os.path.exists(
    dog_dir
), "Expected train/cat and train/dog folders."

cat_files = sorted(
    [
        os.path.join(cat_dir, f)
        for f in os.listdir(cat_dir)
        if f.lower().endswith(".jpg")
    ]
)
dog_files = sorted(
    [
        os.path.join(dog_dir, f)
        for f in os.listdir(dog_dir)
        if f.lower().endswith(".jpg")
    ]
)

fnames = np.array(cat_files + dog_files)
labels = np.array([0] * len(cat_files) + [1] * len(dog_files), dtype=np.int64)

print("n_train:", len(fnames), "n_cat:", len(cat_files), "n_dog:", len(dog_files))
print("example:", fnames[-2], labels[-2])




## === cell 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

from torchvision.transforms import InterpolationMode

train_tfms = transforms.Compose(
    [
        transforms.Resize(
            (sz, sz), interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

test_tfms = transforms.Compose(
    [
        transforms.Resize(
            (sz, sz), interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

test_tfms_flip = transforms.Compose(
    [
        transforms.Resize(
            (sz, sz), interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class ImgDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None, cache=False):
        self.paths = list(paths)
        self.labels = None if labels is None else np.array(labels, dtype=np.int64)
        self.transform = transform
        self.cache = bool(cache)
        self._cache = [None] * len(self.paths) if self.cache else None

    def __len__(self):
        return len(self.paths)

    def _load_img_tensor_uint8(self, p):
        try:
            return read_image(p, mode=ImageReadMode.RGB)  # uint8, CxHxW
        except Exception:
            with Image.open(p) as im:
                im = im.convert("RGB")
                return transforms.PILToTensor()(im)  # uint8, CxHxW

    def __getitem__(self, idx):
        if self.cache:
            x = self._cache[idx]
            if x is None:
                x = self._load_img_tensor_uint8(self.paths[idx])
                self._cache[idx] = x
        else:
            x = self._load_img_tensor_uint8(self.paths[idx])

        x = self.transform(x) if self.transform is not None else x

        if self.labels is None:
            return x, os.path.basename(self.paths[idx])
        return x, int(self.labels[idx])


class ImgDatasetTTAFlip(Dataset):
    def __init__(self, paths, transform_base=None, cache=False):
        self.paths = list(paths)
        self.transform_base = transform_base
        self.cache = bool(cache)
        self._cache = [None] * len(self.paths) if self.cache else None

    def __len__(self):
        return len(self.paths)

    def _load_img_tensor_uint8(self, p):
        try:
            return read_image(p, mode=ImageReadMode.RGB)  # uint8, CxHxW
        except Exception:
            with Image.open(p) as im:
                im = im.convert("RGB")
                return transforms.PILToTensor()(im)

    def __getitem__(self, idx):
        if self.cache:
            x = self._cache[idx]
            if x is None:
                x = self._load_img_tensor_uint8(self.paths[idx])
                self._cache[idx] = x
        else:
            x = self._load_img_tensor_uint8(self.paths[idx])

        x = (
            self.transform_base(x)
            if self.transform_base is not None
            else transforms.ConvertImageDtype(torch.float32)(x)
        )
        return x, os.path.basename(self.paths[idx])




## === cell 4
arch = "resnet50"

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

print("train/valid sizes:", len(train_paths), len(valid_paths))

batch_size = 32

cpu_cnt = os.cpu_count() or 2


def _seed_worker(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


num_workers = min(8, max(2, cpu_cnt - 1))
pin_memory = device.type == "cuda"

train_dl = DataLoader(
    ImgDataset(train_paths, train_labels, transform=train_tfms, cache=False),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)
valid_dl = DataLoader(
    ImgDataset(valid_paths, valid_labels, transform=test_tfms, cache=False),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)




## === cell 5
weights = None
try:
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
except Exception:
    model = models.resnet50(pretrained=True)

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model = model.to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

try:
    model = torch.jit.script(model)
except Exception as e:
    print("TorchScript scripting failed; continuing without it. Reason:", repr(e))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)


def run_one_epoch(model, dl, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    n_items = 0

    _device = device
    _criterion = criterion
    _optimizer = optimizer
    _is_cuda = _device.type == "cuda"

    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for xb, yb in dl:
            xb = xb.to(_device, non_blocking=True)
            if _is_cuda:
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(_device, non_blocking=True)

            if train:
                _optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = _criterion(logits, yb)

            if train:
                loss.backward()
                _optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.detach().argmax(dim=1) == yb).sum().item()
            n_items += bs

    return total_loss / n_items, total_correct / n_items




## === cell 6
start = time.time()
for epoch in range(2):
    tr_loss, tr_acc = run_one_epoch(model, train_dl, train=True)
    va_loss, va_acc = run_one_epoch(model, valid_dl, train=False)
    print(
        f"epoch {epoch+1}/2 | train loss {tr_loss:.4f} acc {tr_acc:.4f} | valid loss {va_loss:.4f} acc {va_acc:.4f}"
    )
print("train time (s):", round(time.time() - start, 2))

torch.save(model.state_dict(), os.path.join(MODEL_PATH, "resnet50_catdog.pth"))




## === cell 7
test_unknown_dir = os.path.join(test_dir, "unknown")
assert os.path.exists(
    test_unknown_dir
), f"Expected test images under: {test_unknown_dir}"

test_files = [
    os.path.join(test_unknown_dir, f)
    for f in os.listdir(test_unknown_dir)
    if f.lower().endswith(".jpg")
]
assert len(test_files) > 0, f"No test images found under: {test_unknown_dir}"

test_files = sorted(
    test_files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
)

print(
    "n_test:",
    len(test_files),
    "first:",
    os.path.basename(test_files[0]),
    "last:",
    os.path.basename(test_files[-1]),
)

test_dl_tta = DataLoader(
    ImgDatasetTTAFlip(test_files, transform_base=test_tfms, cache=False),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)




## === cell 8
def predict_probs_tta_flip(model, dl, n_items):
    model.eval()
    probs_all = np.empty((n_items, 2), dtype=np.float32)
    names_all = [None] * n_items
    write_pos = 0

    _device = device
    _is_cuda = _device.type == "cuda"

    with torch.inference_mode():
        for xb, names in dl:
            bs = xb.size(0)
            xb = xb.to(_device, non_blocking=True)
            if _is_cuda:
                xb = xb.contiguous(memory_format=torch.channels_last)

            xb_flip = xb.flip(dims=[3])

            logits1 = model(xb)
            logits2 = model(xb_flip)

            probs = torch.softmax(logits1, dim=1)
            probs.add_(torch.softmax(logits2, dim=1)).mul_(0.5)

            probs_all[write_pos : write_pos + bs] = probs.cpu().numpy()
            names_all[write_pos : write_pos + bs] = names
            write_pos += bs

    return probs_all, names_all


prob_predictions, names1 = predict_probs_tta_flip(
    model, test_dl_tta, n_items=len(test_files)
)
probs = prob_predictions[:, 1]  # probability of class 1 (dog)

print("probs summary:", pd.Series(probs).describe())




## === cell 9
ids_int = np.fromiter(
    (int(n[:-4]) for n in names1),  # strip ".jpg"
    dtype=np.int64,
    count=len(names1),
)

ans = pd.DataFrame({"id": ids_int, "label": probs})
ans = ans.sort_values("id").reset_index(drop=True)

print("submission shape:", ans.shape)
print(ans.head())

out_path = "submission.csv"
ans.to_csv(out_path, index=False)
print("Wrote:", out_path, "bytes:", os.path.getsize(out_path))
print(pd.read_csv(out_path).head())
