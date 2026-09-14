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
import zipfile
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_CANDIDATES = [
    "../input/dogs-vs-cats-redux-kernels-edition/dogs-vs-cats-redux-kernels-edition",
    "../input/dogs-vs-cats-redux-kernels-edition",
    "../input",
]
BASE_PATH = None
for p in BASE_CANDIDATES:
    if os.path.exists(p):
        if os.path.isdir(os.path.join(p, "train")) and os.path.isdir(
            os.path.join(p, "test")
        ):
            BASE_PATH = p
            break

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not find expected dataset directory with train/ and test/ under ../input"
    )

print("Using BASE_PATH:", BASE_PATH)

OUT_PATH = "./"
SUB_PATH = os.path.join(OUT_PATH, "submission.csv")

sz = 224

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = torch.cuda.is_available()

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

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
    if os.path.isdir(p):
        test_unknown_dir = p
        break

if not (os.path.isdir(train_cat_dir) and os.path.isdir(train_dog_dir)):
    raise FileNotFoundError(
        f"Train folders not found at {train_cat_dir} and {train_dog_dir}"
    )
if test_unknown_dir is None:
    raise FileNotFoundError(f"Test folder not found. Tried: {test_unknown_candidates}")


def _count_jpgs(d):
    c = 0
    with os.scandir(d) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".jpg"):
                c += 1
    return c


print("train_cat_dir:", train_cat_dir, "n=", _count_jpgs(train_cat_dir))
print("train_dog_dir:", train_dog_dir, "n=", _count_jpgs(train_dog_dir))
print("test_dir_used:", test_unknown_dir, "n=", _count_jpgs(test_unknown_dir))



## === cell 2
cat_files = sorted(
    [
        e.path
        for e in os.scandir(train_cat_dir)
        if e.is_file() and e.name.lower().endswith(".jpg")
    ]
)
dog_files = sorted(
    [
        e.path
        for e in os.scandir(train_dog_dir)
        if e.is_file() and e.name.lower().endswith(".jpg")
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

_HAS_TV_READ_IMAGE = hasattr(torchvision, "io") and hasattr(
    torchvision.io, "read_image"
)
try:
    from torchvision.transforms import v2 as T2  # torchvision>=0.15

    _HAS_TV_V2 = True
except Exception:
    T2 = None
    _HAS_TV_V2 = False

try:
    import torchvision.transforms.functional as F
    import torchvision.transforms.functional_tensor as FT  # older versions may not have it

    _HAS_FT = True
except Exception:
    import torchvision.transforms.functional as F

    FT = None
    _HAS_FT = False

if _HAS_TV_READ_IMAGE:

    def _fast_resize_uint8_chw(img_uint8_chw):
        img = F.resize(img_uint8_chw, [sz, sz], antialias=True)
        img = img.to(torch.float32).div_(255.0)
        img = F.normalize(img, mean=mean, std=std)
        return img

    def _fast_resize_uint8_chw_train(img_uint8_chw, do_flip: bool):
        img = F.resize(img_uint8_chw, [sz, sz], antialias=True)
        if do_flip:
            img = torch.flip(img, dims=[2])  # flip W dimension (CHW)
        img = img.to(torch.float32).div_(255.0)
        img = F.normalize(img, mean=mean, std=std)
        return img

else:
    _fast_resize_uint8_chw = None
    _fast_resize_uint8_chw_train = None


class CatsDogsDataset(Dataset):
    def __init__(
        self,
        files,
        labels=None,
        transform=None,
        return_id=False,
        ids=None,
        use_fast_io=False,
        train_mode=False,
        base_seed=42,
    ):
        self.files = list(files)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = transform
        self.return_id = return_id
        self.use_fast_io = bool(use_fast_io and _HAS_TV_READ_IMAGE)
        self.train_mode = bool(train_mode)
        self.base_seed = int(base_seed)
        if ids is not None:
            self.ids = np.asarray(ids, dtype=np.int64)
        else:
            self.ids = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]

        if self.use_fast_io:
            img = torchvision.io.read_image(fp)  # uint8, CHW, RGB
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = Image.open(fp).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        if self.labels is None:
            if self.return_id:
                if self.ids is not None:
                    _id = int(self.ids[idx])
                else:
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
batch_size = 64 if torch.cuda.is_available() else 32

use_fast = bool(_HAS_TV_READ_IMAGE and _fast_resize_uint8_chw is not None)

if use_fast:

    def tr_tfms_tensor(img_uint8_chw):
        do_flip = bool(torch.rand((), dtype=torch.float32) < 0.5)
        return _fast_resize_uint8_chw_train(img_uint8_chw, do_flip)

    def va_tfms_tensor(img_uint8_chw):
        return _fast_resize_uint8_chw(img_uint8_chw)

    tr_tfms = tr_tfms_tensor
    va_tfms = va_tfms_tensor
else:
    tr_tfms = train_tfms
    va_tfms = valid_tfms

train_ds = CatsDogsDataset(
    tr_files,
    tr_y,
    transform=tr_tfms,
    use_fast_io=use_fast,
    train_mode=True,
    base_seed=seed,
)
valid_ds = CatsDogsDataset(
    va_files,
    va_y,
    transform=va_tfms,
    use_fast_io=use_fast,
    train_mode=False,
    base_seed=seed,
)


def _seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

cpu_cnt = os.cpu_count() or 2

if torch.cuda.is_available():
    num_workers = min(8, max(4, cpu_cnt))
    prefetch = 4
else:
    num_workers = min(4, max(2, cpu_cnt // 2))
    prefetch = 2

pin = torch.cuda.is_available()

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=(prefetch if num_workers > 0 else None),
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
    drop_last=True,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=(prefetch if num_workers > 0 else None),
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

print(
    model.__class__.__name__,
    "ready. fast_io:",
    use_fast,
    "num_workers:",
    num_workers,
    "batch:",
    batch_size,
)



## === cell 5
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def run_epoch(model, dl, train=True):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0

    _criterion = criterion
    _optimizer = optimizer
    _device = device
    use_cuda = _device.type == "cuda"
    _use_amp = bool(use_amp and use_cuda)

    if train:
        ctx = torch.enable_grad()
    else:
        ctx = torch.inference_mode()

    with ctx:
        for xb, yb in dl:
            if use_cuda:
                xb = xb.to(_device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                yb = yb.to(_device, non_blocking=True)
            else:
                xb = xb.to(_device)
                yb = yb.to(_device)

            if train:
                _optimizer.zero_grad(set_to_none=True)

            if _use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = _criterion(logits, yb)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(_optimizer)
                    scaler.update()
            else:
                logits = model(xb)
                loss = _criterion(logits, yb)
                if train:
                    loss.backward()
                    _optimizer.step()

            bs = xb.size(0)
            total_loss += float(loss.detach()) * bs
            preds = logits.argmax(dim=1)
            total_correct += int((preds == yb).sum())
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
test_entries = [
    e
    for e in os.scandir(test_unknown_dir)
    if e.is_file() and e.name.lower().endswith(".jpg")
]
test_files = [e.path for e in test_entries]

stems = np.fromiter(
    (int(os.path.splitext(os.path.basename(p))[0]) for p in test_files),
    dtype=np.int64,
    count=len(test_files),
)
order = np.argsort(stems)
test_ids = stems[order]
test_files = [test_files[i] for i in order]

test_ds = CatsDogsDataset(
    test_files,
    labels=None,
    transform=va_tfms,
    return_id=True,
    ids=test_ids,
    use_fast_io=use_fast,
)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=(prefetch if num_workers > 0 else None),
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

print("Test files:", len(test_files))
print("Example test:", test_files[0])



## === cell 7
model.eval()

n_test = len(test_files)
all_ids = np.empty(n_test, dtype=np.int64)
all_probs_dog = np.empty(n_test, dtype=np.float64)

use_cuda = device.type == "cuda"
_use_amp = bool(use_amp and use_cuda)
offset = 0

with torch.inference_mode():
    for xb, ids in test_dl:
        bs = xb.size(0)
        if use_cuda:
            xb = xb.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            if _use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(xb)
            else:
                logits = model(xb)
        else:
            xb = xb.to(device)
            logits = model(xb)

        probs_dog = torch.softmax(logits, dim=1)[:, 1]
        all_ids[offset : offset + bs] = np.asarray(ids)
        all_probs_dog[offset : offset + bs] = probs_dog.detach().cpu().numpy()
        offset += bs

print(
    "Preds:",
    all_probs_dog.shape,
    "IDs:",
    all_ids.shape,
    "prob range:",
    (float(all_probs_dog.min()), float(all_probs_dog.max())),
)

sub = pd.DataFrame(
    {
        "id": all_ids.astype(np.int64, copy=False),
        "label": all_probs_dog.astype(float, copy=False),
    }
)

eps = 1e-6
sub["label"] = sub["label"].clip(eps, 1 - eps)

sub = sub.sort_values("id").reset_index(drop=True)

if sub["id"].duplicated().any():
    raise ValueError("Duplicate ids found in submission.")
if len(sub) != len(test_files):
    raise ValueError(
        f"Submission row count {len(sub)} != number of test files {len(test_files)}"
    )

print(sub.head())
print(
    "Submission rows:",
    len(sub),
    "id range:",
    (int(sub["id"].min()), int(sub["id"].max())),
)

sub.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH, "size bytes:", os.path.getsize(SUB_PATH))
print("Done.")
