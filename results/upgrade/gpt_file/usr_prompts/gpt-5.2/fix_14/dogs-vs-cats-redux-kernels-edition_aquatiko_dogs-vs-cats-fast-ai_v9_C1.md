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

try:
    cpu_cnt = os.cpu_count() or 4
    torch.set_num_threads(min(2, cpu_cnt))
    torch.set_num_interop_threads(1)
except Exception:
    pass

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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


def list_jpgs_fast(d):
    with os.scandir(d) as it:
        return [e.path for e in it if e.is_file() and e.name.endswith(".jpg")]


cat_files = list_jpgs_fast(cat_dir)
dog_files = list_jpgs_fast(dog_dir)

fnames = np.array(cat_files + dog_files, dtype=object)
fnames = fnames[
    np.argsort(
        np.fromiter(
            (os.path.basename(p) for p in fnames), dtype=object, count=len(fnames)
        )
    )
]

labels = np.empty(len(fnames), dtype=np.int64)
cat_prefix = os.path.join(cat_dir, "")
is_cat = np.fromiter(
    (p.startswith(cat_prefix) for p in fnames), dtype=np.bool_, count=len(fnames)
)
labels[is_cat] = 0
labels[~is_cat] = 1

print("n_train:", len(fnames), "label mean (dogs):", labels.mean())
print("example:", fnames[-2], labels[-2])




## === cell 3
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import InterpolationMode

try:
    import torchvision

    if hasattr(torchvision, "set_image_backend"):
        torchvision.set_image_backend("accimage")
except Exception:
    pass

import torchvision.transforms.v2 as v2

normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_tfms = v2.Compose(
    [
        v2.RandomResizedCrop(
            sz, scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR
        ),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ]
)

valid_tfms = v2.Compose(
    [
        v2.Resize(int(sz * 1.14), interpolation=InterpolationMode.BILINEAR),
        v2.CenterCrop(sz),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ]
)


class CatsDogsDataset(Dataset):
    def __init__(self, paths, labels=None, tfms=None, return_id=False):
        self.paths = np.asarray(list(paths), dtype=object)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.tfms = tfms
        self.return_id = return_id

        self._basenames = np.fromiter(
            (os.path.basename(p) for p in self.paths),
            dtype=object,
            count=len(self.paths),
        )
        if return_id:
            self.ids = np.fromiter(
                (int(os.path.splitext(b)[0]) for b in self._basenames),
                dtype=np.int64,
                count=len(self.paths),
            )
        else:
            self.ids = None

    def __len__(self):
        return int(self.paths.shape[0])

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = read_image(p, mode=ImageReadMode.RGB)
        if self.tfms is not None:
            img = self.tfms(img)
        if self.labels is None:
            if self.return_id:
                return img, int(self.ids[idx])
            return img, self._basenames[idx]
        return img, int(self.labels[idx])




## === cell 4
arch = "resnet34"

model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model = model.to(device)

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

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

cpu_cnt = os.cpu_count() or 2

_num_workers = min(8, max(2, cpu_cnt - 1))
prefetch = 4

g = torch.Generator()
g.manual_seed(SEED)


def _seed_worker(worker_id):
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


pin = torch.cuda.is_available()

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=pin,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=prefetch if _num_workers > 0 else None,
    worker_init_fn=_seed_worker if _num_workers > 0 else None,
    generator=g,
)

valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=pin,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=prefetch if _num_workers > 0 else None,
    worker_init_fn=_seed_worker if _num_workers > 0 else None,
)

len(train_ds), len(valid_ds), batch_size




## === cell 6
def run_one_epoch(model, dl, train=True):
    if train:
        model.train()
        ctx_mgr = torch.enable_grad
    else:
        model.eval()
        ctx_mgr = torch.inference_mode

    total_loss = 0.0
    total_correct = 0
    total = 0

    _criterion = criterion
    _optimizer = optimizer
    _device = device
    use_cuda = torch.cuda.is_available()
    argmax = torch.Tensor.argmax

    for xb, yb in dl:
        xb = xb.to(_device, non_blocking=True)
        yb = yb.to(_device, non_blocking=True)

        if use_cuda:
            xb = xb.contiguous(memory_format=torch.channels_last)

        if train:
            _optimizer.zero_grad(set_to_none=True)

        with ctx_mgr():
            out = model(xb)
            loss = _criterion(out, yb)
            if train:
                loss.backward()
                _optimizer.step()

        bs = xb.size(0)
        total_loss += loss.detach().item() * bs
        total_correct += (argmax(out, dim=1) == yb).sum().item()
        total += bs

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
import os


def find_test_images_fast(root):
    fast_candidate = os.path.join(root, "test", "unknown")
    if os.path.isdir(fast_candidate):
        with os.scandir(fast_candidate) as it:
            jpgs = [e.path for e in it if e.is_file() and e.name.endswith(".jpg")]
    else:
        jpgs = []
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_dir():
                            stack.append(e.path)
                        elif e.is_file() and e.name.endswith(".jpg"):
                            jpgs.append(e.path)
            except FileNotFoundError:
                continue

    ids = np.fromiter(
        (int(os.path.splitext(os.path.basename(p))[0]) for p in jpgs),
        dtype=np.int64,
        count=len(jpgs),
    )
    order = np.argsort(ids)
    return [jpgs[i] for i in order]


test_images = find_test_images_fast(test_dir)
print("n_test:", len(test_images))
print("first:", test_images[0] if test_images else None)
print("last :", test_images[-1] if test_images else None)

test_ds = CatsDogsDataset(test_images, labels=None, tfms=valid_tfms, return_id=True)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=prefetch if _num_workers > 0 else None,
    worker_init_fn=_seed_worker if _num_workers > 0 else None,
)




## === cell 8
model.eval()
n_test = len(test_ds)
all_ids = np.empty(n_test, dtype=np.int64)
all_probs = np.empty(n_test, dtype=np.float64)

offset = 0
use_cuda = torch.cuda.is_available()
softmax = torch.softmax

with torch.inference_mode():
    for xb, ids in test_dl:
        xb = xb.to(device, non_blocking=True)
        if use_cuda:
            xb = xb.contiguous(memory_format=torch.channels_last)

        out = model(xb)
        probs = softmax(out, dim=1)[:, 1].detach().cpu().numpy()

        bs = probs.shape[0]
        all_ids[offset : offset + bs] = np.asarray(ids, dtype=np.int64)
        all_probs[offset : offset + bs] = probs
        offset += bs

len(all_ids), int(all_ids.min()) if len(all_ids) else None, (
    int(all_ids.max()) if len(all_ids) else None
)




## === cell 9
ans = pd.DataFrame({"id": all_ids, "label": all_probs})
ans = ans.sort_values("id").reset_index(drop=True)

ans["label"] = ans["label"].clip(1e-6, 1 - 1e-6)

sub_path = "submission.csv"
ans.to_csv(sub_path, index=False)

print(ans.head())
print(ans.tail())
print("Wrote:", sub_path, "rows:", len(ans))
