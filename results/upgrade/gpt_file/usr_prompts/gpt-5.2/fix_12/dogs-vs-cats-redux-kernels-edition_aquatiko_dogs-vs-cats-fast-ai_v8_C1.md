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
import torchvision
from torchvision import transforms

print("CUDA available:", torch.cuda.is_available())
print("Input listing:", os.listdir("../input")[:20])



## === cell 1
PATH = "../input/dogs-vs-cats-redux-kernels-edition/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
os.makedirs(TMP_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

sz = 224

train_cat_dir = os.path.join(PATH, "train", "cat")
train_dog_dir = os.path.join(PATH, "train", "dog")

test_dir_candidates = [
    os.path.join(
        PATH, "test", "test", "unknown"
    ),  # common nested layout in this environment
    os.path.join(PATH, "test", "unknown"),  # flatter layout fallback
    os.path.join(PATH, "test", "test"),  # older kernels layout (direct jpgs)
]
test_dir = next((d for d in test_dir_candidates if os.path.isdir(d)), None)

assert os.path.isdir(train_cat_dir), f"Missing dir: {train_cat_dir}"
assert os.path.isdir(train_dog_dir), f"Missing dir: {train_dog_dir}"
assert test_dir is not None, f"Missing test dir. Tried: {test_dir_candidates}"

sample_sub_path = os.path.join(PATH, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    sample_sub_path = "../input/sample_submission.csv"
print("sample_submission path:", sample_sub_path)
print("Using test_dir:", test_dir)



## === cell 2
cat_files = sorted(
    [
        os.path.join(train_cat_dir, e.name)
        for e in os.scandir(train_cat_dir)
        if e.is_file() and e.name.lower().endswith(".jpg")
    ]
)
dog_files = sorted(
    [
        os.path.join(train_dog_dir, e.name)
        for e in os.scandir(train_dog_dir)
        if e.is_file() and e.name.lower().endswith(".jpg")
    ]
)

fnames = np.array(cat_files + dog_files)
labels = np.array([0] * len(cat_files) + [1] * len(dog_files), dtype=np.int64)

print(
    "Train images:",
    len(fnames),
    "cats:",
    (labels == 0).sum(),
    "dogs:",
    (labels == 1).sum(),
)
print("Example:", fnames[-2], labels[-2])



## === cell 3
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



## === cell 4
try:
    torchvision.set_image_backend("accimage")
    print("torchvision image backend:", torchvision.get_image_backend())
except Exception as e:
    print("Could not set accimage backend, using default. Err:", repr(e))
    print("torchvision image backend:", torchvision.get_image_backend())

from torchvision.io import read_image
from torchvision.transforms import InterpolationMode

train_tfms = transforms.Compose(
    [
        transforms.Resize(
            (sz, sz), interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_tfms = transforms.Compose(
    [
        transforms.Resize(
            (sz, sz), interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _safe_filename_from_path(p: str) -> str:
    st = os.stat(p)
    key = f"{p}|{int(st.st_mtime)}|{int(st.st_size)}"
    import hashlib

    h = hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return h + ".pt"




## === cell 5
import torchvision.transforms.functional as F


def _load_rgb_uint8(p):
    img = read_image(p)  # uint8 [C,H,W]
    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)
    elif img.shape[0] >= 3:
        img = img[:3]
    return img


def _resize_uint8_chw(img_uint8, hw):
    return F.resize(
        img_uint8,
        list(hw),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


_GLOBAL_MMAP_ARGS = {}


def _mmap_worker_init(dat_path, N, H, W):
    global _GLOBAL_MMAP_ARGS
    _GLOBAL_MMAP_ARGS = {"dat_path": dat_path, "N": N, "H": H, "W": W}


def _mmap_worker_write_slice(args):
    start, end, paths_slice = args
    dat_path = _GLOBAL_MMAP_ARGS["dat_path"]
    N = _GLOBAL_MMAP_ARGS["N"]
    H = _GLOBAL_MMAP_ARGS["H"]
    W = _GLOBAL_MMAP_ARGS["W"]
    local_mm = np.memmap(dat_path, dtype=np.uint8, mode="r+", shape=(N, 3, H, W))
    for j, p in enumerate(paths_slice):
        img = _load_rgb_uint8(p)
        img = _resize_uint8_chw(img, (H, W))
        local_mm[start + j] = img.numpy()
    local_mm.flush()
    return (start, end)


def build_or_load_uint8_mmap(paths, mmap_prefix, resize_hw, verbose_every=2000):
    """
    Creates:
      {mmap_prefix}.dat  : raw uint8 tensor data (N,3,H,W)
      {mmap_prefix}.meta : npz with paths hash + shape info
    Returns: np.memmap array with shape (N,3,H,W), dtype=uint8
    """
    paths = list(paths)
    N = len(paths)
    H, W = resize_hw

    if N == 0:
        raise ValueError(
            f"build_or_load_uint8_mmap got 0 paths for prefix={mmap_prefix}"
        )

    dat_path = mmap_prefix + ".dat"
    meta_path = mmap_prefix + ".meta.npz"

    import hashlib

    h = hashlib.md5(usedforsecurity=False)
    for p in paths:
        st = os.stat(p)
        h.update(p.encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
        h.update(str(int(st.st_size)).encode("utf-8"))
    fp = h.hexdigest()

    if os.path.exists(dat_path) and os.path.exists(meta_path):
        try:
            meta = np.load(meta_path, allow_pickle=False)
            if (
                str(meta["fingerprint"].item()) == fp
                and int(meta["N"].item()) == N
                and int(meta["H"].item()) == H
                and int(meta["W"].item()) == W
            ):
                mm = np.memmap(dat_path, dtype=np.uint8, mode="r", shape=(N, 3, H, W))
                return mm
        except Exception:
            pass

    t0 = time.time()

    mm = np.memmap(dat_path, dtype=np.uint8, mode="w+", shape=(N, 3, H, W))
    mm.flush()
    del mm

    cpu_cnt = os.cpu_count() or 2
    n_workers = (
        min(8, max(2, cpu_cnt - 1)) if N >= 2000 else min(4, max(1, cpu_cnt - 1))
    )
    chunk = 512 if N >= 10000 else 256
    tasks = []
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        tasks.append((start, end, paths[start:end]))

    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    with ctx.Pool(
        processes=n_workers,
        initializer=_mmap_worker_init,
        initargs=(dat_path, N, H, W),
    ) as pool:
        done = 0
        for start, end in pool.imap_unordered(
            _mmap_worker_write_slice, tasks, chunksize=1
        ):
            done += end - start
            if verbose_every and (done // verbose_every) != (
                (done - (end - start)) // verbose_every
            ):
                print(f"cached {done}/{N} images to mmap... ({time.time()-t0:.1f}s)")

    np.savez_compressed(meta_path, fingerprint=fp, N=N, H=H, W=W)
    print(f"mmap cache built: {N} images -> {dat_path} in {time.time()-t0:.1f}s")
    mm = np.memmap(dat_path, dtype=np.uint8, mode="r", shape=(N, 3, H, W))
    return mm




## === cell 6
class CatsDogsDataset(Dataset):
    def __init__(
        self,
        paths,
        y=None,
        tfm=None,
        cache=False,
        seed=42,
        cache_mode="decoded",  # "decoded" or "transformed"
        disk_cache_dir=None,  # unused after mmap optimization; kept for API compatibility
        resize_hw=None,  # must be set when using mmap cache
        uint8_mmap=None,  # np.memmap or None
        mmap_offset=0,  # for indexing into a shared mmap
    ):
        self.paths = list(paths)
        self.y = None if y is None else np.asarray(y, dtype=np.int64)
        self.tfm = tfm
        self.cache = bool(cache)
        self.cache_mode = str(cache_mode)
        self.seed = int(seed)
        self.epoch = 0

        self.resize_hw = resize_hw
        self.uint8_mmap = uint8_mmap
        self.mmap_offset = int(mmap_offset)

        self._cached = None
        if self.cache:
            self._cached = [None] * len(self.paths)

        self._has_rhf = False
        self._post_tfm = tfm
        if tfm is not None and isinstance(tfm, transforms.Compose):
            ts = list(tfm.transforms)
            if len(ts) > 1 and isinstance(ts[1], transforms.RandomHorizontalFlip):
                self._has_rhf = True
                post = [ts[0]] + ts[2:]
                self._post_tfm = transforms.Compose(post)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _flip_decision(seed, epoch, idx):
        x = (seed ^ (epoch * 1_000_003) ^ (idx * 9176)) & 0xFFFFFFFF
        x ^= x >> 16
        x = (x * 0x7FEB352D) & 0xFFFFFFFF
        x ^= x >> 15
        x = (x * 0x846CA68B) & 0xFFFFFFFF
        x ^= x >> 16
        return (x & 1) == 1  # p≈0.5 exactly by LSB

    def _get_uint8_chw(self, idx):
        if self.uint8_mmap is not None:
            arr = self.uint8_mmap[self.mmap_offset + idx]  # (3,H,W) uint8
            return torch.from_numpy(np.asarray(arr))  # view/copy-safe
        img = _load_rgb_uint8(self.paths[idx])
        if self.resize_hw is not None:
            img = _resize_uint8_chw(img, self.resize_hw)
        return img

    def __getitem__(self, idx):
        if self.cache and (self._cached[idx] is not None):
            cached = self._cached[idx]
            if self.cache_mode == "decoded":
                img_uint8 = cached
            else:
                img = cached
                if self.y is None:
                    return img, os.path.basename(self.paths[idx])
                return img, int(self.y[idx])
        else:
            img_uint8 = self._get_uint8_chw(idx)

        if self.tfm is None:
            img = img_uint8
        else:
            if self._has_rhf:
                img2 = img_uint8
                if self._flip_decision(self.seed, self.epoch, idx):
                    img2 = torch.flip(img2, dims=[2])
                img = self._post_tfm(img2)
            else:
                img = self.tfm(img_uint8)

        if self.cache:
            self._cached[idx] = img_uint8 if self.cache_mode == "decoded" else img

        if self.y is None:
            return img, os.path.basename(self.paths[idx])
        return img, int(self.y[idx])




## === cell 7
n = len(fnames)
perm = np.random.permutation(n)
valid_size = int(0.1 * n)
valid_idx = perm[:valid_size]
train_idx = perm[valid_size:]

decoded_cache_prefix = os.path.join(TMP_PATH, f"train_resized_uint8_{sz}")
train_uint8_mm = build_or_load_uint8_mmap(
    fnames.tolist(),
    mmap_prefix=decoded_cache_prefix,
    resize_hw=(sz, sz),
    verbose_every=5000,
)

train_ds = CatsDogsDataset(
    fnames[train_idx],
    labels[train_idx],
    tfm=train_tfms,
    cache=True,
    seed=seed,
    cache_mode="decoded",
    disk_cache_dir=None,
    resize_hw=(sz, sz),
    uint8_mmap=train_uint8_mm,
    mmap_offset=0,
)
valid_ds = CatsDogsDataset(
    fnames[valid_idx],
    labels[valid_idx],
    tfm=valid_tfms,
    cache=True,
    seed=seed,
    cache_mode="transformed",
    disk_cache_dir=None,
    resize_hw=(sz, sz),
    uint8_mmap=train_uint8_mm,
    mmap_offset=0,
)

batch_size = 64 if device.type == "cuda" else 32

cpu_cnt = os.cpu_count() or 2
if device.type == "cuda":
    num_workers = min(8, max(2, cpu_cnt - 1))
else:
    num_workers = min(4, max(1, cpu_cnt - 1))

g = torch.Generator()
g.manual_seed(seed)


def _seed_worker(worker_id):
    wseed = seed + worker_id
    random.seed(wseed)
    np.random.seed(wseed)
    torch.manual_seed(wseed)


dl_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else 2,
    generator=g,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

train_dl = DataLoader(train_ds, shuffle=True, drop_last=False, **dl_kwargs)
valid_dl = DataLoader(valid_ds, shuffle=False, drop_last=False, **dl_kwargs)

len(train_ds), len(valid_ds), batch_size, num_workers



## === cell 8
arch_name = "resnet34"
model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model = model.to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def correct_count_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).sum()




## === cell 9
def run_epoch(model, dl, train=True):
    model.train(train)
    total_loss = 0.0
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_n = 0

    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        ctx = torch.enable_grad() if train else torch.inference_mode()
        with ctx:
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.detach()) * bs
        total_correct += correct_count_from_logits(logits.detach(), yb)
        total_n += bs

    return total_loss / max(total_n, 1), (total_correct.item() / max(total_n, 1))


start = time.time()
for epoch in range(1, 3):
    train_ds.set_epoch(epoch)
    tr_loss, tr_acc = run_epoch(model, train_dl, train=True)
    va_loss, va_acc = run_epoch(model, valid_dl, train=False)
    print(
        f"epoch {epoch} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | valid loss {va_loss:.4f} acc {va_acc:.4f}"
    )
print("Training seconds:", round(time.time() - start, 2))

torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{arch_name}_catsdogs.pt"))



## === cell 10
test_files = sorted(
    [
        os.path.join(test_dir, e.name)
        for e in os.scandir(test_dir)
        if e.is_file() and e.name.lower().endswith(".jpg")
    ]
)

if len(test_files) == 0:
    for cand in test_dir_candidates:
        if os.path.isdir(cand):
            tmp = sorted(
                [
                    os.path.join(cand, e.name)
                    for e in os.scandir(cand)
                    if e.is_file() and e.name.lower().endswith(".jpg")
                ]
            )
            if len(tmp) > 0:
                test_dir = cand
                test_files = tmp
                break

assert len(test_files) > 0, f"No test jpgs found. Tried dirs: {test_dir_candidates}"

test_cache_prefix = os.path.join(TMP_PATH, f"test_resized_uint8_{sz}")
test_uint8_mm = build_or_load_uint8_mmap(
    test_files,
    mmap_prefix=test_cache_prefix,
    resize_hw=(sz, sz),
    verbose_every=5000,
)

test_ids_int = np.fromiter(
    (int(os.path.basename(p)[:-4]) for p in test_files),
    dtype=np.int64,
    count=len(test_files),
)

test_ds = CatsDogsDataset(
    test_files,
    y=None,
    tfm=valid_tfms,
    cache=True,
    seed=seed,
    cache_mode="transformed",
    disk_cache_dir=None,
    resize_hw=(sz, sz),
    uint8_mmap=test_uint8_mm,
    mmap_offset=0,
)
test_dl = DataLoader(test_ds, shuffle=False, drop_last=False, **dl_kwargs)

model.eval()
all_probs_dog = []

with torch.inference_mode():
    for xb, _fn in test_dl:
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # dog prob
        all_probs_dog.append(probs)

all_probs_dog = np.concatenate(all_probs_dog, axis=0).astype(np.float64, copy=False)

order = np.argsort(test_ids_int)
ids_sorted = test_ids_int[order]
probs_sorted = all_probs_dog[order]

print(
    "Test preds:",
    len(ids_sorted),
    "min/max prob:",
    float(probs_sorted.min()),
    float(probs_sorted.max()),
)



## === cell 11
ans = pd.DataFrame({"id": ids_sorted, "label": probs_sorted})

sample_sub = pd.read_csv(sample_sub_path)

eps = 1e-6
ans["label"] = ans["label"].astype(np.float64).clip(eps, 1.0 - eps)

ans = sample_sub[["id"]].merge(ans, on="id", how="left")
ans["label"] = ans["label"].astype(float).fillna(0.5).clip(eps, 1.0 - eps)

assert ans["id"].isna().sum() == 0
assert len(ans) == len(sample_sub)
assert (ans["id"].values == sample_sub["id"].values).all()

ans.to_csv("submission.csv", index=False)
print(ans.head())
print("Wrote submission.csv with shape:", ans.shape)
print("submission.csv columns:", list(ans.columns))
print("submission.csv path:", os.path.abspath("submission.csv"))
print("label min/max:", float(ans["label"].min()), float(ans["label"].max()))
