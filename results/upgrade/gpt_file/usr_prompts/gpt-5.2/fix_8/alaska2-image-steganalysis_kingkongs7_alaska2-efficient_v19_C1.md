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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.8097877448852011

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.59713) has done: 'I remove the incompatible runtime `pip install albumentations==0.5.2` (it breaks due to SciPy/Numpy ABI issues) and use the already-installed Albumentations 2.x API while keeping the same augmentation intent and tensor output structure. I also fix the dataset indexing bug (using `.iloc` instead of `.loc` after `reset_index`) and make sure the dataset returns the expected `{"image": tensor}` dict so later code remains unchanged. Finally, I fix the invalid pretrained weight path and load the EfficientNet-B0 weights from the already-installed `torchvision` package (same architecture family, score-reasonable, and runs end-to-end), then generate a valid `submission.csv` with `Id,Label`. These changes are strictly to unblock execution and produce a valid submission; no extra training is introduced.'

# 9. Code solution

## === cell 0
import os
import random
import gc
import time
from glob import glob

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from tqdm.auto import tqdm
from sklearn import metrics

import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
seed = 42
print(f"setting everything to seed {seed}")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)



## === cell 2
data_dir = "../input/alaska2-image-steganalysis"

sample_size = int(
    os.environ.get("ALASKA2_SAMPLE_SIZE", "12000")
)  # per class; adjust via env var if needed
val_size = int(sample_size * 0.25)

train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # label 0 1 2 3

for label, folder in enumerate(folder_names):
    train_filenames = sorted(glob(f"{data_dir}/{folder}/*.jpg")[:sample_size])
    np.random.shuffle(train_filenames)
    train_fn.extend(train_filenames[val_size:])
    train_labels.extend(np.zeros(len(train_filenames[val_size:])) + label)
    val_fn.extend(train_filenames[:val_size])
    val_labels.extend(np.zeros(len(train_filenames[:val_size])) + label)

assert len(train_labels) == len(train_fn), "wrong labels"
assert len(val_labels) == len(val_fn), "wrong labels"

train_df = pd.DataFrame(
    {"ImageFileName": train_fn, "Label": train_labels},
    columns=["ImageFileName", "Label"],
)
train_df["Label"] = train_df["Label"].astype(int)
val_df = pd.DataFrame(
    {"ImageFileName": val_fn, "Label": val_labels}, columns=["ImageFileName", "Label"]
)
val_df["Label"] = val_df["Label"].astype(int)

print(train_df.head())
_ = train_df.Label.hist()



## === cell 3
img_size = 512

BASE_PREPROCESS = A.Compose(
    [
        A.Resize(img_size, img_size, p=1),
        A.ToFloat(max_value=255.0),
    ],
    p=1,
)

TRAIN_STOCHASTIC = A.Compose(
    [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(75, 100), p=0.5),
        ToTensorV2(),
    ],
    p=1,
)

TEST_POST = A.Compose([ToTensorV2()], p=1)

AUGMENTATIONS_TRAIN = None
AUGMENTATIONS_TEST = None

CACHE_DIR = "/kaggle/working/alaska2_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key_from_fns(fns, prefix):
    import hashlib

    h = hashlib.md5()
    step = max(1, len(fns) // 256)
    for fn in fns[::step]:
        h.update(fn.encode("utf-8", errors="ignore"))
    h.update(str(len(fns)).encode())
    h.update(str(img_size).encode())
    return f"{prefix}_{h.hexdigest()}_{len(fns)}_{img_size}"


def build_or_load_base_cache(fns, cache_prefix, dtype=np.float16):
    import multiprocessing as mp

    key = _cache_key_from_fns(list(fns), cache_prefix)
    mmap_path = os.path.join(CACHE_DIR, f"{key}.mmap")
    meta_path = os.path.join(CACHE_DIR, f"{key}.npz")

    if os.path.exists(mmap_path) and os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=False)
        shape = tuple(meta["shape"])
        arr = np.memmap(
            mmap_path, mode="r", dtype=np.dtype(meta["dtype"].item()), shape=shape
        )
        return arr, key

    n = len(fns)
    shape = (n, img_size, img_size, 3)
    arr = np.memmap(mmap_path, mode="w+", dtype=dtype, shape=shape)

    imread_flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION

    def _process_one(args):
        i, fn = args
        im = cv2.imread(fn, imread_flags)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        im = im[:, :, ::-1]  # BGR -> RGB
        im = BASE_PREPROCESS(image=im)["image"]  # float32 HWC in [0,1]
        if dtype == np.float16:
            im = im.astype(np.float16, copy=False)
        else:
            im = im.astype(np.float32, copy=False)
        return i, im

    cpu_cnt = os.cpu_count() or 1
    workers = int(os.environ.get("ALASKA2_CACHE_WORKERS", str(min(8, cpu_cnt))))
    chunksize = int(os.environ.get("ALASKA2_CACHE_CHUNKSIZE", "32"))

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        it = pool.imap_unordered(_process_one, enumerate(fns), chunksize=chunksize)
        for i, im in tqdm(it, total=n, desc=f"caching {cache_prefix}", mininterval=2.0):
            arr[i] = im
    arr.flush()

    np.savez_compressed(
        meta_path,
        shape=np.array(shape, dtype=np.int64),
        dtype=np.array(str(np.dtype(dtype))),
    )
    print(f"Cached {cache_prefix}: {n} images -> {mmap_path} in {time.time()-t0:.1f}s")

    arr = np.memmap(mmap_path, mode="r", dtype=dtype, shape=shape)
    return arr, key




## === cell 4
class Alaska2Dataset(Dataset):
    def __init__(self, df, base_cache, augmentations=None):
        df = df.reset_index(drop=True)
        self.labels = df["Label"].to_numpy(dtype=np.int64, copy=False)
        self.base_cache = base_cache  # memmap/ndarray [N,H,W,3] float16/float32
        self.augment = augmentations

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        im = self.base_cache[idx]  # HWC RGB in [0,1]
        im = np.ascontiguousarray(im)
        if self.augment:
            im = self.augment(image=im)
        else:
            im = {"image": im}
        return im, label




## === cell 5
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b0(weights=weights)
        self.model.classifier = nn.Identity()  # produce 1280-d embedding
        self.dense_output = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.model(x)  # [B, 1280]
        return self.dense_output(feat)




## === cell 6
train_base, _ = build_or_load_base_cache(
    train_df["ImageFileName"].tolist(), "train", dtype=np.float16
)
val_base, _ = build_or_load_base_cache(
    val_df["ImageFileName"].tolist(), "val", dtype=np.float16
)

batch_size = int(os.environ.get("ALASKA2_BATCH_SIZE", "32"))

cpu_cnt = os.cpu_count() or 1
num_workers = int(os.environ.get("ALASKA2_NUM_WORKERS", str(min(8, cpu_cnt))))

train_dataset = Alaska2Dataset(
    train_df, base_cache=train_base, augmentations=TRAIN_STOCHASTIC
)
valid_dataset = Alaska2Dataset(val_df, base_cache=val_base, augmentations=TEST_POST)


def _seed_worker(worker_id):
    worker_seed = (seed + worker_id) % (2**32 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

_loader_kwargs = dict(
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=(4 if num_workers > 0 else None),
)


def _collate_albu(batch):
    ims = torch.stack([b[0]["image"] for b in batch], dim=0)
    labels = torch.as_tensor([b[1] for b in batch], dtype=torch.int64)
    return {"image": ims}, labels


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
    worker_init_fn=_seed_worker,
    generator=g,
    collate_fn=_collate_albu,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size * 2,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    worker_init_fn=_seed_worker,
    generator=g,
    collate_fn=_collate_albu,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

model = Net().to(device)

if device == "cuda":
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile not enabled:", repr(e))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3897117052.py in <cell line: 0>()
      1 # Speed: store base cache as float16 on disk to halve I/O; convert to float32 on GPU as before.
----> 2 train_base, _ = build_or_load_base_cache(
      3     train_df["ImageFileName"].tolist(), "train", dtype=np.float16
      4 )
      5 val_base, _ = build_or_load_base_cache(

/tmp/ipykernel_55/656288874.py in build_or_load_base_cache(fns, cache_prefix, dtype)
     89         it = pool.imap_unordered(_process_one, enumerate(fns), chunksize=chunksize)
     90         # Light tqdm to avoid overhead
---> 91         for i, im in tqdm(it, total=n, desc=f"caching {cache_prefix}", mininterval=2.0):
     92             arr[i] = im
     93     arr.flush()

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/lib/python3.11/multiprocessing/pool.py in <genexpr>(.0)
    449                     result._set_length
    450                 ))
--> 451             return (item for chunk in result for item in chunk)
    452 
    453     def apply_async(self, func, args=(), kwds={}, callback=None,

/usr/lib/python3.11/multiprocessing/pool.py in next(self, timeout)
    871         if success:
    872             return value
--> 873         raise value
    874 
    875     __next__ = next                    # XXX

/usr/lib/python3.11/multiprocessing/pool.py in _handle_tasks(taskqueue, put, outqueue, pool, cache)
    538                         break
    539                     try:
--> 540                         put(task)
    541                     except Exception as e:
    542                         job, idx = task[:2]

/usr/lib/python3.11/multiprocessing/connection.py in send(self, obj)
    204         self._check_closed()
    205         self._check_writable()
--> 206         self._send_bytes(_ForkingPickler.dumps(obj))
    207 
    208     def recv_bytes(self, maxlength=None):

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object 'build_or_load_base_cache.<locals>._process_one'

## === cell 7
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = np.array([0.0, 0.4, 1.0], dtype=np.float64)
    weights = np.array([2.0, 1.0], dtype=np.float64)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_valid, pos_label=1)

    areas = tpr_thresholds[1:] - tpr_thresholds[:-1]
    normalization = float(np.dot(areas, weights))

    competition_metric = 0.0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if not np.any(mask):
            continue

        fpr_m = fpr[mask]
        tpr_m = tpr[mask]

        x_padding = np.linspace(fpr_m[-1], 1.0, 100, dtype=fpr.dtype)
        x = np.concatenate([fpr_m, x_padding])
        y = np.concatenate([tpr_m, np.full_like(x_padding, y_max)])
        y = y - y_min

        competition_metric += float(metrics.auc(x, y)) * float(weight)

    return competition_metric / normalization




## === cell 8
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 2

train_loss, val_loss = [], []

if device == "cuda":
    model = model.to(memory_format=torch.channels_last)

tqdm_mininterval = float(os.environ.get("ALASKA2_TQDM_MININTERVAL", "5.0"))

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-" * 10)
    model.train()
    running_loss = 0.0

    tk0 = tqdm(train_loader, total=int(len(train_loader)), mininterval=tqdm_mininterval)
    for im, labels in tk0:
        inputs = im["image"].to(device, dtype=torch.float32, non_blocking=True)
        if device == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        running_loss += loss_val
        tk0.set_postfix(loss=loss_val)

    epoch_loss = running_loss / max(1, len(train_loader))
    train_loss.append(epoch_loss)
    print("Training Loss: {:.8f}".format(epoch_loss))

    tk1 = tqdm(valid_loader, total=int(len(valid_loader)), mininterval=tqdm_mininterval)
    model.eval()
    running_loss = 0.0

    y_batches, preds_batches = [], []
    with torch.inference_mode():
        for im, labels in tk1:
            inputs = im["image"].to(device, dtype=torch.float32, non_blocking=True)
            if device == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            y_batches.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
            preds_batches.append(F.softmax(outputs, 1).detach().cpu().numpy())

            loss_val = float(loss.item())
            running_loss += loss_val
            tk1.set_postfix(loss=loss_val)

        epoch_loss = running_loss / max(1, len(valid_loader))
        val_loss.append(epoch_loss)

        y = np.concatenate(y_batches, axis=0)
        preds = np.concatenate(preds_batches, axis=0)

        labels_pred = preds.argmax(1)
        acc = (labels_pred == y).mean() * 100.0

        stego_prob = 1.0 - preds[:, 0]

        y_bin = y.copy()
        y_bin[y_bin != 0] = 1
        auc_score = alaska_weighted_auc(y_bin, stego_prob)
        print(f"Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}")

    torch.save(
        model.state_dict(),
        f"epoch_{epoch}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}.pth",
    )



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4141758414.py in <cell line: 0>()
      6 
      7 if device == "cuda":
----> 8     model = model.to(memory_format=torch.channels_last)
      9 
     10 # Speed: reduce tqdm overhead by updating less frequently.

NameError: name 'model' is not defined

## === cell 9
if len(train_loss) > 0 or len(val_loss) > 0:
    plt.figure(figsize=(15, 7))
    plt.plot(train_loss, c="r")
    plt.plot(val_loss, c="b")
    plt.legend(["train_loss", "val_loss"])
    plt.title("Loss Plot")
    plt.show()




## === cell 10
class Alaska2TestDataset(Dataset):
    def __init__(self, df, base_cache, augmentations=None):
        df = df.reset_index(drop=True)
        self.base_cache = base_cache
        self.augment = augmentations

    def __len__(self):
        return self.base_cache.shape[0]

    def __getitem__(self, idx):
        im = self.base_cache[idx]
        im = np.ascontiguousarray(im)
        if self.augment:
            im = self.augment(image=im)
        else:
            im = {"image": im}
        return im


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame(
    {"ImageFileName": list(test_filenames)}, columns=["ImageFileName"]
)

test_base, _ = build_or_load_base_cache(
    test_df["ImageFileName"].tolist(), "test", dtype=np.float16
)

batch_size = int(os.environ.get("ALASKA2_TEST_BATCH_SIZE", "64"))
cpu_cnt = os.cpu_count() or 1
num_workers = int(os.environ.get("ALASKA2_TEST_NUM_WORKERS", str(min(4, cpu_cnt))))
test_dataset = Alaska2TestDataset(
    test_df, base_cache=test_base, augmentations=TEST_POST
)


def _collate_albu_test(batch):
    ims = torch.stack([b["image"] for b in batch], dim=0)
    return {"image": ims}


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker,
    generator=g,
    collate_fn=_collate_albu_test,
)

print("test images:", len(test_df))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1696892839.py in <cell line: 0>()
     24 
     25 # Speed: cache test as float16 on disk too.
---> 26 test_base, _ = build_or_load_base_cache(
     27     test_df["ImageFileName"].tolist(), "test", dtype=np.float16
     28 )

/tmp/ipykernel_55/656288874.py in build_or_load_base_cache(fns, cache_prefix, dtype)
     89         it = pool.imap_unordered(_process_one, enumerate(fns), chunksize=chunksize)
     90         # Light tqdm to avoid overhead
---> 91         for i, im in tqdm(it, total=n, desc=f"caching {cache_prefix}", mininterval=2.0):
     92             arr[i] = im
     93     arr.flush()

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/lib/python3.11/multiprocessing/pool.py in <genexpr>(.0)
    449                     result._set_length
    450                 ))
--> 451             return (item for chunk in result for item in chunk)
    452 
    453     def apply_async(self, func, args=(), kwds={}, callback=None,

/usr/lib/python3.11/multiprocessing/pool.py in next(self, timeout)
    871         if success:
    872             return value
--> 873         raise value
    874 
    875     __next__ = next                    # XXX

/usr/lib/python3.11/multiprocessing/pool.py in _handle_tasks(taskqueue, put, outqueue, pool, cache)
    538                         break
    539                     try:
--> 540                         put(task)
    541                     except Exception as e:
    542                         job, idx = task[:2]

/usr/lib/python3.11/multiprocessing/connection.py in send(self, obj)
    204         self._check_closed()
    205         self._check_writable()
--> 206         self._send_bytes(_ForkingPickler.dumps(obj))
    207 
    208     def recv_bytes(self, maxlength=None):

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object 'build_or_load_base_cache.<locals>._process_one'

## === cell 11
model.eval()

preds = []
tk0 = tqdm(
    test_loader,
    total=len(test_loader),
    mininterval=float(os.environ.get("ALASKA2_TQDM_MININTERVAL", "5.0")),
)
with torch.inference_mode():
    for im in tk0:
        inputs = im["image"].to(device, dtype=torch.float32, non_blocking=True)
        if device == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        x0 = inputs
        x1 = inputs.flip(2)
        x2 = inputs.flip(3)
        x_cat = torch.cat([x0, x1, x2], dim=0)
        out_cat = model(x_cat)
        b = inputs.shape[0]
        out0, out1, out2 = out_cat[:b], out_cat[b : 2 * b], out_cat[2 * b : 3 * b]
        outputs = (0.5 * out0) + (0.25 * out1) + (0.25 * out2)

        preds.append(F.softmax(outputs, 1).cpu().numpy())

preds = np.concatenate(preds, axis=0)

stego_prob = 1.0 - preds[:, 0]

test_df["Id"] = test_df["ImageFileName"].map(os.path.basename)
test_df["Label"] = stego_prob.astype(np.float32)
sub_df = test_df.drop("ImageFileName", axis=1)

sample_path = os.path.join(data_dir, "sample_submission.csv")
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    sub_df = sample_sub[["Id"]].merge(sub_df, on="Id", how="left")
    sub_df["Label"] = sub_df["Label"].fillna(
        float(np.nanmedian(sub_df["Label"].values))
    )

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1444343141.py in <cell line: 0>()
----> 1 model.eval()
      2 
      3 preds = []
      4 tk0 = tqdm(
      5     test_loader,

NameError: name 'model' is not defined
