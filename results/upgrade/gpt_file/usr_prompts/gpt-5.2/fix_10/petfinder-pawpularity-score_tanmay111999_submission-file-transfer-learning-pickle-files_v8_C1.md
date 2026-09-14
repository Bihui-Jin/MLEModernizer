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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

43.0418671430233

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 1
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.ensemble import RandomForestRegressor

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    cpu_cnt = os.cpu_count() or 2
    torch.set_num_threads(max(1, min(cpu_cnt, 8)))
    torch.set_num_interop_threads(max(1, min(cpu_cnt // 2, 4)))
except Exception:
    pass

_CANDIDATE_DIRS = [
    "../input/petfinder-pawpularity-score",
    "/kaggle/input/petfinder-pawpularity-score",
    "/kaggle/data/petfinder-pawpularity-score",
]
DATA_DIR = None
for d in _CANDIDATE_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    DATA_DIR = "../input/petfinder-pawpularity-score"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_submission = pd.read_csv(SAMPLE_SUB)

print(
    "DATA_DIR:",
    DATA_DIR,
    "\ntrain_df:",
    train_df.shape,
    "test_df:",
    test_df.shape,
    "sample_submission:",
    sample_submission.shape,
)
print(
    "train img dir exists:",
    os.path.isdir(TRAIN_IMG_DIR),
    "test img dir exists:",
    os.path.isdir(TEST_IMG_DIR),
)



## === cell 2
assert list(sample_submission.columns) == [
    "Id",
    "Pawpularity",
], "Unexpected submission columns."

train_ids = train_df["Id"].astype(str).tolist()
test_ids = test_df["Id"].astype(str).tolist()

print("Loaded IDs:", len(train_ids), "train and", len(test_ids), "test")



## === cell 3
IMG_SIZE = 128

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGENET_MEAN_ = IMAGENET_MEAN.reshape(1, 1, 3)
IMAGENET_STD_ = IMAGENET_STD.reshape(1, 1, 3)


class PetImageDataset(Dataset):
    def __init__(self, ids, img_dir):
        self.ids = [str(i) for i in ids]
        self.img_dir = img_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        path = os.path.join(self.img_dir, f"{pid}.jpg")

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32) * (1.0 / 255.0)
        img = (img - IMAGENET_MEAN_) / IMAGENET_STD_
        img = np.transpose(img, (2, 0, 1)).copy()  # CHW contiguous for torch.from_numpy
        return torch.from_numpy(img), pid




## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(
    vgg.features, vgg.avgpool
)  # (N, 512, 1, 1) for 128x128
feature_extractor.eval()
for p in feature_extractor.parameters():
    p.requires_grad_(False)
feature_extractor = feature_extractor.to(device)

if device.type == "cuda":
    feature_extractor = feature_extractor.to(memory_format=torch.channels_last)

CACHE_DIR = "./_fe_cache_vgg16_128"
os.makedirs(CACHE_DIR, exist_ok=True)


def _dir_latest_mtime(img_dir, ids, limit=256):
    n = len(ids)
    if n == 0:
        return 0
    sample_idxs = list(range(min(n, limit)))
    if n > limit:
        extra = [n // 4, n // 2, (3 * n) // 4, n - 1]
        for i in extra:
            if 0 <= i < n:
                sample_idxs.append(i)
    latest = 0
    for i in sample_idxs:
        p = os.path.join(img_dir, f"{str(ids[i])}.jpg")
        try:
            mt = int(os.path.getmtime(p))
        except OSError:
            mt = 0
        if mt > latest:
            latest = mt
    return latest


def _cache_fingerprint(ids, img_dir):
    latest_mtime = _dir_latest_mtime(img_dir, ids)
    return f"{os.path.basename(img_dir)}_n{len(ids)}_img{IMG_SIZE}_vgg16avgpool_m{latest_mtime}"


def _try_load_cache(ids, img_dir):
    fp = _cache_fingerprint(ids, img_dir)
    cache_path = os.path.join(CACHE_DIR, f"{fp}.npy")
    cache_ids_path = os.path.join(CACHE_DIR, f"{fp}_ids.npy")
    if not (os.path.exists(cache_path) and os.path.exists(cache_ids_path)):
        return None
    feats = np.load(cache_path, mmap_mode="r")
    out_ids = np.load(cache_ids_path, allow_pickle=True).tolist()
    if out_ids != [str(i) for i in ids]:
        return None
    return feats, out_ids


@torch.inference_mode()
def extract_features(ids, img_dir, batch_size=128, num_workers=4):
    cached = _try_load_cache(ids, img_dir)
    if cached is not None:
        return cached

    ds = PetImageDataset(ids, img_dir)

    def _seed_worker(worker_id):
        seed = SEED + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        drop_last=False,
    )

    fp = _cache_fingerprint(ids, img_dir)
    cache_path = os.path.join(CACHE_DIR, f"{fp}.npy")
    cache_ids_path = os.path.join(CACHE_DIR, f"{fp}_ids.npy")

    feats_arr = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=(len(ds), 512)
    )
    out_ids = [None] * len(ds)

    write_pos = 0
    for xb, pid in dl:
        if device.type == "cuda":
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            xb = xb.to(device)

        fb = feature_extractor(xb)  # (N, 512, 1, 1)
        fb = fb.reshape(fb.shape[0], -1).contiguous().cpu().numpy()  # float32 already
        bsz = fb.shape[0]
        feats_arr[write_pos : write_pos + bsz] = fb
        out_ids[write_pos : write_pos + bsz] = pid
        write_pos += bsz

    np.save(cache_ids_path, np.array(out_ids, dtype=object))
    feats_ro = np.load(cache_path, mmap_mode="r")
    return feats_ro, out_ids




## === cell 5
cpu_cnt = os.cpu_count() or 2

if torch.cuda.is_available():
    num_workers = min(6, max(2, cpu_cnt // 2))
    batch_size = 256
else:
    num_workers = min(4, max(2, cpu_cnt // 2))
    batch_size = 128

train_features, train_ids_out = extract_features(
    train_ids, TRAIN_IMG_DIR, batch_size=batch_size, num_workers=num_workers
)
test_features, test_ids_out = extract_features(
    test_ids, TEST_IMG_DIR, batch_size=batch_size, num_workers=num_workers
)

assert train_ids_out == train_ids, "Train feature extraction order mismatch."
assert test_ids_out == test_ids, "Test feature extraction order mismatch."

print(
    "train_features:",
    np.asarray(train_features).shape,
    "test_features:",
    np.asarray(test_features).shape,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1017831581.py in <cell line: 0>()
      8     batch_size = 128
      9 
---> 10 train_features, train_ids_out = extract_features(
     11     train_ids, TRAIN_IMG_DIR, batch_size=batch_size, num_workers=num_workers
     12 )

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_11/3302051855.py in extract_features(ids, img_dir, batch_size, num_workers)
    111         fb = fb.reshape(fb.shape[0], -1).contiguous().cpu().numpy()  # float32 already
    112         bsz = fb.shape[0]
--> 113         feats_arr[write_pos : write_pos + bsz] = fb
    114         out_ids[write_pos : write_pos + bsz] = pid
    115         write_pos += bsz

ValueError: could not broadcast input array from shape (128,25088) into shape (128,512)

## === cell 6
y_train = train_df["Pawpularity"].astype(float).values

train_features = np.ascontiguousarray(np.asarray(train_features, dtype=np.float32))
test_features = np.ascontiguousarray(np.asarray(test_features, dtype=np.float32))

rfr = RandomForestRegressor(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1,
    max_features=1.0,
    verbose=0,
)

rfr.fit(train_features, y_train)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3775070759.py in <cell line: 0>()
      3 # CHANGE (timeout): ensure C-contiguous float32 arrays (no hidden copies inside sklearn),
      4 # preserving numeric values and model semantics.
----> 5 train_features = np.ascontiguousarray(np.asarray(train_features, dtype=np.float32))
      6 test_features = np.ascontiguousarray(np.asarray(test_features, dtype=np.float32))
      7 

NameError: name 'train_features' is not defined

## === cell 7
pred = rfr.predict(test_features)
pred = np.clip(pred, 0.0, 100.0)

submission = pd.DataFrame({"Id": test_df["Id"].astype(str).values, "Pawpularity": pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1052687382.py in <cell line: 0>()
----> 1 pred = rfr.predict(test_features)
      2 pred = np.clip(pred, 0.0, 100.0)
      3 
      4 submission = pd.DataFrame({"Id": test_df["Id"].astype(str).values, "Pawpularity": pred})
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'rfr' is not defined
