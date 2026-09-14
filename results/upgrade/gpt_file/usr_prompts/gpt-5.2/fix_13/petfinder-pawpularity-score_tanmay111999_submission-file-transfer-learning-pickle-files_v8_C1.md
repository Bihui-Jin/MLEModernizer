# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
    vgg.features,
    nn.AdaptiveAvgPool2d((1, 1)),
)
feature_extractor.eval()
for p in feature_extractor.parameters():
    p.requires_grad_(False)
feature_extractor = feature_extractor.to(device)

if device.type == "cuda":
    feature_extractor = feature_extractor.to(memory_format=torch.channels_last)

CACHE_DIR = "./_fe_cache_vgg16_128"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_fingerprint(ids, img_dir):
    return f"{os.path.basename(img_dir)}_n{len(ids)}_img{IMG_SIZE}_vgg16avgpool1x1"


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
def extract_features(ids, img_dir, batch_size=256, num_workers=4):
    cached = _try_load_cache(ids, img_dir)
    if cached is not None:
        return cached

    ds = PetImageDataset(ids, img_dir)

    def _seed_worker(worker_id):
        seed = SEED + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    gen = torch.Generator()
    gen.manual_seed(SEED)

    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=gen,
    )
    if num_workers > 0:
        dl_kwargs.update(
            dict(
                persistent_workers=True,
                prefetch_factor=2,
                worker_init_fn=_seed_worker,
            )
        )

    dl = DataLoader(ds, **dl_kwargs)

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
        fb = fb.reshape(fb.shape[0], -1).contiguous().cpu().numpy()  # (N, 512)
        bsz = fb.shape[0]
        feats_arr[write_pos : write_pos + bsz] = fb
        out_ids[write_pos : write_pos + bsz] = list(pid)
        write_pos += bsz

    np.save(cache_ids_path, np.array(out_ids, dtype=object))
    feats_ro = np.load(cache_path, mmap_mode="r")
    return feats_ro, out_ids




## === cell 5
cpu_cnt = os.cpu_count() or 2

if torch.cuda.is_available():
    num_workers = min(4, max(2, cpu_cnt // 2))
    batch_size = 512
else:
    num_workers = min(4, max(2, cpu_cnt // 2))
    batch_size = 256

train_features, train_ids_out = extract_features(
    train_ids, TRAIN_IMG_DIR, batch_size=batch_size, num_workers=num_workers
)
test_features, test_ids_out = extract_features(
    test_ids, TEST_IMG_DIR, batch_size=batch_size, num_workers=num_workers
)

assert train_ids_out == train_ids, "Train feature extraction order mismatch."
assert test_ids_out == test_ids, "Test feature extraction order mismatch."

print("train_features:", train_features.shape, "test_features:", test_features.shape)



## === cell 6
y_train = train_df["Pawpularity"].astype(float).values

train_features = np.ascontiguousarray(np.asarray(train_features), dtype=np.float32)
test_features = np.ascontiguousarray(np.asarray(test_features), dtype=np.float32)

import hashlib
import pickle

MODEL_CACHE_DIR = "./_model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

rf_params = dict(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1,
    max_features=1.0,
    verbose=0,
)

h = hashlib.md5()
h.update(str(rf_params).encode("utf-8"))
h.update(str(train_features.shape).encode("utf-8"))
h.update(train_df["Id"].astype(str).iloc[0].encode("utf-8"))
h.update(train_df["Id"].astype(str).iloc[-1].encode("utf-8"))
h.update(str(float(y_train[0])).encode("utf-8"))
h.update(str(float(y_train[-1])).encode("utf-8"))
model_path = os.path.join(MODEL_CACHE_DIR, f"rf_{h.hexdigest()}.pkl")

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        rfr = pickle.load(f)
else:
    rfr = RandomForestRegressor(**rf_params)
    rfr.fit(train_features, y_train)
    with open(model_path, "wb") as f:
        pickle.dump(rfr, f, protocol=pickle.HIGHEST_PROTOCOL)



## === cell 7
pred = rfr.predict(test_features)
pred = np.clip(pred, 0.0, 100.0)

submission = pd.DataFrame({"Id": test_df["Id"].astype(str).values, "Pawpularity": pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
