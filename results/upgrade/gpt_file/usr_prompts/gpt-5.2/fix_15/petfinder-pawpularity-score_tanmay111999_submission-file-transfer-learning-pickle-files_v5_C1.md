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
import warnings

warnings.filterwarnings("ignore")

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.pop(_var, None)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "../input/petfinder-pawpularity-score"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

SUBMISSION_PATH = "submission.csv"

RUN_VALIDATION_SANITY_CHECK = os.environ.get("RUN_VALIDATION_SANITY_CHECK", "1") == "1"

print("Using data dir:", DATA_DIR)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Test CSV exists:", os.path.exists(TEST_CSV))
print("Train image dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)
except Exception:
    pass

os.environ.setdefault("JOBLIB_TEMP_FOLDER", "./_joblib_tmp")
os.makedirs(os.environ["JOBLIB_TEMP_FOLDER"], exist_ok=True)



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

assert "Id" in train_df.columns and "Pawpularity" in train_df.columns
assert "Id" in test_df.columns

print("train_df:", train_df.shape, "test_df:", test_df.shape)




## === cell 2
def ids_to_paths(ids, img_dir):
    ids = np.asarray(ids, dtype=str)
    return np.char.add(np.char.add(img_dir + os.sep, ids), ".jpg").tolist()


train_paths = ids_to_paths(train_df["Id"].values, TRAIN_IMG_DIR)
test_paths = ids_to_paths(test_df["Id"].values, TEST_IMG_DIR)

missing_train = 0
for p in train_paths[:200]:
    if not os.path.exists(p):
        missing_train += 1
missing_test = 0
for p in test_paths[:200]:
    if not os.path.exists(p):
        missing_test += 1

print("Missing (sample of first 200) train images:", missing_train)
print("Missing (sample of first 200) test images:", missing_test)



## === cell 3
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib

IMG_SIZE = (64, 64)  # keep as in provided code (do not change core pipeline parameters)

CACHE_DIR = "./_cache_petfinder"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, f"train_img_{IMG_SIZE[0]}x{IMG_SIZE[1]}.npy")
TEST_CACHE = os.path.join(CACHE_DIR, f"test_img_{IMG_SIZE[0]}x{IMG_SIZE[1]}.npy")
TRAIN_CACHE_META = os.path.join(
    CACHE_DIR, f"train_img_{IMG_SIZE[0]}x{IMG_SIZE[1]}.json"
)
TEST_CACHE_META = os.path.join(CACHE_DIR, f"test_img_{IMG_SIZE[0]}x{IMG_SIZE[1]}.json")


def _fast_signature_from_ids(ids, img_size):
    h = hashlib.sha1()
    h.update(str(img_size).encode("utf-8"))
    h.update((",".join(np.asarray(ids, dtype=str))).encode("utf-8"))
    return h.hexdigest()


def _read_resize_flatten_into(path, out_row, img_w, img_h):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        out_row.fill(0.0)
        return 1
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
    img = img[:, :, ::-1]  # BGR -> RGB
    out_row[:] = img.reshape(-1).astype(np.float32, copy=False) / 255.0
    return 0


def load_and_preprocess(
    paths,
    ids_for_sig,
    img_size=IMG_SIZE,
    cache_path=None,
    cache_meta_path=None,
    max_workers=None,
    chunksize=2048,
):
    """
    Runtime improvements (correctness-preserving):
      - Cache validation is O(1) and uses np.load(mmap_mode="r") for zero-copy reads.
      - Preallocate output array and have workers write directly into rows.
      - Use ThreadPoolExecutor with larger chunksize to reduce scheduling overhead.
    """
    img_w, img_h = img_size
    n = len(paths)
    d = img_h * img_w * 3

    sig = _fast_signature_from_ids(ids_for_sig, img_size)
    if (
        cache_path
        and cache_meta_path
        and os.path.exists(cache_path)
        and os.path.exists(cache_meta_path)
    ):
        try:
            with open(cache_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if (
                meta.get("signature") == sig
                and meta.get("shape") == [n, d]
                and meta.get("dtype") == "float32"
            ):
                arr = np.load(cache_path, mmap_mode="r")
                if arr.shape == (n, d) and arr.dtype == np.float32:
                    return arr, 0
        except Exception:
            pass

    out = np.empty((n, d), dtype=np.float32)
    bad = 0

    if max_workers is None:
        max_workers = max(2, min(8, (os.cpu_count() or 4)))

    def _worker(i):
        return _read_resize_flatten_into(paths[i], out[i], img_w, img_h)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for is_bad in ex.map(_worker, range(n), chunksize=chunksize):
            bad += int(is_bad)

    if cache_path and cache_meta_path:
        np.save(cache_path, out)
        try:
            with open(cache_meta_path, "w", encoding="utf-8") as f:
                json.dump({"signature": sig, "shape": [n, d], "dtype": "float32"}, f)
        except Exception:
            pass

    return out, bad


X_train_img, bad_train = load_and_preprocess(
    train_paths,
    ids_for_sig=train_df["Id"].values,
    cache_path=TRAIN_CACHE,
    cache_meta_path=TRAIN_CACHE_META,
)
X_test_img, bad_test = load_and_preprocess(
    test_paths,
    ids_for_sig=test_df["Id"].values,
    cache_path=TEST_CACHE,
    cache_meta_path=TEST_CACHE_META,
)

y = train_df["Pawpularity"].values.astype(np.float32)

print("X_train_img:", (X_train_img.shape, X_train_img.dtype), "bad_train:", bad_train)
print("X_test_img:", (X_test_img.shape, X_test_img.dtype), "bad_test:", bad_test)



## === cell 4
meta_cols = [c for c in train_df.columns if c not in ("Id", "Pawpularity")]
test_meta_cols = [c for c in test_df.columns if c != "Id"]

assert set(meta_cols) == set(test_meta_cols), "Train/test metadata columns mismatch."
meta_cols = sorted(meta_cols)

X_train_meta = train_df[meta_cols].to_numpy(dtype=np.float32, copy=False)
X_test_meta = test_df[meta_cols].to_numpy(dtype=np.float32, copy=False)

n_train = X_train_meta.shape[0]
n_test = X_test_meta.shape[0]
d_img = X_train_img.shape[1]
d_meta = X_train_meta.shape[1]
d_total = d_img + d_meta

X_train = np.empty((n_train, d_total), dtype=np.float32)
X_test = np.empty((n_test, d_total), dtype=np.float32)

X_train[:, :d_img] = X_train_img
X_train[:, d_img:] = X_train_meta
X_test[:, :d_img] = X_test_img
X_test[:, d_img:] = X_test_meta

del X_train_img, X_test_img, X_train_meta, X_test_meta

if not X_train.flags["C_CONTIGUOUS"]:
    X_train = np.ascontiguousarray(X_train)
if not X_test.flags["C_CONTIGUOUS"]:
    X_test = np.ascontiguousarray(X_test)

print("Final feature shapes:", X_train.shape, X_test.shape)



## === cell 5
idx_all = np.arange(X_train.shape[0], dtype=np.int32)
idx_tr, idx_val = train_test_split(idx_all, test_size=0.1, random_state=RANDOM_STATE)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    min_samples_leaf=2,
    bootstrap=False,
)

X_tr = X_train[idx_tr]
y_tr = y[idx_tr]
model.fit(X_tr, y_tr)

if RUN_VALIDATION_SANITY_CHECK:
    X_va = X_train[idx_val]
    y_va = y[idx_val]
    val_pred = model.predict(X_va)
    rmse = float(np.sqrt(np.mean((val_pred - y_va) ** 2)))
    print("Validation RMSE (sanity check):", rmse)
else:
    print(
        "Validation sanity check skipped (set RUN_VALIDATION_SANITY_CHECK=1 to enable)."
    )



## === cell 6
test_pred = model.predict(X_test).astype(np.float32)
test_pred = np.clip(test_pred, 0.0, 100.0)

print(
    "Pred stats:",
    float(test_pred.min()),
    float(test_pred.mean()),
    float(test_pred.max()),
)



## === cell 7
submission = pd.DataFrame({"Id": test_df["Id"].values, "Pawpularity": test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)

print("Wrote:", SUBMISSION_PATH)
print(submission.head())



## === cell 8
assert os.path.exists(SUBMISSION_PATH)
check = pd.read_csv(SUBMISSION_PATH)
assert list(check.columns) == ["Id", "Pawpularity"]
assert len(check) == len(test_df)
assert check["Pawpularity"].notnull().all()
print("Submission file validated with shape:", check.shape)
