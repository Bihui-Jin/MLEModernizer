# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import pandas as pd

DATA_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str):
    for base in DATA_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in DATA_CANDIDATES:
        path = os.path.join(base, "hotel-id-2021-fgvc8", filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {DATA_CANDIDATES}"
    )


train_path = find_file("train.csv")
sub_path = find_file("sample_submission.csv")

print("Using train:", train_path)
print("Using sample_submission:", sub_path)

train = pd.read_csv(train_path)
sample_sub = pd.read_csv(sub_path)

train.head(), sample_sub.head(), train.shape, sample_sub.shape


## === cell 1
import os

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
global_pred_str = " ".join(global_top5)
print("Global top-5 fallback:", global_pred_str)

_chain_hotel_counts = (
    train.groupby(["chain", "hotel_id"], sort=False).size().rename("cnt").reset_index()
)
_chain_hotel_counts = _chain_hotel_counts.sort_values(
    ["chain", "cnt"], ascending=[True, False], kind="mergesort"
)
_chain_top = _chain_hotel_counts.groupby("chain", sort=False).head(5)
chain_top5 = (
    _chain_top.groupby("chain", sort=False)["hotel_id"]
    .apply(lambda s: s.astype(str).tolist())
    .to_dict()
)

TRAIN_IMAGES_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8/train_images",
    "/kaggle/data/hotel-id-2021-fgvc8/train_images",
    "/kaggle/input/train_images",
    "/kaggle/data/train_images",
    "/kaggle/input/hotel-id-2021-fgvc8/train_images/train_images",
    "/kaggle/data/hotel-id-2021-fgvc8/train_images/train_images",
]
TEST_IMAGES_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/data/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
    "/kaggle/input/hotel-id-2021-fgvc8/test_images/test_images",
    "/kaggle/data/hotel-id-2021-fgvcvc8/test_images/test_images",
    "/kaggle/data/hotel-id-2021-fgvc8/test_images/test_images",
    "/kaggle/input/test_images/test_images",
    "/kaggle/data/test_images/test_images",
]

train_images_root = None
for p in TRAIN_IMAGES_CANDIDATES:
    if os.path.isdir(p):
        train_images_root = p
        break

test_images_root = None
for p in TEST_IMAGES_CANDIDATES:
    if os.path.isdir(p):
        test_images_root = p
        break

print("Using train_images root:", train_images_root)
print("Using test_images root:", test_images_root)

from PIL import Image
import numpy as np


def _ahash_64(image_path: str):
    """64-bit average hash (8x8) -> Python int; returns None if unreadable."""
    try:
        with Image.open(image_path) as im:
            im = im.convert("L").resize((8, 8), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32)
        mean = arr.mean()
        bits = (arr > mean).astype(np.uint8).reshape(-1)
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return int(h)
    except Exception:
        return None


def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


HASH_TRAIN_CAP = 12000  # minimal but useful; adjust cautiously for runtime
_hash_to_chain_counts = {}
_train_image_path_by_name = {}
_hash_ready = False

if train_images_root is not None and test_images_root is not None:
    try:
        for chain_dir in os.listdir(train_images_root):
            chain_path = os.path.join(train_images_root, chain_dir)
            if not os.path.isdir(chain_path):
                continue
            try:
                int(chain_dir)
            except Exception:
                continue
            try:
                for fn in os.listdir(chain_path):
                    if fn not in _train_image_path_by_name:
                        _train_image_path_by_name[fn] = os.path.join(chain_path, fn)
            except Exception:
                continue

        train2 = train[["image", "chain"]].copy()
        train2["image"] = train2["image"].astype(str)
        train2["chain"] = train2["chain"].astype(int)
        train2["path"] = train2["image"].map(_train_image_path_by_name.get)
        train2 = train2[train2["path"].notna()].reset_index(drop=True)

        if len(train2) > HASH_TRAIN_CAP:
            train2 = train2.iloc[:HASH_TRAIN_CAP].copy()

        n_hashed = 0
        for _, row in train2.iterrows():
            h = _ahash_64(row["path"])
            if h is None:
                continue
            ch = int(row["chain"])
            d = _hash_to_chain_counts.get(h)
            if d is None:
                d = {}
                _hash_to_chain_counts[h] = d
            d[ch] = d.get(ch, 0) + 1
            n_hashed += 1

        _hash_ready = n_hashed > 0
        print(
            f"Built train hash index: hashed {n_hashed} train images (cap={HASH_TRAIN_CAP})."
        )
    except Exception as e:
        print(
            "WARNING: hash index build failed; falling back to global-only predictions. Error:",
            repr(e),
        )
        _hash_ready = False
else:
    print(
        "WARNING: train_images/test_images roots not found; falling back to global-only predictions."
    )


def infer_chain_for_test_image(image_name: str):
    """Infer chain via hash NN over indexed train hashes; returns int chain or None."""
    if not _hash_ready or test_images_root is None:
        return None
    test_path = os.path.join(test_images_root, image_name)
    if not os.path.exists(test_path):
        return None
    hq = _ahash_64(test_path)
    if hq is None:
        return None

    if hq in _hash_to_chain_counts:
        counts = _hash_to_chain_counts[hq]
        return max(counts.items(), key=lambda kv: kv[1])[0]

    best_h = None
    best_d = 999
    for hk in _hash_to_chain_counts.keys():
        d = _hamming64(hq, hk)
        if d < best_d:
            best_d = d
            best_h = hk
            if best_d <= 6:  # early accept; does not change semantics, just speeds up
                break

    if best_h is None:
        return None
    if best_d > 10:
        return None
    counts = _hash_to_chain_counts[best_h]
    return max(counts.items(), key=lambda kv: kv[1])[0]


preds = []
n_chain_found = 0
for img in sample_sub["image"].astype(str).tolist():
    ch = infer_chain_for_test_image(img)
    if ch is not None:
        top = chain_top5.get(int(ch))
        if top is not None and len(top) > 0:
            n_chain_found += 1
            combined = []
            for x in list(map(str, top)) + list(map(str, global_top5)):
                if x not in combined:
                    combined.append(x)
                if len(combined) == 5:
                    break
            if len(combined) < 5:
                combined = (combined + global_top5)[:5]
            preds.append(" ".join(combined))
            continue
    preds.append(global_pred_str)

print(
    f"Inferred chain for {n_chain_found}/{len(sample_sub)} test images (others used global fallback)."
)

submission = sample_sub.copy()
submission["hotel_id"] = preds

assert list(submission.columns) == ["image", "hotel_id"]
assert len(submission) == len(sample_sub)
assert submission["hotel_id"].astype(str).str.split().map(len).eq(5).all()

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())


## === cell 2
import subprocess, shlex

print(subprocess.check_output(shlex.split("ls -lha submission.csv")).decode("utf-8"))
print(subprocess.check_output(shlex.split("head -n 5 submission.csv")).decode("utf-8"))
