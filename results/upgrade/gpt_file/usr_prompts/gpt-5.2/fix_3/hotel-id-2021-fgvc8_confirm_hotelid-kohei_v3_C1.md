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
import glob
import math
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/kaggle/input/hotel-id-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

assert {"image", "hotel_id"}.issubset(sub_df.columns)
assert {"image", "hotel_id"}.issubset(train_df.columns)

print("train_df:", train_df.shape, "sub_df:", sub_df.shape)
print("Example train row:", train_df.iloc[0].to_dict())




## === cell 1
def _build_image_index(base_dir: str) -> dict:
    idx = {}
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".jpg"):
                idx.setdefault(fn, os.path.join(root, fn))
    return idx


train_img_index = _build_image_index(TRAIN_IMG_DIR)
test_img_index = _build_image_index(TEST_IMG_DIR)

train_image_to_path = {}
if "chain" in train_df.columns:
    for img, ch in zip(train_df["image"].values, train_df["chain"].values):
        p = os.path.join(TRAIN_IMG_DIR, str(ch), img)
        if os.path.exists(p):
            train_image_to_path[img] = p

missing_train_paths = 0
for img in train_df["image"].values:
    if img not in train_image_to_path:
        p = train_img_index.get(img, "")
        if p:
            train_image_to_path[img] = p
        else:
            missing_train_paths += 1

print(
    "Resolved train paths:", len(train_image_to_path), "Missing:", missing_train_paths
)




## === cell 2
def dct_1d(x: np.ndarray) -> np.ndarray:
    """
    DCT-II for 1D vector x.
    Equivalent to: sum_{n=0..N-1} x[n] * cos(pi/N * (n+0.5) * k)
    Implemented via FFT to avoid building NxN cosine matrices per call.
    """
    x = np.asarray(x, dtype=np.float32)
    N = x.shape[0]
    v = np.empty(2 * N, dtype=np.float32)
    v[:N] = x
    v[N:] = x[::-1]
    V = np.fft.fft(v)[:N]
    k = np.arange(N, dtype=np.float32)
    out = (V * np.exp(-1j * (np.pi * k) / (2.0 * N))).real.astype(np.float32)
    return out


def dct_2d(a: np.ndarray) -> np.ndarray:
    """2D DCT-II via separability (rows then cols), using the fast 1D DCT above."""
    a = np.asarray(a, dtype=np.float32)
    tmp = np.empty_like(a, dtype=np.float32)
    for i in range(a.shape[0]):
        tmp[i, :] = dct_1d(a[i, :])
    out = np.empty_like(tmp, dtype=np.float32)
    for j in range(tmp.shape[1]):
        out[:, j] = dct_1d(tmp[:, j])
    return out


def phash(img_path: str, hash_size: int = 16, highfreq_factor: int = 4) -> np.ndarray:
    """
    DCT-based pHash.
    Produces hash_size*hash_size bits as uint8 array of 0/1.
    """
    img_size = hash_size * highfreq_factor
    with Image.open(img_path) as im:
        im = im.convert("L").resize((img_size, img_size), Image.BILINEAR)
        pixels = np.asarray(im, dtype=np.float32)
    dct = dct_2d(pixels)
    dct_lowfreq = dct[:hash_size, :hash_size]
    med = np.median(dct_lowfreq[1:, 1:])  # ignore DC component for thresholding
    bits = (dct_lowfreq > med).astype(np.uint8).reshape(-1)
    return bits


def hamming_distance(a_bits: np.ndarray, b_bits: np.ndarray) -> int:
    return int(np.count_nonzero(a_bits != b_bits))




## === cell 3
TRAIN_CAP = 20000  # keep identical cap and semantics; speedups make this feasible within timeout

train_subset = train_df.iloc[: min(TRAIN_CAP, len(train_df))].copy()

train_hashes = []
train_hotels = []
used = 0
skipped = 0

for img, hid in zip(train_subset["image"].values, train_subset["hotel_id"].values):
    p = train_image_to_path.get(img, "")
    if not p or not os.path.exists(p):
        skipped += 1
        continue
    try:
        train_hashes.append(phash(p))
        train_hotels.append(str(hid))
        used += 1
    except Exception:
        skipped += 1

train_hashes = (
    np.stack(train_hashes, axis=0)
    if len(train_hashes)
    else np.zeros((0, 256), dtype=np.uint8)
)
train_hotels = np.array(train_hotels, dtype=object)

print("Train hashes:", train_hashes.shape, "used:", used, "skipped:", skipped)

top_popular = train_df["hotel_id"].astype(str).value_counts().head(5).index.tolist()
fallback_pred = " ".join(top_popular + (["0"] * max(0, 5 - len(top_popular))))
fallback_pred = " ".join(fallback_pred.split()[:5])
print("Fallback pred:", fallback_pred)

if train_hashes.shape[0] > 0:
    train_hashes_packed = np.packbits(train_hashes, axis=1)  # (N, 32) uint8
    _POPCOUNT8 = (
        np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
        .sum(axis=1)
        .astype(np.uint8)
    )
else:
    train_hashes_packed = np.zeros((0, 32), dtype=np.uint8)
    _POPCOUNT8 = (
        np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
        .sum(axis=1)
        .astype(np.uint8)
    )



## === cell 4
preds = []
fail_test = 0

for img in sub_df["image"].values:
    test_path = os.path.join(TEST_IMG_DIR, img)
    if not os.path.exists(test_path):
        test_path = test_img_index.get(img, test_path)

    if not os.path.exists(test_path) or train_hashes.shape[0] == 0:
        preds.append(fallback_pred)
        fail_test += 1
        continue

    try:
        q = phash(test_path)  # (256,)
        q_packed = np.packbits(q.reshape(1, -1), axis=1)[0]  # (32,)

        x = np.bitwise_xor(train_hashes_packed, q_packed)  # (N, 32)
        d = _POPCOUNT8[x].sum(axis=1).astype(np.int32)  # (N,)

        topk_idx = np.argsort(d)[:5]
        topk_hotels = train_hotels[topk_idx].tolist()

        if len(topk_hotels) < 5:
            topk_hotels += top_popular
        pred = " ".join(topk_hotels[:5])
        preds.append(pred)
    except Exception:
        preds.append(fallback_pred)
        fail_test += 1

sub_out = sub_df.copy()
sub_out["hotel_id"] = preds

print("Test failures:", fail_test, "Total:", len(sub_out))
print(sub_out.head())

sub_path = "submission.csv"
sub_out.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(open(sub_path, "r", encoding="utf-8").read().splitlines()[:5])
