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

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

assert {"image", "hotel_id"}.issubset(sub_df.columns)
assert {"image", "hotel_id"}.issubset(train_df.columns)

print("train_df:", train_df.shape, "sub_df:", sub_df.shape)
print("Example train row:", train_df.iloc[0].to_dict())




## === cell 1
train_image_to_path = {}
missing_train_paths = 0

if "chain" in train_df.columns:
    _join = os.path.join
    _exists = os.path.exists
    _train_dir = TRAIN_IMG_DIR
    for img, ch in zip(train_df["image"].values, train_df["chain"].values):
        p = _join(_train_dir, str(ch), img)
        if _exists(p):
            train_image_to_path[img] = p
        else:
            missing_train_paths += 1
else:
    missing_train_paths = len(train_df)

print(
    "Resolved train paths:", len(train_image_to_path), "Missing:", missing_train_paths
)




## === cell 2
_PHASH_BUFS = {}  # keyed by img_size, stores a float32 buffer for image pixels
_DCT_MATS = {}  # keyed by N, stores float32 DCT-II (unnormalized) matrix
_DCT_TMP = {}  # keyed by N, stores float32 temporary buffer for DCT computation


def _dct_mat_type2(N: int) -> np.ndarray:
    C = _DCT_MATS.get(N)
    if C is None:
        n = np.arange(N, dtype=np.float32)
        k = np.arange(N, dtype=np.float32)[:, None]
        C = np.cos((np.pi / N) * (n + 0.5) * k).astype(np.float32)
        _DCT_MATS[N] = C
    return C


def _dct_2d_type2_inplace(a: np.ndarray, out: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    N = a.shape[0]
    C = _dct_mat_type2(N)
    tmp = _DCT_TMP.get(N)
    if tmp is None:
        tmp = np.empty((N, N), dtype=np.float32)
        _DCT_TMP[N] = tmp
    np.dot(C, a, out=tmp)
    np.dot(tmp, C.T, out=out)
    return out


def phash(img_path: str, hash_size: int = 16, highfreq_factor: int = 4) -> np.ndarray:
    img_size = hash_size * highfreq_factor  # 64 with defaults

    buf = _PHASH_BUFS.get(img_size)
    if buf is None:
        buf = np.empty((img_size, img_size), dtype=np.float32)
        _PHASH_BUFS[img_size] = buf

    dct_out = _DCT_TMP.get(("out", img_size))
    if dct_out is None:
        dct_out = np.empty((img_size, img_size), dtype=np.float32)
        _DCT_TMP[("out", img_size)] = dct_out

    with Image.open(img_path) as im:
        im = im.convert("L").resize((img_size, img_size), Image.BILINEAR)
        buf[:, :] = np.asarray(im, dtype=np.float32)

    dct = _dct_2d_type2_inplace(buf, dct_out)
    dct_lowfreq = dct[:hash_size, :hash_size]
    med = np.median(dct_lowfreq[1:, 1:])  # ignore DC component
    bits = (dct_lowfreq > med).astype(np.uint8).reshape(-1)
    return bits


def hamming_distance(a_bits: np.ndarray, b_bits: np.ndarray) -> int:
    return int(np.count_nonzero(a_bits != b_bits))




## === cell 3
TRAIN_CAP = 20000  # keep identical cap and semantics

train_subset = train_df.iloc[: min(TRAIN_CAP, len(train_df))].copy()

CACHE_DIR = "/kaggle/working"
CACHE_PATH = os.path.join(CACHE_DIR, f"train_phash_cache_cap{TRAIN_CAP}_hs16_hf4.npz")

train_hashes = None
train_hotels = None
used = 0
skipped = 0

if os.path.exists(CACHE_PATH):
    try:
        data = np.load(CACHE_PATH, allow_pickle=True)
        train_hashes = data["train_hashes"]
        train_hotels = data["train_hotels"]
        used = int(data["used"])
        skipped = int(data["skipped"])
        print(
            "Loaded cached train hashes:",
            train_hashes.shape,
            "used:",
            used,
            "skipped:",
            skipped,
        )
    except Exception as e:
        print("Cache load failed, recomputing:", repr(e))
        train_hashes = None
        train_hotels = None

if train_hashes is None:
    train_hashes_list = []
    train_hotels_list = []

    _phash = phash
    _exists = os.path.exists
    _get_path = train_image_to_path.get
    _append_hash = train_hashes_list.append
    _append_hotel = train_hotels_list.append

    for img, hid in zip(train_subset["image"].values, train_subset["hotel_id"].values):
        p = _get_path(img, "")
        if not p or not _exists(p):
            skipped += 1
            continue
        try:
            _append_hash(_phash(p))
            _append_hotel(str(hid))
            used += 1
        except Exception:
            skipped += 1

    train_hashes = (
        np.stack(train_hashes_list, axis=0)
        if len(train_hashes_list)
        else np.zeros((0, 256), dtype=np.uint8)
    )
    train_hotels = np.array(train_hotels_list, dtype=object)

    try:
        np.savez_compressed(
            CACHE_PATH,
            train_hashes=train_hashes,
            train_hotels=train_hotels,
            used=np.int32(used),
            skipped=np.int32(skipped),
        )
        print("Wrote cache:", CACHE_PATH)
    except Exception as e:
        print("Cache write failed (non-fatal):", repr(e))

print("Train hashes:", train_hashes.shape, "used:", used, "skipped:", skipped)

top_popular = train_df["hotel_id"].astype(str).value_counts().head(5).index.tolist()
fallback_pred = " ".join(top_popular + (["0"] * max(0, 5 - len(top_popular))))
fallback_pred = " ".join(fallback_pred.split()[:5])
print("Fallback pred:", fallback_pred)

_POPCOUNT8 = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)
if train_hashes.shape[0] > 0:
    train_hashes_packed = np.packbits(train_hashes, axis=1)  # (N, 32) uint8
else:
    train_hashes_packed = np.zeros((0, 32), dtype=np.uint8)


def _build_bucket_index(packed: np.ndarray, key_bytes=(0, 10, 20)):
    if packed.shape[0] == 0:
        return {}, key_bytes
    b0, b1, b2 = key_bytes
    keys = (
        (packed[:, b0].astype(np.uint32) << 16)
        | (packed[:, b1].astype(np.uint32) << 8)
        | packed[:, b2].astype(np.uint32)
    )
    order = np.argsort(keys, kind="mergesort")  # stable/deterministic
    keys_sorted = keys[order]
    change = np.empty(keys_sorted.shape[0], dtype=bool)
    change[0] = True
    change[1:] = keys_sorted[1:] != keys_sorted[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], keys_sorted.shape[0]]

    idx = {}
    for s, e in zip(starts.tolist(), ends.tolist()):
        k = int(keys_sorted[s])
        idx[k] = order[s:e].tolist()
    return idx, key_bytes


bucket_index, bucket_bytes = _build_bucket_index(
    train_hashes_packed, key_bytes=(0, 10, 20)
)




## === cell 4
preds = []
fail_test = 0

_packbits = np.packbits
_xor = np.bitwise_xor
_argsort = np.argsort
_test_dir = TEST_IMG_DIR
_phash = phash
_exists = os.path.exists
_popcount = _POPCOUNT8 = _POPCOUNT8  # preserve name usage pattern
_train_packed = train_hashes_packed
_train_hotels = train_hotels
_top_popular = top_popular
_bucket_index = bucket_index
_b0, _b1, _b2 = bucket_bytes

MIN_CANDIDATES_FOR_BUCKET = 200

for img in sub_df["image"].values:
    test_path = os.path.join(_test_dir, img)

    if (not _exists(test_path)) or _train_packed.shape[0] == 0:
        preds.append(fallback_pred)
        fail_test += 1
        continue

    try:
        q = _phash(test_path)  # (256,) uint8
        q_packed = _packbits(q)  # (32,)

        key = (
            (np.uint32(q_packed[_b0]) << 16)
            | (np.uint32(q_packed[_b1]) << 8)
            | np.uint32(q_packed[_b2])
        )
        cand = _bucket_index.get(int(key), None)

        if cand is None or len(cand) < MIN_CANDIDATES_FOR_BUCKET:
            x = _xor(_train_packed, q_packed)  # (N, 32)
            d = _POPCOUNT8[x].sum(axis=1).astype(np.int32)  # (N,)
            topk_idx = _argsort(d)[:5]
        else:
            cand_idx = np.fromiter(cand, dtype=np.int32)
            x = _xor(_train_packed[cand_idx], q_packed)  # (C, 32)
            d = _POPCOUNT8[x].sum(axis=1).astype(np.int32)  # (C,)
            top_local = _argsort(d)[:5]
            topk_idx = cand_idx[top_local]

        topk_hotels = _train_hotels[topk_idx].tolist()
        if len(topk_hotels) < 5:
            topk_hotels += _top_popular
        preds.append(" ".join(topk_hotels[:5]))
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
