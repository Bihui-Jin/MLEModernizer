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
import pandas as pd
from PIL import Image
import numpy as np

CAND_TRAIN = [
    "/kaggle/input/hotel-id-2021-fgvc8/train.csv",
    "/kaggle/input/train.csv",
]
train_path = next((p for p in CAND_TRAIN if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(f"Could not find train.csv in any of: {CAND_TRAIN}")

CAND_TEST_DIRS = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
]
test_dir = next((p for p in CAND_TEST_DIRS if os.path.isdir(p)), None)
if test_dir is None:
    raise FileNotFoundError(f"Could not find test_images/ in any of: {CAND_TEST_DIRS}")

CAND_TRAIN_IMG_DIRS = [
    "/kaggle/input/hotel-id-2021-fgvc8/train_images",
    "/kaggle/input/train_images",
]
train_img_dir = next((p for p in CAND_TRAIN_IMG_DIRS if os.path.isdir(p)), None)
if train_img_dir is None:
    raise FileNotFoundError(
        f"Could not find train_images/ in any of: {CAND_TRAIN_IMG_DIRS}"
    )

CAND_SAMPLE = [
    "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
sample_path = next((p for p in CAND_SAMPLE if os.path.exists(p)), None)

print("Using train_path:", train_path)
print("Using train_img_dir:", train_img_dir)
print("Using test_dir:", test_dir)
print("Using sample_submission:", sample_path)

train_df = pd.read_csv(train_path)
required_cols = {"hotel_id", "image"}
missing = required_cols - set(train_df.columns)
if missing:
    raise ValueError(f"train.csv missing required columns: {sorted(missing)}")

train_df["hotel_id"] = train_df["hotel_id"].astype(str)
train_df["image"] = train_df["image"].astype(str)

global_top5 = train_df["hotel_id"].value_counts().head(5).index.tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + [global_top5[-1]] * 5)[:5]
global_pred_str = " ".join(global_top5)
print("Global fallback top-5 hotel_ids:", global_pred_str)

test_paths = sorted(glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True))
if len(test_paths) == 0:
    raise RuntimeError(
        f"No .jpg files found under {test_dir}. Cannot build submission."
    )

test_img_by_name = {os.path.basename(p): p for p in test_paths}

if sample_path is not None:
    sample_sub = pd.read_csv(sample_path)
    if "image" not in sample_sub.columns:
        raise ValueError("sample_submission.csv missing 'image' column")
    images = sample_sub["image"].astype(str).tolist()
else:
    images = sorted(test_img_by_name.keys())


def _fix5_list(ids):
    ids = [str(x) for x in ids if str(x) != ""]
    if len(ids) == 0:
        ids = global_top5[:]
    if len(ids) < 5:
        ids = ids + [ids[-1]] * (5 - len(ids))
    return ids[:5]


def _fix5_str(s: str) -> str:
    return " ".join(_fix5_list(str(s).split()))


HASH_SIZE = 16  # was 8


def dhash_bits(image_path, hash_size=HASH_SIZE):
    with Image.open(image_path) as im:
        im = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    diff = arr[:, 1:] > arr[:, :-1]  # (hash_size, hash_size) bool
    bits_u8 = diff.reshape(-1).astype(np.uint8, copy=False)
    packed = np.packbits(bits_u8, bitorder="big")  # length = hash_size*hash_size/8
    return packed  # np.uint8 array


def hamming_bytes(packed_a_u8, packed_b_u8, popcnt8_lut):
    return int(popcnt8_lut[np.bitwise_xor(packed_a_u8, packed_b_u8)].sum())




## === cell 1
train_df_local = train_df
has_chain = "chain" in train_df_local.columns

hotel_counts = train_df_local["hotel_id"].value_counts()
top_hotels = set(hotel_counts.head(3500).index.tolist())  # was 2000
cand_train = train_df_local[train_df_local["hotel_id"].isin(top_hotels)].copy()

PER_HOTEL_CAP = 18  # was 12
cand_train = cand_train.groupby("hotel_id", group_keys=False).head(PER_HOTEL_CAP)

MAX_TRAIN_HASH = 22000  # was 14000
if len(cand_train) > MAX_TRAIN_HASH:
    cand_train = cand_train.head(MAX_TRAIN_HASH)


def _build_train_path_from_chain(chain_val, image_name):
    return os.path.join(train_img_dir, str(int(chain_val)), str(image_name))


if has_chain:
    cand_train["path"] = [
        _build_train_path_from_chain(c, img)
        for c, img in zip(cand_train["chain"].values, cand_train["image"].values)
    ]
    exists_mask = np.fromiter(
        (os.path.exists(p) for p in cand_train["path"].values),
        dtype=bool,
        count=len(cand_train),
    )
    cand_train = cand_train.loc[exists_mask].reset_index(drop=True)
else:
    train_paths = glob.glob(os.path.join(train_img_dir, "**", "*.jpg"), recursive=True)
    train_img_by_name = {os.path.basename(p): p for p in train_paths}
    print("Found train images on disk:", len(train_img_by_name))
    cand_train["path"] = cand_train["image"].map(train_img_by_name.get)
    cand_train = cand_train.dropna(subset=["path"]).reset_index(drop=True)

print("Hashing training subset rows:", len(cand_train))

_popcnt8 = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)

train_hashes = []
train_hotels = []
bad_train = 0
for p, hid in zip(cand_train["path"].tolist(), cand_train["hotel_id"].tolist()):
    try:
        train_hashes.append(dhash_bits(p))
        train_hotels.append(hid)
    except Exception:
        bad_train += 1

if len(train_hashes) > 0:
    train_hashes = np.stack(train_hashes, axis=0).astype(np.uint8, copy=False)  # (N, B)
else:
    train_hashes = np.zeros((0, (HASH_SIZE * HASH_SIZE) // 8), dtype=np.uint8)
train_hotels = np.array(train_hotels, dtype=object)

print(
    "Built train hash index:",
    len(train_hashes),
    "hash_bytes:",
    train_hashes.shape[1] if train_hashes.ndim == 2 else None,
    "bad_train:",
    bad_train,
)

KNN_K = 25  # neighbors to vote from
TOPN_OUT = 5  # MAP@5 requirement

preds = []
n_hashed = 0
n_fallback = 0

_global_top5 = global_top5
_global_pred_str = global_pred_str
_train_hotels = train_hotels
_train_hashes = train_hashes
_train_hashes_len = len(train_hashes)
_KNN_K = KNN_K
_TOPN_OUT = TOPN_OUT
_test_img_by_name = test_img_by_name
_fix5_list_local = _fix5_list
_popcnt8_local = _popcnt8

_INV_DENOM = 1.0  # keep simple and stable
_EPS = 1e-9

if _train_hashes_len > 0:
    xor_buf = np.empty_like(_train_hashes, dtype=np.uint8)  # (N, B)
else:
    xor_buf = None

for img in images:
    if _train_hashes_len == 0:
        preds.append(_global_pred_str)
        n_fallback += 1
        continue

    p = _test_img_by_name.get(img)
    if p is None:
        preds.append(_global_pred_str)
        n_fallback += 1
        continue

    try:
        th = dhash_bits(p)
    except Exception:
        preds.append(_global_pred_str)
        n_fallback += 1
        continue

    n_hashed += 1

    np.bitwise_xor(_train_hashes, th, out=xor_buf)  # uint8 (N,B)

    dists = _popcnt8_local[xor_buf].sum(axis=1, dtype=np.uint16)  # (N,)

    k = _KNN_K if _KNN_K < _train_hashes_len else _train_hashes_len
    nn_idx = np.argpartition(dists, k - 1)[:k]

    vote = {}
    for idx in nn_idx:
        hid = _train_hotels[idx]
        dist = int(dists[idx])
        w = 1.0 / (dist + _INV_DENOM + _EPS)
        cur = vote.get(hid)
        if cur is None:
            vote[hid] = [1, dist, w]
        else:
            cur[0] += 1
            if dist < cur[1]:
                cur[1] = dist
            cur[2] += w

    ranked = sorted(vote.items(), key=lambda kv: (-kv[1][0], -kv[1][2], kv[1][1]))
    top_ids = [hid for hid, _ in ranked[:_TOPN_OUT]]

    out = []
    seen = set()
    for hid in top_ids:
        if hid not in seen:
            out.append(hid)
            seen.add(hid)
    for hid in _global_top5:
        if len(out) >= _TOPN_OUT:
            break
        if hid not in seen:
            out.append(hid)
            seen.add(hid)

    preds.append(" ".join(_fix5_list_local(out)))

sub = pd.DataFrame({"image": images, "hotel_id": preds})

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(sub), "cols:", list(sub.columns))
print("Hashed test images:", n_hashed, "fallback:", n_fallback)
print(sub.head())
