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
from pathlib import Path

BASE_INPUT = Path("/kaggle/input")

print("Listing /kaggle/input (top-level):")
if BASE_INPUT.exists():
    for p in sorted(BASE_INPUT.iterdir()):
        try:
            sz = p.stat().st_size if p.is_file() else 0
        except Exception:
            sz = 0
        print(
            f" - {p.name}{'/' if p.is_dir() else ''}\t{sz/1024/1024:.2f} MB"
            if p.is_file()
            else f" - {p.name}/"
        )
else:
    raise FileNotFoundError("/kaggle/input not found")




## === cell 1
import pandas as pd

CANDIDATE_ROOTS = [
    Path("/kaggle/input/hotel-id-2021-fgvc8"),
    Path("/kaggle/data/hotel-id-2021-fgvc8"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]


def find_first_existing(rel_path: str) -> Path:
    for r in CANDIDATE_ROOTS:
        p = r / rel_path
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {rel_path} under candidate roots: {CANDIDATE_ROOTS}"
    )


train_csv = find_first_existing("train.csv")
sample_sub_csv = find_first_existing("sample_submission.csv")

print("Using train.csv:", train_csv)
print("Using sample_submission.csv:", sample_sub_csv)

train_df = pd.read_csv(train_csv)
sub_df = pd.read_csv(sample_sub_csv)

required_train_cols = {"image", "hotel_id"}
required_sub_cols = {"image", "hotel_id"}
if not required_train_cols.issubset(train_df.columns):
    raise ValueError(
        f"train.csv missing required columns: {required_train_cols - set(train_df.columns)}"
    )
if not required_sub_cols.issubset(sub_df.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns: {required_sub_cols - set(sub_df.columns)}"
    )

global_top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + global_top5 * 5)[:5]

print("Global fallback top-5 hotel_ids:", global_top5)




## === cell 2
from PIL import Image
import numpy as np
from collections import Counter


def find_images_dir(name: str) -> Path:
    candidates = []
    for r in CANDIDATE_ROOTS:
        candidates.append(r / name)
        candidates.append(r / "hotel-id-2021-fgvc8" / name)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find images dir for {name} under {CANDIDATE_ROOTS}"
    )


train_images_dir = find_images_dir("train_images")
test_images_dir = find_images_dir("test_images")

print("Using train_images_dir:", train_images_dir)
print("Using test_images_dir:", test_images_dir)


def ahash64(img_path: Path) -> np.uint64:
    with Image.open(img_path) as im:
        im = im.convert("L").resize((8, 8), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    m = arr.mean()
    bits = (arr > m).astype(np.uint8).reshape(-1)  # length 64, row-major
    packed = np.packbits(bits, bitorder="big")  # shape (8,)
    return packed.view(">u8")[0].astype(np.uint64)


def popcount_u64_scalar(x: np.uint64) -> int:
    v = int(x)
    c = 0
    while v:
        v &= v - 1
        c += 1
    return c


def hamming_u64(a: np.uint64, b: np.uint64) -> int:
    return popcount_u64_scalar(np.bitwise_xor(a, b))


train_df = train_df.copy()
train_df["image"] = train_df["image"].astype(str)
train_df["hotel_id"] = train_df["hotel_id"].astype(str)

MAX_TRAIN_HASHES = 30000  # keep identical behavior/limit
train_df_sorted = train_df.sort_values("image").reset_index(drop=True)
train_df_use = train_df_sorted.iloc[
    : min(MAX_TRAIN_HASHES, len(train_df_sorted))
].copy()

train_df_use["chain"] = (
    train_df_sorted.loc[train_df_use.index, "chain"].values
    if "chain" in train_df_sorted.columns
    else 0
)


def resolve_train_path(chain, image_name: str) -> Path:
    return train_images_dir / str(chain) / image_name


train_paths = []
missing_train = 0
for ch, img in zip(train_df_use["chain"].values, train_df_use["image"].values):
    p = resolve_train_path(ch, img)
    if not p.exists():
        missing_train += 1
    train_paths.append(p)
train_df_use["path"] = train_paths
print(f"Train subset size: {len(train_df_use)} (missing paths: {missing_train})")

train_hashes = []
train_hotels = []
kept_paths = []
bad = 0
for p, hid in zip(train_df_use["path"].values, train_df_use["hotel_id"].values):
    try:
        p = Path(p)
        if not p.exists():
            bad += 1
            continue
        h = ahash64(p)
        train_hashes.append(h)
        train_hotels.append(hid)
        kept_paths.append(str(p))
    except Exception:
        bad += 1

train_hashes = np.array(train_hashes, dtype=np.uint64)
train_hotels = np.array(train_hotels, dtype=object)
print(f"Computed train hashes: {len(train_hashes)} (skipped: {bad})")

sub_df = sub_df.copy()
sub_df["image"] = sub_df["image"].astype(str)


def resolve_test_path(image_name: str) -> Path:
    p = test_images_dir / image_name
    if p.exists():
        return p
    alt = test_images_dir / "test_images" / image_name
    if alt.exists():
        return alt
    return p


test_paths = [resolve_test_path(x) for x in sub_df["image"].values]
missing_test = sum(0 if p.exists() else 1 for p in test_paths)
print(f"Test images: {len(test_paths)} (missing paths: {missing_test})")

BUCKET_BITS = 16
bucket_shift = np.uint64(64 - BUCKET_BITS)

buckets = {}
for idx, h in enumerate(train_hashes):
    key = int(np.right_shift(h, bucket_shift))
    buckets.setdefault(key, []).append(idx)

bucket_neighbors = {}
for key in buckets.keys():
    merged = []
    for k in (key, key ^ 1, key ^ 2, key ^ 3):
        lst = buckets.get(k)
        if lst:
            merged.extend(lst)
    bucket_neighbors[key] = merged


def get_candidate_indices(h: np.uint64):
    key = int(np.right_shift(h, bucket_shift))
    cand = bucket_neighbors.get(key)
    if not cand:
        step = max(1, len(train_hashes) // 5000)
        cand = list(range(0, len(train_hashes), step))[:5000]
    return cand


_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_many_u64(hq: np.uint64, hs: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(hs, hq).view(np.uint64)
    xb = x.view(np.uint8).reshape(-1, 8)  # little-endian bytes; popcount sum invariant
    return _POPCNT8[xb].sum(axis=1).astype(np.int16)


TOPK_NEIGHBORS = 50  # identical
preds = []
have_train = len(train_hashes) > 0

for p in test_paths:
    try:
        if (not p.exists()) or (not have_train):
            preds.append(" ".join(global_top5))
            continue

        hq = ahash64(p)
        cand_idx = get_candidate_indices(hq)
        cand_hashes = train_hashes[np.asarray(cand_idx, dtype=np.int32)]

        dvec = hamming_many_u64(hq, cand_hashes)
        order = np.argsort(dvec, kind="mergesort")
        topn = min(TOPK_NEIGHBORS, order.size)
        top_local = order[:topn]

        scores = Counter()
        for j in top_local:
            d = int(dvec[j])
            i = cand_idx[int(j)]
            scores[train_hotels[i]] += 64 - d

        ordered = [hid for hid, _ in scores.most_common(5)]
        if len(ordered) < 5:
            for hid in global_top5:
                if hid not in ordered:
                    ordered.append(hid)
                if len(ordered) == 5:
                    break
        preds.append(" ".join(ordered[:5]))
    except Exception:
        preds.append(" ".join(global_top5))

submission = pd.DataFrame({"image": sub_df["image"].values, "hotel_id": preds})

assert submission.shape[0] == sub_df.shape[0]
assert list(submission.columns) == ["image", "hotel_id"]
assert submission["hotel_id"].astype(str).str.split().map(len).eq(5).all()

out_path = Path("submission.csv")
submission.to_csv(out_path, index=False)

print("Wrote:", out_path.resolve())
print(submission.head())
