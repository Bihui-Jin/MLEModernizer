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


def dhash64(image_path, hash_size=8):
    with Image.open(image_path) as im:
        im = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return np.uint64(h)


def hamming64(a, b):
    return int((np.uint64(a) ^ np.uint64(b)).bit_count())




## === cell 1

train_paths = glob.glob(os.path.join(train_img_dir, "**", "*.jpg"), recursive=True)
train_img_by_name = {os.path.basename(p): p for p in train_paths}
print("Found train images on disk:", len(train_img_by_name))

hotel_counts = train_df["hotel_id"].value_counts()
top_hotels = set(
    hotel_counts.head(2000).index.tolist()
)  # cap to control runtime/memory
cand_train = train_df[train_df["hotel_id"].isin(top_hotels)].copy()

PER_HOTEL_CAP = 8
cand_train = cand_train.groupby("hotel_id", group_keys=False).head(PER_HOTEL_CAP)

MAX_TRAIN_HASH = 14000
if len(cand_train) > MAX_TRAIN_HASH:
    cand_train = cand_train.head(MAX_TRAIN_HASH)

cand_train["path"] = cand_train["image"].map(train_img_by_name.get)
cand_train = cand_train.dropna(subset=["path"]).reset_index(drop=True)
print("Hashing training subset rows:", len(cand_train))

train_hashes = []
train_hotels = []
bad_train = 0
for p, hid in zip(cand_train["path"].tolist(), cand_train["hotel_id"].tolist()):
    try:
        train_hashes.append(dhash64(p))
        train_hotels.append(hid)
    except Exception:
        bad_train += 1

train_hashes = np.array(train_hashes, dtype=np.uint64)
train_hotels = np.array(train_hotels, dtype=object)
print("Built train hash index:", len(train_hashes), "bad_train:", bad_train)

KNN_K = 25  # neighbors to vote from
TOPN_OUT = 5  # MAP@5 requirement

preds = []
n_hashed = 0
n_fallback = 0

for img in images:
    p = test_img_by_name.get(img)
    if p is None or len(train_hashes) == 0:
        preds.append(global_pred_str)
        n_fallback += 1
        continue

    try:
        th = dhash64(p)
        n_hashed += 1
    except Exception:
        preds.append(global_pred_str)
        n_fallback += 1
        continue

    dists = np.empty(len(train_hashes), dtype=np.int16)
    for i in range(len(train_hashes)):
        dists[i] = hamming64(th, train_hashes[i])

    nn_idx = np.argpartition(dists, min(KNN_K, len(dists)) - 1)[
        : min(KNN_K, len(dists))
    ]
    nn_hotels = train_hotels[nn_idx]

    vote = {}
    for idx in nn_idx:
        hid = train_hotels[idx]
        dist = int(dists[idx])
        if hid not in vote:
            vote[hid] = [0, dist]  # count, best_dist
        vote[hid][0] += 1
        if dist < vote[hid][1]:
            vote[hid][1] = dist

    ranked = sorted(vote.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
    top_ids = [hid for hid, _ in ranked[:TOPN_OUT]]

    out = []
    seen = set()
    for hid in top_ids:
        if hid not in seen:
            out.append(hid)
            seen.add(hid)
    for hid in global_top5:
        if len(out) >= TOPN_OUT:
            break
        if hid not in seen:
            out.append(hid)
            seen.add(hid)

    preds.append(" ".join(_fix5_list(out)))

sub = pd.DataFrame({"image": images, "hotel_id": preds})

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(sub), "cols:", list(sub.columns))
print("Hashed test images:", n_hashed, "fallback:", n_fallback)
print(sub.head())
