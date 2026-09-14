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
import random
import pandas as pd

BASE_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]
BASE = next((p for p in BASE_CANDIDATES if os.path.exists(p)), None)
print("Using BASE:", BASE)


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


train_csv = find_first_existing(
    [
        os.path.join(BASE, "train.csv") if BASE else None,
        "/kaggle/input/hotel-id-2021-fgvc8/train.csv",
        "/kaggle/data/hotel-id-2021-fgvc8/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)
sample_sub_csv = find_first_existing(
    [
        os.path.join(BASE, "sample_submission.csv") if BASE else None,
        "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv",
        "/kaggle/data/hotel-id-2021-fgvc8/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

TEST_IMG_ROOT = find_first_existing(
    [
        os.path.join(BASE, "test_images") if BASE else None,
        "/kaggle/input/hotel-id-2021-fgvc8/test_images",
        "/kaggle/data/hotel-id-2021-fgvc8/test_images",
        "/kaggle/input/test_images",
        "/kaggle/data/test_images",
    ]
)

TRAIN_IMG_ROOT = find_first_existing(
    [
        os.path.join(BASE, "train_images") if BASE else None,
        "/kaggle/input/hotel-id-2021-fgvc8/train_images",
        "/kaggle/data/hotel-id-2021-fgvc8/train_images",
        "/kaggle/input/train_images",
        "/kaggle/data/train_images",
    ]
)

assert train_csv is not None, "train.csv not found in expected locations."
assert (
    sample_sub_csv is not None
), "sample_submission.csv not found in expected locations."
assert TEST_IMG_ROOT is not None, "test_images folder not found in expected locations."
assert (
    TRAIN_IMG_ROOT is not None
), "train_images folder not found in expected locations."

print("train_csv:", train_csv)
print("sample_sub_csv:", sample_sub_csv)
print("TRAIN_IMG_ROOT:", TRAIN_IMG_ROOT)
print("TEST_IMG_ROOT:", TEST_IMG_ROOT)

train = pd.read_csv(train_csv)
sample_sub = pd.read_csv(sample_sub_csv)

print("train shape:", train.shape)
print("sample_sub shape:", sample_sub.shape)
print("train columns:", train.columns.tolist())
print("sample_sub columns:", sample_sub.columns.tolist())



## === cell 1
from PIL import Image

train2 = train.copy()
train2["hotel_id"] = train2["hotel_id"].astype(str)
train2["chain"] = train2["chain"].fillna(0).astype(int)

global_counts = train2["hotel_id"].value_counts()
global_top5 = global_counts.index[:5].tolist()


def pad_to_5(hids, fallback):
    out = []
    for x in hids:
        if x not in out:
            out.append(x)
        if len(out) == 5:
            return out
    for x in fallback:
        if x not in out:
            out.append(x)
        if len(out) == 5:
            return out
    uniq = train2["hotel_id"].unique().tolist()
    for x in uniq:
        if x not in out:
            out.append(x)
        if len(out) == 5:
            return out
    return out


chain_top = {}
for ch, grp in train2.groupby("chain", sort=False):
    top = grp["hotel_id"].value_counts().index[:5].tolist()
    chain_top[int(ch)] = pad_to_5(top, global_top5)


def build_test_image_chain_lookup(test_root: str):
    mapping = {}
    try:
        entries = os.listdir(test_root)
    except FileNotFoundError:
        return mapping

    for name in entries:
        full = os.path.join(test_root, name)
        if os.path.isdir(full):
            ch = int(name) if name.isdigit() else 0
            try:
                for fn in os.listdir(full):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        mapping[fn] = ch
            except FileNotFoundError:
                continue
        else:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                mapping[name] = 0
    return mapping


test_img_to_chain = build_test_image_chain_lookup(TEST_IMG_ROOT)
print("Built test image->chain mapping size:", len(test_img_to_chain))


def resolve_train_image_path(image_name: str, chain: int) -> str:
    return os.path.join(TRAIN_IMG_ROOT, str(int(chain)), str(image_name))


def resolve_test_image_path(image_name: str) -> str:
    p1 = os.path.join(TEST_IMG_ROOT, str(image_name))
    if os.path.exists(p1):
        return p1
    ch = test_img_to_chain.get(str(image_name), None)
    if ch is not None:
        p2 = os.path.join(TEST_IMG_ROOT, str(int(ch)), str(image_name))
        if os.path.exists(p2):
            return p2
    return p1  # best-effort fallback


def ahash_8x8(image_path: str):
    try:
        with Image.open(image_path) as im:
            im = im.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
            px = list(im.getdata())
        meanv = sum(px) / 64.0
        bits = 0
        for v in px:
            bits = (bits << 1) | (1 if v > meanv else 0)
        return f"{bits:016x}"
    except Exception:
        return None


RANDOM_SEED = 123
MAX_TRAIN_HASH_IMAGES = 20000  # cap for runtime; raise cautiously if you have more time
random.seed(RANDOM_SEED)

train_idx = list(range(len(train2)))
random.shuffle(train_idx)
train_idx = train_idx[: min(MAX_TRAIN_HASH_IMAGES, len(train_idx))]

hash_counts = {}  # hash -> dict(hotel_id -> count)
missing_train_images = 0
failed_train_hash = 0

for i in train_idx:
    row = train2.iloc[i]
    img = str(row["image"])
    ch = int(row["chain"])
    hid = str(row["hotel_id"])
    p = resolve_train_image_path(img, ch)
    if not os.path.exists(p):
        missing_train_images += 1
        continue
    h = ahash_8x8(p)
    if h is None:
        failed_train_hash += 1
        continue
    d = hash_counts.get(h)
    if d is None:
        d = {}
        hash_counts[h] = d
    d[hid] = d.get(hid, 0) + 1

hash_top5 = {}
for h, d in hash_counts.items():
    top = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    top_ids = [k for k, _ in top[:5]]
    hash_top5[h] = pad_to_5(top_ids, global_top5)

print("Train hash stats:")
print("  Used train rows:", len(train_idx))
print("  Unique hashes:", len(hash_top5))
print("  Missing train image files:", missing_train_images)
print("  Failed hash computations:", failed_train_hash)
print("Global top-5 used for fallback:", global_top5)

preds = []
missing_chain_hits = 0
missing_image_hits = 0
missing_hash_hits = 0
failed_test_hash = 0

for img in sample_sub["image"].astype(str).tolist():
    tp = resolve_test_image_path(img)
    if not os.path.exists(tp):
        missing_image_hits += 1
        h = None
    else:
        h = ahash_8x8(tp)
        if h is None:
            failed_test_hash += 1

    top5 = None
    if h is not None:
        top5 = hash_top5.get(h)

    if top5 is None:
        missing_hash_hits += 1
        top5 = global_top5

    if img in test_img_to_chain:
        ch = test_img_to_chain[img]
    else:
        ch = 0
    if ch not in chain_top:
        if ch != 0:
            missing_chain_hits += 1
        ch = 0

    if ch != 0:
        chain_list = chain_top.get(ch, global_top5)
        final5 = pad_to_5(chain_list + top5, global_top5)
    else:
        final5 = pad_to_5(top5, global_top5)

    preds.append(" ".join(final5))

sub = sample_sub.copy()
sub["hotel_id"] = preds

assert list(sub.columns) == [
    "image",
    "hotel_id",
], "Submission columns must be exactly: image, hotel_id"
assert len(sub) == len(sample_sub), "Submission row count must match sample_submission."

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(sub.head())
print("Test-time diagnostics:")
print(
    "  Test images missing from folder path (unexpected in hidden):", missing_image_hits
)
print("  Test hash missing (fallback to global top-5):", missing_hash_hits)
print("  Test hash failed computations:", failed_test_hash)
print("  Chains in mapping:", len(chain_top))
print(
    "  Test images with unknown/unseen chain mapping (fallback to chain 0):",
    missing_chain_hits,
)
