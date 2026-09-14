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

BASE_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(relpath: str) -> str:
    for base in BASE_CANDIDATES:
        p = os.path.join(base, relpath)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {relpath} under any of: {BASE_CANDIDATES}")


def find_dir(relpath: str) -> str:
    for base in BASE_CANDIDATES:
        p = os.path.join(base, relpath)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        f"Could not find dir {relpath} under any of: {BASE_CANDIDATES}"
    )


train_csv = find_file("train.csv")
sample_csv = find_file("sample_submission.csv")

train = pd.read_csv(train_csv)
sample = pd.read_csv(sample_csv)

assert {"image", "hotel_id"}.issubset(
    train.columns
), f"train columns: {train.columns.tolist()}"
assert {"image", "hotel_id"}.issubset(
    sample.columns
), f"sample columns: {sample.columns.tolist()}"

print("train:", train.shape, "sample:", sample.shape)
print("train head:\n", train.head())
print("sample head:\n", sample.head())



## === cell 1
import numpy as np
from PIL import Image, ImageOps
from collections import defaultdict, Counter

TRAIN_IMG_CANDIDATES = [
    "train_images",
    "hotel-id-2021-fgvc8/train_images",
    "hotel-id-2021-fgvc8/train_images/train_images",
]
TEST_IMG_CANDIDATES = [
    "test_images",
    "hotel-id-2021-fgvc8/test_images",
    "hotel-id-2021-fgvcvc8/test_images/test_images",  # harmless if missing
    "hotel-id-2021-fgvc8/test_images/test_images",
]

train_img_dir = None
for rp in TRAIN_IMG_CANDIDATES:
    try:
        train_img_dir = find_dir(rp)
        break
    except FileNotFoundError:
        pass
if train_img_dir is None:
    raise FileNotFoundError("Could not locate train_images directory.")

test_img_dir = None
for rp in TEST_IMG_CANDIDATES:
    try:
        test_img_dir = find_dir(rp)
        break
    except FileNotFoundError:
        pass
if test_img_dir is None:
    raise FileNotFoundError("Could not locate test_images directory.")

print("train_img_dir:", train_img_dir)
print("test_img_dir :", test_img_dir)

top5_global = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(top5_global) < 5:
    top5_global = top5_global + ["0"] * (5 - len(top5_global))
top5_global_str = " ".join(top5_global[:5])


def image_path_train(chain: int, image_name: str) -> str:
    return os.path.join(train_img_dir, str(chain), image_name)


def image_path_test(image_name: str) -> str:
    return os.path.join(test_img_dir, image_name)


def dhash64(img: Image.Image, hash_size: int = 8) -> np.uint64:
    img = ImageOps.grayscale(img).resize((hash_size + 1, hash_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.int16)
    diff = (arr[:, 1:] > arr[:, :-1]).astype(np.uint8)  # shape (8,8), values 0/1
    bits = diff.reshape(-1)  # row-major
    packed = np.packbits(bits, bitorder="big")  # 8 bytes, MSB-first within each byte
    return np.frombuffer(packed.tobytes(), dtype=">u8")[0].astype(np.uint64)


_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_all_u64(
    query_hash_u64: np.uint64, train_hashes_u64: np.ndarray
) -> np.ndarray:
    x = np.bitwise_xor(train_hashes_u64, query_hash_u64)  # uint64 vector
    xb = x.view(np.uint8).reshape(
        -1, 8
    )  # little-endian bytes are fine for popcount sum
    return _POPCNT8[xb].sum(axis=1).astype(np.uint8)


MAX_TRAIN_HASH = 35000  # keep as-is (core logic / constant)
train_small = train.copy()
if len(train_small) > MAX_TRAIN_HASH:
    vc = train_small["hotel_id"].value_counts()
    keep_hotels = set(vc.head(2000).index)  # common hotels
    train_small = train_small[train_small["hotel_id"].isin(keep_hotels)]
    if len(train_small) > MAX_TRAIN_HASH:
        train_small = train_small.iloc[:MAX_TRAIN_HASH].copy()

print("Using train images for hashing:", len(train_small))

train_hashes = []
train_hotels = []
hash_to_hotel_votes = defaultdict(Counter)

failed = 0
for row in train_small.itertuples(index=False):
    img_name = getattr(row, "image")
    chain = getattr(row, "chain")
    hid = str(getattr(row, "hotel_id"))
    p = image_path_train(chain, img_name)
    try:
        with Image.open(p) as im:
            h = dhash64(im)
        train_hashes.append(h)
        train_hotels.append(hid)
        hash_to_hotel_votes[int(h)][hid] += 1
    except Exception:
        failed += 1

train_hashes = np.array(train_hashes, dtype=np.uint64)
train_hotels = np.array(train_hotels, dtype=object)

print("Computed train hashes:", len(train_hashes), "failed:", failed)

if len(train_hashes) < 1000:
    print("Too few train hashes computed; falling back to global top5 baseline.")
    sub = sample[["image"]].copy()
    sub["hotel_id"] = top5_global_str
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv:", sub.shape)
else:
    KNN = 120  # number of neighbors to consider for voting
    BEST = 5

    sub = sample[["image"]].copy()
    preds = []

    test_failed = 0
    train_hashes_u64 = train_hashes  # local binding (micro-optimization)

    for img_name in sub["image"].tolist():
        p = image_path_test(img_name)
        try:
            with Image.open(p) as im:
                th = dhash64(im)

            votes = Counter()
            exact = hash_to_hotel_votes.get(int(th))
            if exact:
                votes.update(exact)

            if len(votes) < 20:
                dists = hamming_all_u64(th, train_hashes_u64)
                idx = np.argpartition(dists, KNN)[:KNN]
                for j in idx:
                    hid = train_hotels[j]
                    w = int(65 - int(dists[j]))  # max 64 bits; shift to positive weight
                    if w < 1:
                        w = 1
                    votes[hid] += w

            best = [hid for hid, _ in votes.most_common(BEST)]
            for hid in top5_global:
                if len(best) >= BEST:
                    break
                if hid not in best:
                    best.append(hid)
            if len(best) < BEST:
                best += ["0"] * (BEST - len(best))
            preds.append(" ".join(best[:BEST]))
        except Exception:
            test_failed += 1
            preds.append(top5_global_str)

    sub["hotel_id"] = preds

    assert sub.shape[0] == sample.shape[0]
    assert list(sub.columns) == ["image", "hotel_id"]
    assert sub["hotel_id"].str.split().str.len().eq(5).all()

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv:", sub.shape, "test_failed:", test_failed)
    print(sub.head())



## === cell 2
import subprocess

subprocess.run(
    ["bash", "-lc", "ls -lha submission.csv && head -n 5 submission.csv"], check=False
)
