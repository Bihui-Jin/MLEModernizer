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
from collections import defaultdict, Counter

import pandas as pd
from PIL import Image

DATA_ROOT = "/kaggle/input/hotel-id-2021-fgvc8"

train_csv = os.path.join(DATA_ROOT, "train.csv")
sample_csv = os.path.join(DATA_ROOT, "sample_submission.csv")

if not os.path.exists(train_csv):
    train_csv = "/kaggle/input/train.csv"
if not os.path.exists(sample_csv):
    sample_csv = "/kaggle/input/sample_submission.csv"

assert os.path.exists(train_csv), f"train.csv not found at {train_csv}"
assert os.path.exists(sample_csv), f"sample_submission.csv not found at {sample_csv}"

train = pd.read_csv(train_csv)
sample = pd.read_csv(sample_csv)

assert (
    "hotel_id" in train.columns
    and "image" in train.columns
    and "chain" in train.columns
)
assert list(sample.columns) == [
    "image",
    "hotel_id",
], f"Unexpected sample columns: {sample.columns.tolist()}"

SEED = 42
random.seed(SEED)



## === cell 1
topk = 5

global_top_hotels = (
    train["hotel_id"].value_counts().head(topk).index.astype(str).tolist()
)
if len(global_top_hotels) < topk:
    global_top_hotels = (global_top_hotels + [global_top_hotels[-1]] * topk)[:topk]

train_images_root = os.path.join(DATA_ROOT, "train_images")
test_images_root = os.path.join(DATA_ROOT, "test_images")

if not os.path.exists(train_images_root):
    train_images_root = "/kaggle/input/train_images"
if not os.path.exists(test_images_root):
    test_images_root = "/kaggle/input/test_images"

if not os.path.exists(train_images_root):
    alt = "/kaggle/input/hotel-id-2021-fgvc8/train_images"
    if os.path.exists(alt):
        train_images_root = alt
if not os.path.exists(test_images_root):
    alt = "/kaggle/input/hotel-id-2021-fgvc8/test_images"
    if os.path.exists(alt):
        test_images_root = alt

assert os.path.exists(
    train_images_root
), f"train_images not found at {train_images_root}"
assert os.path.exists(test_images_root), f"test_images not found at {test_images_root}"

train_img_to_chain = {}
if os.path.exists(train_images_root):
    for entry in os.scandir(train_images_root):
        if entry.is_dir() and entry.name.isdigit():
            cid = entry.name
            for f in os.scandir(entry.path):
                if f.is_file():
                    train_img_to_chain[f.name] = cid


def dhash64(path, hash_size=8):
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize(
                (hash_size + 1, hash_size), Image.Resampling.BILINEAR
            )
            px = im.tobytes()  # length = (hash_size+1)*hash_size
        w = hash_size + 1
        h = 0
        bit = 0
        for r in range(hash_size):
            off = r * w
            row = px[off : off + w]
            for c in range(hash_size):
                h |= (1 if row[c] > row[c + 1] else 0) << bit
                bit += 1
        return h
    except Exception:
        return None


def hamming64(a, b):
    return (a ^ b).bit_count()


MAX_TOTAL_TRAIN_HASH = 25000
MAX_PER_HOTEL = 4

hotel_to_imgs = defaultdict(list)
for img, hid in zip(train["image"].astype(str), train["hotel_id"].astype(str)):
    hotel_to_imgs[hid].append(img)

hotels = list(hotel_to_imgs.keys())
random.shuffle(hotels)

selected_train_imgs = []
for hid in hotels:
    imgs = hotel_to_imgs[hid]
    if len(imgs) > MAX_PER_HOTEL:
        imgs = sorted(imgs)
        step = max(1, len(imgs) // MAX_PER_HOTEL)
        imgs = imgs[::step][:MAX_PER_HOTEL]
    selected_train_imgs.extend([(img, hid) for img in imgs])
    if len(selected_train_imgs) >= MAX_TOTAL_TRAIN_HASH:
        selected_train_imgs = selected_train_imgs[:MAX_TOTAL_TRAIN_HASH]
        break

bucket_bits = 16
bucket_mask = (1 << bucket_bits) - 1

train_hashes = []  # list of (hash_int, hotel_id)
buckets = defaultdict(list)  # prefix -> indices into train_hashes

for img, hid in selected_train_imgs:
    chain_folder = train_img_to_chain.get(img, None)
    if chain_folder is None:
        continue
    p = os.path.join(train_images_root, chain_folder, img)
    if not os.path.exists(p):
        continue
    h = dhash64(p)
    if h is None:
        continue
    idx = len(train_hashes)
    train_hashes.append((h, hid))
    buckets[h & bucket_mask].append(idx)

print(
    f"Hashed train images: {len(train_hashes)} (from selected {len(selected_train_imgs)})"
)
print(f"Buckets: {len(buckets)}")



## === cell 2
import heapq

ALL_BUCKETS = set(buckets.keys())
_bucket_neighbor_cache = {}
_shortlist_bucket_neighbors_default = 2
for base in ALL_BUCKETS:
    cand = []
    for delta in range(
        -_shortlist_bucket_neighbors_default, _shortlist_bucket_neighbors_default + 1
    ):
        b = base + delta
        if b in buckets:
            cand.extend(buckets[b])
    _bucket_neighbor_cache[base] = cand


def predict_for_hash(th, k=5, shortlist_bucket_neighbors=2, topn_neighbors=50):
    if th is None or len(train_hashes) == 0:
        return global_top_hotels[:k]

    base = th & bucket_mask

    if (
        shortlist_bucket_neighbors == _shortlist_bucket_neighbors_default
        and base in _bucket_neighbor_cache
    ):
        cand = _bucket_neighbor_cache[base]
    else:
        cand = []
        for delta in range(-shortlist_bucket_neighbors, shortlist_bucket_neighbors + 1):
            b = base + delta
            if b in buckets:
                cand.extend(buckets[b])

    if not cand:
        step = max(1, len(train_hashes) // 4000)
        cand = range(0, len(train_hashes), step)

    dists = []
    append = dists.append
    th_local = th
    train_hashes_local = train_hashes
    ham = hamming64
    for idx in cand:
        h, hid = train_hashes_local[idx]
        append((ham(th_local, h), hid))

    if len(dists) > topn_neighbors:
        dists = heapq.nsmallest(topn_neighbors, dists, key=lambda x: x[0])
    else:
        dists.sort(key=lambda x: x[0])

    votes = Counter()
    for rank, (d, hid) in enumerate(dists):
        votes[hid] += 1.0 / (1.0 + d) + 0.01 * (topn_neighbors - rank) / topn_neighbors

    pred = [hid for hid, _ in votes.most_common(k)]
    if len(pred) < k:
        pred = (pred + global_top_hotels)[:k]
    return pred[:k]


test_paths = {}
for f in os.scandir(test_images_root):
    if f.is_file():
        test_paths[f.name] = f.path
    elif f.is_dir():
        for ff in os.scandir(f.path):
            if ff.is_file():
                test_paths[ff.name] = ff.path

sample_images = sample["image"].astype(str).tolist()
test_hash_cache = {}
missing_paths = 0
for img in sample_images:
    p = test_paths.get(img)
    if p is None or not os.path.exists(p):
        missing_paths += 1
        continue
    test_hash_cache[img] = dhash64(p)

preds = []
for img in sample_images:
    th = test_hash_cache.get(img, None)
    if th is None:
        top_hotels = global_top_hotels
    else:
        top_hotels = predict_for_hash(th, k=topk)
    preds.append(" ".join([str(x) for x in top_hotels]))

print(f"Missing test paths for {missing_paths} / {len(sample)} images")

sub = sample.copy()
sub["hotel_id"] = preds

assert len(sub) == len(sample)
assert sub["hotel_id"].notnull().all()

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)

print(f"Wrote {sub_path} with shape {sub.shape}")
print(sub.head())



## === cell 3
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(6):
        print(f.readline().rstrip("\n"))
