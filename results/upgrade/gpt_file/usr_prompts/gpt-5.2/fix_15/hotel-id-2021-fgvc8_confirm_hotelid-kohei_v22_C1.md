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

train["image"] = train["image"].astype(str)
train["hotel_id"] = train["hotel_id"].astype(str)
train["chain"] = train["chain"].astype(str)

chain_top_hotels = {}
for ch, vc in train.groupby("chain")["hotel_id"].value_counts().groupby(level=0):
    chain_top_hotels[ch] = vc.head(50).index.get_level_values(1).astype(str).tolist()

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

img_to_chain_from_csv = dict(zip(train["image"].values, train["chain"].values))

hotel_to_chain = (
    train.drop_duplicates("hotel_id")[["hotel_id", "chain"]]
    .set_index("hotel_id")["chain"]
    .to_dict()
)

try:
    import cv2  # available on Kaggle; if not, we fall back to PIL
except Exception:
    cv2 = None


def dhash64(path, hash_size=8):
    try:
        if cv2 is not None:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if im is None:
                return None
            im = cv2.resize(
                im, (hash_size + 1, hash_size), interpolation=cv2.INTER_LINEAR
            )
            h = 0
            bit = 0
            for r in range(hash_size):
                row = im[r]
                for c in range(hash_size):
                    h |= (1 if int(row[c]) > int(row[c + 1]) else 0) << bit
                    bit += 1
            return h
        else:
            with Image.open(path) as im:
                im = im.convert("L").resize(
                    (hash_size + 1, hash_size), Image.Resampling.BILINEAR
                )
                px = im.tobytes()
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


MAX_TOTAL_TRAIN_HASH = 60000
MAX_PER_HOTEL = 8

hotel_to_imgs = defaultdict(list)
for img, hid in zip(train["image"].values, train["hotel_id"].values):
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

bucket_bits = 12
bucket_shift = 64 - bucket_bits

train_hashes = []  # list of (hash_int, hotel_id)
buckets = defaultdict(list)  # prefix -> indices into train_hashes

os_path_join = os.path.join
exists = os.path.exists
dhash64_local = dhash64
append_train_hashes = train_hashes.append
buckets_local = buckets
bucket_shift_local = bucket_shift


def _build_image_path_index_recursive(root, max_depth=6):
    out = {}
    stack = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    if entry.is_file():
                        out[entry.name] = entry.path
                    elif entry.is_dir() and depth < max_depth:
                        stack.append((entry.path, depth + 1))
        except Exception:
            continue
    return out


train_paths = _build_image_path_index_recursive(train_images_root, max_depth=6)

hashed_missing_train_paths = 0
for img, hid in selected_train_imgs:
    p = train_paths.get(img)
    if p is None or not exists(p):
        hashed_missing_train_paths += 1
        continue
    h = dhash64_local(p)
    if h is None:
        continue
    idx = len(train_hashes)
    append_train_hashes((h, hid))
    buckets_local[(h >> bucket_shift_local)].append(idx)

print(
    f"Hashed train images: {len(train_hashes)} (from selected {len(selected_train_imgs)}); "
    f"missing_paths_in_index={hashed_missing_train_paths}"
)
print(f"Buckets: {len(buckets)}")




## === cell 2
import numpy as np

train_hash_vals = [h for h, _hid in train_hashes]
train_hash_hids = [hid for _h, hid in train_hashes]

train_hash_vals_u64 = np.asarray(train_hash_vals, dtype=np.uint64)
train_hash_hids_py = train_hash_hids

ALL_BUCKETS = set(buckets.keys())
_bucket_neighbor_cache = {}


def _prefix_neighbors_hamming3(prefix: int, bits: int):
    out = [prefix]
    for i in range(bits):
        out.append(prefix ^ (1 << i))
    for i in range(bits):
        pi = prefix ^ (1 << i)
        for j in range(i + 1, bits):
            out.append(pi ^ (1 << j))
    for i in range(bits):
        pi = prefix ^ (1 << i)
        for j in range(i + 1, bits):
            pij = pi ^ (1 << j)
            for k in range(j + 1, bits):
                out.append(pij ^ (1 << k))
    return out


prefix_space = 1 << bucket_bits
for base in range(prefix_space):
    cand = []
    for b in _prefix_neighbors_hamming3(base, bucket_bits):
        if b in buckets:
            cand.extend(buckets[b])
    _bucket_neighbor_cache[base] = cand

_sorted_bucket_keys = (
    np.array(sorted(ALL_BUCKETS), dtype=np.int32)
    if ALL_BUCKETS
    else np.array([], dtype=np.int32)
)


def _nearest_cached_bucket_key(base: int):
    if _sorted_bucket_keys.size == 0:
        return None
    pos = int(np.searchsorted(_sorted_bucket_keys, base))
    if pos <= 0:
        return int(_sorted_bucket_keys[0])
    if pos >= _sorted_bucket_keys.size:
        return int(_sorted_bucket_keys[-1])
    left = int(_sorted_bucket_keys[pos - 1])
    right = int(_sorted_bucket_keys[pos])
    return left if (base - left) <= (right - base) else right


def _popcount_u64(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = x + (x >> np.uint64(8))
    x = x + (x >> np.uint64(16))
    x = x + (x >> np.uint64(32))
    return (x & np.uint64(0x7F)).astype(np.int16)


ANCHORS = 32
_anchor_hashes = []
if len(train_hash_vals) > 0:
    step = max(1, len(train_hash_vals) // ANCHORS)
    _anchor_hashes = [train_hash_vals[i] for i in range(0, len(train_hash_vals), step)][
        :ANCHORS
    ]
_anchor_prefixes = [int(h >> bucket_shift) for h in _anchor_hashes]


def predict_for_hash(
    th,
    k=5,
    shortlist_bucket_neighbors=2,  # kept for API compatibility
    topn_neighbors=120,  # kept: more NN evidence for voting; same logic.
    test_chain=None,
    chain_boost=0.20,
    exact_match_bonus=0.15,
):
    if th is None or len(train_hash_vals_u64) == 0:
        if test_chain is not None and test_chain in chain_top_hotels:
            base = chain_top_hotels[test_chain]
            out = base[:k]
            if len(out) < k:
                out = (out + global_top_hotels)[:k]
            return out[:k]
        return global_top_hotels[:k]

    base = int(th >> bucket_shift)
    cand = list(_bucket_neighbor_cache.get(base, []))

    if _anchor_prefixes:
        for ap in _anchor_prefixes:
            cand.extend(_bucket_neighbor_cache.get(ap, []))

    if not cand:
        step = max(1, len(train_hash_vals_u64) // 6000)
        cand = range(0, len(train_hash_vals_u64), step)

    if isinstance(cand, range):
        cand_idx = np.fromiter(cand, dtype=np.int32)
    else:
        cand_idx = np.asarray(cand, dtype=np.int32)
        if cand_idx.size:
            cand_idx = np.unique(cand_idx)

    x = np.bitwise_xor(np.uint64(th), train_hash_vals_u64[cand_idx])
    d = _popcount_u64(x).astype(np.int16)

    if d.size > topn_neighbors:
        part = np.argpartition(d, topn_neighbors - 1)[:topn_neighbors]
        sel = part[np.argsort(d[part], kind="stable")]
    else:
        sel = np.argsort(d, kind="stable")

    votes = Counter()

    if exact_match_bonus > 0:
        exact_pos = np.where(d == 0)[0]
        if exact_pos.size:
            for j in exact_pos.tolist():
                hid0 = train_hash_hids_py[int(cand_idx[j])]
                bonus = exact_match_bonus
                if test_chain is not None and hotel_to_chain.get(hid0) == test_chain:
                    bonus *= 1.0 + chain_boost
                votes[hid0] += bonus

    for rank_pos, j in enumerate(sel.tolist()):
        dist = int(d[j])
        hid = train_hash_hids_py[int(cand_idx[j])]

        base_w = (
            1.0 / (1.0 + dist) + 0.01 * (topn_neighbors - rank_pos) / topn_neighbors
        )

        if test_chain is not None and hotel_to_chain.get(hid) == test_chain:
            base_w *= 1.0 + chain_boost

        votes[hid] += base_w

    pred = [hid for hid, _ in votes.most_common(k)]

    if len(pred) < k:
        if test_chain is not None and test_chain in chain_top_hotels:
            for hid in chain_top_hotels[test_chain]:
                if hid not in votes:
                    pred.append(hid)
                    if len(pred) >= k:
                        break

    if len(pred) < k:
        for hid in global_top_hotels:
            if hid not in pred:
                pred.append(hid)
            if len(pred) >= k:
                break

    return pred[:k]


def _infer_chain_from_test_path(p: str):
    return None


sample_images = sample["image"].astype(str).tolist()
test_hash_cache = {}
test_chain_cache = {}
missing_paths = 0

test_paths = _build_image_path_index_recursive(test_images_root, max_depth=6)

dhash64_local = dhash64
exists = os.path.exists
os_path_join = os.path.join
test_images_root_local = test_images_root

for img in sample_images:
    p = os_path_join(test_images_root_local, img)
    if not exists(p):
        p = test_paths.get(img)
    if p is None or not exists(p):
        missing_paths += 1
        continue
    test_hash_cache[img] = dhash64_local(p)
    test_chain_cache[img] = _infer_chain_from_test_path(p)

preds = []
append_pred = preds.append
global_top_hotels_local = global_top_hotels
predict_local = predict_for_hash

for img in sample_images:
    th = test_hash_cache.get(img)
    tc = test_chain_cache.get(img)
    if th is None:
        if tc is not None and tc in chain_top_hotels:
            top_hotels = chain_top_hotels[tc][:topk]
            if len(top_hotels) < topk:
                top_hotels = (top_hotels + global_top_hotels_local)[:topk]
        else:
            top_hotels = global_top_hotels_local
    else:
        top_hotels = predict_local(th, k=topk, test_chain=tc)
    append_pred(" ".join(top_hotels))

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
