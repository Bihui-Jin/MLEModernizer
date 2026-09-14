# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.7445482866043597

# 6. Current score

0.00067

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a Kaggle score because it depends on external wheel files and a `peko` package entrypoint that are not present in the provided environment, so it never reliably produces a valid `submission.csv`. To keep the “core logic” minimal while making it run end-to-end, I switch to a deterministic, dependency-free baseline that always generates a correctly formatted submission by using the most frequent `hotel_id` values from `train.csv` as the top-5 predictions for every test image. This won’t be competitive with your target MAP@5, but it unblock scoring and give you a stable starting point to iterate from. The output be written as `submission.csv` with the required `image,hotel_id` columns and 9756 rows aligned to `sample_submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.7445), so we need a real retrieval signal while keeping changes minimal and dependency-free. The smallest legitimate improvement is to replace the “global most frequent hotels” guess with a chain-aware prior: for each test image, predict the top-5 most frequent hotels within the most likely chain for that image, and fall back to global top-5 if the chain is unknown. We can infer the chain of a test image from its folder name under `test_images/` (chains are the subfolders, consistent with how train images are organized), which preserves evaluation semantics and uses only provided data. This keeps the approach simple (frequency baseline) but should move MAP@5 substantially toward your target compared to predicting the same 5 hotels for every image.'
- What this solution (achieved 0.00097) has done: 'The timeout is dominated by two hot spots: (1) repeated PIL image opens/resizes for hashing and (2) per-test-image Python loops computing Hamming distances and vote aggregation. I keep the exact dhash/Hamming/voting logic, but reduce overhead by switching image hashing to OpenCV (much faster decode+resize) with a PIL fallback, by precomputing candidate index lists once, and by vectorizing Hamming distance computation with NumPy bit operations (exactly equivalent). I also avoid rebuilding intermediate Python lists (like `dists`) when we only need the top-N neighbors, using `np.argpartition` + a small final sort, preserving identical neighbor selection semantics. Paths and outputs remain unchanged.'
- What this solution (achieved 0.00029) has done: 'Your current MAP@5 is far below the target, so we need a real (but still minimal) improvement to the existing dHash+Hamming retrieval without changing the overall approach. The biggest issue is bucket selection: using `h & ((1<<16)-1)` makes “neighbor buckets” meaningless (adjacent integers are not similar in Hamming space), so candidate retrieval is noisy; switching to a contiguous-bit prefix bucket makes nearby buckets actually correlated, improving nearest-neighbor quality while preserving the same hashing + distance + voting logic. I also add a small chain-aware prior in the final ranking: if a test image is in a chain subfolder, we boost votes for hotels from that same chain, which uses only provided metadata and keeps semantics intact. All paths stay the same and the notebook still writes a valid `submission.csv`.'
- What this solution (achieved 0.00051) has done: 'Your current score is far below the target, so we need a real retrieval signal improvement while keeping the same dHash→Hamming→vote ranking core intact. The biggest issue is that “neighbor buckets” built by ±delta over a numeric prefix don’t correspond to Hamming-near hashes, so many true neighbors are never considered; I keep the same bucket idea but instead enumerate buckets within Hamming distance 1 of the prefix (flip one prefix bit), which is a minimal change that makes candidate recall much higher. I also add a tiny “same-hotel exact hash prior” by counting hotel IDs among exact-hash matches (distance 0) before the weighted vote, preserving the same voting semantics but stabilizing top-5 when hashes collide appropriately. These changes are localized (candidate generation + vote init) and keep the rest of your pipeline, paths, and submission format unchanged.'
- What this solution (achieved 0.00058) has done: 'Your current MAP@5 is far below the target, so we need to increase retrieval quality without changing the core dHash→Hamming→vote ranking logic. The biggest gap is candidate recall: restricting to Hamming-1 prefix buckets is too tight, so I minimally expand candidate buckets to include prefixes within Hamming distance 2 (still using the same prefix-bucketing idea), which should bring more true neighbors into consideration. I also add a small, safe chain-aware fallback: if we can infer a test image’s chain, we backfill remaining slots from that chain’s most frequent hotels before the global prior, improving top-5 relevance when visual retrieval is weak. Output paths and submission format remain unchanged (`submission.csv`, `image,hotel_id`).'
- What this solution (achieved 0.00096) has done: 'Your MAP@5 is far below the target, so we should improve retrieval quality without changing the core dHash→Hamming→vote ranking approach. The biggest low-risk gain is to increase candidate recall by using a slightly larger train hash index (more unique hotels and more images per hotel) while keeping the same hashing and voting semantics, and to make bucket lookup more robust by falling back to the nearest existing cached bucket when a test prefix wasn’t seen in train. I also expand prefix-neighbor enumeration from Hamming≤2 to Hamming≤3 (still the same “prefix bucketing + neighbor prefixes” logic) which is a localized change that mainly affects candidate generation. These minimal changes should move the score upward toward your target while staying within time limits by keeping all distance computations vectorized.'
- What this solution (achieved 0.00067) has done: 'Your current score is far below the target, so we should increase MAP@5 by improving nearest-neighbor candidate recall while keeping the same dHash→Hamming→vote ranking core unchanged. The lowest-risk issue is that the bucket-neighbor cache is only built for prefixes seen in train; when a test prefix is unseen, we fall back to the numerically “nearest” prefix which is not correlated in Hamming space. I minimally extend the cache to also include Hamming≤3 neighbor-prefix lists for *all* 2^bucket_bits prefixes (4096), so every test image gets a meaningful candidate set without changing hashing, distance, or voting semantics. I also increase `topn_neighbors` slightly (keeping the same vote logic) so the final vote aggregation has more evidence when candidates are noisier.'

# 9. Code solution

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
img_to_chain_local = img_to_chain_from_csv
train_images_root_local = train_images_root
append_train_hashes = train_hashes.append
buckets_local = buckets
bucket_shift_local = bucket_shift

for img, hid in selected_train_imgs:
    chain_folder = img_to_chain_local.get(img)
    if chain_folder is None:
        continue
    p = os_path_join(train_images_root_local, chain_folder, img)
    if not exists(p):
        continue
    h = dhash64_local(p)
    if h is None:
        continue
    idx = len(train_hashes)
    append_train_hashes((h, hid))
    buckets_local[(h >> bucket_shift_local)].append(idx)

print(
    f"Hashed train images: {len(train_hashes)} (from selected {len(selected_train_imgs)})"
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


def predict_for_hash(
    th,
    k=5,
    shortlist_bucket_neighbors=2,  # kept for API compatibility
    topn_neighbors=120,  # Change (score↑, minimal): slightly more NN evidence for voting; same logic.
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
    cand = _bucket_neighbor_cache.get(base, [])

    if not cand:
        step = max(1, len(train_hash_vals_u64) // 6000)
        cand = range(0, len(train_hash_vals_u64), step)

    if isinstance(cand, range):
        cand_idx = np.fromiter(cand, dtype=np.int32)
    else:
        cand_idx = np.asarray(cand, dtype=np.int32)

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


def _build_test_path_index(root):
    out = {}
    for f in os.scandir(root):
        if f.is_file():
            out[f.name] = f.path
        elif f.is_dir():
            for ff in os.scandir(f.path):
                if ff.is_file():
                    out[ff.name] = ff.path
    return out


def _infer_chain_from_test_path(p: str):
    if p is None:
        return None
    parent = os.path.basename(os.path.dirname(p))
    root_base = os.path.basename(os.path.normpath(test_images_root))
    if parent == root_base:
        return None
    return parent


sample_images = sample["image"].astype(str).tolist()
test_hash_cache = {}
test_chain_cache = {}
missing_paths = 0

test_paths = None
dhash64_local = dhash64
exists = os.path.exists
os_path_join = os.path.join
test_images_root_local = test_images_root

for img in sample_images:
    p = os_path_join(test_images_root_local, img)
    if not exists(p):
        if test_paths is None:
            test_paths = _build_test_path_index(test_images_root_local)
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
