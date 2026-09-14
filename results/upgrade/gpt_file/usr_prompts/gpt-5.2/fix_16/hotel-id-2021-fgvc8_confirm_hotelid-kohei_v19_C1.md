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
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)

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


HASH_SIZE = 16  # keep identical


def dhash_bits(image_path, hash_size=HASH_SIZE):
    with Image.open(image_path) as im:
        im = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    diff = arr[:, 1:] > arr[:, :-1]  # (hash_size, hash_size) bool
    bits_u8 = diff.reshape(-1).astype(np.uint8, copy=False)
    packed = np.packbits(bits_u8, bitorder="big")
    return packed  # np.uint8 array




## === cell 1
train_df_local = train_df
has_chain = "chain" in train_df_local.columns

hotel_counts = train_df_local["hotel_id"].value_counts()
top_hotels = set(hotel_counts.head(3500).index.tolist())  # keep identical
cand_train = train_df_local[train_df_local["hotel_id"].isin(top_hotels)].copy()

PER_HOTEL_CAP = 30  # was 18

cand_train = cand_train.groupby("hotel_id", group_keys=False).head(PER_HOTEL_CAP)

MAX_TRAIN_HASH = 60000  # was 22000
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

cache_dir = "/kaggle/working"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(
    cache_dir,
    f"train_dhash_cache_h{HASH_SIZE}_top3500_cap{PER_HOTEL_CAP}_max{MAX_TRAIN_HASH}.npz",
)

train_hashes = None
train_hotels = None
bad_train = 0

if os.path.exists(train_cache_path):
    try:
        npz = np.load(train_cache_path, allow_pickle=True)
        train_hashes = npz["hashes"].astype(np.uint8, copy=False)
        train_hotels = npz["hotels"].astype(object, copy=False)
        bad_train = int(npz["bad_train"]) if "bad_train" in npz else 0
        print("Loaded train hash cache:", train_cache_path, "N:", len(train_hashes))
    except Exception as e:
        print("Failed to load cache, recomputing. Error:", repr(e))
        train_hashes = None
        train_hotels = None

if train_hashes is None or train_hotels is None:
    paths_list = cand_train["path"].tolist()
    hotels_list = cand_train["hotel_id"].tolist()
    train_hash_list = []
    train_hash_list_append = train_hash_list.append
    train_hotel_list = []
    train_hotel_list_append = train_hotel_list.append
    bad_train = 0
    for p, hid in zip(paths_list, hotels_list):
        try:
            train_hash_list_append(dhash_bits(p))
            train_hotel_list_append(hid)
        except Exception:
            bad_train += 1

    if len(train_hash_list) > 0:
        train_hashes = np.stack(train_hash_list, axis=0).astype(np.uint8, copy=False)
    else:
        train_hashes = np.zeros((0, (HASH_SIZE * HASH_SIZE) // 8), dtype=np.uint8)
    train_hotels = np.array(train_hotel_list, dtype=object)

    try:
        np.savez_compressed(
            train_cache_path,
            hashes=train_hashes,
            hotels=train_hotels,
            bad_train=np.int32(bad_train),
        )
        print("Saved train hash cache:", train_cache_path)
    except Exception as e:
        print("Failed to save cache (non-fatal):", repr(e))

print(
    "Built train hash index:",
    len(train_hashes),
    "hash_bytes:",
    train_hashes.shape[1] if train_hashes.ndim == 2 else None,
    "bad_train:",
    bad_train,
)

KNN_K = 25  # keep identical
TOPN_OUT = 5  # keep identical

hotel_codes, hotel_uniques = pd.factorize(train_hotels, sort=False)
hotel_codes = hotel_codes.astype(np.int32, copy=False)
n_classes = int(hotel_codes.max()) + 1 if len(hotel_codes) else 0
hotel_uniques = np.asarray(hotel_uniques, dtype=object)

PREFIX3_BYTES = 3
PREFIX2_BYTES = 2
PREFIX1_BYTES = 1
MIN_BUCKET_SIZE = max(4 * KNN_K, 200)

_train_hashes = train_hashes
_train_hashes_len = int(len(train_hashes))
hash_bytes = (HASH_SIZE * HASH_SIZE) // 8

if _train_hashes_len > 0:
    pref3 = (
        (_train_hashes[:, 0].astype(np.uint32) << 16)
        | (_train_hashes[:, 1].astype(np.uint32) << 8)
        | _train_hashes[:, 2].astype(np.uint32)
    )
    pref2 = (_train_hashes[:, 0].astype(np.uint16) << 8) | _train_hashes[:, 1].astype(
        np.uint16
    )
    pref1 = _train_hashes[:, 0].astype(np.uint16)

    def _build_index(keys):
        order = np.argsort(keys, kind="mergesort")
        ks = keys[order]
        uniq, start, cnt = np.unique(ks, return_index=True, return_counts=True)
        return order, uniq, start, cnt

    pref3_order, pref3_uniq, pref3_start, pref3_cnt = _build_index(pref3)
    pref2_order, pref2_uniq, pref2_start, pref2_cnt = _build_index(pref2)
    pref1_order, pref1_uniq, pref1_start, pref1_cnt = _build_index(pref1)

    pref3_slices = {
        int(k): (int(s), int(c))
        for k, s, c in zip(
            pref3_uniq.tolist(), pref3_start.tolist(), pref3_cnt.tolist()
        )
    }
    pref2_slices = {
        int(k): (int(s), int(c))
        for k, s, c in zip(
            pref2_uniq.tolist(), pref2_start.tolist(), pref2_cnt.tolist()
        )
    }
    pref1_slices = {
        int(k): (int(s), int(c))
        for k, s, c in zip(
            pref1_uniq.tolist(), pref1_start.tolist(), pref1_cnt.tolist()
        )
    }
else:
    pref3_order = pref3_uniq = pref3_start = pref3_cnt = None
    pref2_order = pref2_uniq = pref2_start = pref2_cnt = None
    pref1_order = pref1_uniq = pref1_start = pref1_cnt = None
    pref3_slices = {}
    pref2_slices = {}
    pref1_slices = {}


def _bucket_indices_fast(key, order, slice_map):
    sc = slice_map.get(int(key))
    if sc is None:
        return None
    s, c = sc
    return order[s : s + c]


_counts_buf = np.zeros((n_classes,), dtype=np.int16) if n_classes else None
_min_dist_buf = np.empty((n_classes,), dtype=np.int32) if n_classes else None
_wsum_buf = np.zeros((n_classes,), dtype=np.float64) if n_classes else None
_touched = np.empty((KNN_K,), dtype=np.int32)
_touched_n = 0

preds = []
n_hashed = 0
n_fallback = 0

_global_top5 = global_top5
_global_pred_str = global_pred_str
_popcnt8_local = _popcnt8
_hotel_codes = hotel_codes
_hotel_uniques = hotel_uniques
_n_classes = n_classes
_KNN_K = int(KNN_K)
_TOPN_OUT = int(TOPN_OUT)
_test_img_by_name = test_img_by_name
_fix5_list_local = _fix5_list

_INV_DENOM = 1.0
_EPS = 1e-9

xor_buf_full = (
    np.empty((_train_hashes_len, hash_bytes), dtype=np.uint8)
    if _train_hashes_len > 0
    else None
)
xor_buf_sub = (
    np.empty((_KNN_K * 256, hash_bytes), dtype=np.uint8)
    if _train_hashes_len > 0
    else None
)


def _hash_one(img_name):
    p = _test_img_by_name.get(img_name)
    if p is None:
        return img_name, None, False
    try:
        return img_name, dhash_bits(p), True
    except Exception:
        return img_name, None, False


max_workers = min(32, (os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    hashed = list(ex.map(_hash_one, images))

test_hash_by_img = {img: th for img, th, ok in hashed if ok}
test_ok_set = {img for img, th, ok in hashed if ok}

NEAR_DUP_HAMMING_THRESH = 4

for img in images:
    if img not in test_ok_set or _train_hashes_len == 0:
        preds.append(_global_pred_str)
        n_fallback += 1
        continue

    th = test_hash_by_img[img]
    n_hashed += 1

    cand_idx = None
    if _train_hashes_len > 0:
        key3 = (np.uint32(th[0]) << 16) | (np.uint32(th[1]) << 8) | np.uint32(th[2])
        cand_idx = _bucket_indices_fast(key3, pref3_order, pref3_slices)
        if cand_idx is None or cand_idx.size < MIN_BUCKET_SIZE:
            key2 = (np.uint16(th[0]) << 8) | np.uint16(th[1])
            cand_idx = _bucket_indices_fast(key2, pref2_order, pref2_slices)
        if cand_idx is None or cand_idx.size < MIN_BUCKET_SIZE:
            key1 = np.uint16(th[0])
            cand_idx = _bucket_indices_fast(key1, pref1_order, pref1_slices)
        if cand_idx is None or cand_idx.size < MIN_BUCKET_SIZE:
            cand_idx = None

    if cand_idx is None:
        np.bitwise_xor(_train_hashes, th, out=xor_buf_full)
        dists = _popcnt8_local[xor_buf_full].sum(axis=1, dtype=np.uint16)
        k = _KNN_K if _KNN_K < _train_hashes_len else _train_hashes_len
        nn_idx = np.argpartition(dists, k - 1)[:k]
        nn_codes = _hotel_codes[nn_idx]
        nn_dists = dists[nn_idx].astype(np.int32, copy=False)
        min_pos = int(nn_dists.argmin()) if k > 0 else -1
        nearest_hotel = _hotel_uniques[int(nn_codes[min_pos])] if min_pos >= 0 else None
        nearest_dist = int(nn_dists[min_pos]) if min_pos >= 0 else 1_000_000
    else:
        tr_sub = _train_hashes[cand_idx]
        nsub = int(tr_sub.shape[0])
        if xor_buf_sub.shape[0] < nsub:
            xor_buf_sub = np.empty((nsub, hash_bytes), dtype=np.uint8)
        np.bitwise_xor(tr_sub, th, out=xor_buf_sub[:nsub])
        dists_sub = _popcnt8_local[xor_buf_sub[:nsub]].sum(axis=1, dtype=np.uint16)
        k = _KNN_K if _KNN_K < nsub else nsub
        nn_loc = np.argpartition(dists_sub, k - 1)[:k]
        nn_idx = cand_idx[nn_loc]
        nn_codes = _hotel_codes[nn_idx]
        nn_dists = dists_sub[nn_loc].astype(np.int32, copy=False)
        min_pos = int(nn_dists.argmin()) if k > 0 else -1
        nearest_hotel = _hotel_uniques[int(nn_codes[min_pos])] if min_pos >= 0 else None
        nearest_dist = int(nn_dists[min_pos]) if min_pos >= 0 else 1_000_000

    if _n_classes == 0:
        preds.append(_global_pred_str)
        n_fallback += 1
        continue

    _touched_n = 0

    for c in nn_codes:
        c = int(c)
        if _counts_buf[c] == 0 and _wsum_buf[c] == 0.0:
            _touched[_touched_n] = c
            _touched_n += 1
            _min_dist_buf[c] = 1_000_000
        _counts_buf[c] += 1

    for c, d in zip(nn_codes, nn_dists):
        c = int(c)
        d = int(d)
        if d < _min_dist_buf[c]:
            _min_dist_buf[c] = d
        _wsum_buf[c] += 1.0 / (float(d) + _INV_DENOM + _EPS)

    touched_classes = _touched[:_touched_n]
    counts_t = _counts_buf[touched_classes]
    wsum_t = _wsum_buf[touched_classes]
    mind_t = _min_dist_buf[touched_classes]
    order = np.lexsort((mind_t, -wsum_t, -counts_t))
    best_classes = touched_classes[order[:_TOPN_OUT]]
    top_ids = _hotel_uniques[best_classes].tolist()

    out = []
    seen = set()

    if nearest_hotel is not None and nearest_dist <= NEAR_DUP_HAMMING_THRESH:
        out.append(str(nearest_hotel))
        seen.add(str(nearest_hotel))

    for hid in top_ids:
        hid = str(hid)
        if hid not in seen:
            out.append(hid)
            seen.add(hid)
    for hid in _global_top5:
        if len(out) >= _TOPN_OUT:
            break
        hid = str(hid)
        if hid not in seen:
            out.append(hid)
            seen.add(hid)

    preds.append(" ".join(_fix5_list_local(out)))

    for c in touched_classes:
        c = int(c)
        _counts_buf[c] = 0
        _wsum_buf[c] = 0.0

sub = pd.DataFrame({"image": images, "hotel_id": preds})

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(sub), "cols:", list(sub.columns))
print("Hashed test images:", n_hashed, "fallback:", n_fallback)
print(sub.head())
