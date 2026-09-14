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
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/kaggle/input/hotel-id-2021-fgvc8"
train_csv = os.path.join(BASE, "train.csv")
sample_csv = os.path.join(BASE, "sample_submission.csv")
train_img_dir = os.path.join(BASE, "train_images")
test_img_dir = os.path.join(BASE, "test_images")

assert os.path.exists(train_csv), f"Missing train.csv at {train_csv}"
assert os.path.exists(sample_csv), f"Missing sample_submission.csv at {sample_csv}"
assert os.path.isdir(train_img_dir), f"Missing train_images dir at {train_img_dir}"
assert os.path.isdir(test_img_dir), f"Missing test_images dir at {test_img_dir}"

train_df = pd.read_csv(train_csv, usecols=["image", "chain", "hotel_id"])
sample = pd.read_csv(sample_csv)

top5_global = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
pred_str_global = " ".join(top5_global)

print("Global Top-5 prior hotel_ids:", top5_global)
print("Train rows:", len(train_df), "Sample rows:", len(sample))



## === cell 1
import numpy as np
from collections import Counter

_POPCOUNT8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def dhash64(image_path, hash_size=8):
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)

        diff = arr[:, 1:] > arr[:, :-1]  # (8,8) bool
        bits_u8 = diff.reshape(-1).astype(np.uint8, copy=False)
        packed = np.packbits(bits_u8, bitorder="big")  # 8 bytes
        h = np.frombuffer(packed.tobytes(), dtype=">u8")[0].astype(
            np.uint64, copy=False
        )
        return np.uint64(h)
    except Exception:
        return None


def _u64_bytes_view(u64_arr: np.ndarray) -> np.ndarray:
    u64_arr = np.asarray(u64_arr, dtype=np.uint64)
    return u64_arr.view(np.uint8).reshape(-1, 8)


def hamming_u64_vec_uint64_from_bytes(
    train_hash_bytes_u8: np.ndarray, th_uint64: np.uint64
) -> np.ndarray:
    th_u = np.uint64(th_uint64)
    th_bytes = np.frombuffer(th_u.tobytes(), dtype=np.uint8)  # (8,)
    xored = np.bitwise_xor(train_hash_bytes_u8, th_bytes)  # (N,8)
    return _POPCOUNT8[xored].sum(axis=1).astype(np.int16, copy=False)


def train_image_path(chain, image_name):
    return os.path.join(train_img_dir, str(chain), image_name)


def test_image_path(image_name):
    return os.path.join(test_img_dir, image_name)




## === cell 2
from concurrent.futures import ThreadPoolExecutor

MAX_TRAIN_INDEX = 87798  # full train.csv size; deterministic full coverage
K_NEIGHBORS = 50
TOPK_SUBMIT = 5

if len(train_df) > MAX_TRAIN_INDEX:
    train_subset = train_df.iloc[:MAX_TRAIN_INDEX].copy()
else:
    train_subset = train_df.copy()

TRAIN_INDEX_CACHE = "/kaggle/working/train_index_dhash64_full.npz"


def _hash_train_row(tup):
    chain, image_name, hotel_id = tup
    p = train_image_path(chain, image_name)
    h = dhash64(p)
    if h is None:
        return None
    return (np.uint64(h), str(hotel_id))


def _load_train_index_cache(path: str):
    try:
        if not os.path.exists(path):
            return None
        z = np.load(path, allow_pickle=False)
        hashes = z["hashes"].astype(np.uint64, copy=False)
        hotels = z["hotels"].astype("U", copy=False)
        missing = int(z["missing"])
        return hashes, hotels, missing
    except Exception:
        return None


def _save_train_index_cache(
    path: str, hashes: np.ndarray, hotels: np.ndarray, missing: int
):
    np.savez_compressed(
        path,
        hashes=np.asarray(hashes, dtype=np.uint64),
        hotels=np.asarray(hotels, dtype="U"),
        missing=np.int64(missing),
    )


cached = _load_train_index_cache(TRAIN_INDEX_CACHE)
if cached is not None:
    train_hashes, train_hotels, train_missing = cached
    print("Loaded cached train index:", TRAIN_INDEX_CACHE)
else:
    train_tuples = list(
        zip(
            train_subset["chain"].values,
            train_subset["image"].values,
            train_subset["hotel_id"].values,
        )
    )

    max_workers = min(8, (os.cpu_count() or 2))
    train_hashes_list = []
    train_hotels_list = []
    train_missing = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for out in ex.map(_hash_train_row, train_tuples, chunksize=512):
            if out is None:
                train_missing += 1
            else:
                h, hid = out
                train_hashes_list.append(h)
                train_hotels_list.append(hid)

    train_hashes = np.array(train_hashes_list, dtype=np.uint64)
    train_hotels = np.array(train_hotels_list, dtype="U")
    _save_train_index_cache(
        TRAIN_INDEX_CACHE, train_hashes, train_hotels, train_missing
    )
    print("Saved cached train index:", TRAIN_INDEX_CACHE)

assert len(train_hashes) == len(train_hotels)
print("Indexed train images:", len(train_hashes), "Missing/unreadable:", train_missing)

pop_rank = [k for k, _ in Counter(train_hotels.tolist()).most_common(50)]
if len(pop_rank) < 5:
    pop_rank = top5_global[:]

train_hash_bytes = _u64_bytes_view(train_hashes)


def _prefix_keys_from_u64(
    u64_arr: np.ndarray, prefix_bytes: int, offset: int = 0
) -> np.ndarray:
    b = _u64_bytes_view(u64_arr)
    if prefix_bytes == 1:
        return b[:, offset].astype(np.uint16, copy=False)
    elif prefix_bytes == 2:
        return (
            (b[:, offset].astype(np.uint16) << 8) | b[:, offset + 1].astype(np.uint16)
        ).astype(np.uint16, copy=False)
    elif prefix_bytes == 3:
        return (
            (b[:, offset].astype(np.uint32) << 16)
            | (b[:, offset + 1].astype(np.uint32) << 8)
            | b[:, offset + 2].astype(np.uint32)
        ).astype(np.uint32, copy=False)
    else:
        return (
            (b[:, offset].astype(np.uint32) << 24)
            | (b[:, offset + 1].astype(np.uint32) << 16)
            | (b[:, offset + 2].astype(np.uint32) << 8)
            | b[:, offset + 3].astype(np.uint32)
        ).astype(np.uint32, copy=False)


def _build_prefix_buckets_compact_arrays(
    u64_arr: np.ndarray, prefix_bytes: int, offset: int = 0
):
    keys = _prefix_keys_from_u64(u64_arr, prefix_bytes, offset=offset)
    order = np.argsort(keys, kind="mergesort")  # stable/deterministic
    keys_sorted = keys[order]
    uniq, start_idx, counts = np.unique(
        keys_sorted, return_index=True, return_counts=True
    )
    starts = start_idx.astype(np.int32, copy=False)
    ends = (start_idx + counts).astype(np.int32, copy=False)
    return uniq, starts, ends, order.astype(np.int32, copy=False)


PREFIX_BYTES_MAIN = 2
PREFIX_BYTES_FALLBACK = 1

uniq2_front, starts2_front, ends2_front, prefix_order2_front = (
    _build_prefix_buckets_compact_arrays(train_hashes, PREFIX_BYTES_MAIN, offset=0)
)
uniq2_back, starts2_back, ends2_back, prefix_order2_back = (
    _build_prefix_buckets_compact_arrays(train_hashes, PREFIX_BYTES_MAIN, offset=6)
)

uniq1_front, starts1_front, ends1_front, prefix_order1_front = (
    _build_prefix_buckets_compact_arrays(train_hashes, PREFIX_BYTES_FALLBACK, offset=0)
)
uniq1_back, starts1_back, ends1_back, prefix_order1_back = (
    _build_prefix_buckets_compact_arrays(train_hashes, PREFIX_BYTES_FALLBACK, offset=7)
)

print(
    "Prefix buckets built (2 bytes): front",
    len(uniq2_front),
    "| back",
    len(uniq2_back),
    "|| (1 byte): front",
    len(uniq1_front),
    "| back",
    len(uniq1_back),
)

max_workers = min(8, (os.cpu_count() or 2))



## === cell 3
preds = []
test_missing = 0


def predict_from_neighbors(nei_hotels, nei_dists, topk=5):
    if len(nei_hotels) == 0:
        return pred_str_global

    nei_hotels = np.asarray(nei_hotels, dtype="U")
    nei_dists = np.asarray(nei_dists, dtype=np.int16)

    uniq, inv, counts = np.unique(nei_hotels, return_inverse=True, return_counts=True)

    best = np.full(len(uniq), 32767, dtype=np.int16)
    np.minimum.at(best, inv, nei_dists)

    sumd = np.zeros(len(uniq), dtype=np.int32)
    np.add.at(sumd, inv, nei_dists.astype(np.int32, copy=False))

    order = np.lexsort((uniq.astype(str), sumd, -counts, best))
    ranked = uniq[order][:topk].tolist()

    if len(ranked) < topk:
        for h in pop_rank:
            if h not in ranked:
                ranked.append(h)
            if len(ranked) == topk:
                break

    return " ".join(ranked[:topk])


def _list_test_images(test_dir: str):
    names = []
    try:
        for fn in os.listdir(test_dir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                names.append(fn)
    except Exception:
        names = []
    names.sort()  # deterministic
    return names


test_images_list = _list_test_images(test_img_dir)
if len(test_images_list) == 0:
    test_images_list = sample["image"].astype(str).tolist()

print("Test images found:", len(test_images_list))

_TEST_HASH_MEMO = {}


def _hash_test_image(img_name: str):
    h = _TEST_HASH_MEMO.get(img_name, None)
    if h is None and img_name not in _TEST_HASH_MEMO:
        tp = test_image_path(img_name)
        h = dhash64(tp)
        _TEST_HASH_MEMO[img_name] = h
    return img_name, h


_BIT_MASKS8 = (1 << np.arange(8, dtype=np.uint16)).astype(np.uint16)


def _probe_keys_2bytes(b0: int, b1: int) -> np.ndarray:
    v = (np.uint16(b0) << 8) | np.uint16(b1)

    out = np.empty(137, dtype=np.uint16)  # 1 + C(16,1) + C(16,2) = 137
    out[0] = v
    pos = 1

    out[pos : pos + 8] = v ^ (_BIT_MASKS8 << 8)
    pos += 8
    out[pos : pos + 8] = v ^ _BIT_MASKS8
    pos += 8

    k = 0
    for i in range(7):
        mi = np.uint16(1 << i)
        for j in range(i + 1, 8):
            out[pos + k] = v ^ (np.uint16((mi | (1 << j)) << 8))
            k += 1
    pos += k

    k = 0
    for i in range(7):
        mi = np.uint16(1 << i)
        for j in range(i + 1, 8):
            out[pos + k] = v ^ np.uint16(mi | (1 << j))
            k += 1
    pos += k

    cross = (v ^ (_BIT_MASKS8[:, None] << 8) ^ _BIT_MASKS8[None, :]).reshape(-1)
    out[pos : pos + 64] = cross
    pos += 64

    out = np.unique(out)
    return out


def _get_bucket_slice(
    uniq_keys: np.ndarray, starts: np.ndarray, ends: np.ndarray, key: int
):
    i = np.searchsorted(uniq_keys, key)
    if i < uniq_keys.size and int(uniq_keys[i]) == int(key):
        return int(starts[i]), int(ends[i])
    return None


def _gather_candidates(th_u: np.uint64):
    if len(train_hashes) == 0:
        return np.empty(0, dtype=np.int32)

    b = np.frombuffer(np.uint64(th_u).tobytes(), dtype=np.uint8)
    b0f, b1f = int(b[0]), int(b[1])
    b0b, b1b = int(b[6]), int(b[7])

    out_chunks = []
    total = 0
    limit = max(K_NEIGHBORS * 10, 4096)

    def add_from(uniq_keys, starts, ends, order_arr, keys_arr):
        nonlocal total
        for k in keys_arr:
            se = _get_bucket_slice(uniq_keys, starts, ends, int(k))
            if se is None:
                continue
            s, e = se
            if e > s:
                out_chunks.append(order_arr[s:e])
                total += e - s
                if total >= limit:
                    return True
        return False

    add_from(
        uniq2_front,
        starts2_front,
        ends2_front,
        prefix_order2_front,
        _probe_keys_2bytes(b0f, b1f),
    )
    add_from(
        uniq2_back,
        starts2_back,
        ends2_back,
        prefix_order2_back,
        _probe_keys_2bytes(b0b, b1b),
    )

    if total == 0:
        se = _get_bucket_slice(uniq1_front, starts1_front, ends1_front, b0f)
        if se is not None:
            s, e = se
            if e > s:
                return prefix_order1_front[s:e].astype(np.int32, copy=False)
        se = _get_bucket_slice(uniq1_back, starts1_back, ends1_back, b1b)
        if se is not None:
            s, e = se
            if e > s:
                return prefix_order1_back[s:e].astype(np.int32, copy=False)
        return np.empty(0, dtype=np.int32)

    cand_idx = np.concatenate(out_chunks).astype(np.int32, copy=False)
    if cand_idx.size > 1:
        cand_idx = np.unique(cand_idx)
    return cand_idx


with ThreadPoolExecutor(max_workers=max_workers) as ex:
    hashed_iter = ex.map(_hash_test_image, test_images_list, chunksize=256)

    for img_name, th in hashed_iter:
        if th is None or len(train_hashes) == 0:
            test_missing += 1
            preds.append(pred_str_global)
            continue

        th_u = np.uint64(th)
        cand_idx = _gather_candidates(th_u)

        if cand_idx.size > 0:
            dists_c = hamming_u64_vec_uint64_from_bytes(
                train_hash_bytes[cand_idx], th_u
            )
            if len(dists_c) > K_NEIGHBORS:
                kth = K_NEIGHBORS - 1
                idx_local = np.argpartition(dists_c, kth)[:K_NEIGHBORS]
            else:
                idx_local = np.arange(len(dists_c))

            if idx_local.size > 1:
                idx_local = idx_local[np.argsort(dists_c[idx_local], kind="mergesort")]

            idx = cand_idx[idx_local]
            nei_hotels = train_hotels[idx]
            nei_dists = dists_c[idx_local]
            preds.append(
                predict_from_neighbors(nei_hotels, nei_dists, topk=TOPK_SUBMIT)
            )
            continue

        dists = hamming_u64_vec_uint64_from_bytes(train_hash_bytes, th_u)
        if len(dists) > K_NEIGHBORS:
            kth = K_NEIGHBORS - 1
            idx = np.argpartition(dists, kth)[:K_NEIGHBORS]
        else:
            idx = np.arange(len(dists))

        if idx.size > 1:
            idx = idx[np.argsort(dists[idx], kind="mergesort")]

        nei_hotels = train_hotels[idx]
        nei_dists = dists[idx]
        preds.append(predict_from_neighbors(nei_hotels, nei_dists, topk=TOPK_SUBMIT))

print("Test images unreadable:", test_missing)
print("Example prediction:", test_images_list[0], "->", preds[0])



## === cell 4
sub = pd.DataFrame({"image": test_images_list, "hotel_id": preds})

assert list(sub.columns) == [
    "image",
    "hotel_id",
], "Submission must have columns: image, hotel_id"
assert len(sub) == len(
    test_images_list
), "Submission row count must match discovered test images"
assert (
    sub["hotel_id"].astype(str).str.split().map(len).min() >= 1
), "Each prediction must be non-empty"

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())



## === cell 5
import os

assert os.path.exists("submission.csv"), "submission.csv was not created"
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
