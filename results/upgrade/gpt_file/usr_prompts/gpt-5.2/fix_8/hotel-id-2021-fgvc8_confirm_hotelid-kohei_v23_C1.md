# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7712701018775774

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a Kaggle score because it depends on external wheels (`pekolib*`) that are not present in your provided dataset paths, so the pipeline fails before producing a valid `submission.csv`. To keep the core “retrieval -> top-5 hotel_id string” semantics while making it run end-to-end, I’m replacing the missing dependency with a minimal baseline that always outputs the 5 most frequent `hotel_id` values from `train.csv` for every test image. This generate a correctly-formatted `submission.csv` reliably within the time limit (and typically yields a non-zero MAP@5), allowing you to obtain an initial score and iterate further toward the target. All paths are kept within `/kaggle/input` and `/kaggle/working`, and the output file is exactly `submission.csv` with the required columns.'

# 9. Code solution

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
print("Train rows:", len(train_df), "Test rows:", len(sample))




## === cell 1
import numpy as np
from collections import Counter

_POPCOUNT8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)

HASH_CACHE_DIR = "/kaggle/working/dhash_cache"
os.makedirs(HASH_CACHE_DIR, exist_ok=True)


def _cache_path(prefix: str, key: str) -> str:
    sub = key[:2]
    d = os.path.join(HASH_CACHE_DIR, prefix, sub)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.npy")


def dhash64(image_path, hash_size=8):
    try:
        bname = os.path.basename(image_path)
        prefix = "tr" if (os.sep + "train_images" + os.sep) in image_path else "te"
        cp = _cache_path(prefix, bname)
        if os.path.exists(cp):
            arr = np.load(cp, allow_pickle=False)
            return np.uint64(arr[()])

        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
        diff = arr[:, 1:] > arr[:, :-1]  # shape (8,8) of booleans
        bits_u8 = diff.reshape(-1).astype(np.uint8, copy=False)  # row-major flatten
        packed = np.packbits(bits_u8, bitorder="big")  # 8 bytes
        h = np.frombuffer(packed.tobytes(), dtype=">u8")[0].astype(
            np.uint64, copy=False
        )

        np.save(cp, h, allow_pickle=False)
        return h
    except Exception:
        return None


def _u64_bytes_view(u64_arr: np.ndarray) -> np.ndarray:
    return u64_arr.view(np.uint8).reshape(-1, 8)


def hamming_u64_vec_uint64_from_bytes(
    train_hash_bytes_u8: np.ndarray, th_uint64: np.uint64
) -> np.ndarray:
    th_u = np.uint64(th_uint64)
    th_bytes = th_u.view(np.uint8)  # shape (8,)
    xored = np.bitwise_xor(train_hash_bytes_u8, th_bytes)  # (N,8) uint8
    return _POPCOUNT8[xored].sum(axis=1).astype(np.int16, copy=False)


def train_image_path(chain, image_name):
    return os.path.join(train_img_dir, str(chain), image_name)


def test_image_path(image_name):
    return os.path.join(test_img_dir, image_name)




## === cell 2
from concurrent.futures import ThreadPoolExecutor

MAX_TRAIN_INDEX = 50000  # cap for speed/memory; adjust upward if your runtime allows
K_NEIGHBORS = 50  # retrieve this many nearest hashes to vote hotel_ids
TOPK_SUBMIT = 5

train_subset = train_df.iloc[: min(len(train_df), MAX_TRAIN_INDEX)].copy()


def _hash_train_row(tup):
    chain, image_name, hotel_id = tup
    p = train_image_path(chain, image_name)
    h = dhash64(p)
    if h is None:
        return None
    return (np.uint64(h), str(hotel_id))


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
    for out in ex.map(_hash_train_row, train_tuples, chunksize=256):
        if out is None:
            train_missing += 1
        else:
            h, hid = out
            train_hashes_list.append(h)
            train_hotels_list.append(hid)

train_hashes = np.array(train_hashes_list, dtype=np.uint64)
train_hotels = np.array(train_hotels_list, dtype=object)

assert len(train_hashes) == len(train_hotels)
print("Indexed train images:", len(train_hashes), "Missing/unreadable:", train_missing)

pop_rank = [k for k, _ in Counter(train_hotels.tolist()).most_common(50)]
if len(pop_rank) < 5:
    pop_rank = top5_global[:]  # ultimate fallback

train_hash_bytes = _u64_bytes_view(train_hashes)


def _prefix_keys_from_u64(u64_arr: np.ndarray, prefix_bytes: int) -> np.ndarray:
    b = _u64_bytes_view(u64_arr)
    if prefix_bytes == 1:
        return b[:, 0].astype(np.uint16, copy=False)
    elif prefix_bytes == 2:
        return ((b[:, 0].astype(np.uint16) << 8) | b[:, 1].astype(np.uint16)).astype(
            np.uint16, copy=False
        )
    elif prefix_bytes == 3:
        return (
            (b[:, 0].astype(np.uint32) << 16)
            | (b[:, 1].astype(np.uint32) << 8)
            | b[:, 2].astype(np.uint32)
        ).astype(np.uint32, copy=False)
    else:
        return (
            (b[:, 0].astype(np.uint32) << 24)
            | (b[:, 1].astype(np.uint32) << 16)
            | (b[:, 2].astype(np.uint32) << 8)
            | b[:, 3].astype(np.uint32)
        ).astype(np.uint32, copy=False)


def _build_prefix_buckets_compact(u64_arr: np.ndarray, prefix_bytes: int):
    keys = _prefix_keys_from_u64(u64_arr, prefix_bytes)
    order = np.argsort(keys, kind="mergesort")  # stable/deterministic
    keys_sorted = keys[order]

    uniq, start_idx, counts = np.unique(
        keys_sorted, return_index=True, return_counts=True
    )

    starts = start_idx.astype(np.int32, copy=False)
    ends = (start_idx + counts).astype(np.int32, copy=False)

    buckets = {
        int(k): (s, e) for k, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist())
    }
    return buckets, order.astype(np.int32, copy=False)


PREFIX_BYTES_START = 2
prefix_buckets, prefix_order = _build_prefix_buckets_compact(
    train_hashes, PREFIX_BYTES_START
)
print("Prefix buckets built:", len(prefix_buckets), "prefix_bytes:", PREFIX_BYTES_START)




## === cell 3
preds = []
test_missing = 0


def predict_from_neighbors(nei_hotels, nei_dists, topk=5):
    if len(nei_hotels) == 0:
        return pred_str_global

    nei_hotels = np.asarray(nei_hotels, dtype=object)
    nei_dists = np.asarray(nei_dists, dtype=np.int16)

    uniq, inv, counts = np.unique(nei_hotels, return_inverse=True, return_counts=True)

    best = np.full(len(uniq), 32767, dtype=np.int16)
    np.minimum.at(best, inv, nei_dists)

    order = np.lexsort((uniq.astype(str), best, -counts))
    ranked = uniq[order][:topk].tolist()

    if len(ranked) < topk:
        for h in pop_rank:
            if h not in ranked:
                ranked.append(h)
            if len(ranked) == topk:
                break

    return " ".join(ranked[:topk])


test_images_list = sample["image"].astype(str).tolist()


def _hash_test_image(img_name: str):
    tp = test_image_path(img_name)
    return img_name, dhash64(tp)


def _prefix_key_from_hash(th_u: np.uint64, prefix_bytes: int) -> int:
    b = np.uint64(th_u).view(np.uint8)
    if prefix_bytes == 1:
        return int(b[0])
    if prefix_bytes == 2:
        return int((np.uint16(b[0]) << 8) | np.uint16(b[1]))
    if prefix_bytes == 3:
        return int((np.uint32(b[0]) << 16) | (np.uint32(b[1]) << 8) | np.uint32(b[2]))
    return int(
        (np.uint32(b[0]) << 24)
        | (np.uint32(b[1]) << 16)
        | (np.uint32(b[2]) << 8)
        | np.uint32(b[3])
    )


def _gather_candidates(th_u: np.uint64):
    if len(train_hashes) == 0:
        return np.empty(0, dtype=np.int32)

    key0 = _prefix_key_from_hash(th_u, PREFIX_BYTES_START)
    seen = set()

    MAX_VISIT = 4096
    visit = 0
    delta = 0

    slices = []
    total = 0

    while visit < MAX_VISIT and total < K_NEIGHBORS:
        if delta == 0:
            k = key0
            delta = 1
        else:
            k = key0 + delta
            delta = -delta
            if delta > 0:
                delta += 1

        if k in seen:
            continue
        seen.add(k)
        visit += 1

        se = prefix_buckets.get(k)
        if se is not None:
            s, e = se
            if e > s:
                slices.append((s, e))
                total += e - s
                if total >= K_NEIGHBORS:
                    break

    if not slices:
        return np.empty(0, dtype=np.int32)

    out = np.empty(total, dtype=np.int32)
    pos = 0
    for s, e in slices:
        n = e - s
        out[pos : pos + n] = prefix_order[s:e]
        pos += n
    return out


with ThreadPoolExecutor(max_workers=max_workers) as ex:
    hashed_iter = ex.map(_hash_test_image, test_images_list, chunksize=128)

    for img_name, th in hashed_iter:
        if th is None or len(train_hashes) == 0:
            test_missing += 1
            preds.append(pred_str_global)
            continue

        th_u = np.uint64(th)
        cand_idx = _gather_candidates(th_u)

        if cand_idx.size >= K_NEIGHBORS:
            dists_c = hamming_u64_vec_uint64_from_bytes(
                train_hash_bytes[cand_idx], th_u
            )
            idx_local = np.argpartition(dists_c, K_NEIGHBORS)[:K_NEIGHBORS]
            idx = cand_idx[idx_local]
            nei_hotels = train_hotels[idx]
            nei_dists = dists_c[idx_local]
            preds.append(
                predict_from_neighbors(nei_hotels, nei_dists, topk=TOPK_SUBMIT)
            )
            continue

        dists = hamming_u64_vec_uint64_from_bytes(train_hash_bytes, th_u)
        if len(dists) > K_NEIGHBORS:
            idx = np.argpartition(dists, K_NEIGHBORS)[:K_NEIGHBORS]
        else:
            idx = np.arange(len(dists))
        nei_hotels = train_hotels[idx]
        nei_dists = dists[idx]
        preds.append(predict_from_neighbors(nei_hotels, nei_dists, topk=TOPK_SUBMIT))

print("Test images unreadable:", test_missing)
print("Example prediction:", sample["image"].iloc[0], "->", preds[0])




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1573092633.py in <cell line: 0>()
    117 
    118         th_u = np.uint64(th)
--> 119         cand_idx = _gather_candidates(th_u)
    120 
    121         if cand_idx.size >= K_NEIGHBORS:

/tmp/ipykernel_11/1573092633.py in _gather_candidates(th_u)
     59         return np.empty(0, dtype=np.int32)
     60 
---> 61     key0 = _prefix_key_from_hash(th_u, PREFIX_BYTES_START)
     62     seen = set()
     63 

/tmp/ipykernel_11/1573092633.py in _prefix_key_from_hash(th_u, prefix_bytes)
     37 
     38 def _prefix_key_from_hash(th_u: np.uint64, prefix_bytes: int) -> int:
---> 39     b = np.uint64(th_u).view(np.uint8)
     40     if prefix_bytes == 1:
     41         return int(b[0])

ValueError: Changing the dtype of a 0d array is only supported if the itemsize is unchanged

## === cell 4
sub = sample.copy()
sub["hotel_id"] = preds

assert list(sub.columns) == [
    "image",
    "hotel_id",
], "Submission must have columns: image, hotel_id"
assert len(sub) == len(sample), "Submission row count must match sample_submission"
assert (
    sub["hotel_id"].astype(str).str.split().map(len).min() >= 1
), "Each prediction must be non-empty"

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2016397981.py in <cell line: 0>()
      1 sub = sample.copy()
----> 2 sub["hotel_id"] = preds
      3 
      4 assert list(sub.columns) == [
      5     "image",

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (9756)

## === cell 5
import os

assert os.path.exists("submission.csv"), "submission.csv was not created"
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/466289385.py in <cell line: 0>()
      1 import os
      2 
----> 3 assert os.path.exists("submission.csv"), "submission.csv was not created"
      4 with open("submission.csv", "r", encoding="utf-8") as f:
      5     for _ in range(5):

AssertionError: submission.csv was not created
