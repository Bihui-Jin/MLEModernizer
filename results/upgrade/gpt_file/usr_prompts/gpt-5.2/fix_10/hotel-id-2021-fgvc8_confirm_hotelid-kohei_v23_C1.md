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

0.7712701018775774

# 6. Current score

0.00303

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a Kaggle score because it depends on external wheels (`pekolib*`) that are not present in your provided dataset paths, so the pipeline fails before producing a valid `submission.csv`. To keep the core “retrieval -> top-5 hotel_id string” semantics while making it run end-to-end, I’m replacing the missing dependency with a minimal baseline that always outputs the 5 most frequent `hotel_id` values from `train.csv` for every test image. This generate a correctly-formatted `submission.csv` reliably within the time limit (and typically yields a non-zero MAP@5), allowing you to obtain an initial score and iterate further toward the target. All paths are kept within `/kaggle/input` and `/kaggle/working`, and the output file is exactly `submission.csv` with the required columns.'
- What this solution (achieved 0.00228) has done: 'I fix the runtime error caused by calling `.view(np.uint8)` on a scalar `np.uint64`, by converting hashes to bytes using a safe `np.frombuffer(..., dtype=np.uint8)` approach. I also harden the hash read/write cache so it always stores/loads a true `uint64` scalar, avoiding shape/dtype surprises across runs. These changes are score-neutral (they don’t change the retrieval/voting logic), but they unblock inference so a valid `submission.csv` is produced. Finally, I add a small safety guard so `np.argpartition` is called with a valid `kth` index.'
- What this solution (achieved 0.00303) has done: 'Your current score is far below the target, so we need a genuine but minimal improvement without changing the overall “dHash retrieval + neighbor voting -> top-5 string” approach. The biggest issue is that your candidate gathering only searches nearby numeric prefix keys, which is not meaningful for byte prefixes and often returns weak/empty candidate sets; we instead use a deterministic multi-probe over the first 2 bytes (plus 1-byte fallback) by enumerating small Hamming-radius variants of those bytes, which keeps the same hashing + Hamming distance core but yields much better neighbors. We also switch the ranking to prioritize smallest best-distance first (then counts) to better match “nearest neighbor” retrieval semantics for MAP@5, while keeping the same inputs/outputs. These are targeted changes that should substantially raise MAP@5 toward your target while still finishing within the time limit.'

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


def _load_cached_u64(cp: str):
    try:
        arr = np.load(cp, allow_pickle=False)
        if isinstance(arr, np.ndarray):
            if arr.shape == ():
                return np.uint64(arr.item())
            if arr.size == 1:
                return np.uint64(arr.reshape(-1)[0].item())
        return np.uint64(arr)
    except Exception:
        return None


def _save_cached_u64(cp: str, h: np.uint64):
    np.save(cp, np.array(h, dtype=np.uint64), allow_pickle=False)


def dhash64(image_path, hash_size=8):
    try:
        bname = os.path.basename(image_path)
        prefix = "tr" if (os.sep + "train_images" + os.sep) in image_path else "te"
        cp = _cache_path(prefix, bname)
        if os.path.exists(cp):
            h = _load_cached_u64(cp)
            if h is not None:
                return np.uint64(h)

        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)

        diff = arr[:, 1:] > arr[:, :-1]  # (8,8) bool
        bits_u8 = diff.reshape(-1).astype(np.uint8, copy=False)
        packed = np.packbits(bits_u8, bitorder="big")  # 8 bytes
        h = np.frombuffer(packed.tobytes(), dtype=">u8")[0].astype(
            np.uint64, copy=False
        )

        _save_cached_u64(cp, h)
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
    th_bytes = np.frombuffer(th_u.tobytes(), dtype=np.uint8)  # shape (8,)
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
        int(k): (int(s), int(e))
        for k, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist())
    }
    return buckets, order.astype(np.int32, copy=False)


PREFIX_BYTES_MAIN = 2
PREFIX_BYTES_FALLBACK = 1
prefix_buckets2, prefix_order2 = _build_prefix_buckets_compact(
    train_hashes, PREFIX_BYTES_MAIN
)
prefix_buckets1, prefix_order1 = _build_prefix_buckets_compact(
    train_hashes, PREFIX_BYTES_FALLBACK
)
print(
    "Prefix buckets built:",
    len(prefix_buckets2),
    "prefix_bytes:",
    PREFIX_BYTES_MAIN,
    "| fallback buckets:",
    len(prefix_buckets1),
)


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

    order = np.lexsort((uniq.astype(str), -counts, best))
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


def _prefix_key_from_bytes(b0: int, b1: int) -> int:
    return int((np.uint16(b0) << 8) | np.uint16(b1))


def _probe_keys_2bytes(b0: int, b1: int):
    keys = []
    seen = set()

    def add(k):
        if k not in seen:
            seen.add(k)
            keys.append(k)

    add(_prefix_key_from_bytes(b0, b1))

    for bit in range(8):
        add(_prefix_key_from_bytes(b0 ^ (1 << bit), b1))
    for bit in range(8):
        add(_prefix_key_from_bytes(b0, b1 ^ (1 << bit)))

    for bit0 in range(8):
        for bit1 in range(8):
            add(_prefix_key_from_bytes(b0 ^ (1 << bit0), b1 ^ (1 << bit1)))

    return keys


def _gather_candidates(th_u: np.uint64):
    if len(train_hashes) == 0:
        return np.empty(0, dtype=np.int32)

    b = np.frombuffer(np.uint64(th_u).tobytes(), dtype=np.uint8)
    b0 = int(b[0])
    b1 = int(b[1])

    cand = []
    total = 0
    for k in _probe_keys_2bytes(b0, b1):
        se = prefix_buckets2.get(k)
        if se is None:
            continue
        s, e = se
        if e > s:
            cand.append((s, e))
            total += e - s
            if total >= max(K_NEIGHBORS * 8, 2048):  # cap candidate pool size for speed
                break

    if not cand:
        k1 = b0
        se = prefix_buckets1.get(k1)
        if se is None:
            return np.empty(0, dtype=np.int32)
        s, e = se
        if e <= s:
            return np.empty(0, dtype=np.int32)
        total = e - s
        out = np.empty(total, dtype=np.int32)
        out[:] = prefix_order1[s:e]
        return out

    out = np.empty(total, dtype=np.int32)
    pos = 0
    for s, e in cand:
        n = e - s
        out[pos : pos + n] = prefix_order2[s:e]
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

        if cand_idx.size > 0:
            dists_c = hamming_u64_vec_uint64_from_bytes(
                train_hash_bytes[cand_idx], th_u
            )
            if len(dists_c) > K_NEIGHBORS:
                kth = K_NEIGHBORS - 1
                idx_local = np.argpartition(dists_c, kth)[:K_NEIGHBORS]
            else:
                idx_local = np.arange(len(dists_c))
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
        nei_hotels = train_hotels[idx]
        nei_dists = dists[idx]
        preds.append(predict_from_neighbors(nei_hotels, nei_dists, topk=TOPK_SUBMIT))

print("Test images unreadable:", test_missing)
print("Example prediction:", sample["image"].iloc[0], "->", preds[0])


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


## === cell 5
import os

assert os.path.exists("submission.csv"), "submission.csv was not created"
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
