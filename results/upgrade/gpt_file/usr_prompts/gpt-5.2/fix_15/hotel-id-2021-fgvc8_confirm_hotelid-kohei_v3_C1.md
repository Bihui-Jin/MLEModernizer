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
import math
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/kaggle/input/hotel-id-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

assert {"image", "hotel_id"}.issubset(sub_df.columns)
assert {"image", "hotel_id"}.issubset(train_df.columns)

print("train_df:", train_df.shape, "sub_df:", sub_df.shape)
print("Example train row:", train_df.iloc[0].to_dict())




## === cell 1
train_image_to_path = {}
missing_train_paths = 0

if "chain" in train_df.columns:
    _join = os.path.join
    _exists = os.path.exists
    _train_dir = TRAIN_IMG_DIR
    for img, ch in zip(train_df["image"].values, train_df["chain"].values):
        p = _join(_train_dir, str(ch), img)
        if _exists(p):
            train_image_to_path[img] = p
        else:
            missing_train_paths += 1
else:
    missing_train_paths = len(train_df)

print(
    "Resolved train paths:", len(train_image_to_path), "Missing:", missing_train_paths
)




## === cell 2
_PHASH_BUFS = {}  # keyed by img_size -> float32 (img_size, img_size)
_DCT_MAT = {}  # keyed by N -> float32 (N, N) DCT-II matrix (unnormalized)
_DCT_TMP_F32 = {}  # keyed by N -> float32 (N, N) temp
_DCT_OUT_F32 = {}  # keyed by N -> float32 (N, N) output


def _get_dct_mat(N: int) -> np.ndarray:
    C = _DCT_MAT.get(N)
    if C is None:
        n = np.arange(N, dtype=np.float32)
        k = np.arange(N, dtype=np.float32)[:, None]
        C = np.cos((np.pi / N) * (n[None, :] + 0.5) * k).astype(np.float32)
        _DCT_MAT[N] = C
    return C


def _dct_2d_type2_fast(a_f32: np.ndarray, out_f32: np.ndarray) -> np.ndarray:
    N = a_f32.shape[0]
    C = _get_dct_mat(N)
    tmp = _DCT_TMP_F32.get(N)
    if tmp is None:
        tmp = np.empty((N, N), dtype=np.float32)
        _DCT_TMP_F32[N] = tmp

    np.dot(C, a_f32, out=tmp)
    np.dot(tmp, C.T, out=out_f32)
    return out_f32


def phash(img_path: str, hash_size: int = 16, highfreq_factor: int = 4) -> np.ndarray:
    img_size = hash_size * highfreq_factor  # 64 with defaults

    buf = _PHASH_BUFS.get(img_size)
    if buf is None:
        buf = np.empty((img_size, img_size), dtype=np.float32)
        _PHASH_BUFS[img_size] = buf

    dct_out = _DCT_OUT_F32.get(img_size)
    if dct_out is None:
        dct_out = np.empty((img_size, img_size), dtype=np.float32)
        _DCT_OUT_F32[img_size] = dct_out

    with Image.open(img_path) as im:
        im = im.convert("L").resize((img_size, img_size), Image.BILINEAR)
        buf[:, :] = np.asarray(im, dtype=np.float32)

    dct = _dct_2d_type2_fast(buf, dct_out)
    dct_lowfreq = dct[:hash_size, :hash_size]
    med = np.median(dct_lowfreq[1:, 1:])  # ignore DC component
    bits = (dct_lowfreq > med).astype(np.uint8, copy=False).reshape(-1)
    return bits


def hamming_distance(a_bits: np.ndarray, b_bits: np.ndarray) -> int:
    return int(np.count_nonzero(a_bits != b_bits))




## === cell 3
TRAIN_CAP = 20000  # keep identical cap and semantics

train_subset = train_df.iloc[: min(TRAIN_CAP, len(train_df))].copy()

CACHE_DIR = "/kaggle/working"
CACHE_PATH = os.path.join(CACHE_DIR, f"train_phash_cache_cap{TRAIN_CAP}_hs16_hf4.npz")

train_hashes_packed = None
train_hotels = None
used = 0
skipped = 0

if os.path.exists(CACHE_PATH):
    try:
        data = np.load(CACHE_PATH, allow_pickle=True)
        if "train_hashes_packed" in data:
            train_hashes_packed = data["train_hashes_packed"]
        else:
            train_hashes = data["train_hashes"]
            train_hashes_packed = np.ascontiguousarray(
                np.packbits(train_hashes, axis=1)
            )
        train_hotels = data["train_hotels"]
        used = int(data["used"])
        skipped = int(data["skipped"])
        print(
            "Loaded cached train hashes (packed):",
            train_hashes_packed.shape,
            "used:",
            used,
            "skipped:",
            skipped,
        )
    except Exception as e:
        print("Cache load failed, recomputing:", repr(e))
        train_hashes_packed = None
        train_hotels = None

if train_hashes_packed is None:
    import multiprocessing as mp

    imgs = train_subset["image"].values
    hids = train_subset["hotel_id"].values

    paths = np.empty(len(imgs), dtype=object)
    _get_path = train_image_to_path.get
    for i, img in enumerate(imgs):
        paths[i] = _get_path(img, "")

    out_hotels = np.empty((len(imgs),), dtype=object)
    ok = np.zeros((len(imgs),), dtype=bool)
    packed_tmp = np.empty((len(imgs), 32), dtype=np.uint8)

    def _init_worker():
        _ = _get_dct_mat(64)
        if 64 not in _PHASH_BUFS:
            _PHASH_BUFS[64] = np.empty((64, 64), dtype=np.float32)
        if 64 not in _DCT_OUT_F32:
            _DCT_OUT_F32[64] = np.empty((64, 64), dtype=np.float32)
        if 64 not in _DCT_TMP_F32:
            _DCT_TMP_F32[64] = np.empty((64, 64), dtype=np.float32)

    def _worker_chunk(args):
        start, end = args
        n = end - start
        hps = np.empty((n, 32), dtype=np.uint8)
        hid_s = np.empty((n,), dtype=object)
        oks = np.zeros((n,), dtype=bool)
        for j, i in enumerate(range(start, end)):
            p = paths[i]
            if not p:
                continue
            try:
                h = phash(p)  # (256,) uint8
                hps[j, :] = np.packbits(h)  # (32,) uint8
                hid_s[j] = str(hids[i])
                oks[j] = True
            except Exception:
                pass
        return start, end, hps, hid_s, oks

    nproc = min(8, max(1, (os.cpu_count() or 2) - 1))
    chunk = 1024
    tasks = [(s, min(s + chunk, len(imgs))) for s in range(0, len(imgs), chunk)]

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nproc, initializer=_init_worker) as pool:
        for start, end, hps, hid_s, oks in pool.imap_unordered(
            _worker_chunk, tasks, chunksize=4
        ):
            sl = slice(start, end)
            ok[sl] = oks
            packed_tmp[sl, :] = hps
            out_hotels[sl] = hid_s

    used = int(ok.sum())
    skipped = int(len(ok) - used)

    if used:
        train_hashes_packed = np.ascontiguousarray(packed_tmp[ok])
        train_hotels = out_hotels[ok]
    else:
        train_hashes_packed = np.zeros((0, 32), dtype=np.uint8)
        train_hotels = np.array([], dtype=object)

    try:
        np.savez_compressed(
            CACHE_PATH,
            train_hashes_packed=train_hashes_packed,
            train_hotels=train_hotels,
            used=np.int32(used),
            skipped=np.int32(skipped),
        )
        print("Wrote cache:", CACHE_PATH)
    except Exception as e:
        print("Cache write failed (non-fatal):", repr(e))

print(
    "Train hashes (packed):",
    train_hashes_packed.shape,
    "used:",
    used,
    "skipped:",
    skipped,
)

top_popular = train_df["hotel_id"].astype(str).value_counts().head(5).index.tolist()
fallback_pred = " ".join(top_popular + (["0"] * max(0, 5 - len(top_popular))))
fallback_pred = " ".join(fallback_pred.split()[:5])
print("Fallback pred:", fallback_pred)

_POPCOUNT8 = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)


def _build_bucket_index_vectorized(packed: np.ndarray, key_bytes=(0, 10, 20, 30)):
    if packed.shape[0] == 0:
        return None
    b0, b1, b2, b3 = key_bytes
    keys = (
        (packed[:, b0].astype(np.uint32) << 24)
        | (packed[:, b1].astype(np.uint32) << 16)
        | (packed[:, b2].astype(np.uint32) << 8)
        | packed[:, b3].astype(np.uint32)
    )
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]

    change = np.empty(keys_sorted.shape[0], dtype=bool)
    change[0] = True
    change[1:] = keys_sorted[1:] != keys_sorted[:-1]
    starts = np.flatnonzero(change).astype(np.int32)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = keys_sorted.shape[0]

    uniq_keys = keys_sorted[starts].astype(np.uint32)
    return (uniq_keys, starts, ends, order, tuple(key_bytes))


bucket_struct = _build_bucket_index_vectorized(
    train_hashes_packed, key_bytes=(0, 10, 20, 30)
)




## === cell 4
TEST_CACHE_PATH = os.path.join(CACHE_DIR, "test_phash_cache_hs16_hf4_packed32.npz")
test_hashes_packed = None
test_ok = None
test_imgs = sub_df["image"].values

if os.path.exists(TEST_CACHE_PATH):
    try:
        td = np.load(TEST_CACHE_PATH, allow_pickle=True)
        if (
            "images" in td
            and "hashes_packed" in td
            and len(td["images"]) == len(test_imgs)
        ):
            if np.all(td["images"] == test_imgs):
                test_hashes_packed = td["hashes_packed"]
                test_ok = td["ok"].astype(bool)
                print(
                    "Loaded cached test hashes (packed):",
                    test_hashes_packed.shape,
                    "ok:",
                    int(test_ok.sum()),
                )
    except Exception as e:
        print("Test cache load failed (non-fatal):", repr(e))
        test_hashes_packed = None
        test_ok = None

if test_hashes_packed is None:
    import multiprocessing as mp

    test_hashes_packed = np.empty((len(test_imgs), 32), dtype=np.uint8)
    test_ok = np.zeros((len(test_imgs),), dtype=bool)

    paths = np.empty(len(test_imgs), dtype=object)
    _join = os.path.join
    _exists = os.path.exists
    for i, img in enumerate(test_imgs):
        p = _join(TEST_IMG_DIR, img)
        paths[i] = p if _exists(p) else ""

    def _init_worker_test():
        _ = _get_dct_mat(64)
        if 64 not in _PHASH_BUFS:
            _PHASH_BUFS[64] = np.empty((64, 64), dtype=np.float32)
        if 64 not in _DCT_OUT_F32:
            _DCT_OUT_F32[64] = np.empty((64, 64), dtype=np.float32)
        if 64 not in _DCT_TMP_F32:
            _DCT_TMP_F32[64] = np.empty((64, 64), dtype=np.float32)

    def _worker_test_chunk(args):
        start, end = args
        n = end - start
        hs = np.empty((n, 32), dtype=np.uint8)
        oks = np.zeros((n,), dtype=bool)
        for j, i in enumerate(range(start, end)):
            p = paths[i]
            if not p:
                continue
            try:
                hs[j, :] = np.packbits(phash(p))
                oks[j] = True
            except Exception:
                pass
        return start, end, hs, oks

    nproc = min(8, max(1, (os.cpu_count() or 2) - 1))
    chunk = 1024
    tasks = [(s, min(s + chunk, len(paths))) for s in range(0, len(paths), chunk)]

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nproc, initializer=_init_worker_test) as pool:
        for start, end, hs, oks in pool.imap_unordered(
            _worker_test_chunk, tasks, chunksize=4
        ):
            sl = slice(start, end)
            test_hashes_packed[sl, :] = hs
            test_ok[sl] = oks

    try:
        np.savez_compressed(
            TEST_CACHE_PATH,
            images=test_imgs,
            hashes_packed=test_hashes_packed,
            ok=test_ok.astype(np.uint8),
        )
        print("Wrote test cache:", TEST_CACHE_PATH)
    except Exception as e:
        print("Test cache write failed (non-fatal):", repr(e))




## === cell 5
import multiprocessing as mp

preds = [""] * len(test_imgs)
fail_test = 0

_train_packed = np.ascontiguousarray(train_hashes_packed)
_train_hotels = train_hotels
_top_popular = top_popular
N_train = _train_packed.shape[0]


def _topk5_indices_from_dist(d: np.ndarray) -> np.ndarray:
    if d.shape[0] <= 5:
        return np.argsort(d, kind="mergesort")
    idx = np.argpartition(d, 5)[:5]
    return idx[np.argsort(d[idx], kind="mergesort")]


def _bucket_slice(bucket_struct, key_u32: np.uint32):
    if bucket_struct is None:
        return None
    uniq_keys, starts, ends, order, _ = bucket_struct
    i = np.searchsorted(uniq_keys, key_u32)
    if i < uniq_keys.shape[0] and uniq_keys[i] == key_u32:
        s = int(starts[i])
        e = int(ends[i])
        return s, e, order
    return None


_G = {}


def _init_match_worker(
    train_packed,
    train_hotels,
    bucket_struct,
    popcount8,
    top_popular,
    fallback_pred,
    test_hashes_packed,
    test_ok,
):
    _G["train_packed"] = train_packed
    _G["train_hotels"] = train_hotels
    _G["bucket_struct"] = bucket_struct
    _G["popcount8"] = popcount8
    _G["top_popular"] = top_popular
    _G["fallback_pred"] = fallback_pred
    _G["test_hashes_packed"] = test_hashes_packed
    _G["test_ok"] = test_ok


def _match_chunk(args):
    start, end = args
    out_preds = [""] * (end - start)
    out_failed = np.zeros((end - start,), dtype=bool)

    tp = _G["train_packed"]
    th = _G["train_hotels"]
    pc = _G["popcount8"]
    bs = _G["bucket_struct"]
    fb = _G["fallback_pred"]
    pop = _G["top_popular"]
    tq = _G["test_hashes_packed"]
    tok = _G["test_ok"]

    for j, i in enumerate(range(start, end)):
        if (not bool(tok[i])) or (tp.shape[0] == 0):
            out_preds[j] = fb
            out_failed[j] = True
            continue
        try:
            q_packed = tq[i]  # already packed (32,) uint8

            key = (
                (np.uint32(q_packed[0]) << 24)
                | (np.uint32(q_packed[10]) << 16)
                | (np.uint32(q_packed[20]) << 8)
                | np.uint32(q_packed[30])
            )
            sl = _bucket_slice(bs, key)

            if sl is None:
                x = np.bitwise_xor(tp, q_packed)  # (N,32)
                d = pc[x].sum(axis=1).astype(np.int32, copy=False)
                topk_idx = _topk5_indices_from_dist(d)
            else:
                s, e, order = sl
                cand_idx = order[s:e]
                x = np.bitwise_xor(tp[cand_idx], q_packed)  # (C,32)
                d = pc[x].sum(axis=1).astype(np.int32, copy=False)
                top_local = _topk5_indices_from_dist(d)
                topk_idx = cand_idx[top_local]

            topk_hotels = th[topk_idx].tolist()
            if len(topk_hotels) < 5:
                topk_hotels += pop
            out_preds[j] = " ".join(topk_hotels[:5])
        except Exception:
            out_preds[j] = fb
            out_failed[j] = True

    return start, end, out_preds, out_failed


if len(test_imgs) == 0:
    preds = []
else:
    ctx = mp.get_context("fork")
    nproc = min(8, max(1, (os.cpu_count() or 2) - 1))

    chunk = 1024
    tasks = [
        (s, min(s + chunk, len(test_imgs))) for s in range(0, len(test_imgs), chunk)
    ]

    with ctx.Pool(
        processes=nproc,
        initializer=_init_match_worker,
        initargs=(
            _train_packed,
            _train_hotels,
            bucket_struct,
            _POPCOUNT8,
            _top_popular,
            fallback_pred,
            test_hashes_packed,
            test_ok,
        ),
    ) as pool:
        for start, end, out_preds, out_failed in pool.imap_unordered(
            _match_chunk, tasks, chunksize=4
        ):
            preds[start:end] = out_preds
            fail_test += int(out_failed.sum())

sub_out = sub_df.copy()
sub_out["hotel_id"] = preds

print("Test failures:", fail_test, "Total:", len(sub_out))
print(sub_out.head())

sub_path = "submission.csv"
sub_out.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(open(sub_path, "r", encoding="utf-8").read().splitlines()[:5])
