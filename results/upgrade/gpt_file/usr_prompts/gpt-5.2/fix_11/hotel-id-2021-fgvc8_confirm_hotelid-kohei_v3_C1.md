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
_FFT_WORK = {}  # keyed by N -> complex64 (2N, N) temp for row FFTs
_FFT_WORK2 = {}  # keyed by N -> complex64 (2N, N) temp for col FFTs
_DCT_TMP_F32 = {}  # keyed by N -> float32 (N, N) temp (after row dct)
_DCT_OUT_F32 = {}  # keyed by N -> float32 (N, N) output
_DCT_W = {}  # keyed by N -> complex64 (N,) twiddle exp(-j*pi*k/(2N))


def _get_twiddle(N: int) -> np.ndarray:
    w = _DCT_W.get(N)
    if w is None:
        k = np.arange(N, dtype=np.float32)
        w = np.exp(-1j * np.pi * k / (2.0 * N)).astype(np.complex64)
        _DCT_W[N] = w
    return w


def _dct2_axis0_inplace(
    a_f32: np.ndarray, out_f32: np.ndarray, work_c64: np.ndarray, w_c64: np.ndarray
):
    """
    Compute unnormalized DCT-II along axis=0 for each column independently.

    Uses the standard FFT trick: for length N signal x,
    form y of length 2N: y[:N]=x, y[N:]=x[::-1], then:
    DCT-II(k) = Re( exp(-j*pi*k/(2N)) * FFT(y)[k] ), for k=0..N-1

    This yields an unnormalized DCT-II (same family as the cosine matrix used before).
    """
    N = a_f32.shape[0]
    work_c64[:N, :] = a_f32.astype(np.complex64, copy=False)
    work_c64[N:, :] = a_f32[::-1, :].astype(np.complex64, copy=False)
    F = np.fft.fft(work_c64, axis=0)
    out_f32[:, :] = (F[:N, :] * w_c64[:, None]).real.astype(np.float32, copy=False)
    return out_f32


def _dct_2d_type2_fast(a_f32: np.ndarray, out_f32: np.ndarray) -> np.ndarray:
    N = a_f32.shape[0]

    work1 = _FFT_WORK.get(N)
    if work1 is None:
        work1 = np.empty((2 * N, N), dtype=np.complex64)
        _FFT_WORK[N] = work1

    work2 = _FFT_WORK2.get(N)
    if work2 is None:
        work2 = np.empty((2 * N, N), dtype=np.complex64)
        _FFT_WORK2[N] = work2

    tmp = _DCT_TMP_F32.get(N)
    if tmp is None:
        tmp = np.empty((N, N), dtype=np.float32)
        _DCT_TMP_F32[N] = tmp

    w = _get_twiddle(N)

    _dct2_axis0_inplace(a_f32.T, tmp.T, work1, w)
    _dct2_axis0_inplace(tmp, out_f32, work2, w)
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

train_hashes = None
train_hotels = None
used = 0
skipped = 0

if os.path.exists(CACHE_PATH):
    try:
        data = np.load(CACHE_PATH, allow_pickle=True)
        train_hashes = data["train_hashes"]
        train_hotels = data["train_hotels"]
        used = int(data["used"])
        skipped = int(data["skipped"])
        print(
            "Loaded cached train hashes:",
            train_hashes.shape,
            "used:",
            used,
            "skipped:",
            skipped,
        )
    except Exception as e:
        print("Cache load failed, recomputing:", repr(e))
        train_hashes = None
        train_hotels = None

if train_hashes is None:
    import multiprocessing as mp

    imgs = train_subset["image"].values
    hids = train_subset["hotel_id"].values

    paths = np.empty(len(imgs), dtype=object)
    _get_path = train_image_to_path.get
    for i, img in enumerate(imgs):
        paths[i] = _get_path(img, "")

    out_hashes = np.empty((len(imgs), 256), dtype=np.uint8)
    out_hotels = np.empty((len(imgs),), dtype=object)
    ok = np.zeros((len(imgs),), dtype=bool)

    def _init_worker():
        _ = _get_twiddle(64)
        if 64 not in _PHASH_BUFS:
            _PHASH_BUFS[64] = np.empty((64, 64), dtype=np.float32)
        if 64 not in _DCT_OUT_F32:
            _DCT_OUT_F32[64] = np.empty((64, 64), dtype=np.float32)
        if 64 not in _DCT_TMP_F32:
            _DCT_TMP_F32[64] = np.empty((64, 64), dtype=np.float32)
        if 64 not in _FFT_WORK:
            _FFT_WORK[64] = np.empty((128, 64), dtype=np.complex64)
        if 64 not in _FFT_WORK2:
            _FFT_WORK2[64] = np.empty((128, 64), dtype=np.complex64)

    def _worker(args):
        i, p, hid = args
        if not p:
            return i, None, None, False
        try:
            return i, phash(p), str(hid), True
        except Exception:
            return i, None, None, False

    nproc = min(8, max(1, (os.cpu_count() or 2) - 1))
    chunksize = 128  # CHANGE (timeout fix): larger chunks reduce IPC overhead; results unchanged.

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nproc, initializer=_init_worker) as pool:
        for i, h, hid_s, is_ok in pool.imap_unordered(
            _worker,
            ((i, paths[i], hids[i]) for i in range(len(imgs))),
            chunksize=chunksize,
        ):
            if is_ok:
                out_hashes[i, :] = h
                out_hotels[i] = hid_s
                ok[i] = True

    used = int(ok.sum())
    skipped = int(len(ok) - used)

    if used:
        train_hashes = out_hashes[ok]
        train_hotels = out_hotels[ok]
    else:
        train_hashes = np.zeros((0, 256), dtype=np.uint8)
        train_hotels = np.array([], dtype=object)

    try:
        np.savez_compressed(
            CACHE_PATH,
            train_hashes=train_hashes,
            train_hotels=train_hotels,
            used=np.int32(used),
            skipped=np.int32(skipped),
        )
        print("Wrote cache:", CACHE_PATH)
    except Exception as e:
        print("Cache write failed (non-fatal):", repr(e))

print("Train hashes:", train_hashes.shape, "used:", used, "skipped:", skipped)

top_popular = train_df["hotel_id"].astype(str).value_counts().head(5).index.tolist()
fallback_pred = " ".join(top_popular + (["0"] * max(0, 5 - len(top_popular))))
fallback_pred = " ".join(fallback_pred.split()[:5])
print("Fallback pred:", fallback_pred)

_POPCOUNT8 = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)

if train_hashes.shape[0] > 0:
    train_hashes_packed = np.packbits(train_hashes, axis=1)  # (N, 32) uint8
else:
    train_hashes_packed = np.zeros((0, 32), dtype=np.uint8)


def _build_bucket_index_vectorized(packed: np.ndarray, key_bytes=(0, 10, 20, 30)):
    if packed.shape[0] == 0:
        return None
    b = np.array(key_bytes, dtype=np.int64)
    keys = (
        (packed[:, b[0]].astype(np.uint32) << 24)
        | (packed[:, b[1]].astype(np.uint32) << 16)
        | (packed[:, b[2]].astype(np.uint32) << 8)
        | packed[:, b[3]].astype(np.uint32)
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
TEST_CACHE_PATH = os.path.join(CACHE_DIR, "test_phash_cache_hs16_hf4.npz")
test_hashes = None
test_ok = None
test_imgs = sub_df["image"].values

if os.path.exists(TEST_CACHE_PATH):
    try:
        td = np.load(TEST_CACHE_PATH, allow_pickle=True)
        if "images" in td and "hashes" in td and len(td["images"]) == len(test_imgs):
            if np.all(td["images"] == test_imgs):
                test_hashes = td["hashes"]
                test_ok = td["ok"].astype(bool)
                print(
                    "Loaded cached test hashes:",
                    test_hashes.shape,
                    "ok:",
                    int(test_ok.sum()),
                )
    except Exception as e:
        print("Test cache load failed (non-fatal):", repr(e))
        test_hashes = None
        test_ok = None

if test_hashes is None:
    test_hashes = np.empty((len(test_imgs), 256), dtype=np.uint8)
    test_ok = np.zeros((len(test_imgs),), dtype=bool)

    for i, img in enumerate(test_imgs):
        p = os.path.join(TEST_IMG_DIR, img)
        if not os.path.exists(p):
            continue
        try:
            test_hashes[i, :] = phash(p)
            test_ok[i] = True
        except Exception:
            pass

    try:
        np.savez_compressed(
            TEST_CACHE_PATH,
            images=test_imgs,
            hashes=test_hashes,
            ok=test_ok.astype(np.uint8),
        )
        print("Wrote test cache:", TEST_CACHE_PATH)
    except Exception as e:
        print("Test cache write failed (non-fatal):", repr(e))



## === cell 5
preds = []
fail_test = 0

_packbits = np.packbits
_xor = np.bitwise_xor
_train_packed = np.ascontiguousarray(train_hashes_packed)
_train_hotels = train_hotels
_top_popular = top_popular

N_train = _train_packed.shape[0]

_x_full = np.empty((N_train, 32), dtype=np.uint8) if N_train else None
_d_full = np.empty((N_train,), dtype=np.int32) if N_train else None


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


for idx_img, img in enumerate(test_imgs):
    if N_train == 0 or not bool(test_ok[idx_img]):
        preds.append(fallback_pred)
        fail_test += 1
        continue

    try:
        q = test_hashes[idx_img]  # (256,) uint8
        q_packed = _packbits(q)  # (32,) uint8

        key = (
            (np.uint32(q_packed[0]) << 24)
            | (np.uint32(q_packed[10]) << 16)
            | (np.uint32(q_packed[20]) << 8)
            | np.uint32(q_packed[30])
        )
        sl = _bucket_slice(bucket_struct, key)

        if sl is None:
            np.bitwise_xor(_train_packed, q_packed, out=_x_full)
            np.sum(_POPCOUNT8[_x_full], axis=1, dtype=np.int32, out=_d_full)
            topk_idx = _topk5_indices_from_dist(_d_full)
        else:
            s, e, order = sl
            cand_idx = order[s:e]
            x = _xor(_train_packed[cand_idx], q_packed)  # (C, 32)
            d = _POPCOUNT8[x].sum(axis=1).astype(np.int32, copy=False)  # (C,)
            top_local = _topk5_indices_from_dist(d)
            topk_idx = cand_idx[top_local]

        topk_hotels = _train_hotels[topk_idx].tolist()
        if len(topk_hotels) < 5:
            topk_hotels += _top_popular
        preds.append(" ".join(topk_hotels[:5]))
    except Exception:
        preds.append(fallback_pred)
        fail_test += 1

sub_out = sub_df.copy()
sub_out["hotel_id"] = preds

print("Test failures:", fail_test, "Total:", len(sub_out))
print(sub_out.head())

sub_path = "submission.csv"
sub_out.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(open(sub_path, "r", encoding="utf-8").read().splitlines()[:5])
