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

0.7406331565210046

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it depends on private wheel files (`/kaggle/input/pekolib*`) that aren’t present in the provided dataset tree, so it fails before producing a valid `submission.csv`. To get a valid submission end-to-end with minimal, safe changes, I remove those external installs and replace them with a simple, fully self-contained baseline that reads `train.csv`, builds a frequency-based top-5 hotel list, and writes a correctly-formatted `submission.csv`. This won’t reach the target MAP@5, but it unblock scoring and provide a stable baseline that we can incrementally improve afterward. I also make the input path robust by preferring `/kaggle/input/hotel-id-2021-fgvc8/` and falling back to `/kaggle/data/hotel-id-2021-fgvc8/` if needed.'
- What this solution (achieved 0.0014) has done: 'Your current 0.00209 score comes from predicting the same global top-5 hotels for every test image, which is a very weak baseline for MAP@5. With minimal changes and without introducing any new modeling/training loops, we can leverage the provided `chain` metadata by predicting the top-5 hotels *within the most likely chain* for each test image, inferred from the test image’s folder name (the chain subdirectory in `test_images/`). When a test image isn’t in a chain folder or the chain is unknown, we fall back to the global top-5, preserving robustness and submission validity. This keeps the approach “frequency top-k” (same core logic) but makes it conditional, which should substantially increase MAP@5 toward your target.'
- What this solution (achieved 0.0014) has done: 'Your current approach is bottlenecked by `infer_chain_for_image`: it repeatedly scans the entire `test_images` directory for every single test image, which is extremely slow and can lead to timeouts/partial runs and effectively-random fallbacks—both hurting MAP@5. I keep the same “frequency top-5, conditioned on inferred chain, else global top-5” core logic, but make chain inference deterministic and fast by building a one-time `image -> chain` lookup from the `test_images` folder structure. This should materially improve the score toward your target by correctly using chain-specific priors for the full hidden test set. I also make sure the mapping handles both possible layouts (`test_images/<chain>/<image>` and flat `test_images/<image>`) while still writing a valid `submission.csv`.'
- What this solution (achieved 0.00201) has done: 'Your current solution is still essentially predicting chain-conditioned popularity, but the “chain from test folder” signal is wrong for this dataset because test images are not organized into chain subfolders (and the hidden test won’t be either), so most images fall back to chain 0 and you get near-random MAP@5. To move the score toward the target with minimal changes while preserving the same frequency-based core logic, I instead infer a coarse “visual prior” from the filename itself: use the first 2 hex characters of the image id as a bucket, learn top-5 hotels per bucket from train, and predict the bucket’s top-5 for each test image, falling back to global top-5. This keeps the same type of model (top-k frequency lookup + fallback) but makes it actually condition on a signal that exists in both train and test, which should materially increase MAP@5 from ~0.0014. I also keep the existing chain logic intact but make it secondary (only used if chain can be inferred), so the submission remains robust and valid.'
- What this solution (achieved 0.00209) has done: 'The timeout is dominated by expensive filesystem scans (building per-chain filename sets by reading entire directories) and repeated PIL image open/resize operations done serially. I keep the exact hashing + vote logic, but eliminate full directory enumerations by using direct `os.path.exists` checks with a small LRU cache, and build the test chain mapping using direct existence checks rather than scanning directories. I also parallelize hash computation safely using a thread pool (PIL image decode/resize releases the GIL) while keeping determinism (fixed seed and ordered result collection). These changes reduce I/O overhead and utilize available CPU without altering the prediction algorithm or fallbacks.'

# 9. Code solution

## === cell 0
import os
import random
import pandas as pd

BASE_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]
BASE = next((p for p in BASE_CANDIDATES if os.path.exists(p)), None)
print("Using BASE:", BASE)


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


train_csv = find_first_existing(
    [
        os.path.join(BASE, "train.csv") if BASE else None,
        "/kaggle/input/hotel-id-2021-fgvc8/train.csv",
        "/kaggle/data/hotel-id-2021-fgvc8/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)
sample_sub_csv = find_first_existing(
    [
        os.path.join(BASE, "sample_submission.csv") if BASE else None,
        "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv",
        "/kaggle/data/hotel-id-2021-fgvc8/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

TEST_IMG_ROOT = find_first_existing(
    [
        os.path.join(BASE, "test_images") if BASE else None,
        "/kaggle/input/hotel-id-2021-fgvc8/test_images",
        "/kaggle/data/hotel-id-2021-fgvc8/test_images",
        "/kaggle/input/test_images",
        "/kaggle/data/test_images",
    ]
)

TRAIN_IMG_ROOT = find_first_existing(
    [
        os.path.join(BASE, "train_images") if BASE else None,
        "/kaggle/input/hotel-id-2021-fgvc8/train_images",
        "/kaggle/data/hotel-id-2021-fgvc8/train_images",
        "/kaggle/input/train_images",
        "/kaggle/data/train_images",
    ]
)

assert train_csv is not None, "train.csv not found in expected locations."
assert (
    sample_sub_csv is not None
), "sample_submission.csv not found in expected locations."
assert TEST_IMG_ROOT is not None, "test_images folder not found in expected locations."
assert (
    TRAIN_IMG_ROOT is not None
), "train_images folder not found in expected locations."

print("train_csv:", train_csv)
print("sample_sub_csv:", sample_sub_csv)
print("TRAIN_IMG_ROOT:", TRAIN_IMG_ROOT)
print("TEST_IMG_ROOT:", TEST_IMG_ROOT)

train = pd.read_csv(train_csv)
sample_sub = pd.read_csv(sample_sub_csv)

print("train shape:", train.shape)
print("sample_sub shape:", sample_sub.shape)
print("train columns:", train.columns.tolist())
print("sample_sub columns:", sample_sub.columns.tolist())



## === cell 1
from PIL import Image
import numpy as np
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

train2 = train.copy()
train2["hotel_id"] = train2["hotel_id"].astype(str)
train2["chain"] = train2["chain"].fillna(0).astype(int)

global_counts = train2["hotel_id"].value_counts()
global_top5 = global_counts.index[:5].tolist()
ALL_UNIQ_HOTELS = train2["hotel_id"].unique().tolist()

_fallback_stream = global_top5 + ALL_UNIQ_HOTELS


def pad_to_5(hids, fallback):
    out = []
    for x in hids:
        if x not in out:
            out.append(x)
        if len(out) == 5:
            return out
    for x in fallback:
        if x not in out:
            out.append(x)
        if len(out) == 5:
            return out
    for x in ALL_UNIQ_HOTELS:
        if x not in out:
            out.append(x)
        if len(out) == 5:
            return out
    return out


def pad_to_5_fast(hids):
    out = []
    seen = set()
    for x in hids:
        if x not in seen:
            out.append(x)
            seen.add(x)
            if len(out) == 5:
                return out
    for x in _fallback_stream:
        if x not in seen:
            out.append(x)
            seen.add(x)
            if len(out) == 5:
                return out
    return out


chain_top = {}
for ch, grp in train2.groupby("chain", sort=False):
    top = grp["hotel_id"].value_counts().index[:5].tolist()
    chain_top[int(ch)] = pad_to_5(top, global_top5)


def resolve_test_image_path(image_name: str) -> str:
    return os.path.join(TEST_IMG_ROOT, str(image_name))


_Image_open = Image.open
_asarray = np.asarray
_packbits = np.packbits
_BILINEAR = Image.Resampling.BILINEAR


def ahash(image_path: str, hash_size: int = 16):
    try:
        with _Image_open(image_path) as im:
            im = im.convert("L").resize((hash_size, hash_size), _BILINEAR)
            px = _asarray(im, dtype=np.uint8).reshape(-1)
        meanv = float(px.mean())
        bits = (px > meanv).astype(np.uint8, copy=False)
        packed = _packbits(bits, bitorder="big")
        val = int.from_bytes(packed.tobytes(), byteorder="big", signed=False)
        width = (hash_size * hash_size) // 4
        return f"{val:0{width}x}"
    except Exception:
        return None


RANDOM_SEED = 123
random.seed(RANDOM_SEED)

MAX_TRAIN_HASH_IMAGES = 80000  # keep identical core behavior

train_idx = list(range(len(train2)))
random.shuffle(train_idx)
train_idx = train_idx[: min(MAX_TRAIN_HASH_IMAGES, len(train_idx))]

train_img_list = train2["image"].astype(str).tolist()
train_chain_list = train2["chain"].astype(int).tolist()
train_hid_list = train2["hotel_id"].astype(str).tolist()

hash_counts = defaultdict(Counter)
missing_train_images = 0
failed_train_hash = 0

_isfile = os.path.isfile

_G_train_img_list = None
_G_train_chain_list = None
_G_train_hid_list = None
_G_train_root = None


def _init_train_worker(img_list, chain_list, hid_list, train_root):
    global _G_train_img_list, _G_train_chain_list, _G_train_hid_list, _G_train_root
    _G_train_img_list = img_list
    _G_train_chain_list = chain_list
    _G_train_hid_list = hid_list
    _G_train_root = train_root


def _train_one_idx(i: int):
    img = _G_train_img_list[i]
    ch = int(_G_train_chain_list[i])
    hid = _G_train_hid_list[i]
    p = f"{_G_train_root}/{ch}/{img}"
    if not _isfile(p):
        return (None, hid, "missing")
    h = ahash(p, hash_size=16)
    if h is None:
        return (None, hid, "failed")
    return (h, hid, "ok")


cpu = os.cpu_count() or 4
train_workers = min(8, cpu)

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

CHUNK = 2048
with ProcessPoolExecutor(
    max_workers=train_workers,
    initializer=_init_train_worker,
    initargs=(train_img_list, train_chain_list, train_hid_list, TRAIN_IMG_ROOT),
) as ex:
    futures = []
    for s in range(0, len(train_idx), CHUNK):
        futures.append(
            ex.submit(
                lambda idxs: [_train_one_idx(i) for i in idxs], train_idx[s : s + CHUNK]
            )
        )
    for fut in as_completed(futures):
        for h, hid, status in fut.result():
            if status == "missing":
                missing_train_images += 1
                continue
            if status == "failed":
                failed_train_hash += 1
                continue
            hash_counts[h][hid] += 1

hash_top5 = {}
for h, ctr in hash_counts.items():
    top_ids = [k for k, _ in ctr.most_common(5)]
    hash_top5[h] = pad_to_5_fast(top_ids)

print("Train hash stats:")
print("  Used train rows:", len(train_idx))
print("  Unique hashes:", len(hash_top5))
print("  Missing train image files:", missing_train_images)
print("  Failed hash computations:", failed_train_hash)
print("Global top-5 used for fallback:", global_top5)

global_top5_padded = pad_to_5_fast(global_top5)

preds = []
missing_image_hits = 0
missing_hash_hits = 0
failed_test_hash = 0

test_images_arr = sample_sub["image"].astype(str).values


def _test_one(img):
    p = f"{TEST_IMG_ROOT}/{img}"
    if not _isfile(p):
        return (img, None, "missing")
    h = ahash(p, hash_size=16)
    if h is None:
        return (img, None, "failed")
    return (img, h, "ok")


test_workers = min(16, cpu)
with ThreadPoolExecutor(max_workers=test_workers) as ex:
    for img, h, st in ex.map(_test_one, test_images_arr, chunksize=1024):
        if st == "missing":
            missing_image_hits += 1
            h = None
        elif st == "failed":
            failed_test_hash += 1

        top5 = hash_top5.get(h) if h is not None else None
        if top5 is None:
            missing_hash_hits += 1
            top5 = global_top5_padded

        final5 = pad_to_5_fast(top5)
        preds.append(" ".join(final5))

sub = sample_sub.copy()
sub["hotel_id"] = preds

assert list(sub.columns) == [
    "image",
    "hotel_id",
], "Submission columns must be exactly: image, hotel_id"
assert len(sub) == len(sample_sub), "Submission row count must match sample_submission."

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(sub.head())
print("Test-time diagnostics:")
print(
    "  Test images missing from folder path (unexpected in hidden):", missing_image_hits
)
print("  Test hash missing (fallback to global top-5):", missing_hash_hits)
print("  Test hash failed computations:", failed_test_hash)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
_pickle.PicklingError: Can't pickle <function <lambda> at 0x7fffac0fe980>: attribute lookup <lambda> on __main__ failed
"""

The above exception was the direct cause of the following exception:

PicklingError                             Traceback (most recent call last)
/tmp/ipykernel_11/2137066902.py in <cell line: 0>()
    161         )
    162     for fut in as_completed(futures):
--> 163         for h, hid, status in fut.result():
    164             if status == "missing":
    165                 missing_train_images += 1

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/multiprocessing/queues.py in _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem)
    242 
    243                         # serialize the data before acquiring the lock
--> 244                         obj = _ForkingPickler.dumps(obj)
    245                         if wacquire is None:
    246                             send_bytes(obj)

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

PicklingError: Can't pickle <function <lambda> at 0x7fffac0fe980>: attribute lookup <lambda> on __main__ failed
