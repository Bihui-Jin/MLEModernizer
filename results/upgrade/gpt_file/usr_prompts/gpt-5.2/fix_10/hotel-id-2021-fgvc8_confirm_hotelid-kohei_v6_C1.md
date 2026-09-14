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

0.7948219247284642

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your notebook currently can’t yield a score because it depends on private wheel inputs (`pekolib*`) that are not present in your provided `/kaggle/input/` tree, so it fails before producing a valid `submission.csv`. To make it run end-to-end and generate a correct submission file, I remove those missing-wheel installs and replace the external `peko.subs.hotelid.v6` call with a minimal, fully self-contained baseline that reads `sample_submission.csv` and writes a valid `submission.csv` in the required format. This won’t aim for the target MAP@5 yet (since no model is available in the current code), but it unblock scoring so you can report a `current_score`, after which we can make minimal score-improving changes toward the target. Paths are kept within `/kaggle/input` and the working directory, and the output file name remains `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current 0.0 score comes from predicting the same invalid/degenerate hotel list (“0 0 0 0 0”) for every test image, which yields near-zero MAP@5. With minimal change and no external dependencies or model training, we can move substantially toward the target by using a strong frequency baseline: predict the five most common `hotel_id` values from `train.csv` for every test image. This preserves your lightweight “generate submission from CSVs only” core logic, but replaces the constant placeholder with a statistically grounded default that typically scores much higher on this competition. The code below also keeps your robust path fallback and still writes a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current baseline predicts the same global top-5 hotels for every image, which stays far below the target because it ignores the strong “chain” structure in this dataset. To move the score substantially upward while keeping the same lightweight “CSV-only submission generation” core logic (no model training), I switch to a chain-aware frequency baseline: predict the top-5 hotels within each chain, and for each test image infer its chain from the test image’s parent folder name. If a chain can’t be inferred or has too few examples, we fall back to the global top-5. This is a minimal, fast change that usually yields a large MAP@5 gain on Hotel-ID compared with a global-only prior.'
- What this solution (achieved 0.00113) has done: 'Main bottlenecks are (1) Python-level loops over ~12k train images + ~10k test images doing PIL open/resize and (2) an O(N_train) Hamming distance scan per test image (10k × 12k ≈ 120M distances), which is too slow in pure NumPy on CPU. To preserve the exact same dhash + weighted kNN voting logic, the key optimization is to replace the per-test full scan with an exact hash-bucket prefilter: index train hashes by a small prefix (e.g., 16 bits), and for each query only compute Hamming distances against candidates in the same bucket (still exact for those candidates, but much fewer comparisons). Additionally, speed up image decoding by avoiding `.copy()` and repeated allocations, and build the test file map without `os.walk` overhead by directly listing the directory (and only falling back to walk if needed). These changes keep the algorithm semantics (same hash, same distance, same voting) while cutting runtime enough to fit 600s.'

# 9. Code solution

## === cell 0
import os
import random
from collections import Counter
from PIL import Image
import pandas as pd

print("Listing /kaggle/input:")
print("\n".join(os.listdir("/kaggle/input")))

candidate_sample_paths = [
    "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
sample_path = next((p for p in candidate_sample_paths if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any of: {candidate_sample_paths}"
    )

candidate_train_paths = [
    "/kaggle/input/hotel-id-2021-fgvc8/train.csv",
    "/kaggle/input/train.csv",
]
train_path = next((p for p in candidate_train_paths if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(
        f"Could not find train.csv in any of: {candidate_train_paths}"
    )

candidate_train_img_dirs = [
    "/kaggle/input/hotel-id-2021-fgvc8/train_images",
    "/kaggle/input/train_images",
]
train_img_dir = next((p for p in candidate_train_img_dirs if os.path.isdir(p)), None)
if train_img_dir is None:
    raise FileNotFoundError(
        f"Could not find train_images directory in any of: {candidate_train_img_dirs}"
    )

candidate_test_img_dirs = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
]
test_img_dir = next((p for p in candidate_test_img_dirs if os.path.isdir(p)), None)
if test_img_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images directory in any of: {candidate_test_img_dirs}"
    )

sub = pd.read_csv(sample_path)
if list(sub.columns) != ["image", "hotel_id"]:
    raise ValueError(f"Unexpected columns in sample submission: {sub.columns.tolist()}")

train = pd.read_csv(train_path, usecols=["image", "chain", "hotel_id"])

print(f"Using train_images dir: {train_img_dir}")
print(f"Using test_images dir: {test_img_dir}")
print(f"Train shape: {train.shape}, Test(sub) shape: {sub.shape}")




## === cell 1
import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None
    print("WARNING: cv2 not available; will fall back to PIL (slower). Error:", repr(e))


def dhash64_from_gray8(gray8_2d: np.ndarray, hash_size: int = 8) -> int:
    if cv2 is None:
        im = Image.fromarray(gray8_2d, mode="L").resize(
            (hash_size + 1, hash_size), Image.Resampling.BILINEAR
        )
        a = np.asarray(im, dtype=np.uint8)
    else:
        a = cv2.resize(
            gray8_2d,
            (hash_size + 1, hash_size),
            interpolation=cv2.INTER_LINEAR,
        )
        if a.dtype != np.uint8:
            a = a.astype(np.uint8, copy=False)

    diff = (a[:, :-1] > a[:, 1:]).reshape(-1).astype(np.uint8, copy=False)
    bits = np.packbits(diff, bitorder="little")
    return int(np.frombuffer(bits.tobytes(), dtype="<u8")[0])


def safe_open_gray8(path: str):
    if not path:
        return None
    if cv2 is not None:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return g
    try:
        with Image.open(path) as im:
            im.load()
            return np.asarray(im.convert("L"), dtype=np.uint8)
    except Exception:
        return None


global_top5 = train["hotel_id"].value_counts().head(5).index.tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 * 5)[:5]
global_default_pred = " ".join(map(str, global_top5))
print(f"Global top5 hotel_ids: {global_top5}")

SEED = 123
random.seed(SEED)
np.random.seed(SEED)

max_index_images = 88000  # near full train; improves score vs 12k, while still feasible
per_hotel_cap = (
    1000  # effectively "no cap" for most hotels, but avoids extreme domination
)


train_sorted = train.copy()
hotel_counts = train_sorted["hotel_id"].value_counts()
train_sorted["hotel_freq"] = train_sorted["hotel_id"].map(hotel_counts).astype(np.int32)
train_sorted = train_sorted.sort_values(
    ["hotel_freq"], ascending=False, kind="mergesort"
)

train_sorted["_rank_in_hotel"] = train_sorted.groupby("hotel_id", sort=False).cumcount()
train_sorted = train_sorted[train_sorted["_rank_in_hotel"] < per_hotel_cap]

paths = train_sorted["chain"].astype(str).values  # avoid repeated astype in loop
imgs = train_sorted["image"].values
hids = train_sorted["hotel_id"].astype(np.int32).values

img_paths = np.char.add(
    np.char.add(np.char.add(train_img_dir, os.sep), paths), np.char.add(os.sep, imgs)
)
exists_mask = np.fromiter(
    (os.path.exists(p) for p in img_paths), dtype=np.bool_, count=len(img_paths)
)
train_sorted = train_sorted.loc[exists_mask].head(max_index_images)

selected_rows = list(
    zip(
        train_sorted["image"].tolist(),
        train_sorted["chain"].astype(int).tolist(),
        train_sorted["hotel_id"].astype(int).tolist(),
        (
            train_sorted["chain"].astype(str).radd(train_img_dir + os.sep)
            + os.sep
            + train_sorted["image"]
        ).tolist(),
    )
)

print(
    f"Indexed training images: {len(selected_rows)} (max_index_images={max_index_images}, per_hotel_cap={per_hotel_cap})"
)

train_hashes = []
bad_train = 0

_safe_open = safe_open_gray8
_dhash = dhash64_from_gray8

for _, _, hid, path in selected_rows:
    g = _safe_open(path)
    if g is None:
        bad_train += 1
        continue
    train_hashes.append((_dhash(g), int(hid)))
print(f"Train hashes computed: {len(train_hashes)} (failed: {bad_train})")

if len(train_hashes) < 500:
    print(
        "WARNING: Too few train hashes; falling back to global prior for all predictions."
    )
    sub["hotel_id"] = global_default_pred
    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape={sub.shape}")
else:
    k_nn = 50
    preds = []
    bad_test = 0

    th_arr = np.fromiter(
        (h for h, _ in train_hashes), dtype=np.uint64, count=len(train_hashes)
    )
    tid_arr = np.fromiter(
        (hid for _, hid in train_hashes), dtype=np.int32, count=len(train_hashes)
    )

    def hamming_all(qh_uint64: np.uint64, th_uint64_arr: np.ndarray) -> np.ndarray:
        xor = np.bitwise_xor(th_uint64_arr, qh_uint64)
        xb = xor.view(np.uint8).reshape(-1, 8)
        return np.unpackbits(xb, axis=1).sum(axis=1, dtype=np.uint16)

    PREFIX_BITS = 12
    _shift = np.uint64(64 - PREFIX_BITS)
    prefixes = (th_arr >> _shift).astype(np.uint16, copy=False)
    sort_idx = np.argsort(prefixes, kind="mergesort")
    prefixes_sorted = prefixes[sort_idx]
    th_sorted = th_arr[sort_idx]
    tid_sorted = tid_arr[sort_idx]

    max_pfx = 1 << PREFIX_BITS
    starts = np.full(max_pfx, -1, dtype=np.int32)
    ends = np.full(max_pfx, -1, dtype=np.int32)
    uniq_pfx, start_idx = np.unique(prefixes_sorted, return_index=True)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = len(prefixes_sorted)
    starts[uniq_pfx] = start_idx.astype(np.int32, copy=False)
    ends[uniq_pfx] = end_idx.astype(np.int32, copy=False)

    test_path_map = {}
    try:
        for fn in os.listdir(test_img_dir):
            if fn.endswith(".jpg"):
                test_path_map[fn] = os.path.join(test_img_dir, fn)
    except Exception:
        test_path_map = {}

    if len(test_path_map) == 0:
        for root, _, files in os.walk(test_img_dir):
            for fn in files:
                if fn.endswith(".jpg"):
                    test_path_map[fn] = os.path.join(root, fn)

    _th_arr = th_arr
    _tid_arr = tid_arr
    _th_sorted = th_sorted
    _tid_sorted = tid_sorted
    _starts = starts
    _ends = ends

    for img in sub["image"].tolist():
        test_path = test_path_map.get(img, "")
        g = _safe_open(test_path) if test_path else None
        if g is None:
            bad_test += 1
            preds.append(global_default_pred)
            continue

        qh = np.uint64(_dhash(g))
        qprefix = int((qh >> _shift) & np.uint64((1 << PREFIX_BITS) - 1))

        s = int(_starts[qprefix])
        if s == -1:
            cand_th = _th_arr
            cand_tid = _tid_arr
        else:
            e = int(_ends[qprefix])
            if (e - s) < 200:
                cand_th = _th_arr
                cand_tid = _tid_arr
            else:
                cand_th = _th_sorted[s:e]
                cand_tid = _tid_sorted[s:e]

        dists = hamming_all(qh, cand_th)

        if len(dists) > k_nn:
            idx = np.argpartition(dists, k_nn - 1)[:k_nn]
            idx = idx[np.argsort(dists[idx], kind="mergesort")]
        else:
            idx = np.argsort(dists, kind="mergesort")

        nn_hids = cand_tid[idx].astype(np.int64, copy=False)
        nn_d = dists[idx].astype(np.float64, copy=False)
        w = 1.0 / (1.0 + nn_d)

        order = np.argsort(nn_hids, kind="mergesort")
        nn_hids_s = nn_hids[order]
        w_s = w[order]
        uniq, start = np.unique(nn_hids_s, return_index=True)
        sums = np.add.reduceat(w_s, start)

        ranked = uniq[np.argsort(-sums, kind="mergesort")].tolist()
        out5 = (ranked + global_top5)[:5]
        preds.append(" ".join(map(str, out5)))

    sub["hotel_id"] = preds
    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)

    print(f"Bad/unreadable test images (fallback to global): {bad_test} / {len(sub)}")
    print(f"Wrote {out_path} with shape={sub.shape} and columns={sub.columns.tolist()}")
    print(sub.head())




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3756739331.py in <cell line: 0>()
     83 
     84 img_paths = np.char.add(
---> 85     np.char.add(np.char.add(train_img_dir, os.sep), paths), np.char.add(os.sep, imgs)
     86 )
     87 # os.path.exists is not vectorized; use list comprehension in C-accelerated loop over Python,

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U47' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 2
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(6):
        print(f.readline().rstrip("\n"))

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/623693436.py in <cell line: 0>()
----> 1 with open("submission.csv", "r", encoding="utf-8") as f:
      2     for _ in range(6):
      3         print(f.readline().rstrip("\n"))

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
