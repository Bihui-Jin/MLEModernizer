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

0.7445482866043589

# 6. Current score

0.00186

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a Kaggle score because it relies on private wheel inputs (`pekolib*`) that aren’t present in the provided `/kaggle/input/` tree, so it fails before producing a valid `submission.csv`. To move the score toward the target, the smallest safe change is to (1) make the pipeline self-contained with only standard installed packages and (2) generate a valid baseline submission CSV in the required format by predicting the 5 most frequent `hotel_id`s from `train.csv` for every test image in `sample_submission.csv`. This run end-to-end in the Kaggle environment, produce a valid submission, and yield a non-zero MAP@5 (likely below the target, but it enables iteration toward it). Paths remain on `/kaggle/input/...` and output is written to `submission.csv`.'
- What this solution (achieved 0.00186) has done: 'Your timeout is dominated by per-image PIL open/resize/convert work executed ~9.7k times for test and up to 30k times for train, plus Python-loop overhead in hashing and candidate lookup. I keep the exact hashing/matching logic but make it faster by (1) avoiding repeated function/global lookups, (2) speeding up image decoding/resizing via `Image.reduce()` + `draft()` when possible (equivalent output to current pipeline for 8x8 target), (3) parallelizing hash computation for train/test images with a thread pool (I/O bound), and (4) replacing dictionary bucket-building loops with vectorized numpy sorting/grouping while preserving identical bucket semantics. These changes reduce wall time substantially without changing the algorithm, thresholds, TOPK, or scoring.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

BASE_INPUT = Path("/kaggle/input")

print("Listing /kaggle/input (top-level):")
if BASE_INPUT.exists():
    for p in sorted(BASE_INPUT.iterdir()):
        try:
            sz = p.stat().st_size if p.is_file() else 0
        except Exception:
            sz = 0
        print(
            f" - {p.name}{'/' if p.is_dir() else ''}\t{sz/1024/1024:.2f} MB"
            if p.is_file()
            else f" - {p.name}/"
        )
else:
    raise FileNotFoundError("/kaggle/input not found")




## === cell 1
import pandas as pd

CANDIDATE_ROOTS = [
    Path("/kaggle/input/hotel-id-2021-fgvc8"),
    Path("/kaggle/data/hotel-id-2021-fgvc8"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]


def find_first_existing(rel_path: str) -> Path:
    for r in CANDIDATE_ROOTS:
        p = r / rel_path
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {rel_path} under candidate roots: {CANDIDATE_ROOTS}"
    )


train_csv = find_first_existing("train.csv")
sample_sub_csv = find_first_existing("sample_submission.csv")

print("Using train.csv:", train_csv)
print("Using sample_submission.csv:", sample_sub_csv)

train_df = pd.read_csv(train_csv)
sub_df = pd.read_csv(sample_sub_csv)

required_train_cols = {"image", "hotel_id"}
required_sub_cols = {"image", "hotel_id"}
if not required_train_cols.issubset(train_df.columns):
    raise ValueError(
        f"train.csv missing required columns: {required_train_cols - set(train_df.columns)}"
    )
if not required_sub_cols.issubset(sub_df.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns: {required_sub_cols - set(sub_df.columns)}"
    )

global_top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + global_top5 * 5)[:5]

print("Global fallback top-5 hotel_ids:", global_top5)




## === cell 2
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def find_images_dir(name: str) -> Path:
    candidates = []
    for r in CANDIDATE_ROOTS:
        candidates.append(r / name)
        candidates.append(r / "hotel-id-2021-fgvc8" / name)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find images dir for {name} under {CANDIDATE_ROOTS}"
    )


train_images_dir = find_images_dir("train_images")
test_images_dir = find_images_dir("test_images")

print("Using train_images_dir:", train_images_dir)
print("Using test_images_dir:", test_images_dir)

_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def ahash64(img_path: str) -> np.uint64:
    with Image.open(img_path) as im:
        try:
            im.draft("L", (8, 8))
        except Exception:
            pass
        im = im.convert("L")

        try:
            w, h = im.size
            if w > 64 or h > 64:
                factor = max(1, min(w // 64, h // 64))
                if factor > 1:
                    im = im.reduce(factor)
        except Exception:
            pass

        im = im.resize((8, 8), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)

    m = arr.mean()
    bits = (arr > m).astype(np.uint8).reshape(-1)  # length 64, row-major
    packed = np.packbits(bits, bitorder="big")  # shape (8,)
    return packed.view(">u8")[0].astype(np.uint64)


def hamming_many_u64(hq: np.uint64, hs: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(hs, hq).view(np.uint64)
    xb = x.view(np.uint8).reshape(-1, 8)
    return _POPCNT8[xb].sum(axis=1).astype(np.int16)


train_df = train_df.copy()
train_df["image"] = train_df["image"].astype(str)
train_df["hotel_id"] = train_df["hotel_id"].astype(str)

MAX_TRAIN_HASHES = 30000  # keep identical behavior/limit
train_df_sorted = train_df.sort_values("image").reset_index(drop=True)
train_df_use = train_df_sorted.iloc[
    : min(MAX_TRAIN_HASHES, len(train_df_sorted))
].copy()

train_df_use["chain"] = (
    train_df_sorted.loc[train_df_use.index, "chain"].values
    if "chain" in train_df_sorted.columns
    else 0
)

train_images_dir_str = str(train_images_dir)
chains = train_df_use["chain"].astype(str).values
imgs = train_df_use["image"].values
train_paths = [
    os.path.join(train_images_dir_str, ch, img) for ch, img in zip(chains, imgs)
]
train_df_use["path"] = train_paths
missing_train = sum(0 if os.path.exists(p) else 1 for p in train_paths)
print(f"Train subset size: {len(train_df_use)} (missing paths: {missing_train})")


def _hash_one_train(args):
    p, hid = args
    if not os.path.exists(p):
        return None
    try:
        return (ahash64(p), hid)
    except Exception:
        return None


train_pairs = list(zip(train_df_use["path"].values, train_df_use["hotel_id"].values))
train_hashes_list = []
train_hotels_list = []
bad = 0

max_workers = min(32, (os.cpu_count() or 4) * 2)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = [ex.submit(_hash_one_train, a) for a in train_pairs]
    for fu in as_completed(futures):
        out = fu.result()
        if out is None:
            bad += 1
        else:
            h, hid = out
            train_hashes_list.append(h)
            train_hotels_list.append(hid)

train_hashes = np.array(train_hashes_list, dtype=np.uint64)
train_hotels = np.array(train_hotels_list, dtype=object)
print(f"Computed train hashes: {len(train_hashes)} (skipped: {bad})")

sub_df = sub_df.copy()
sub_df["image"] = sub_df["image"].astype(str)

test_images_dir_str = str(test_images_dir)
alt_test_dir_str = os.path.join(test_images_dir_str, "test_images")
test_paths = []
missing_test = 0
for x in sub_df["image"].values:
    p = os.path.join(test_images_dir_str, x)
    if os.path.exists(p):
        test_paths.append(p)
    else:
        alt = os.path.join(alt_test_dir_str, x)
        if os.path.exists(alt):
            test_paths.append(alt)
        else:
            test_paths.append(p)
            missing_test += 1
print(f"Test images: {len(test_paths)} (missing paths: {missing_test})")

BUCKET_BITS = 16
bucket_shift = np.uint64(64 - BUCKET_BITS)

have_train = len(train_hashes) > 0
if have_train:
    keys = np.right_shift(train_hashes, bucket_shift).astype(np.uint32)
    order = np.argsort(keys, kind="mergesort")  # stable, deterministic
    keys_sorted = keys[order]
    uniq_keys, start_idx, counts = np.unique(
        keys_sorted, return_index=True, return_counts=True
    )
    buckets = {}
    for k, s, c in zip(uniq_keys.tolist(), start_idx.tolist(), counts.tolist()):
        buckets[int(k)] = order[s : s + c].astype(np.int32, copy=False)

    bucket_neighbors = {}
    for key in buckets.keys():
        merged = []
        for kk in (key, key ^ 1, key ^ 2, key ^ 3):
            arr = buckets.get(kk)
            if arr is not None and arr.size:
                merged.append(arr)
        bucket_neighbors[key] = np.concatenate(merged) if merged else None
else:
    buckets = {}
    bucket_neighbors = {}

_fallback_cand = None


def get_candidate_indices(h: np.uint64):
    key = int(np.right_shift(h, bucket_shift))
    cand = bucket_neighbors.get(key)
    if cand is None:
        global _fallback_cand
        if _fallback_cand is None:
            step = max(1, len(train_hashes) // 5000)
            stop = min(len(train_hashes), step * 5000)
            _fallback_cand = np.arange(0, stop, step, dtype=np.int32)
        cand = _fallback_cand
    return cand


TOPK_NEIGHBORS = 50  # identical
preds = []

global_top5_str = " ".join(global_top5)
train_hotels_local = train_hotels
train_hashes_local = train_hashes


def _hash_one_test(p):
    if not os.path.exists(p):
        return None
    try:
        return ahash64(p)
    except Exception:
        return None


test_hashes = [None] * len(test_paths)
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    fut_to_i = {ex.submit(_hash_one_test, p): i for i, p in enumerate(test_paths)}
    for fu in as_completed(fut_to_i):
        i = fut_to_i[fu]
        test_hashes[i] = fu.result()

for hq in test_hashes:
    try:
        if (hq is None) or (not have_train):
            preds.append(global_top5_str)
            continue

        cand_idx = get_candidate_indices(hq)
        cand_hashes = train_hashes_local[cand_idx]

        dvec = hamming_many_u64(hq, cand_hashes)
        order = np.argsort(dvec, kind="mergesort")
        topn = min(TOPK_NEIGHBORS, order.size)
        top_local = order[:topn]

        sel_cand_idx = cand_idx[top_local]
        sel_hotels = train_hotels_local[sel_cand_idx]
        sel_scores = (64 - dvec[top_local]).astype(np.int32)

        uniq_hotels, inv = np.unique(sel_hotels, return_inverse=True)
        summed = np.bincount(inv, weights=sel_scores, minlength=uniq_hotels.shape[0])

        if uniq_hotels.shape[0] > 5:
            top5_idx = np.argpartition(-summed, 5)[:5]
            top5_idx = top5_idx[np.argsort(-summed[top5_idx], kind="mergesort")]
        else:
            top5_idx = np.argsort(-summed, kind="mergesort")

        ordered = [str(uniq_hotels[i]) for i in top5_idx[:5]]

        if len(ordered) < 5:
            for hid in global_top5:
                if hid not in ordered:
                    ordered.append(hid)
                if len(ordered) == 5:
                    break
        preds.append(" ".join(ordered[:5]))
    except Exception:
        preds.append(global_top5_str)

submission = pd.DataFrame({"image": sub_df["image"].values, "hotel_id": preds})

assert submission.shape[0] == sub_df.shape[0]
assert list(submission.columns) == ["image", "hotel_id"]
assert submission["hotel_id"].astype(str).str.split().map(len).eq(5).all()

out_path = Path("submission.csv")
submission.to_csv(out_path, index=False)

print("Wrote:", out_path.resolve())
print(submission.head())
