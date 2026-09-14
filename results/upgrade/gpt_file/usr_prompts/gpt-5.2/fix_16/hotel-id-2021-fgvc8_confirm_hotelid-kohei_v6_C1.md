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

0.7948219247284642

# 6. Current score

0.00196

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your notebook currently can’t yield a score because it depends on private wheel inputs (`pekolib*`) that are not present in your provided `/kaggle/input/` tree, so it fails before producing a valid `submission.csv`. To make it run end-to-end and generate a correct submission file, I remove those missing-wheel installs and replace the external `peko.subs.hotelid.v6` call with a minimal, fully self-contained baseline that reads `sample_submission.csv` and writes a valid `submission.csv` in the required format. This won’t aim for the target MAP@5 yet (since no model is available in the current code), but it unblock scoring so you can report a `current_score`, after which we can make minimal score-improving changes toward the target. Paths are kept within `/kaggle/input` and the working directory, and the output file name remains `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current 0.0 score comes from predicting the same invalid/degenerate hotel list (“0 0 0 0 0”) for every test image, which yields near-zero MAP@5. With minimal change and no external dependencies or model training, we can move substantially toward the target by using a strong frequency baseline: predict the five most common `hotel_id` values from `train.csv` for every test image. This preserves your lightweight “generate submission from CSVs only” core logic, but replaces the constant placeholder with a statistically grounded default that typically scores much higher on this competition. The code below also keeps your robust path fallback and still writes a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current baseline predicts the same global top-5 hotels for every image, which stays far below the target because it ignores the strong “chain” structure in this dataset. To move the score substantially upward while keeping the same lightweight “CSV-only submission generation” core logic (no model training), I switch to a chain-aware frequency baseline: predict the top-5 hotels within each chain, and for each test image infer its chain from the test image’s parent folder name. If a chain can’t be inferred or has too few examples, we fall back to the global top-5. This is a minimal, fast change that usually yields a large MAP@5 gain on Hotel-ID compared with a global-only prior.'
- What this solution (achieved 0.00113) has done: 'Main bottlenecks are (1) Python-level loops over ~12k train images + ~10k test images doing PIL open/resize and (2) an O(N_train) Hamming distance scan per test image (10k × 12k ≈ 120M distances), which is too slow in pure NumPy on CPU. To preserve the exact same dhash + weighted kNN voting logic, the key optimization is to replace the per-test full scan with an exact hash-bucket prefilter: index train hashes by a small prefix (e.g., 16 bits), and for each query only compute Hamming distances against candidates in the same bucket (still exact for those candidates, but much fewer comparisons). Additionally, speed up image decoding by avoiding `.copy()` and repeated allocations, and build the test file map without `os.walk` overhead by directly listing the directory (and only falling back to walk if needed). These changes keep the algorithm semantics (same hash, same distance, same voting) while cutting runtime enough to fit 600s.'
- What this solution (achieved 0.00248) has done: 'The timeout is dominated by (1) building train image paths with a slow row-wise `DataFrame.apply`, (2) hashing up to ~88k training images one-by-one with repeated function lookups and without parallel I/O/decode, and (3) computing Hamming distances via `unpackbits` for every candidate set. I keep the exact same dhash + kNN + weighted-vote logic, but speed it up by vectorizing path construction, parallelizing train-hash computation with a deterministic thread pool (OpenCV releases the GIL during decode/resize), and replacing the Hamming computation with an equivalent POPCNT-based lookup-table on bytes (same distances, much faster than `unpackbits`). I also avoid extra Python overhead in the per-test loop by pre-binding locals and using precomputed arrays, without changing any prediction semantics. All file paths and fallback behavior remain unchanged.'
- What this solution (achieved 0.00115) has done: 'Your current 0.00248 suggests the pipeline is running but the retrieval signal is still extremely weak; the biggest minimal gain (without changing the core dhash+kNN+vote approach) is to stop mixing unrelated hotels by building the kNN search within the *same chain* as the query. We can infer a test image’s chain from its parent folder when available, and otherwise fall back to global search; for train images we already know chain from `train.csv`. This keeps the same hash, same Hamming distance, same kNN size, same weighting/voting—only reduces candidate pollution—so it should move MAP@5 substantially upward toward your target. I also fix the prefix-bucket logic bug where small buckets were incorrectly falling back to the full train set, which defeats the intended prefilter and hurts both speed and quality.'
- What this solution (achieved 0.00115) has done: 'Your current MAP@5 is near-zero because the test-chain inference is effectively always `None` (you join `test_img_dir` with filename, so the parent is `test_images`, not a chain folder), which makes the “same-chain” filtering never trigger and forces a very noisy global retrieval. I make a minimal, metric-relevant fix: infer the test chain from the image filename by looking it up in `train.csv` (since many test images overlap with train in this competition’s setup), and only fall back to folder-based inference if not found. Additionally, I fix the prefix-bucket fallback so it does not revert to the full chain/global set when a bucket is empty; instead it use the chain/global top-5 frequency prior (a safer MAP@5 default than global retrieval on unrelated images). Core logic remains dhash + Hamming kNN + weighted vote; we’re only correcting chain routing and a fallback choice that currently destroys precision.'
- What this solution (achieved 0.00196) has done: 'Your score is far below the target, so we should improve retrieval quality without changing the core dhash+kNN+weighted-vote approach. The biggest minimal fix is that the prefix-bucket prefilter is too strict: when a test hash’s 12-bit prefix has no matches, you fall back to a frequency prior, discarding useful near-prefix neighbors; we instead expand to neighboring prefixes by small Hamming distance on the prefix (same model, just a safer candidate set). Additionally, we stop accidentally excluding most chains from chain_to_index (threshold 200 is too high) and ensure we always include a chain index when it has enough images to be meaningful. These two changes keep the same hash, same distance, same kNN, same voting—only improve candidate selection so MAP@5 moves toward the target.'
- What this solution (achieved 0.00196) has done: 'Your current score (0.00196) is far below the target (0.7948), so we should improve recall/precision without changing the dhash→Hamming kNN→weighted-vote core logic. The main minimal, metric-relevant issue is that test image paths are discovered only at the top of `test_images/` (no recursion unless `os.listdir` fails), which miss almost all hidden-test images when Kaggle provides them in chain subfolders—leading to unreadable images and global-fallback predictions. I make test path discovery always recurse (fast enough for ~10k files) and keep chain inference consistent from the folder name, falling back to `img2chain` when available. This should drastically reduce fallback-to-global and move MAP@5 upward toward your target while preserving the same hashing, distance, kNN, and voting semantics.'

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

try:
    _PIL_BILINEAR = Image.Resampling.BILINEAR  # Pillow>=9
except Exception:
    _PIL_BILINEAR = Image.BILINEAR  # older Pillow


def dhash64_from_gray8(gray8_2d: np.ndarray, hash_size: int = 8) -> int:
    if cv2 is None:
        im = Image.fromarray(gray8_2d, mode="L").resize(
            (hash_size + 1, hash_size), _PIL_BILINEAR
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

chain_top5_map = {}
vc = (
    train.groupby("chain")["hotel_id"]
    .value_counts()
    .rename("cnt")
    .reset_index()
    .sort_values(["chain", "cnt"], ascending=[True, False], kind="mergesort")
)
for c, grp in vc.groupby("chain", sort=False):
    top = grp["hotel_id"].head(5).astype(int).tolist()
    if len(top) < 5:
        top = (top + global_top5)[:5]
    chain_top5_map[int(c)] = top

SEED = 123
random.seed(SEED)
np.random.seed(SEED)

max_index_images = 88000
per_hotel_cap = 1000

train_sorted = train.copy()
hotel_counts = train_sorted["hotel_id"].value_counts()
train_sorted["hotel_freq"] = train_sorted["hotel_id"].map(hotel_counts).astype(np.int32)
train_sorted = train_sorted.sort_values(
    ["hotel_freq"], ascending=False, kind="mergesort"
)

train_sorted["_rank_in_hotel"] = train_sorted.groupby("hotel_id", sort=False).cumcount()
train_sorted = train_sorted[train_sorted["_rank_in_hotel"] < per_hotel_cap]

chain_str = train_sorted["chain"].astype(np.int64).astype(str)
train_sorted["path"] = (
    (train_img_dir.rstrip("/") + "/")
    + chain_str
    + "/"
    + train_sorted["image"].astype(str)
)

exists_mask = train_sorted["path"].map(os.path.exists)
train_sorted = train_sorted.loc[exists_mask].head(max_index_images)

selected_rows = list(
    zip(
        train_sorted["image"].tolist(),
        train_sorted["chain"].astype(int).tolist(),
        train_sorted["hotel_id"].astype(int).tolist(),
        train_sorted["path"].tolist(),
    )
)

print(
    f"Indexed training images: {len(selected_rows)} (max_index_images={max_index_images}, per_hotel_cap={per_hotel_cap})"
)

train_hashes = []
bad_train = 0

_safe_open = safe_open_gray8
_dhash = dhash64_from_gray8

from concurrent.futures import ThreadPoolExecutor


def _hash_one(path_chain_hid):
    path, chain, hid = path_chain_hid
    g = _safe_open(path)
    if g is None:
        return None
    return (_dhash(g), int(chain), int(hid))


paths_chain_hids = [(path, chain, hid) for _, chain, hid, path in selected_rows]

max_workers = min(8, (os.cpu_count() or 2))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    results = list(ex.map(_hash_one, paths_chain_hids, chunksize=64))

for r in results:
    if r is None:
        bad_train += 1
    else:
        train_hashes.append(r)

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
        (h for h, _, _ in train_hashes), dtype=np.uint64, count=len(train_hashes)
    )
    tchain_arr = np.fromiter(
        (c for _, c, _ in train_hashes), dtype=np.int32, count=len(train_hashes)
    )
    tid_arr = np.fromiter(
        (hid for _, _, hid in train_hashes), dtype=np.int32, count=len(train_hashes)
    )

    _POPCNT8 = (
        np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
        .sum(axis=1)
        .astype(np.uint8)
    )

    def hamming_all(qh_uint64: np.uint64, th_uint64_arr: np.ndarray) -> np.ndarray:
        xor = np.bitwise_xor(th_uint64_arr, qh_uint64)
        xb = xor.view(np.uint8).reshape(-1, 8)
        return _POPCNT8[xb].sum(axis=1, dtype=np.uint16)

    PREFIX_BITS = 12
    _shift = np.uint64(64 - PREFIX_BITS)
    _mask_pfx = np.uint64((1 << PREFIX_BITS) - 1)

    def build_prefix_index(hashes_uint64: np.ndarray, ids_int32: np.ndarray):
        prefixes = (hashes_uint64 >> _shift).astype(np.uint16, copy=False)
        sort_idx = np.argsort(prefixes, kind="mergesort")
        prefixes_sorted = prefixes[sort_idx]
        h_sorted = hashes_uint64[sort_idx]
        id_sorted = ids_int32[sort_idx]

        max_pfx = 1 << PREFIX_BITS
        starts = np.full(max_pfx, -1, dtype=np.int32)
        ends = np.full(max_pfx, -1, dtype=np.int32)

        uniq_pfx, start_idx = np.unique(prefixes_sorted, return_index=True)
        end_idx = np.empty_like(start_idx)
        end_idx[:-1] = start_idx[1:]
        end_idx[-1] = len(prefixes_sorted)

        starts[uniq_pfx] = start_idx.astype(np.int32, copy=False)
        ends[uniq_pfx] = end_idx.astype(np.int32, copy=False)

        return h_sorted, id_sorted, starts, ends

    chain_to_index = {}
    unique_chains = np.unique(tchain_arr)
    MIN_CHAIN_INDEX = 50
    for c in unique_chains.tolist():
        m = tchain_arr == c
        if int(m.sum()) < MIN_CHAIN_INDEX:
            continue
        h_c = th_arr[m]
        id_c = tid_arr[m]
        h_sorted_c, id_sorted_c, starts_c, ends_c = build_prefix_index(h_c, id_c)
        chain_to_index[int(c)] = (h_sorted_c, id_sorted_c, starts_c, ends_c)

    th_sorted_g, tid_sorted_g, starts_g, ends_g = build_prefix_index(th_arr, tid_arr)

    img2chain = dict(
        zip(train["image"].astype(str).tolist(), train["chain"].astype(int).tolist())
    )

    test_path_map = {}
    test_chain_map = {}
    for root, _, files in os.walk(test_img_dir):
        parent = os.path.basename(root)
        chain_from_folder = int(parent) if parent.isdigit() else None
        for fn in files:
            if fn.endswith(".jpg"):
                p = os.path.join(root, fn)
                test_path_map[fn] = p
                ch = img2chain.get(fn, None)
                if ch is None:
                    ch = chain_from_folder
                test_chain_map[fn] = ch

    print(f"Discovered test images: {len(test_path_map)} under {test_img_dir}")

    _global_top5 = global_top5
    _global_default_pred = global_default_pred
    _hamming_all = hamming_all
    _argpartition = np.argpartition
    _argsort = np.argsort
    _unique = np.unique
    _reduceat = np.add.reduceat
    _chain_top5_map = chain_top5_map

    def _collect_bucket(h_sorted, id_sorted, starts, ends, pfx: int):
        s = int(starts[pfx])
        if s == -1:
            return None, None
        e = int(ends[pfx])
        return h_sorted[s:e], id_sorted[s:e]

    def _expand_prefixes(pfx: int):
        neigh = [pfx]
        for b in range(PREFIX_BITS):
            neigh.append(pfx ^ (1 << b))
        for b1 in range(PREFIX_BITS):
            for b2 in range(b1 + 1, PREFIX_BITS):
                neigh.append(pfx ^ (1 << b1) ^ (1 << b2))
        seen = set()
        out = []
        for x in neigh:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    TARGET_CANDS = 3000  # keep bounded for speed; still preserves core logic (just candidate selection)

    for img in sub["image"].tolist():
        test_path = test_path_map.get(img, "")
        g = _safe_open(test_path) if test_path else None
        if g is None:
            bad_test += 1
            preds.append(_global_default_pred)
            continue

        qh = np.uint64(_dhash(g))
        qprefix = int((qh >> _shift) & _mask_pfx)

        qchain = test_chain_map.get(img, None)

        if qchain is not None and int(qchain) in chain_to_index:
            h_sorted, id_sorted, starts, ends = chain_to_index[int(qchain)]
            cand_h_list = []
            cand_id_list = []
            total = 0
            for p in _expand_prefixes(qprefix):
                ch, ci = _collect_bucket(h_sorted, id_sorted, starts, ends, p)
                if ch is None:
                    continue
                cand_h_list.append(ch)
                cand_id_list.append(ci)
                total += ch.shape[0]
                if total >= TARGET_CANDS:
                    break
            if total == 0:
                preds.append(
                    " ".join(map(str, _chain_top5_map.get(int(qchain), _global_top5)))
                )
                continue
            cand_th = np.concatenate(cand_h_list, axis=0)
            cand_tid = np.concatenate(cand_id_list, axis=0)
        else:
            cand_h_list = []
            cand_id_list = []
            total = 0
            for p in _expand_prefixes(qprefix):
                ch, ci = _collect_bucket(th_sorted_g, tid_sorted_g, starts_g, ends_g, p)
                if ch is None:
                    continue
                cand_h_list.append(ch)
                cand_id_list.append(ci)
                total += ch.shape[0]
                if total >= TARGET_CANDS:
                    break
            if total == 0:
                preds.append(_global_default_pred)
                continue
            cand_th = np.concatenate(cand_h_list, axis=0)
            cand_tid = np.concatenate(cand_id_list, axis=0)

        dists = _hamming_all(qh, cand_th)

        if len(dists) > k_nn:
            idx = _argpartition(dists, k_nn - 1)[:k_nn]
            idx = idx[_argsort(dists[idx], kind="mergesort")]
        else:
            idx = _argsort(dists, kind="mergesort")

        nn_hids = cand_tid[idx].astype(np.int64, copy=False)
        nn_d = dists[idx].astype(np.float64, copy=False)
        w = 1.0 / (1.0 + nn_d)

        order = _argsort(nn_hids, kind="mergesort")
        nn_hids_s = nn_hids[order]
        w_s = w[order]
        uniq, start = _unique(nn_hids_s, return_index=True)
        sums = _reduceat(w_s, start)

        ranked = uniq[_argsort(-sums, kind="mergesort")].tolist()
        out5 = (ranked + _global_top5)[:5]
        preds.append(" ".join(map(str, out5)))

    sub["hotel_id"] = preds
    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)

    print(f"Bad/unreadable test images (fallback to global): {bad_test} / {len(sub)}")
    print(f"Wrote {out_path} with shape={sub.shape} and columns={sub.columns.tolist()}")
    print(sub.head())



## === cell 2
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(6):
        print(f.readline().rstrip("\n"))
