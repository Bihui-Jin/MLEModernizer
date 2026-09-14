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

0.7449797928769869

# 6. Current score

0.00207

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently cannot yield a Kaggle score because it depends on private Kaggle Dataset wheels (`/kaggle/input/pekolib*`) that are not present in your environment, so it fails before producing a valid `submission.csv`. I keep the “generate a submission.csv from the provided data” core goal, but replace the missing-package calls with a minimal, fully self-contained baseline that always runs using only the provided `train.csv` + `sample_submission.csv`. To move score upward from “not yielded” toward your target, the baseline predict the global top-5 most frequent `hotel_id` values from the training metadata (a common safe baseline for MAP@5) and write the correctly formatted submission file. This keeps runtime well under 600s and ensures the submission rows align exactly with `sample_submission.csv`.'
- What this solution (achieved 0.00209) has done: 'The timeout is dominated by `infer_chain_from_fs`, which for every test image repeatedly scans all subdirectories under `test_images_root` and does `os.path.exists` checks—an O(N_images × N_dirs) filesystem walk. We can preserve identical prediction logic by building a one-time mapping from `image_name -> chain` by scanning the directory tree once, then doing O(1) lookups per image. We also speed up the `chain_top5` computation by replacing the Python `apply(lambda ...)` with a vectorized `groupby().head(5)` pattern that produces the same top-5-per-chain list. All outputs/paths and fallback behavior remain the same.'
- What this solution (achieved 0.00243) has done: 'The timeout is dominated by repeatedly opening and hashing many train images (up to 12k) serially with PIL, plus per-image filesystem `exists()` checks and per-test-image nearest-neighbor scans. To keep the same hashing logic and decision rules, I (1) avoid per-row `os.path.exists` by building paths from a precomputed set of existing files, (2) parallelize the train-image hashing with a thread pool (I/O-bound due to image decode), (3) replace the per-test-image chunk loop for nearest neighbor with one vectorized popcount over all keys (same exact min-distance result), and (4) ensure caches/dtypes are efficient and deterministic. These changes keep identical core semantics (same aHash, same thresholds, same top-5 construction) while substantially reducing wall time.'
- What this solution (achieved 0.00207) has done: 'Your current score (0.00243) is far below the target (0.74498), so we need a real lift while keeping the same overall approach (chain-prior + hash-based chain inference + top-5 completion). The biggest minimal win is to stop arbitrarily truncating the hash index to the first 12k training rows (which are timestamp/CSV-order biased) and instead build the same hash->chain index from a larger, more representative sample spread across chains. I keep the same aHash, same nearest-neighbor rule and threshold, and the same chain_top5/global fallback logic, but increase the hashing cap moderately and sample deterministically per-chain to improve chain inference coverage without changing model semantics. This should move MAP@5 upward substantially while staying within the 600s budget by keeping parallel hashing and avoiding expensive filesystem operations.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

DATA_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str):
    for base in DATA_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in DATA_CANDIDATES:
        path = os.path.join(base, "hotel-id-2021-fgvc8", filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {DATA_CANDIDATES}"
    )


train_path = find_file("train.csv")
sub_path = find_file("sample_submission.csv")

print("Using train:", train_path)
print("Using sample_submission:", sub_path)

train = pd.read_csv(train_path)
sample_sub = pd.read_csv(sub_path)

train.head(), sample_sub.head(), train.shape, sample_sub.shape



## === cell 1
import os

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
global_pred_str = " ".join(global_top5)
print("Global top-5 fallback:", global_pred_str)

_chain_hotel_counts = (
    train.groupby(["chain", "hotel_id"], sort=False).size().rename("cnt").reset_index()
)
_chain_hotel_counts = _chain_hotel_counts.sort_values(
    ["chain", "cnt"], ascending=[True, False], kind="mergesort"
)
_chain_top = _chain_hotel_counts.groupby("chain", sort=False).head(5)
chain_top5 = (
    _chain_top.groupby("chain", sort=False)["hotel_id"]
    .apply(lambda s: s.astype(str).tolist())
    .to_dict()
)

TRAIN_IMAGES_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8/train_images",
    "/kaggle/data/hotel-id-2021-fgvc8/train_images",
    "/kaggle/input/train_images",
    "/kaggle/data/train_images",
    "/kaggle/input/hotel-id-2021-fgvc8/train_images/train_images",
    "/kaggle/data/hotel-id-2021-fgvc8/train_images/train_images",
]
TEST_IMAGES_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/data/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
    "/kaggle/input/hotel-id-2021-fgvc8/test_images/test_images",
    "/kaggle/data/hotel-id-2021-fgvcvc8/test_images/test_images",
    "/kaggle/data/hotel-id-2021-fgvc8/test_images/test_images",
    "/kaggle/input/test_images/test_images",
    "/kaggle/data/test_images/test_images",
]

train_images_root = None
for p in TRAIN_IMAGES_CANDIDATES:
    if os.path.isdir(p):
        train_images_root = p
        break

test_images_root = None
for p in TEST_IMAGES_CANDIDATES:
    if os.path.isdir(p):
        test_images_root = p
        break

print("Using train_images root:", train_images_root)
print("Using test_images root:", test_images_root)

from PIL import Image
import numpy as np


def _ahash_64(image_path: str):
    """64-bit average hash (8x8) -> Python int; returns None if unreadable."""
    try:
        with Image.open(image_path) as im:
            im = im.convert("L").resize((8, 8), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32).reshape(-1)
        mean = float(arr.mean())
        bits = (arr > mean).astype(np.uint8)
        return int(np.packbits(bits, bitorder="big").view(np.uint64)[0])
    except Exception:
        return None


def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


HASH_TRAIN_CAP = (
    60000  # moderate increase to improve coverage; still feasible with threading
)
PER_CHAIN_CAP = (
    800  # limit per chain to spread coverage across chains deterministically
)

_hash_to_chain_counts = {}
_hash_ready = False

_hash_keys_u64 = None
_hash_keys_py = None
_popcount_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_u64_vectorized(x_u64: np.ndarray) -> np.ndarray:
    """Exact popcount for uint64 array using byte LUT; deterministic and fast."""
    xb = x_u64.view(np.uint8).reshape(-1, 8)
    return _popcount_lut[xb].sum(axis=1).astype(np.int16)


def _build_existing_train_files_map(train_root: str):
    chain_to_files = {}
    try:
        for ch in os.listdir(train_root):
            ch_dir = os.path.join(train_root, ch)
            if not os.path.isdir(ch_dir):
                continue
            try:
                chain_to_files[ch] = set(os.listdir(ch_dir))
            except Exception:
                continue
    except Exception:
        return {}
    return chain_to_files


from concurrent.futures import ThreadPoolExecutor

if train_images_root is not None and test_images_root is not None:
    try:
        train2 = train.loc[:, ["image", "chain"]].copy()
        train2["image"] = train2["image"].astype(str)
        train2["chain"] = train2["chain"].astype(int)

        train2 = (
            train2.sort_values(["chain", "image"], kind="mergesort")
            .groupby("chain", sort=False)
            .head(PER_CHAIN_CAP)
            .reset_index(drop=True)
        )

        if len(train2) > HASH_TRAIN_CAP:
            train2 = train2.iloc[:HASH_TRAIN_CAP].copy()

        train2["chain_str"] = train2["chain"].astype(str)

        existing_map = _build_existing_train_files_map(train_images_root)

        if existing_map:
            exists_mask = [
                (cs in existing_map) and (img in existing_map[cs])
                for cs, img in zip(
                    train2["chain_str"].tolist(), train2["image"].tolist()
                )
            ]
            train2 = train2.loc[exists_mask].reset_index(drop=True)
        else:
            train2["path"] = (
                train_images_root.rstrip("/")
                + "/"
                + train2["chain_str"]
                + "/"
                + train2["image"]
            )
            exists_mask = train2["path"].map(os.path.exists)
            train2 = train2[exists_mask].reset_index(drop=True)

        if "path" not in train2.columns:
            train2["path"] = (
                train_images_root.rstrip("/")
                + "/"
                + train2["chain_str"]
                + "/"
                + train2["image"]
            )

        def _hash_one(path_and_chain):
            path, chain_val = path_and_chain
            h = _ahash_64(path)
            return h, int(chain_val)

        paths = train2["path"].tolist()
        chains = train2["chain"].tolist()

        max_workers = min(32, (os.cpu_count() or 4) * 2)

        n_hashed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for h, ch in ex.map(_hash_one, zip(paths, chains), chunksize=64):
                if h is None:
                    continue
                d = _hash_to_chain_counts.get(h)
                if d is None:
                    d = {}
                    _hash_to_chain_counts[h] = d
                d[ch] = d.get(ch, 0) + 1
                n_hashed += 1

        _hash_ready = n_hashed > 0
        if _hash_ready:
            _hash_keys_py = list(_hash_to_chain_counts.keys())
            _hash_keys_u64 = np.array(_hash_keys_py, dtype=np.uint64)

        print(
            f"Built train hash index: hashed {n_hashed} train images (cap={HASH_TRAIN_CAP}, per_chain={PER_CHAIN_CAP})."
        )
    except Exception as e:
        print(
            "WARNING: hash index build failed; falling back to global-only predictions. Error:",
            repr(e),
        )
        _hash_ready = False
else:
    print(
        "WARNING: train_images/test_images roots not found; falling back to global-only predictions."
    )

_test_hash_cache = {}


def infer_chain_for_test_image(image_name: str):
    """Infer chain via hash NN over indexed train hashes; returns int chain or None."""
    if not _hash_ready or test_images_root is None:
        return None

    hq = _test_hash_cache.get(image_name, "__MISS__")
    if hq == "__MISS__":
        test_path = os.path.join(test_images_root, image_name)
        if not os.path.exists(test_path):
            _test_hash_cache[image_name] = None
            return None
        hq = _ahash_64(test_path)
        _test_hash_cache[image_name] = hq

    if hq is None:
        return None

    if hq in _hash_to_chain_counts:
        counts = _hash_to_chain_counts[hq]
        return max(counts.items(), key=lambda kv: kv[1])[0]

    if _hash_keys_u64 is None or _hash_keys_u64.size == 0:
        return None

    q = np.uint64(hq)

    dists = _popcount_u64_vectorized(_hash_keys_u64 ^ q)
    j = int(dists.argmin())
    best_d = int(dists[j])
    if best_d > 10:
        return None

    best_h = int(_hash_keys_u64[j])
    counts = _hash_to_chain_counts[best_h]
    return max(counts.items(), key=lambda kv: kv[1])[0]


global_top5_str = [str(x) for x in global_top5]

preds = []
n_chain_found = 0
images = sample_sub["image"].astype(str).tolist()

_chain_pred_str = {}
for ch, top in chain_top5.items():
    combined = []
    seen = set()
    for x in list(map(str, top)) + global_top5_str:
        if x not in seen:
            combined.append(x)
            seen.add(x)
        if len(combined) == 5:
            break
    if len(combined) < 5:
        combined = (combined + global_top5_str)[:5]
    _chain_pred_str[int(ch)] = " ".join(combined)

for img in images:
    ch = infer_chain_for_test_image(img)
    if ch is not None:
        pred = _chain_pred_str.get(int(ch))
        if pred is not None:
            n_chain_found += 1
            preds.append(pred)
            continue
    preds.append(global_pred_str)

print(
    f"Inferred chain for {n_chain_found}/{len(sample_sub)} test images (others used global fallback)."
)

submission = sample_sub.copy()
submission["hotel_id"] = preds

assert list(submission.columns) == ["image", "hotel_id"]
assert len(submission) == len(sample_sub)
assert submission["hotel_id"].astype(str).str.split().map(len).eq(5).all()

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())



## === cell 2
import subprocess, shlex

print(subprocess.check_output(shlex.split("ls -lha submission.csv")).decode("utf-8"))
print(subprocess.check_output(shlex.split("head -n 5 submission.csv")).decode("utf-8"))
