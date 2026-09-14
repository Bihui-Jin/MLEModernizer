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

0.00239

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently cannot yield a Kaggle score because it depends on private Kaggle Dataset wheels (`/kaggle/input/pekolib*`) that are not present in your environment, so it fails before producing a valid `submission.csv`. I keep the “generate a submission.csv from the provided data” core goal, but replace the missing-package calls with a minimal, fully self-contained baseline that always runs using only the provided `train.csv` + `sample_submission.csv`. To move score upward from “not yielded” toward your target, the baseline predict the global top-5 most frequent `hotel_id` values from the training metadata (a common safe baseline for MAP@5) and write the correctly formatted submission file. This keeps runtime well under 600s and ensures the submission rows align exactly with `sample_submission.csv`.'
- What this solution (achieved 0.00209) has done: 'The timeout is dominated by `infer_chain_from_fs`, which for every test image repeatedly scans all subdirectories under `test_images_root` and does `os.path.exists` checks—an O(N_images × N_dirs) filesystem walk. We can preserve identical prediction logic by building a one-time mapping from `image_name -> chain` by scanning the directory tree once, then doing O(1) lookups per image. We also speed up the `chain_top5` computation by replacing the Python `apply(lambda ...)` with a vectorized `groupby().head(5)` pattern that produces the same top-5-per-chain list. All outputs/paths and fallback behavior remain the same.'
- What this solution (achieved 0.00243) has done: 'The timeout is dominated by repeatedly opening and hashing many train images (up to 12k) serially with PIL, plus per-image filesystem `exists()` checks and per-test-image nearest-neighbor scans. To keep the same hashing logic and decision rules, I (1) avoid per-row `os.path.exists` by building paths from a precomputed set of existing files, (2) parallelize the train-image hashing with a thread pool (I/O-bound due to image decode), (3) replace the per-test-image chunk loop for nearest neighbor with one vectorized popcount over all keys (same exact min-distance result), and (4) ensure caches/dtypes are efficient and deterministic. These changes keep identical core semantics (same aHash, same thresholds, same top-5 construction) while substantially reducing wall time.'
- What this solution (achieved 0.00207) has done: 'Your current score (0.00243) is far below the target (0.74498), so we need a real lift while keeping the same overall approach (chain-prior + hash-based chain inference + top-5 completion). The biggest minimal win is to stop arbitrarily truncating the hash index to the first 12k training rows (which are timestamp/CSV-order biased) and instead build the same hash->chain index from a larger, more representative sample spread across chains. I keep the same aHash, same nearest-neighbor rule and threshold, and the same chain_top5/global fallback logic, but increase the hashing cap moderately and sample deterministically per-chain to improve chain inference coverage without changing model semantics. This should move MAP@5 upward substantially while staying within the 600s budget by keeping parallel hashing and avoiding expensive filesystem operations.'
- What this solution (achieved 0.00349) has done: 'Your current score (0.00207) is far below the target (0.74498), so we need a real lift while keeping the same chain-prior + aHash-nearest-neighbor inference core logic intact. The biggest issue is that aHash+chain inference is fundamentally weak (and often wrong) for this task, so most predictions devolve to near-random chain top-5/global top-5, which yields very low MAP@5. A minimal but high-impact improvement that preserves the overall “retrieve neighbors then vote” approach is to switch the retrieval target from `chain` to `hotel_id` (still using the same aHash, same nearest-neighbor with a Hamming threshold, same top-5 completion with global priors), and to build the hash index over `hotel_id` directly. This keeps the same style of logic while making predictions align with the metric’s label space, and should move the score substantially upward toward the target without changing training loops/models (there are none) and staying within the time limit.'
- What this solution (achieved 0.00361) has done: 'Main bottlenecks are (1) listing every filename in many large chain folders (`os.listdir` + building big `set`s) and (2) per-image Python overhead during inference. We keep the exact same hashing + nearest-neighbor voting logic, but replace the expensive “pre-list all files” existence check with a single vectorized `os.path.exists` pass on constructed paths (same semantics, far less I/O/memory). We also avoid repeated dict lookups/boxing in the hashing loop, tighten caching sentinels, and reuse preallocated arrays where safe, while keeping determinism and identical ranking/tie-break behavior. Net effect: much less filesystem scanning and lower Python overhead, which should bring runtime under 600s.'
- What this solution (achieved 0.00239) has done: 'Your current score (0.00361) is far below the target (0.74498), so we need a meaningful lift while keeping your existing “aHash retrieval + vote + global top-5 fallback” core logic intact. The lowest-risk high-impact fix is to stop using aHash to retrieve directly into `hotel_id` space (which is too noisy) and instead retrieve into `chain` first (more stable visually), then output that chain’s top-5 hotels (which matches the metric label space better). Concretely: build the hash index as `hash -> chain` (instead of `hash -> hotel_id`), infer a ranked list of candidate chains per test image using the same nearest-neighbor + weighted vote, then map the best chain to its precomputed top-5 hotels (fallback to global top-5 if chain inference fails). This preserves the same hashing, same NN distance rule, same voting structure, and same submission format, but should move MAP@5 substantially upward toward your target.'

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

train_meta = train.loc[:, ["chain", "hotel_id"]].copy()
train_meta["chain"] = train_meta["chain"].astype(int)
train_meta["hotel_id"] = train_meta["hotel_id"].astype(int)

_chain_top5_df = (
    train_meta.groupby(["chain", "hotel_id"], sort=False)
    .size()
    .rename("cnt")
    .reset_index()
    .sort_values(
        ["chain", "cnt", "hotel_id"], ascending=[True, False, True], kind="mergesort"
    )
)
_chain_top5_df = (
    _chain_top5_df.groupby("chain", sort=False).head(5).reset_index(drop=True)
)

chain_to_top5_hotels = (
    _chain_top5_df.groupby("chain", sort=False)["hotel_id"]
    .apply(lambda s: [str(x) for x in s.tolist()])
    .to_dict()
)

print("Prepared chain->top5 map for", len(chain_to_top5_hotels), "chains")

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


HASH_TRAIN_CAP = 60000
PER_CHAIN_CAP = 800

_hash_to_chain_counts = {}
_hash_ready = False

_hash_keys_u64 = None
_hash_keys_py = None

_popcount_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_u64_vectorized(x_u64: np.ndarray) -> np.ndarray:
    """Exact popcount for uint64 array using byte LUT; deterministic and fast."""
    xb = x_u64.view(np.uint8).reshape(-1, 8)
    return _popcount_lut[xb].sum(axis=1).astype(np.int16)


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
        train2["path"] = (
            train_images_root.rstrip("/")
            + "/"
            + train2["chain_str"]
            + "/"
            + train2["image"]
        )
        exists_mask = train2["path"].map(os.path.exists)
        train2 = train2.loc[exists_mask].reset_index(drop=True)

        def _hash_one(path_and_chain):
            path, chain_val = path_and_chain
            h = _ahash_64(path)
            return h, int(chain_val)

        paths = train2["path"].tolist()
        chains = train2["chain"].tolist()

        max_workers = min(16, max(1, (os.cpu_count() or 4)))

        n_hashed = 0
        get_bucket = _hash_to_chain_counts.get
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for h, ch in ex.map(_hash_one, zip(paths, chains), chunksize=128):
                if h is None:
                    continue
                d = get_bucket(h)
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
            f"Built train hash index (hash->chain): hashed {n_hashed} train images (cap={HASH_TRAIN_CAP}, per_chain={PER_CHAIN_CAP})."
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

TOPK_NEIGHBORS = 25
HAMMING_MAX_DIST = 14


def infer_chains_for_test_image(image_name: str):
    """Infer ranked chain candidates via hash NN voting; returns list[int] (best-first)."""
    if not _hash_ready or test_images_root is None:
        return []

    if image_name in _test_hash_cache:
        hq = _test_hash_cache[image_name]
    else:
        test_path = os.path.join(test_images_root, image_name)
        if not os.path.exists(test_path):
            _test_hash_cache[image_name] = None
            return []
        hq = _ahash_64(test_path)
        _test_hash_cache[image_name] = hq

    if hq is None:
        return []

    counts_exact = _hash_to_chain_counts.get(hq)
    if counts_exact is not None:
        ranked = sorted(counts_exact.items(), key=lambda kv: (-kv[1], kv[0]))
        return [ch for ch, _ in ranked]

    if _hash_keys_u64 is None or _hash_keys_u64.size == 0:
        return []

    q = np.uint64(hq)
    dists = _popcount_u64_vectorized(_hash_keys_u64 ^ q)

    K = min(TOPK_NEIGHBORS, int(_hash_keys_u64.size))
    nn_idx = np.argpartition(dists, K - 1)[:K]
    nn_idx = nn_idx[np.argsort(dists[nn_idx], kind="mergesort")]

    chain_votes = {}
    get_counts = _hash_to_chain_counts.get
    ham_max = HAMMING_MAX_DIST
    for j in nn_idx:
        dj = int(dists[j])
        if dj > ham_max:
            break
        best_h = int(_hash_keys_u64[int(j)])
        counts = get_counts(best_h, {})
        w = (ham_max - dj) + 1
        for ch, c in counts.items():
            chain_votes[ch] = chain_votes.get(ch, 0) + int(c) * w

    if not chain_votes:
        return []

    ranked = sorted(chain_votes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ch for ch, _ in ranked]


def _safe_chain_int(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
            x = int(float(x))
        else:
            if isinstance(x, (float, np.floating)) and not np.isfinite(x):
                return None
            x = int(x)
        return x
    except Exception:
        return None


global_top5_str = [str(x) for x in global_top5]

preds = []
n_any_found = 0
n_chain_mapped = 0
images = sample_sub["image"].astype(str).tolist()

for img in images:
    chains = infer_chains_for_test_image(img)

    out = []
    seen = set()

    picked = False
    if chains:
        for ch in chains:
            ch = _safe_chain_int(ch)
            if ch is None:
                continue
            top5_hotels = chain_to_top5_hotels.get(ch)
            if top5_hotels:
                for s in top5_hotels:
                    if s not in seen:
                        out.append(s)
                        seen.add(s)
                    if len(out) == 5:
                        break
                if len(out) > 0:
                    picked = True
                    n_chain_mapped += 1
                break  # use best chain that has a top-5 list

        if picked:
            n_any_found += 1

    for s in global_top5_str:
        if s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) == 5:
            break

    if len(out) != 5:
        out = (out + global_top5_str)[:5]
        if len(out) < 5:
            out = (out + [global_top5_str[0]] * 5)[:5]

    preds.append(" ".join(out))

print(
    f"Produced non-empty chain-mapped list for {n_chain_mapped}/{len(sample_sub)} test images."
)
print(
    f"Produced non-empty candidate list for {n_any_found}/{len(sample_sub)} test images."
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
import subprocess, shlex, os

if os.path.exists("submission.csv"):
    print(
        subprocess.check_output(shlex.split("ls -lha submission.csv")).decode("utf-8")
    )
    print(
        subprocess.check_output(shlex.split("head -n 5 submission.csv")).decode("utf-8")
    )
else:
    raise FileNotFoundError("submission.csv was not created.")
