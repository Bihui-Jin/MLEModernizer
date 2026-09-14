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

0.7687168476888083

# 6. Current score

0.00502

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it depends on external wheels/modules (`pekolib`) that aren’t present in the provided filesystem, so it won’t reliably run end-to-end or create `submission.csv`. I replace that dependency with a minimal, self-contained baseline that always produces a valid `submission.csv` in the required `image,hotel_id` format using only the provided `train.csv` and `sample_submission.csv`. To move the score upward (vs. a blank/failed run) while keeping logic simple and within time, I predict the global top-5 most frequent `hotel_id` from the training data for every test image. This won’t hit the target MAP@5, but it generate a valid submission and produce a non-zero baseline score, which is the smallest legitimate step toward the target given the current “no submission” state.'
- What this solution (achieved 0.00209) has done: 'Your current baseline predicts the same global top-5 hotels for every image, which is valid but far from the target MAP@5. To move the score upward with minimal logic change and without using images/models, I generate *chain-aware* top-5 candidates by leveraging the `chain` field in `train.csv` and the fact that test images are stored in chain subfolders (the parent directory name). This keeps the approach as “frequency-based top-5” but makes it conditional on the inferred chain, which should materially increase MAP@5 vs the global prior while remaining fast and deterministic. I also keep a safe fallback to the global top-5 when the chain cannot be inferred or has no stats.'
- What this solution (achieved 0.00209) has done: 'Your current approach is valid but likely underperforms because many test images won’t have a numeric parent folder (or won’t match your `fn`-only mapping), causing frequent fallback to the global top-5. I make a minimal, score-relevant fix: build the test image “chain” mapping using the *relative path* (to robustly detect chain folders) and also match both `image` and `test_images/<chain>/<image>` naming patterns. I also add a tiny amount of smoothing by using global priors only when chain has too few unique hotels, otherwise keep chain top-5 unchanged—this preserves the same frequency-based core logic while improving recall. Output format and paths remain the same and it still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.00209) has done: 'Your current score is low because most test images likely don’t get a usable inferred `chain` from the directory structure, so you fall back to a weak global top-5 too often. I keep the same frequency-based “chain-aware top-5” core logic, but make the chain inference more robust by (1) also extracting chain from the immediate parent folder name (common layout), and (2) building the mapping keyed by the exact `image` strings in `sample_submission.csv` (so we match whether Kaggle provides bare filenames or paths). I also make the “chain too small” fallback less aggressive (only fall back when the chain has no candidates), which should increase MAP@5 while still being deterministic and fast. The output remains a valid `submission.csv` with `image,hotel_id` and exactly 5 IDs per row.'
- What this solution (achieved 0.00209) has done: 'Your current logic is sound but likely underperforms because many test images won’t get a chain assigned due to directory layout differences (e.g., `test_images/test_images/<chain>/<image>.jpg`), and because `os.walk` may traverse nested folders where the first numeric folder isn’t found reliably. I keep the exact same “frequency-based top-5, chain-aware with global fallback” core approach, but make chain inference deterministic and more robust by explicitly globbing for image files and extracting the nearest numeric parent chain id. I also add a safe “mixing” fallback that appends missing items from the global top-5 (instead of fully replacing with global), which preserves chain specificity while ensuring always-5 unique-ish candidates. This should increase MAP@5 materially from ~0.002 by reducing unnecessary global fallback without changing the fundamental model-free method.'
- What this solution (achieved 0.00209) has done: 'Your current frequency-based chain-aware baseline is valid but likely missing many chain matches because it only maps by basename and depends on globbing patterns that may not cover the nested `test_images/test_images/...` layout reliably. I keep the exact same core logic (per-chain top-5 with a global fallback) but make chain inference deterministic and higher-recall by scanning the actual test image files with `os.walk`, extracting the nearest numeric parent chain id, and building an `image -> chain` map keyed by the exact `image` strings from `sample_submission.csv` (plus basename fallback). I also make the fallback slightly less lossy by always “merge-filling” from global top-5 (instead of fully switching to global) when a chain exists but has no top-5 entry. These are minimal changes intended to materially increase MAP@5 from ~0.002 by reducing unnecessary global-only predictions, without changing the modeling approach.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should increase it with the smallest legitimate change that preserves your frequency-based, chain-aware top-5 logic. The biggest likely issue is that you’re inferring `chain` only from numeric parent folders, but the provided `test_images` layout often has no chain folders at all, making your chain-aware logic rarely apply and collapsing to a weak global prior. I keep the same approach but add a second, metadata-driven candidate source: a per-`hotel_id` co-occurrence prior conditioned on the chain (computed from `train.csv`), then merge-fill with chain-top5 and global-top5 to always produce 5 IDs. This is still “count-based top-5 retrieval” (no image model), but typically improves MAP@5 a lot versus global-only when chain inference is missing.'
- What this solution (achieved 0.00215) has done: 'Your current score is far below the target, so we need a real signal beyond the global/chain frequency prior while keeping the same “non-image, count-based candidate generation” core logic. The biggest gain with minimal conceptual change is to leverage *visual similarity* without training a model: compute simple image hashes (dHash) for train and test images using PIL (available) and vote the top-5 hotels among the nearest hashed train images. To stay within time, we cap the number of train images indexed per chain and only compare within the inferred chain when available (fallback to global chain=0 index), and we still merge-fill with your existing chain/global top-5 for robustness. This keeps the overall approach as fast retrieval + frequency priors, but should materially improve MAP@5 versus always predicting priors.'
- What this solution (achieved 0.00239) has done: 'Your current score is far below the target, so we should increase it with the smallest changes that strengthen the existing “dHash nearest-neighbor + chain/global priors” logic. The biggest issue is that `resolve_test_path()` often fails because it only checks a couple of path patterns; when it fails you fall back to priors, collapsing MAP@5. I (1) build a fast `basename -> actual_full_path` map for all test jpgs once and use it for guaranteed path resolution, and (2) make the NN voting slightly more metric-aligned by weighting closer neighbors higher (still the same retrieval/voting core idea). These changes keep your architecture and approach intact, but should materially reduce fallback-to-prior and improve MAP@5.'
- What this solution (achieved 0.00315) has done: 'The timeout is dominated by Python-level image hashing and repeated per-image nearest-neighbor work; we keep the exact same hashing + voting logic, but remove avoidable overhead. Specifically, we (1) stop walking the entire test image tree and instead build paths directly from the submission list plus a single fast fallback scan, (2) parallelize dhash computation for both train and test images using a deterministic thread pool (PIL releases the GIL during decoding/resizing), and (3) precompute test hashes once and reuse them during prediction to avoid repeated disk I/O. All changes preserve the same distance computation, NN_K selection, and weighted-vote ranking; only scheduling/caching/vectorization is improved.'
- What this solution (achieved 0.00502) has done: 'Your current score is far below the target, so we should increase it with the smallest changes that preserve your exact “dHash NN voting + chain/global priors” core logic. The biggest score limiter is likely that many test images are being hashed against the weak chain=0 index because chain inference from test paths is often missing/incorrect in the hidden test layout. I (1) strengthen chain inference by also using the parent folder immediately under `test_images` when it’s numeric and (2) index a small set of “global” train hashes across all chains (not just chain=0) for better fallback NN matches when chain is unknown, while keeping NN_K, hashing, and weighted voting unchanged. This should reduce prior-only predictions and improve MAP@5 without changing the modeling approach.'

# 9. Code solution

## === cell 0
import os
import heapq  # kept (core approach uses NN_K top-k; we no longer need heap for speed, but keep import harmless)
import pandas as pd
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

BASE_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for base in BASE_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
        path2 = os.path.join(base, "hotel-id-2021-fgvc8", filename)
        if os.path.exists(path2):
            return path2
        path3 = os.path.join(base, "hotel-id-2021-fgvc8", filename)
        if os.path.exists(path3):
            return path3
    raise FileNotFoundError(
        f"Could not find {filename} under candidates: {BASE_CANDIDATES}"
    )


def find_dir(dirname: str) -> str:
    for base in BASE_CANDIDATES:
        p1 = os.path.join(base, dirname)
        if os.path.isdir(p1):
            return p1
        p2 = os.path.join(base, "hotel-id-2021-fgvc8", dirname)
        if os.path.isdir(p2):
            return p2
    raise FileNotFoundError(
        f"Could not find directory {dirname} under candidates: {BASE_CANDIDATES}"
    )


def top5_strict_from_counts(series: pd.Series) -> list:
    top = series.value_counts().head(5).index.astype(str).tolist()
    if len(top) == 0:
        return []
    while len(top) < 5:
        top.append(top[-1])
    return top[:5]


def merge_top5(primary: list, fallback: list) -> list:
    out = []
    for x in primary:
        xs = str(x)
        if xs not in out:
            out.append(xs)
        if len(out) == 5:
            return out
    for x in fallback:
        xs = str(x)
        if xs not in out:
            out.append(xs)
        if len(out) == 5:
            return out
    while len(out) < 5 and len(out) > 0:
        out.append(out[-1])
    return out[:5]


def dhash64_from_image(im: Image.Image, hash_size: int = 8):
    im = im.resize((hash_size + 1, hash_size), Image.BILINEAR)
    b = memoryview(im.tobytes())
    w = hash_size + 1
    h = 0
    bit = 1
    for r in range(hash_size):
        row_off = r * w
        for c in range(hash_size):
            if b[row_off + c] > b[row_off + c + 1]:
                h |= bit
            bit <<= 1
    return h


def dhash128_from_path(img_path: str, hash_size: int = 8):
    try:
        with Image.open(img_path) as im0:
            im_rgb = im0.convert("RGB")
            im_gray = im_rgb.convert("L")
            h1 = dhash64_from_image(im_gray, hash_size=hash_size)

            im_hsv = im_rgb.convert("HSV")
            v = im_hsv.split()[2]
            h2 = dhash64_from_image(v, hash_size=hash_size)

            return (h1, h2)
    except Exception:
        return None


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def hamming128(a, b) -> int:
    return hamming64(a[0], b[0]) + hamming64(a[1], b[1])


_POPCOUNT_NIBBLE = np.array(
    [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4], dtype=np.uint8
)


def popcount_uint64(arr_uint64: np.ndarray) -> np.ndarray:
    a = np.asarray(arr_uint64, dtype=np.uint64)
    b = a.view(np.uint8).reshape(a.size, 8)
    lo = b & 0x0F
    hi = b >> 4
    return (
        (_POPCOUNT_NIBBLE[lo] + _POPCOUNT_NIBBLE[hi])
        .sum(axis=1)
        .astype(np.uint16, copy=False)
    )


train_csv = find_file("train.csv")
sample_sub_csv = find_file("sample_submission.csv")

print("Using train.csv:", train_csv)
print("Using sample_submission.csv:", sample_sub_csv)

train = pd.read_csv(train_csv)
sub = pd.read_csv(sample_sub_csv)

train["hotel_id"] = train["hotel_id"].astype(str)
train["chain"] = train["chain"].astype(int)

global_top5 = top5_strict_from_counts(train["hotel_id"])
if len(global_top5) == 0:
    raise RuntimeError("No hotel_id found in training data.")

chain_top5 = {}
vc = train.groupby("chain")["hotel_id"].value_counts()
for chain_id in vc.index.get_level_values(0).unique().tolist():
    top = vc.loc[chain_id].head(5).index.astype(str).tolist()
    if len(top) == 0:
        continue
    while len(top) < 5:
        top.append(top[-1])
    chain_top5[int(chain_id)] = top[:5]

chain_hotelid_to_top5_other = {}
chain_hotel_counts = (
    train.groupby(["chain", "hotel_id"]).size().rename("cnt").reset_index()
)
for ch, g in chain_hotel_counts.groupby("chain", sort=False):
    ordered = g.sort_values("cnt", ascending=False)["hotel_id"].astype(str).tolist()
    for hid in ordered:
        others = [x for x in ordered if x != hid][:5]
        if len(others) > 0:
            while len(others) < 5:
                others.append(others[-1])
            chain_hotelid_to_top5_other[(int(ch), str(hid))] = others[:5]

test_images_dir = find_dir("test_images")
train_images_dir = find_dir("train_images")
print("Using test_images dir:", test_images_dir)
print("Using train_images dir:", train_images_dir)

sub_images = sub["image"].astype(str).tolist()
sub_image_set = set(sub_images)
sub_basename_set = set(os.path.basename(x) for x in sub_images)


def infer_chain_from_path(p: str):
    try:
        rel = os.path.relpath(p, test_images_dir)
        parts = rel.split(os.sep)
        if len(parts) >= 2 and parts[0].isdigit():
            return int(parts[0])
    except Exception:
        pass

    d = os.path.dirname(p)
    while True:
        base = os.path.basename(d)
        if base.isdigit():
            return int(base)
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


img2chain_by_exact = {}
img2chain_by_basename = {}

test_basename_to_fullpath = {}
unresolved_basenames = set(sub_basename_set)

for img_key in sub_images:
    base = os.path.basename(img_key)
    if base not in unresolved_basenames:
        continue
    p1 = os.path.join(test_images_dir, img_key)
    if os.path.exists(p1):
        test_basename_to_fullpath[base] = p1
        ch = infer_chain_from_path(p1)
        if ch is not None:
            img2chain_by_basename.setdefault(base, ch)
            img2chain_by_exact.setdefault(img_key, ch)
        unresolved_basenames.discard(base)
        continue
    p2 = os.path.join(test_images_dir, base)
    if os.path.exists(p2):
        test_basename_to_fullpath[base] = p2
        ch = infer_chain_from_path(p2)
        if ch is not None:
            img2chain_by_basename.setdefault(base, ch)
            img2chain_by_exact.setdefault(img_key, ch)
        unresolved_basenames.discard(base)

walk_count = 0
hit_exact = len(img2chain_by_exact)
hit_base = len(img2chain_by_basename)

if unresolved_basenames:
    for root, _, files in os.walk(test_images_dir):
        for fn in files:
            if not fn.lower().endswith(".jpg"):
                continue
            walk_count += 1
            if fn not in unresolved_basenames:
                continue
            full_path = os.path.join(root, fn)
            test_basename_to_fullpath[fn] = full_path
            ch = infer_chain_from_path(full_path)
            if ch is not None:
                img2chain_by_basename.setdefault(fn, ch)
                hit_base = len(img2chain_by_basename)
            unresolved_basenames.discard(fn)
        if not unresolved_basenames:
            break

for img_key in sub_images:
    if img_key in img2chain_by_exact:
        continue
    base = os.path.basename(img_key)
    ch = img2chain_by_basename.get(base)
    if ch is not None:
        img2chain_by_exact[img_key] = ch
hit_exact = len(img2chain_by_exact)

print("Fallback os.walk scanned jpg files:", walk_count)
print(
    "Inferred chain for",
    hit_base,
    "unique test basenames out of",
    len(sub_basename_set),
)
print(
    "Inferred chain for",
    hit_exact,
    "exact submission image strings out of",
    len(sub_image_set),
)
print(
    "Built basename->fullpath for",
    len(test_basename_to_fullpath),
    "test images out of",
    len(sub_basename_set),
)

MAX_TRAIN_PER_CHAIN = 1400
MAX_TRAIN_GLOBAL = 6000

NN_K = 25  # unchanged (keeps the same voting approach/semantics)

chains_to_index = set(img2chain_by_exact.values()) | set(img2chain_by_basename.values())
chains_to_index.add(0)

train_hash_index = {}  # chain -> list of (hash:(int,int), hotel_id:str)
total_hashed = 0


def _hash_train_item(args):
    img_path, hid = args
    h = dhash128_from_path(img_path)
    if h is None:
        return None
    return (h, hid)


max_workers = min(32, (os.cpu_count() or 4) + 4)

for ch in sorted(chains_to_index):
    dfc = train.loc[train["chain"] == int(ch), ["image", "hotel_id"]].copy()
    if dfc.empty:
        continue

    dfc["__cnt"] = dfc.groupby("hotel_id")["hotel_id"].transform("size")
    dfc = dfc.sort_values("__cnt", ascending=False).drop(columns="__cnt")

    images = dfc["image"].astype(str).to_numpy()
    hids = dfc["hotel_id"].astype(str).to_numpy()

    chain_dir = os.path.join(train_images_dir, str(ch))

    paths_hids = []
    for img, hid in zip(images, hids):
        if len(paths_hids) >= MAX_TRAIN_PER_CHAIN:
            break
        img_path = os.path.join(chain_dir, img)
        if os.path.exists(img_path):
            paths_hids.append((img_path, hid))

    if not paths_hids:
        continue

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        hashed = list(ex.map(_hash_train_item, paths_hids, chunksize=32))

    entries = [x for x in hashed if x is not None]
    if entries:
        train_hash_index[int(ch)] = entries
        total_hashed += len(entries)

dfg = train.loc[:, ["chain", "image", "hotel_id"]].copy()
dfg["__cnt"] = dfg.groupby("hotel_id")["hotel_id"].transform("size")
dfg = dfg.sort_values("__cnt", ascending=False).drop(columns="__cnt")

paths_hids_global = []
for ch, img, hid in zip(
    dfg["chain"].to_numpy(),
    dfg["image"].astype(str).to_numpy(),
    dfg["hotel_id"].astype(str).to_numpy(),
):
    if len(paths_hids_global) >= MAX_TRAIN_GLOBAL:
        break
    img_path = os.path.join(train_images_dir, str(int(ch)), img)
    if os.path.exists(img_path):
        paths_hids_global.append((img_path, hid))

if paths_hids_global:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        hashed_g = list(ex.map(_hash_train_item, paths_hids_global, chunksize=32))
    entries_g = [x for x in hashed_g if x is not None]
    if entries_g:
        train_hash_index[-1] = entries_g  # -1 denotes global index
        total_hashed += len(entries_g)

print(
    "Built train hash entries:",
    total_hashed,
    "across chains:",
    sorted(train_hash_index.keys())[:10],
    "...",
)

train_hash_packed = {}  # chain -> (h1_uint64[N], h2_uint64[N], hid_obj[N])
for ch, entries in train_hash_index.items():
    n = len(entries)
    h1 = np.empty(n, dtype=np.uint64)
    h2 = np.empty(n, dtype=np.uint64)
    hid = np.empty(n, dtype=object)
    for i, (hh, hhid) in enumerate(entries):
        h1[i] = np.uint64(hh[0])
        h2[i] = np.uint64(hh[1])
        hid[i] = hhid
    train_hash_packed[int(ch)] = (h1, h2, hid)

test_hash_cache = {}


def resolve_test_path(img_key: str) -> str:
    base = os.path.basename(img_key)
    p = test_basename_to_fullpath.get(base)
    if p is not None and os.path.exists(p):
        return p

    candidates = [
        os.path.join(test_images_dir, img_key),
        os.path.join(test_images_dir, base),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    ch = img2chain_by_exact.get(img_key, None)
    if ch is None:
        ch = img2chain_by_basename.get(base, None)
    if ch is not None:
        p = os.path.join(test_images_dir, str(ch), base)
        if os.path.exists(p):
            return p
    return None


chain_prior_top5 = {}
for ch, anchors in chain_top5.items():
    mixed = merge_top5(anchors, [])
    for a in anchors:
        others = chain_hotelid_to_top5_other.get((int(ch), str(a)), None)
        if others is not None:
            mixed = merge_top5(mixed, others)
    chain_prior_top5[int(ch)] = merge_top5(mixed, global_top5)


def _hash_test_item(img_key: str):
    tp = resolve_test_path(img_key)
    if tp is None:
        return (img_key, None)
    h = dhash128_from_path(tp)
    return (img_key, h)


imgs_list = sub["image"].astype(str).tolist()
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    test_hash_by_key = dict(ex.map(_hash_test_item, imgs_list, chunksize=64))


def predict_top5_for_image(img_key: str) -> list:
    ch = img2chain_by_exact.get(img_key, None)
    if ch is None:
        ch = img2chain_by_basename.get(os.path.basename(img_key), None)

    if ch is not None and int(ch) in chain_prior_top5:
        prior_top5 = chain_prior_top5[int(ch)]
    else:
        prior_top5 = global_top5

    packed = None
    if ch is not None and int(ch) in train_hash_packed:
        packed = train_hash_packed[int(ch)]
    else:
        if -1 in train_hash_packed:
            packed = train_hash_packed[-1]
        elif 0 in train_hash_packed:
            packed = train_hash_packed[0]

    if not packed:
        return prior_top5

    th = test_hash_by_key.get(img_key)
    if th is None:
        return prior_top5

    h1_arr, h2_arr, hid_arr = packed
    q1 = np.uint64(th[0])
    q2 = np.uint64(th[1])

    d = popcount_uint64(np.bitwise_xor(h1_arr, q1)).astype(
        np.int32, copy=False
    ) + popcount_uint64(np.bitwise_xor(h2_arr, q2)).astype(np.int32, copy=False)

    n = int(d.shape[0])
    k = NN_K if n >= NN_K else n
    if k <= 0:
        return prior_top5

    idx_k = np.argpartition(d, k - 1)[:k]
    idx_k = idx_k[np.argsort(d[idx_k], kind="stable")]
    d_k = d[idx_k].astype(np.int64, copy=False)
    hid_k = hid_arr[idx_k]

    vote = {}
    for dist, hid in zip(d_k.tolist(), hid_k.tolist()):
        w = 1.0 / (1.0 + float(dist))
        vote[hid] = vote.get(hid, 0.0) + w

    nn_ranked = sorted(vote.items(), key=lambda x: (-x[1], x[0]))
    nn_top = [hid for hid, _ in nn_ranked][:5]

    return merge_top5(nn_top, prior_top5)


sub = sub[["image"]].copy()

preds = []
for img in imgs_list:
    top5 = predict_top5_for_image(img)
    preds.append(" ".join(top5))

sub["hotel_id"] = preds

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Rows:", len(sub), "Cols:", list(sub.columns))




## === cell 1
import pandas as pd

chk = pd.read_csv("submission.csv")
assert list(chk.columns) == [
    "image",
    "hotel_id",
], "Submission columns must be: image, hotel_id"
assert chk["image"].notnull().all(), "All images must be present"
assert chk["hotel_id"].notnull().all(), "All predictions must be present"
assert (
    chk["hotel_id"].astype(str).str.split().str.len() == 5
).all(), "Each prediction must contain 5 space-delimited hotel_ids"
print("submission.csv looks valid.")
print(chk.head())
