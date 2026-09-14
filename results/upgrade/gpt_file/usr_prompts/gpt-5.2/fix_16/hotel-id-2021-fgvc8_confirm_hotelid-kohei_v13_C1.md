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

0.7698703376273452

# 6. Current score

0.00151

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00232) has done: 'I keep your current approach of running `python -m peko.subs.hotelid.v13`, but make the notebook reliably produce a valid `submission.csv` in the expected working directory and fail fast with actionable errors if it doesn’t. The main reason you’re getting “Not yielded” is likely that the script writes the submission somewhere else (or under a different name), so Kaggle can’t find/upload it. I add a minimal “find-and-copy” step that searches common output locations for a CSV with the required `image,hotel_id` header, copies it to `/kaggle/working/submission.csv`, and validates row count/columns against `sample_submission.csv`. This doesn’t change model logic, but it unblocks scoring and ensures the output is submission-valid.'
- What this solution (achieved 0.00232) has done: 'I fix the failing dependency installation by making it robust to the actual wheel layout in `/kaggle/input` (your current glob points to non-existent files), while keeping the core logic of running `python -m peko.subs.hotelid.v13` unchanged. Then I keep your existing “find-and-copy” validation step to guarantee a correctly formatted `/kaggle/working/submission.csv`. This should both unblock execution and recover the intended score (your current 0.00232 is consistent with a broken/placeholder submission rather than a real model output). The changes are minimal and confined to installation/path handling and submission file validation.'
- What this solution (achieved 0.00209) has done: 'I remove the hard dependency on external `pekolib` wheels (which aren’t present in your environment) and replace the failing `python -m peko.subs.hotelid.v13` call with a minimal, self-contained baseline that uses only the provided `train.csv` labels to generate valid MAP@5-style predictions. This fixes the runtime error and guarantees a correctly formatted `/kaggle/working/submission.csv` with the exact required columns and row order from `sample_submission.csv`. To improve score vs. the current near-zero placeholder behavior, the baseline predicts the five most frequent `hotel_id`s from the training set for every test image (a common safe fallback that usually beats random/empty outputs). The existing submission validation/copy logic is kept, but now it always find the generated CSV.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should improve the predictions while keeping the same “no-image-model” core idea (label-only baseline) intact. The main issue is that predicting the same global top-5 hotels for every image is too weak; a minimal, legitimate upgrade is to condition those top-5 guesses on the *test image’s chain folder* (the dataset stores images under `test_images/{chain}/...`). We infer `chain` for each test image by scanning `/kaggle/input/**/test_images` once, then for each chain predict that chain’s most frequent 5 `hotel_id`s from train (fallback to global top-5 for unknown/zero chains). This keeps the approach simple, preserves evaluation semantics, and should move MAP@5 materially toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should improve prediction quality while keeping the same “label-only + chain-conditioned top-k” core approach. The biggest issue is that inferring `chain` for test images by scanning subfolders can fail (or be incomplete) in the hidden test environment, which collapses predictions back to the weak global top-5. I make chain inference robust by (1) parsing chain from the sample’s `image` path if it contains folder components and (2) falling back to a fast recursive scan that maps filenames to chain folders when needed. This keeps the same model-free logic but should substantially reduce the “unknown chain” rate and move MAP@5 upward toward your target, while still writing a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we need a legitimate signal beyond global/chain priors while keeping the same “no-image-model” baseline family. The smallest meaningful upgrade is to add *visual near-duplicate retrieval* using simple perceptual hashes computed from the images with only stdlib + PIL: build an index of training image hashes per hotel_id, then for each test image retrieve the nearest hashes and vote hotels into a top-5 list (falling back to your existing chain-top5/global-top5 when retrieval is weak). This preserves your overall approach (no ML training loop, still produces top-5 strings) but should move MAP@5 substantially upward versus 0.002. I also keep runtime bounded by sampling a capped number of train images per hotel and caching hashes to disk in `/kaggle/working` so repeated runs are faster, while still producing a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.00209) has done: 'The timeout is dominated by hashing thousands of training images and repeatedly doing small Python-loop work during retrieval. I keep the same dhash64 + bucketed Hamming retrieval logic and the same fallback strategy, but make it faster by (1) eliminating expensive train `groupby`/concat sampling in favor of a deterministic, vectorized per-hotel selection, (2) parallelizing only the *missing* train-image hash computation with a thread pool (I/O-bound), (3) precomputing per-bucket hotel_id arrays to avoid Python list extensions inside the per-test-image loop, and (4) cutting redundant filesystem scans and Python overhead in the submission loop. These changes preserve the exact algorithm and scoring semantics, only reducing repeated work and moving equivalent computation into faster/vectorized paths.'
- What this solution (achieved 0.00151) has done: 'Your score is far below the target, so we need a real signal that matches the competition (image-based retrieval) while keeping your current “dhash retrieval + chain/global fallback + top-5 voting” core logic intact. The main bug hurting score is that chain inference is effectively disabled (`need_scan = False`), so most hidden-test images likely end up with `unknown_chain` and weak priors; we re-enable robust chain mapping by scanning `test_images` (cached) and also scanning the hidden-test flat layout if Kaggle provides it. Separately, your current retrieval only searches 3 hash buckets (bucket±1), which is too restrictive; we expand to a small fixed radius (±4) to increase recall with negligible semantic change and controlled runtime. Finally, we broaden the train indexing slightly (more hotels/images per hotel) to improve nearest-neighbor coverage while keeping the same hashing/indexing approach and staying within the 600s budget thanks to caching and the existing thread pool.'
- What this solution (achieved 0.00149) has done: 'Your current score (0.00151) is far below the target (0.76987), so we should improve retrieval quality while keeping the same “dhash retrieval + bucketed Hamming + chain/global fallback” core logic. The biggest likely reason for near-zero MAP@5 is that the hidden test images are not being found (path scan only checks the top level of a `test_images` dir), causing retrieval to fail almost always and falling back to weak priors. I make hidden-test path discovery robust by recursively indexing all images under any discovered `test_images` directories (cached), and I also build `image_to_chain` from those paths so chain conditioning works even when sample paths don’t include folders. These changes don’t alter your model/retrieval method; they just ensure it actually runs on the real hidden test set and uses the intended chain-aware fallback, which should move score materially upward toward the target.'
- What this solution (achieved 0.00151) has done: 'I fix the Pillow image-loading bug that causes hashing threads to crash (`AttributeError: 'NoneType' object has no attribute 'seek'`) by ensuring we compute the hash while the image file handle is still open (convert/copy inside the context manager). Then I make the `test_images` discovery and basename→path index recursive (cached) so hidden-test images are actually found and retrieval doesn’t silently fail back to weak priors. Finally, I keep your existing dhash+bucketed retrieval + chain/global fallback logic unchanged, but ensure the pipeline always completes and writes a valid `/kaggle/working/submission.csv` with correct columns and row order.'

# 9. Code solution

## === cell 0
import pandas as pd
from pathlib import Path

WORKING = Path("/kaggle/working")
WORKING.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv")
if not TRAIN_CSV.exists():
    TRAIN_CSV = Path("/kaggle/input/train.csv")

SAMPLE_CSV = Path("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv")
if not SAMPLE_CSV.exists():
    SAMPLE_CSV = Path("/kaggle/input/sample_submission.csv")

TEST_IMAGES_DIR = Path("/kaggle/input/hotel-id-2021-fgvc8/test_images")
if not TEST_IMAGES_DIR.exists():
    TEST_IMAGES_DIR = Path("/kaggle/input/test_images")
if not TEST_IMAGES_DIR.exists():
    candidates = list(Path("/kaggle/input").rglob("test_images"))
    candidates = [p for p in candidates if p.is_dir()]
    if candidates:
        candidates.sort(key=lambda p: len(str(p)))
        TEST_IMAGES_DIR = candidates[0]

TRAIN_IMAGES_DIR = Path("/kaggle/input/hotel-id-2021-fgvc8/train_images")
if not TRAIN_IMAGES_DIR.exists():
    TRAIN_IMAGES_DIR = Path("/kaggle/input/train_images")
if not TRAIN_IMAGES_DIR.exists():
    candidates = list(Path("/kaggle/input").rglob("train_images"))
    candidates = [p for p in candidates if p.is_dir()]
    if candidates:
        candidates.sort(key=lambda p: len(str(p)))
        TRAIN_IMAGES_DIR = candidates[0]

ALL_TEST_IMAGES_DIRS = [TEST_IMAGES_DIR] if TEST_IMAGES_DIR.exists() else []
HIDDEN_TEST_IMAGES_DIR = TEST_IMAGES_DIR if TEST_IMAGES_DIR.exists() else None

if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"train.csv not found at {TRAIN_CSV}")
if not SAMPLE_CSV.exists():
    raise FileNotFoundError(f"sample_submission.csv not found at {SAMPLE_CSV}")

train = pd.read_csv(TRAIN_CSV)
sample = pd.read_csv(SAMPLE_CSV)

required_cols = {"chain", "hotel_id", "image"}
missing = required_cols - set(train.columns)
if missing:
    raise ValueError(f"train.csv missing required columns: {sorted(missing)}")

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + global_top5[:1] * 5)[:5]
global_pred_str = " ".join(global_top5)

train["chain"] = train["chain"].fillna(0).astype(int)
chain_top5 = {}
vc = train.groupby("chain")["hotel_id"].value_counts()
for chain_id in vc.index.get_level_values(0).unique().tolist():
    topk = vc.loc[chain_id].head(5)
    ids = topk.index.astype(str).tolist()
    if len(ids) < 5:
        ids = (ids + global_top5)[:5]
    chain_top5[int(chain_id)] = ids


def parse_chain_from_image_field(image_field: str):
    s = str(image_field)
    if "/" not in s:
        return None
    parts = s.split("/")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0])
    except Exception:
        return None


image_to_chain = {}
parsed_from_sample = 0
for img in sample["image"].astype(str).tolist():
    ch = parse_chain_from_image_field(img)
    if ch is not None:
        image_to_chain[Path(img).name] = ch
        parsed_from_sample += 1

print("Global top-5 hotel_ids used (fallback):", global_top5)
print(f"Parsed chain from sample image paths for {parsed_from_sample} rows.")
print(
    f"test_images dir used (primary): {TEST_IMAGES_DIR if TEST_IMAGES_DIR.exists() else 'NOT FOUND (using global fallback)'}"
)
print(
    f"train_images dir used: {TRAIN_IMAGES_DIR if TRAIN_IMAGES_DIR.exists() else 'NOT FOUND (hash retrieval disabled)'}"
)



## === cell 1
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "PIL is required (usually available as Pillow on Kaggle)."
    ) from e

import numpy as np

HASH_CACHE = WORKING / "phash_cache_v2.json"
INDEX_CACHE = WORKING / "phash_index_v2.json"
TEST_PATHS_CACHE = WORKING / "test_basename_to_path_v4.json"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image_path(p: Path) -> bool:
    return p.is_file() and (p.suffix.lower() in IMG_EXTS)


TEST_BASENAME_TO_PATH = {}
if TEST_PATHS_CACHE.exists():
    try:
        with open(TEST_PATHS_CACHE, "r") as f:
            TEST_BASENAME_TO_PATH = {k: Path(v) for k, v in json.load(f).items()}
    except Exception:
        TEST_BASENAME_TO_PATH = {}

if not TEST_BASENAME_TO_PATH:
    t_scan = time.time()
    built = 0
    if TEST_IMAGES_DIR.exists():
        for p in TEST_IMAGES_DIR.rglob("*"):
            if _is_image_path(p):
                if p.name not in TEST_BASENAME_TO_PATH:
                    TEST_BASENAME_TO_PATH[p.name] = p
                    built += 1

    with open(TEST_PATHS_CACHE, "w") as f:
        json.dump(
            {k: str(v) for k, v in TEST_BASENAME_TO_PATH.items()},
            f,
            separators=(",", ":"),
        )

    print(
        f"Built TEST_BASENAME_TO_PATH for {len(TEST_BASENAME_TO_PATH)} files in {time.time()-t_scan:.1f}s"
    )
else:
    print(
        f"Loaded TEST_BASENAME_TO_PATH for {len(TEST_BASENAME_TO_PATH)} files from cache"
    )


def _parse_chain_from_path(p: Path):
    parts = p.parts
    for i in range(len(parts) - 2):
        if parts[i] == "test_images":
            cand = parts[i + 1]
            try:
                return int(cand)
            except Exception:
                return None
    return None


scanned_chain = 0
for base, p in TEST_BASENAME_TO_PATH.items():
    ch = _parse_chain_from_path(p)
    if ch is not None:
        if base not in image_to_chain:
            image_to_chain[base] = ch
            scanned_chain += 1

print(f"Added chain mappings from recursive test_images scan: {scanned_chain}")
print(f"Total chain mappings available (by basename): {len(image_to_chain)}")

_DHASH_HASH_SIZE = 8
_DHASH_POWERS = np.uint64(1) << np.arange(
    _DHASH_HASH_SIZE * _DHASH_HASH_SIZE, dtype=np.uint64
)


def dhash64(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.frombuffer(g.tobytes(), dtype=np.uint8).reshape((hash_size, hash_size + 1))
    bits = (arr[:, :-1] > arr[:, 1:]).reshape(-1).astype(np.uint64)
    if hash_size == _DHASH_HASH_SIZE:
        powers = _DHASH_POWERS
    else:
        powers = np.uint64(1) << np.arange(hash_size * hash_size, dtype=np.uint64)
    return int((bits * powers).sum(dtype=np.uint64))


def safe_image_hash(path: Path):
    try:
        with Image.open(path) as im:
            try:
                im.draft("L", (_DHASH_HASH_SIZE + 1, _DHASH_HASH_SIZE))
            except Exception:
                pass
            im2 = im.convert("RGB").copy()
        return dhash64(im2)
    except Exception:
        return None


def resolve_train_image_path(chain_id: int, image_name: str) -> Path:
    return TRAIN_IMAGES_DIR / str(chain_id) / image_name


def resolve_test_image_path(chain_id: int, image_name: str) -> Path:
    return TEST_IMAGES_DIR / str(chain_id) / image_name


MAX_HOTELS = 6000
MAX_IMAGES_PER_HOTEL = 24
TOPK_RETRIEVAL = 60
VOTE_TOPN = 5
BUCKET_RADIUS = 4

t0 = time.time()

hotel_freq = train["hotel_id"].value_counts()
top_hotels = hotel_freq.head(MAX_HOTELS).index.tolist()
train_small = train[train["hotel_id"].isin(top_hotels)].copy()

train_small = train_small.sort_values(
    ["hotel_id", "timestamp", "image"], kind="mergesort"
).reset_index(drop=True)
gsize = (
    train_small.groupby("hotel_id", sort=False)["image"].transform("size").to_numpy()
)
gpos = train_small.groupby("hotel_id", sort=False).cumcount().to_numpy()

n = gsize.astype(np.int64)
m = int(MAX_IMAGES_PER_HOTEL)

keep = n <= m
if np.any(~keep):
    denom = m - 1
    num = gpos * denom
    i = np.rint(num / np.maximum(1, (n - 1))).astype(np.int64)
    i = np.clip(i, 0, m - 1)
    target_pos = np.rint(i * (n - 1) / denom).astype(np.int64)
    keep = keep | (gpos == target_pos)

train_small = train_small.loc[keep].reset_index(drop=True)

if HASH_CACHE.exists():
    try:
        with open(HASH_CACHE, "r") as f:
            phash_cache = json.load(f)
    except Exception:
        phash_cache = {}
else:
    phash_cache = {}

bucket_index = None
indexed_count = 0
meta = {}
if INDEX_CACHE.exists():
    try:
        with open(INDEX_CACHE, "r") as f:
            index_data = json.load(f)
        meta = index_data.get("meta", {})
        if (
            meta.get("MAX_HOTELS") == MAX_HOTELS
            and meta.get("MAX_IMAGES_PER_HOTEL") == MAX_IMAGES_PER_HOTEL
            and meta.get("hash") == "dhash64"
        ):
            bucket_index = {
                int(k): [(int(h), str(hid)) for h, hid in v]
                for k, v in index_data["bucket_index"].items()
            }
            indexed_count = int(index_data.get("indexed_count", 0))
    except Exception:
        bucket_index = None

need_rebuild = (bucket_index is None) or (indexed_count < len(train_small))

if need_rebuild:
    bucket_index = defaultdict(list)

    rows = list(train_small.itertuples(index=False))
    missing_tasks = []
    computed = 0
    missing_imgs = 0

    for row in rows:
        img_name = str(row.image)
        chain_id = int(row.chain)
        hotel_id = str(row.hotel_id)
        key = f"tr::{chain_id}/{img_name}"
        if key in phash_cache:
            h = int(phash_cache[key])
            bucket = (h >> 48) & 0xFFFF
            bucket_index[int(bucket)].append((int(h), hotel_id))
            computed += 1
        else:
            missing_tasks.append((key, chain_id, img_name, hotel_id))

    def _compute_one(task):
        key, chain_id, img_name, hotel_id = task
        img_path = resolve_train_image_path(chain_id, img_name)
        h = safe_image_hash(img_path)
        if h is None:
            return None
        bucket = (int(h) >> 48) & 0xFFFF
        return (key, int(h), int(bucket), hotel_id)

    max_workers = min(32, (os.cpu_count() or 4) * 2)
    if missing_tasks:
        chunksize = 64
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for res in ex.map(_compute_one, missing_tasks, chunksize=chunksize):
                if res is None:
                    missing_imgs += 1
                    continue
                key, h, bucket, hotel_id = res
                phash_cache[key] = int(h)
                bucket_index[int(bucket)].append((int(h), str(hotel_id)))
                computed += 1

    with open(HASH_CACHE, "w") as f:
        json.dump(phash_cache, f, separators=(",", ":"))
    serializable_index = {str(k): v for k, v in bucket_index.items()}
    with open(INDEX_CACHE, "w") as f:
        json.dump(
            {
                "bucket_index": serializable_index,
                "indexed_count": computed,
                "meta": {
                    "MAX_HOTELS": MAX_HOTELS,
                    "MAX_IMAGES_PER_HOTEL": MAX_IMAGES_PER_HOTEL,
                    "hash": "dhash64",
                },
            },
            f,
            separators=(",", ":"),
        )

    print(
        f"Built hash index: {computed} train images hashed, missing/unreadable={missing_imgs}, buckets={len(bucket_index)}"
    )
else:
    print(
        f"Loaded hash index from cache: buckets={len(bucket_index)}; indexed_count={indexed_count}"
    )

bucket_h_arr = {}
bucket_hid_arr = {}
for b, lst in bucket_index.items():
    if not lst:
        continue
    hs = np.fromiter((int(x[0]) for x in lst), dtype=np.uint64, count=len(lst))
    hids = np.array([str(x[1]) for x in lst], dtype=object)
    bucket_h_arr[int(b)] = hs
    bucket_hid_arr[int(b)] = hids

bucket_window_h = {}
bucket_window_hid = {}
for b in bucket_h_arr.keys():
    hs_list = []
    hid_list = []
    for delta in range(-BUCKET_RADIUS, BUCKET_RADIUS + 1):
        nb = (int(b) + delta) & 0xFFFF
        hs = bucket_h_arr.get(nb)
        if hs is not None and hs.size:
            hs_list.append(hs)
            hid_list.append(bucket_hid_arr[nb])
    if hs_list:
        bucket_window_h[int(b)] = np.concatenate(hs_list, axis=0)
        bucket_window_hid[int(b)] = np.concatenate(hid_list, axis=0)

bucket_window_hid_codes = {}
bucket_window_code_to_hid = {}
for b, hids in bucket_window_hid.items():
    uniq = np.unique(hids)
    code_map = {hid: i for i, hid in enumerate(uniq.tolist())}
    codes = np.fromiter(
        (code_map[x] for x in hids.tolist()), dtype=np.int32, count=hids.size
    )
    bucket_window_hid_codes[int(b)] = codes
    bucket_window_code_to_hid[int(b)] = uniq.astype(object)

print(f"Hash/index prep time: {time.time() - t0:.1f}s")

if hasattr(np, "bit_count"):
    _np_bitcount = np.bit_count

    def _hamming_vec(th_u64: np.uint64, hs_u64: np.ndarray) -> np.ndarray:
        return _np_bitcount(np.bitwise_xor(hs_u64, th_u64)).astype(np.int16, copy=False)

else:

    def _hamming_vec(th_u64: np.uint64, hs_u64: np.ndarray) -> np.ndarray:
        xor = np.bitwise_xor(hs_u64, th_u64)
        return np.fromiter(
            (int(x).bit_count() for x in xor.tolist()), dtype=np.int16, count=len(xor)
        )


def retrieval_top5_for_test(img_name: str, chain_id: int | None):
    test_path = None
    if chain_id is not None and TEST_IMAGES_DIR.exists():
        p = resolve_test_image_path(int(chain_id), img_name)
        if p.exists():
            test_path = p
    if test_path is None:
        test_path = TEST_BASENAME_TO_PATH.get(img_name, None)
    if test_path is None:
        return None

    key = f"te::{(chain_id if chain_id is not None else 'na')}/{img_name}"
    if key in phash_cache:
        th = int(phash_cache[key])
    else:
        h = safe_image_hash(test_path)
        if h is None:
            return None
        th = int(h)
        phash_cache[key] = int(th)

    bucket = (th >> 48) & 0xFFFF

    hs_cat = bucket_window_h.get(int(bucket))
    if hs_cat is None or hs_cat.size == 0:
        return None

    dists = _hamming_vec(np.uint64(th), hs_cat)

    if dists.size > TOPK_RETRIEVAL:
        idx = np.argpartition(dists, TOPK_RETRIEVAL - 1)[:TOPK_RETRIEVAL]
        idx = idx[np.argsort(dists[idx], kind="mergesort")]
    else:
        idx = np.argsort(dists, kind="mergesort")

    codes = bucket_window_hid_codes[int(bucket)][idx]
    dsel = dists[idx].astype(np.int16, copy=False)
    ranks = np.arange(idx.size, dtype=np.int16)
    w = np.maximum(1, 64 - dsel).astype(np.int32) + (TOPK_RETRIEVAL - ranks).clip(
        min=0
    ).astype(np.int32)
    scores = np.bincount(
        codes, weights=w, minlength=bucket_window_code_to_hid[int(bucket)].size
    )
    if scores.size == 0:
        return None

    best_codes = np.argsort(-scores, kind="mergesort")[:VOTE_TOPN]
    best = [
        str(bucket_window_code_to_hid[int(bucket)][c])
        for c in best_codes.tolist()
        if scores[c] > 0
    ]

    if chain_id is not None:
        pad = chain_top5.get(int(chain_id), global_top5)
    else:
        pad = global_top5

    out = []
    for x in best + list(map(str, pad)) + list(map(str, global_top5)):
        if x not in out:
            out.append(x)
        if len(out) == 5:
            break
    if len(out) < 5:
        out = (out + list(map(str, global_top5)))[:5]
    return out




## === cell 2
sub = sample.copy()

preds = []
unknown_chain = 0
used_retrieval = 0
retrieval_failed = 0

images = sub["image"].astype(str).to_numpy()

for img in images:
    base = Path(img).name
    ch = image_to_chain.get(base, None)
    if ch is None:
        unknown_chain += 1

    top5 = retrieval_top5_for_test(base, ch)
    if top5 is not None:
        preds.append(" ".join(top5))
        used_retrieval += 1
    else:
        retrieval_failed += 1
        if ch is None:
            preds.append(global_pred_str)
        else:
            ids = chain_top5.get(int(ch), global_top5)
            preds.append(" ".join(list(map(str, ids))))

sub["hotel_id"] = preds

out_path = WORKING / "submission.csv"
sub.to_csv(out_path, index=False)

print(f"Unknown chain (fallback risk): {unknown_chain} / {len(sub)}")
print(f"Used retrieval for: {used_retrieval} / {len(sub)}")
print(f"Retrieval failed for: {retrieval_failed} / {len(sub)}")
print(f"Wrote submission to: {out_path}")
print(sub.head(3).to_string(index=False))

with open(HASH_CACHE, "w") as f:
    json.dump(phash_cache, f, separators=(",", ":"))



## === cell 3
import pandas as pd
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT_SAMPLE = Path("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv")
if not INPUT_SAMPLE.exists():
    INPUT_SAMPLE = Path("/kaggle/input/sample_submission.csv")

dst = WORKING / "submission.csv"
if not dst.exists():
    raise FileNotFoundError(f"Expected submission at {dst} not found.")

sample_chk = pd.read_csv(INPUT_SAMPLE)
sub_chk = pd.read_csv(dst)

if list(sub_chk.columns) != ["image", "hotel_id"]:
    raise ValueError(f"submission.csv has wrong columns: {list(sub_chk.columns)}")

if sub_chk.shape[0] != sample_chk.shape[0]:
    raise ValueError(
        f"submission.csv row count {sub_chk.shape[0]} != sample_submission row count {sample_chk.shape[0]}"
    )

if not sub_chk["image"].equals(sample_chk["image"]):
    if set(sub_chk["image"]) == set(sample_chk["image"]):
        sub_chk = sub_chk.set_index("image").loc[sample_chk["image"]].reset_index()
        sub_chk.to_csv(dst, index=False)
    else:
        raise ValueError(
            "submission.csv images do not match sample_submission images (set mismatch)."
        )

print(f"Using submission file: {dst}")
print(sub_chk.head(3).to_string(index=False))
print("\nsubmission.csv written to /kaggle/working/submission.csv")
