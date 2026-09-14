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

0.6702281720973288

# 6. Current score

0.00125

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it depends on external wheels (`pekolib*`) that are not present in the provided dataset tree, so the pipeline fails before producing `submission.csv`. I remove those installs and replace them with a minimal, self-contained submission generator that always runs end-to-end on the provided `/kaggle/input/hotel-id-2021-fgvc8/` data and writes a valid `submission.csv` in the required `image,hotel_id` format. To get a non-trivial baseline MAP@5 without changing “modeling” complexity, it predict the 5 most frequent `hotel_id` values from `train.csv` for every test image (a common safe baseline that should score above random and, crucially, yields a valid submission). This moves you from “no score” to a real score, which is the necessary first step toward the target.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.6702), so we should improve the baseline while keeping the “predict 5 hotel_ids per image” core logic intact. The smallest meaningful step is to replace the global top-5 hotel prior with a chain-aware prior: for each test image, infer its `chain` from the test image folder name (0–89) and then predict the top-5 `hotel_id` within that chain from `train.csv`. This uses only provided metadata and file structure (no extra models), preserves evaluation semantics, and should move MAP@5 substantially toward the target compared to a single global prior. We also add safe fallbacks when a chain folder is missing/unknown, ensuring a valid submission is always produced.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.6702), so we need a meaningful lift while keeping your “chain-aware top-5 prior” core logic. The biggest issue is that the notebook likely fails to infer chains for most test images because the hidden test set uses a different folder layout than your existence-check loop, so you silently fall back to the global prior almost everywhere. I replace per-image `os.path.exists` probing with a fast, robust lookup that builds an `image -> chain` map by scanning the actual `test_images` directory structure once, and then uses that map for predictions. This preserves the same prediction strategy (top-5 within inferred chain, else global top-5) but should substantially increase chain coverage and move MAP@5 toward the target.'
- What this solution (achieved 0.00209) has done: 'Your current 0.00209 is far below the 0.6702 target, and the main reason is that your chain inference almost always falls back to the global prior because the test images are not organized into numeric chain subfolders in the released/hidden test set. The smallest meaningful improvement that preserves your “metadata-prior top-5” core logic is to add a timestamp-aware split and a hotel-to-hotel co-occurrence prior: for each `hotel_id`, learn its most common “neighbors” within the same chain from training (using near-in-time images), then predict a chain prior list expanded/re-ranked by these neighbors. This still uses only `train.csv` metadata (no image model), keeps the same submission semantics (5 IDs per image), and typically lifts MAP@5 significantly versus pure frequency priors. We also keep your robust submission writing and add a strict guarantee of exactly 5 unique IDs per row.'
- What this solution (achieved 0.00184) has done: 'Your current MAP@5 (0.00209) is extremely far below the target (0.6702), and the main culprit is that the “chain-aware” prior almost never activates on the hidden test because test images are not stored under numeric chain folders—so you mostly submit the weak global top-5. The smallest meaningful improvement that preserves your “metadata-only priors” core logic is to (1) infer chain from the test image filename prefix using the `sample_submission.csv` structure (common in this dataset) and (2) when chain is unknown, back off to a stronger per-hotel popularity prior (top-5 hotels overall, but with chain-conditioned smoothing) while guaranteeing exactly 5 unique IDs. These changes keep the same modeling approach (frequency/neighbor priors, no image model), but should significantly increase the fraction of rows using a relevant chain prior, moving your score materially toward the target.'
- What this solution (achieved 0.00119) has done: 'Your current MAP@5 is far below the target, and the main reason is that the code still can’t infer a meaningful `chain` for most hidden test images (so it mostly falls back to a weak global top-5). To move score upward without changing the “metadata-only priors + neighbors” core logic, I add a robust `image -> chain` inference by scanning the actual `test_images` tree for *both* possible layouts (flat and chain-subfoldered) and also parsing chain from the *relative path* when available. Then I strengthen the unknown-chain fallback slightly by using a “chain=0 prior” (hotels that are common among non-chain/unknown in train) before falling back to global top-5, which is still the same frequency-prior approach. I also guarantee exactly 5 unique IDs per row deterministically.'
- What this solution (achieved 0.00085) has done: 'Main runtime is dominated by (1) hashing up to 25k training images and (2) per-test-image candidate building + Hamming computations, plus unnecessary repeated filesystem `exists()` checks and Python-loop overhead in `dhash_64` and `Counter`. I keep the exact retrieval + fallback logic, but make equivalent low-level optimizations: speed up `dhash_64` bit packing using `np.packbits`, remove repeated `os.path.exists` calls by pre-indexing the test image paths once, avoid building large Python lists/Counter by using `np.unique(..., return_counts=True)` for voting, and reduce candidate-index concatenation overhead by preallocating. These changes preserve semantics and determinism while cutting constant factors substantially so the notebook fits under 600 seconds.'
- What this solution (achieved 0.00132) has done: 'Your current score is far below target, so we need a real quality lift without changing the core “dHash retrieval + vote, else chain/priors fallback” approach. The biggest accuracy issue in your retrieval is that the vote is based only on raw neighbor counts, which can be noisy; a minimal, metric-aligned improvement is to weight votes by similarity (lower Hamming distance) while keeping the same retrieved set and the same top-N selection. I also add a small, safe chain-aware prior blending step only inside the retrieval branch to stabilize cases where the top matches are split across many hotels, without changing the fallback logic. These changes keep the same architecture/loops and should move MAP@5 meaningfully upward toward the target while preserving runtime constraints.'
- What this solution (achieved 0.00148) has done: 'Main bottlenecks are (1) hashing up to 25k train images with PIL (I/O bound) and (2) per-test-image Python overhead inside the retrieval loop (building `set(prior_list)` and list-comprehension bonus each time, repeated path existence checks). I keep the exact hashing + retrieval/voting logic, but reduce constant factors by caching pure computations (per-chain prior sets/bonuses and per-chain top5), avoiding repeated `os.path.exists` calls with a prebuilt train path set, and parallelizing *test* hashing with the same ThreadPool approach used for train hashing. These are provably equivalent (same dhash, same candidate selection, same scoring), just moving repeated work out of the hot loops and doing independent per-image work concurrently. This should cut wall time substantially without changing evaluation semantics.'
- What this solution (achieved 0.00139) has done: 'Your current score (0.00148) is extremely far below the target (0.6702), so we need a meaningful quality lift while keeping the same “dHash retrieval + weighted vote, else chain/priors fallback” core logic. The biggest issue is that your retrieval candidate generation is too restrictive (16-bit prefix buckets), so you rarely retrieve truly similar images and the fallback prior dominates; we can safely widen the candidate pool by reducing the prefix to 12 bits and slightly increasing TOP_MATCHES while keeping the same hashing, voting, and post-processing. I also cap per-chain prior bonus computation overhead by prebuilding a compact per-chain prior array and keep determinism/semantics identical. These are minimal parameter-level changes that should increase the chance the correct hotel appears in the top-5 and move MAP@5 upward toward the target.'
- What this solution (achieved 0.00125) has done: 'Your current score (0.00139) is far below the target (0.6702), so we need a meaningful lift while preserving your existing “dHash retrieval + weighted vote, else chain/priors fallback” logic. The main issue is that your retrieval hash is too lossy (8×8=64-bit dHash), making nearest-neighbor voting mostly noise; a minimal, same-approach improvement is to increase hash resolution (still dHash, same method) to 16×16 (256 bits) and update Hamming distance accordingly. This keeps the identical pipeline structure (hash → candidate bucket → Hamming → weighted vote → top5) but substantially increases discriminative power and should move MAP@5 upward. I also keep runtime bounded by reducing candidate minimums and TOP_MATCHES slightly to offset the heavier 256-bit distance computation, without changing the overall training/inference approach or adding new dependencies.'

# 9. Code solution

## === cell 0
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from PIL import Image

DATA_DIR = "/kaggle/input/hotel-id-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SUB_PATH = "submission.csv"

print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))

train = pd.read_csv(TRAIN_CSV)
sample = pd.read_csv(SAMPLE_SUB)

train["hotel_id"] = train["hotel_id"].astype(int)
train["chain"] = train["chain"].astype(int)
train["timestamp"] = (
    pd.to_numeric(train["timestamp"], errors="coerce").fillna(0).astype("int64")
)

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 * 5)[:5]

unknown_chain_top5 = (
    train.loc[train["chain"] == 0, "hotel_id"]
    .value_counts()
    .head(5)
    .index.astype(str)
    .tolist()
)
if len(unknown_chain_top5) < 5:
    unknown_chain_top5 = (unknown_chain_top5 + global_top5)[:5]
unknown_chain_top5 = (unknown_chain_top5 * 5)[:5]

K_PRIOR = 50
chain_prior = (
    train.groupby("chain")["hotel_id"]
    .value_counts()
    .groupby(level=0)
    .head(K_PRIOR)
    .reset_index(name="cnt")
)
chain_prior_map = {}
for ch, grp in chain_prior.groupby("chain"):
    chain_prior_map[int(ch)] = grp["hotel_id"].astype(int).tolist()

NEIGHBORS_PER_HOTEL = 10
neighbor_counts = {}
for ch, dfc in (
    train[["chain", "hotel_id", "timestamp"]]
    .sort_values(["chain", "timestamp"])
    .groupby("chain", sort=False)
):
    h = dfc["hotel_id"].to_numpy()
    for a, b in zip(h[:-1], h[1:]):
        if a == b:
            continue
        key_a = (int(ch), int(a))
        key_b = (int(ch), int(b))
        da = neighbor_counts.get(key_a)
        if da is None:
            da = {}
            neighbor_counts[key_a] = da
        db = neighbor_counts.get(key_b)
        if db is None:
            db = {}
            neighbor_counts[key_b] = db
        da[int(b)] = da.get(int(b), 0) + 1
        db[int(a)] = db.get(int(a), 0) + 1

neighbor_top_map = {}
for key, d in neighbor_counts.items():
    top = sorted(d.items(), key=lambda x: (-x[1], x[0]))[:NEIGHBORS_PER_HOTEL]
    neighbor_top_map[key] = [hid for hid, _ in top]

_chain_prefix_re = re.compile(r"^(\d{1,3})[^0-9]")
_chain_prefix_alt_re = re.compile(r"^(\d{1,3})")


def infer_chain_from_filename(fn: str):
    base = os.path.basename(fn)
    m = _chain_prefix_re.match(base)
    if m is None:
        m = _chain_prefix_alt_re.match(base)
    if m is None:
        return None
    try:
        ch = int(m.group(1))
    except Exception:
        return None
    if 0 <= ch <= 500:
        return ch
    return None


def build_test_image_chain_maps(test_img_dir: str):
    fn_to_chain = {}
    rel_to_chain = {}

    if not os.path.isdir(test_img_dir):
        return fn_to_chain, rel_to_chain

    candidate_roots = [test_img_dir]
    nested = os.path.join(test_img_dir, "test_images")
    if os.path.isdir(nested):
        candidate_roots.append(nested)

    for root_base in candidate_roots:
        for root, dirs, files in os.walk(root_base):
            parent = os.path.basename(root)
            try:
                ch = int(parent)
            except ValueError:
                ch = None

            for fn in files:
                if not fn.lower().endswith(".jpg"):
                    continue
                if ch is not None and fn not in fn_to_chain:
                    fn_to_chain[fn] = ch
                rel = os.path.relpath(os.path.join(root, fn), root_base).replace(
                    "\\", "/"
                )
                if ch is not None and rel not in rel_to_chain:
                    rel_to_chain[rel] = ch

    return fn_to_chain, rel_to_chain


image_to_chain_from_fn, image_to_chain_from_rel = build_test_image_chain_maps(
    TEST_IMG_DIR
)


def infer_chain_for_image(img_name: str):
    ch = image_to_chain_from_fn.get(img_name)
    if ch is not None:
        return int(ch)

    ch = image_to_chain_from_rel.get(img_name.replace("\\", "/"))
    if ch is not None:
        return int(ch)

    norm = img_name.replace("\\", "/")
    parts = norm.split("/")
    if len(parts) >= 2:
        try:
            ch2 = int(parts[-2])
            return ch2
        except Exception:
            pass

    return infer_chain_from_filename(img_name)


def make_top5_for_chain(ch: int):
    base = chain_prior_map.get(ch, [])
    if not base:
        return [int(x) for x in global_top5]

    seed = base[:10]
    cand = []
    for hid in seed:
        cand.append(int(hid))
        cand.extend([int(x) for x in neighbor_top_map.get((ch, int(hid)), [])])

    seen = set()
    out = []
    for hid in cand:
        if hid in seen:
            continue
        seen.add(hid)
        out.append(hid)
        if len(out) >= 5:
            return out

    for hid in base:
        hid = int(hid)
        if hid in seen:
            continue
        seen.add(hid)
        out.append(hid)
        if len(out) >= 5:
            return out

    for s in global_top5:
        hid = int(s)
        if hid in seen:
            continue
        seen.add(hid)
        out.append(hid)
        if len(out) >= 5:
            return out

    return (out * 5)[:5]


def finalize_top5(top5_int, fallback_list_str):
    seen = set()
    out = []
    for hid in top5_int:
        hid = int(hid)
        if hid in seen:
            continue
        seen.add(hid)
        out.append(str(hid))
        if len(out) == 5:
            return out
    for s in fallback_list_str:
        if s in out:
            continue
        out.append(s)
        if len(out) == 5:
            return out
    return (out * 5)[:5]


_cached_chain_top5_str = {}
for ch in chain_prior_map.keys():
    _cached_chain_top5_str[int(ch)] = finalize_top5(
        make_top5_for_chain(int(ch)), global_top5
    )

print("Loaded train/sample.")
print("Global top5:", global_top5)



## === cell 1
HASH_SIZE = 16  # 16x16 => 256 comparisons => 256-bit hash stored as Python int


def dhash(img: Image.Image, hash_size: int = HASH_SIZE) -> int:
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.uint8)  # (hash_size, hash_size+1)
    diff = (
        (a[:, :-1] > a[:, 1:]).reshape(-1).astype(np.uint8)
    )  # hash_size*hash_size bits
    packed = np.packbits(diff, bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little", signed=False)


def build_test_image_path_map(test_img_dir: str):
    path_map = {}
    if not os.path.isdir(test_img_dir):
        return path_map

    candidate_roots = [test_img_dir]
    nested = os.path.join(test_img_dir, "test_images")
    if os.path.isdir(nested):
        candidate_roots.append(nested)

    for root_base in candidate_roots:
        for root, _, files in os.walk(root_base):
            for fn in files:
                if fn.lower().endswith(".jpg") and fn not in path_map:
                    path_map[fn] = os.path.join(root, fn)
    return path_map


_test_path_map = build_test_image_path_map(TEST_IMG_DIR)


def resolve_image_path(base_dir: str, image_name: str) -> str:
    p = _test_path_map.get(image_name)
    if p is not None:
        return p
    p1 = os.path.join(base_dir, image_name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(base_dir, "test_images", image_name)
    if os.path.exists(p2):
        return p2
    p3 = os.path.join(base_dir, image_name.replace("\\", "/"))
    if os.path.exists(p3):
        return p3
    p4 = os.path.join(base_dir, "test_images", image_name.replace("\\", "/"))
    if os.path.exists(p4):
        return p4
    return p1  # fallback


def resolve_train_path(train_img_dir: str, chain: int, image_name: str) -> str:
    return os.path.join(train_img_dir, str(chain), image_name)


MAX_TRAIN_IMAGES_TOTAL = 25000
MAX_IMAGES_PER_HOTEL = 5

TOP_MATCHES = 60  # was 80

t0 = time.time()

hotel_freq = train["hotel_id"].value_counts()
train_sorted = train.copy()
train_sorted["freq"] = train_sorted["hotel_id"].map(hotel_freq)
train_sorted = train_sorted.sort_values(["freq", "timestamp"], ascending=[False, False])

_existing_train_paths = set()
if os.path.isdir(TRAIN_IMG_DIR):
    for ch_name in os.listdir(TRAIN_IMG_DIR):
        ch_dir = os.path.join(TRAIN_IMG_DIR, ch_name)
        if not os.path.isdir(ch_dir):
            continue
        try:
            int(ch_name)
        except Exception:
            continue
        for fn in os.listdir(ch_dir):
            if fn.lower().endswith(".jpg"):
                _existing_train_paths.add(os.path.join(ch_dir, fn))

selected = []
per_hotel_count = defaultdict(int)
for row in train_sorted.itertuples(index=False):
    if len(selected) >= MAX_TRAIN_IMAGES_TOTAL:
        break
    hid = int(row.hotel_id)
    if per_hotel_count[hid] >= MAX_IMAGES_PER_HOTEL:
        continue
    ch = int(row.chain)
    img_name = str(row.image)
    path = resolve_train_path(TRAIN_IMG_DIR, ch, img_name)
    if path not in _existing_train_paths:
        continue
    selected.append((path, hid))
    per_hotel_count[hid] += 1


def _hash_one(item):
    path, hid = item
    try:
        with Image.open(path) as im:
            h = dhash(im)
        return (h, hid, 0)
    except Exception:
        return (0, 0, 1)


n_workers = min(32, (os.cpu_count() or 4) * 2)
train_hashes = []
read_fail = 0
with ThreadPoolExecutor(max_workers=n_workers) as ex:
    for h, hid, failed in ex.map(_hash_one, selected, chunksize=128):
        read_fail += failed
        if not failed:
            train_hashes.append((h, hid))

print(
    f"Built train hashes: {len(train_hashes)} (unique hotels: {len(per_hotel_count)}) read_fail={read_fail} time={time.time()-t0:.1f}s"
)

USE_RETRIEVAL = len(train_hashes) >= 500

if USE_RETRIEVAL:
    train_hash_arr = np.array([h for h, _ in train_hashes], dtype=object)
    train_hid_arr = np.fromiter(
        (hid for _, hid in train_hashes), dtype=np.int32, count=len(train_hashes)
    )

    PREFIX_BITS = 12
    _prefix_shift = (HASH_SIZE * HASH_SIZE) - PREFIX_BITS  # 256 - 12

    def _get_prefix_int(h: int) -> int:
        return int((h >> _prefix_shift) & ((1 << PREFIX_BITS) - 1))

    train_prefix = np.fromiter(
        (_get_prefix_int(int(h)) for h in train_hash_arr),
        dtype=np.uint16,
        count=len(train_hash_arr),
    )

    order = np.argsort(train_prefix, kind="stable")
    train_prefix_sorted = train_prefix[order]
    uniq, start_idx, counts = np.unique(
        train_prefix_sorted, return_index=True, return_counts=True
    )
    prefix_to_range = {
        int(p): (int(s), int(s + c)) for p, s, c in zip(uniq, start_idx, counts)
    }
    train_order = order

    _pc8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount16(xu16: np.ndarray) -> np.ndarray:
        lo = (xu16 & np.uint16(0xFF)).astype(np.uint8)
        hi = (xu16 >> np.uint16(8)).astype(np.uint8)
        return (_pc8[lo] + _pc8[hi]).astype(np.uint8)

    existing_prefixes = uniq.astype(np.uint16)
    _prefix_neighbors_cache = {}

    def _get_prefix_neighbors_for(p: int):
        v = _prefix_neighbors_cache.get(p)
        if v is not None:
            return v
        d = _popcount16(np.bitwise_xor(existing_prefixes, np.uint16(p))).astype(
            np.uint8
        )
        ord2 = np.lexsort((existing_prefixes.astype(np.uint32), d.astype(np.uint32)))
        out = (existing_prefixes[ord2], d[ord2])
        _prefix_neighbors_cache[p] = out
        return out

    def get_candidate_indices_for_hash(th_int: int, min_candidates: int) -> np.ndarray:
        p = _get_prefix_int(th_int)
        neigh, dists = _get_prefix_neighbors_for(p)

        total = 0
        last_dist = -1
        used = []
        for pref, dd in zip(neigh, dists):
            dd = int(dd)
            if dd != last_dist and total >= min_candidates:
                break
            last_dist = dd
            rng = prefix_to_range.get(int(pref))
            if rng is None:
                continue
            s, e = rng
            used.append((s, e))
            total += e - s

        if total == 0:
            return np.empty((0,), dtype=np.int64)

        out = np.empty((total,), dtype=np.int64)
        pos = 0
        for s, e in used:
            seg = train_order[s:e]
            n = seg.size
            out[pos : pos + n] = seg
            pos += n
        return out

    def hamming_int(a: int, b: int) -> int:
        return (a ^ b).bit_count()




## === cell 2
images = sample["image"].astype(str).tolist()

preds = [""] * len(images)
t1 = time.time()
SIM_POWER = 2.0
W_PRIOR = 0.30

_chain_prior_arr = {}
for ch, lst in chain_prior_map.items():
    if lst:
        _chain_prior_arr[int(ch)] = np.asarray(lst[:K_PRIOR], dtype=np.int32)


def _predict_one(idx_img):
    idx, img_name = idx_img
    top5 = None
    test_read_fail_local = 0

    if USE_RETRIEVAL:
        path = resolve_image_path(TEST_IMG_DIR, img_name)
        try:
            with Image.open(path) as im:
                th_int = int(dhash(im))

            cand_idx = get_candidate_indices_for_hash(
                th_int, min_candidates=max(TOP_MATCHES * 50, 4000)
            )
            if cand_idx.size == 0:
                raise RuntimeError("No candidates")

            cand_hid = train_hid_arr[cand_idx]
            d = np.fromiter(
                (hamming_int(int(train_hash_arr[i]), th_int) for i in cand_idx),
                dtype=np.int16,
                count=cand_idx.size,
            )

            if d.size > TOP_MATCHES:
                topk = np.argpartition(d, TOP_MATCHES - 1)[:TOP_MATCHES]
            else:
                topk = np.arange(d.size)
            topk = topk[np.argsort(d[topk], kind="stable")]

            top_hids = cand_hid[topk].astype(np.int32, copy=False)
            top_d = d[topk].astype(np.float32, copy=False)

            w = 1.0 / np.power(1.0 + top_d, SIM_POWER)

            uh, inv = np.unique(top_hids, return_inverse=True)
            score = np.bincount(inv, weights=w, minlength=uh.size).astype(np.float32)

            ch = infer_chain_for_image(img_name)
            if ch is not None:
                prior_arr = _chain_prior_arr.get(int(ch))
                if prior_arr is not None and prior_arr.size:
                    prior_bonus = np.isin(uh, prior_arr).astype(
                        np.float32
                    ) * np.float32(W_PRIOR)
                    score = score + prior_bonus

            ordv = np.lexsort((uh.astype(np.int64), (-score).astype(np.float64)))
            top5_int = uh[ordv][:5].astype(int).tolist()

            top5 = finalize_top5(top5_int, global_top5)
        except Exception:
            test_read_fail_local = 1
            top5 = None

    if top5 is None:
        ch = infer_chain_for_image(img_name)
        if ch is not None:
            top5 = _cached_chain_top5_str.get(int(ch))
            if top5 is None:
                top5_int = make_top5_for_chain(int(ch))
                top5 = finalize_top5(top5_int, global_top5)
                _cached_chain_top5_str[int(ch)] = top5
        else:
            top5 = finalize_top5([int(x) for x in unknown_chain_top5], global_top5)

    return idx, " ".join(top5), test_read_fail_local


test_read_fail = 0
n_workers_pred = min(32, (os.cpu_count() or 4) * 2)
with ThreadPoolExecutor(max_workers=n_workers_pred) as ex:
    for idx, pred, failed in ex.map(_predict_one, enumerate(images), chunksize=64):
        preds[idx] = pred
        test_read_fail += failed

sub = pd.DataFrame({"image": images, "hotel_id": preds})
assert list(sub.columns) == ["image", "hotel_id"]
assert len(sub) == len(sample)
sub.to_csv(SUB_PATH, index=False)

print("Wrote:", SUB_PATH, "rows:", len(sub))
print(sub.head())
print(
    f"Retrieval enabled: {USE_RETRIEVAL}, test_read_fail={test_read_fail}, time_pred={time.time()-t1:.1f}s"
)



## === cell 3
with open(SUB_PATH, "r", encoding="utf-8") as f:
    for i in range(6):
        print(f.readline().rstrip("\n"))
