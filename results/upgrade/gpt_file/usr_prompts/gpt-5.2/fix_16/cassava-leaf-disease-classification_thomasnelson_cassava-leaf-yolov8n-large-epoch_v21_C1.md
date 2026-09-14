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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.870806890299184

# 6. Current score

0.21151

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The crash is coming from a protobuf/TensorFlow incompatibility in this Python 3.13 environment (protobuf’s `MessageFactory.GetPrototype` is missing), which prevents TensorFlow from importing and therefore stops training, leaving `model` undefined for inference. The minimal fix is to avoid TensorFlow entirely and switch to a pure-Python baseline that still produces a valid `submission.csv`. To move accuracy toward the target without external ML libraries, we use a simple but strong prior for this dataset: predict the most frequent class from `train.csv` for all test images (a common baseline that typically lands around the class prior accuracy). This keeps the pipeline deterministic, fast, and guarantees a correctly formatted submission.'
- What this solution (achieved 0.11584) has done: 'You’re currently using a pure majority-class baseline, which caps accuracy near the largest class prior (matching your ~0.611 score). To move closer to the 0.8708 target without changing the overall “no-TensorFlow / fast / deterministic” approach, the smallest legitimate improvement is to use a slightly more informative prior: predict class-4 (“healthy”) for all test images. In Cassava, “healthy” is typically the dominant label and commonly yields a higher baseline than the empirical majority in some splits/kernels; this change is minimal (just the chosen constant) and preserves the same submission semantics. I keep the majority-label computation and printouts for transparency, but override the final prediction label to 4 to push the score upward toward your target.'
- What this solution (achieved 0.16854) has done: 'Your current submission predicts a single constant class (4), which is too weak and leaves a large gap to the 0.8708 target. To move accuracy upward with minimal, fully deterministic changes and without introducing ML libraries, we keep the same “pure-Python/pandas” approach but switch from a constant predictor to a nearest-neighbor lookup in a simple perceptual-hash space computed from the provided JPEGs. This preserves the overall pipeline structure (read CSVs → create labels → write submission.csv) while using actual image content to make more informed predictions. We also include a safe fallback to the majority label if an image can’t be read, ensuring a valid submission is always produced within the time limit.'
- What this solution (achieved 0.1648) has done: 'Your current 1-NN dHash uses very few prototypes (250 per class) and a single hash scale, which makes it underfit and yields a low score. To move accuracy upward toward the 0.8708 target while preserving the same core approach (pure-Python image hashing + nearest-neighbor), I (1) increase the number of per-class prototypes (still deterministic, still fast enough), and (2) use a tiny multi-scale ensemble of dHash (two sizes) with the same 1-NN rule but a summed Hamming distance, which is a minimal extension of your existing distance computation. I also ensure the prototype selection is deterministic but more representative by sampling evenly across the whole class with a fixed seed (instead of only `head()`), without changing any learning loop (there is none). These changes should materially improve the score while keeping the pipeline and submission semantics identical.'
- What this solution (achieved 0.18572) has done: 'Your current 1-NN dHash baseline is far below the target, so we should improve feature quality without changing the overall “pure-Python image hashing + nearest-neighbor” core logic. The biggest issue is that your dHash resize is incorrect (it should be width=size+1, height=size), and fixing it alone materially improve nearest-neighbor behavior while keeping the same model semantics. To further nudge accuracy upward with minimal risk, I keep 1-NN but (a) use a standard pHash in addition to dHash and (b) weight the distances lightly, still preserving “hash features + Hamming distance + nearest prototype” logic. I also make prototype selection slightly more representative by including a few deterministic “head” samples plus seeded random samples, which improves coverage without adding training loops or external libs.'
- What this solution (achieved 0.19058) has done: 'Your current 1-NN hash search is dominated by noisy neighbors because it compares every test image against a large, unfiltered prototype set; the smallest “same-core-logic” improvement is to keep the exact same hashes + Hamming distance + 1-NN decision, but add a cheap prefilter that only evaluates the best candidates under a single coarse hash first. Concretely, we (1) add a 32-bit “bucket key” derived from `dhash8` and build a dictionary from bucket→prototype indices, and (2) at inference time, search prototypes from the matching bucket(s) (and a small deterministic set of nearby buckets via a few fixed bit-flips) before falling back to the full list if needed. This tends to improve accuracy (fewer bad comparisons) while staying deterministic, preserving the model semantics (still 1-NN on the same distances), and also speeds up enough to allow a slightly larger prototype set safely within the time limit. All paths and submission format stay identical and the script still always writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.19656) has done: 'Your current score (0.19058) is far below the target (0.87081), so we should increase accuracy with the smallest change that preserves your core “pure-Python hash features + Hamming distance + nearest-prototype” logic. The biggest low-risk gain is to switch from 1-NN to a tiny k-NN vote over the same candidate set and same distances, which reduces sensitivity to a single noisy neighbor while keeping identical features and distance computation. To keep runtime under control, we reuse your bucket prefilter and only vote over the top-k nearest among the filtered candidates. We also keep the same majority-label fallback and submission writing exactly as before.'
- What this solution (achieved 0.19619) has done: 'Your current score is far below the target, so we should improve accuracy while keeping the same core “pure-Python hash features + (weighted) Hamming distance + k-NN vote over prototypes” logic. The most leverage with minimal risk is to make the bucket prefilter less “brittle” by (1) using a slightly shorter bucket key so more true neighbors land in the candidate set, and (2) flipping a deterministic set of higher-impact bits (including some higher-order bits) instead of only low positions. To keep runtime predictable under 600s, we keep the same prototype count and k-NN, but raise MAX_CANDS moderately to benefit from the improved recall. These changes only affect candidate retrieval (not the feature extraction, distance, or voting semantics) and should move accuracy upward toward your target.'
- What this solution (achieved 0.21001) has done: 'Your current gap to the target is large (0.196 → 0.871), so the smallest safe way to push accuracy upward while preserving your exact “hash features + weighted Hamming distance + bucket-prefilter + k-NN vote” core logic is to improve candidate recall and reduce noisy distance contributions without changing the model family. I (1) make the bucket key a bit shorter and broaden the deterministic nearby-bucket search so true neighbors are less likely to be missed, and (2) down-weight the very noisy `dhash16` term (256 bits) so it doesn’t dominate distances when images are resized differently. I also slightly increase `KNN_K` and `MAX_CANDS` to stabilize voting while staying within the same inference loop and keeping runtime under the 600s constraint. The script still run end-to-end and write `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.21151) has done: 'Your current score (0.21001) is far below the target (0.87081), so we should improve accuracy while keeping the exact same core approach: hash features → weighted Hamming distances → bucket-prefilter → k-NN inverse-distance vote. The minimal high-leverage adjustment is to increase candidate recall (so true neighbors aren’t missed) by (1) shortening the bucket key a bit more and (2) expanding the deterministic nearby-bucket flips list, while keeping the same distance/voting logic. To avoid a blow-up in runtime, we also cap per-bucket candidate expansion deterministically and keep the same prototype budget, only raising `MAX_CANDS` moderately. These changes only affect which prototypes are compared (not how hashes/distances/votes are computed) and should move accuracy upward toward the target.'

# 9. Code solution

## === cell 0
import os
import sys
import random
from csv import writer

import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)



## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/working/cassava-leaf-disease-classification",
    "/kaggle/working/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]

DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.isfile(os.path.join(r, "train.csv")) and os.path.isfile(
        os.path.join(r, "sample_submission.csv")
    ):
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset root with train.csv and sample_submission.csv. "
        f"Tried: {CANDIDATE_ROOTS}"
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

CAND_TRAIN_IMG_DIRS = [
    os.path.join(DATA_ROOT, "train_images"),
    os.path.join(DATA_ROOT, "cassava-leaf-disease-classification", "train_images"),
]
CAND_TEST_IMG_DIRS = [
    os.path.join(DATA_ROOT, "test_images"),
    os.path.join(DATA_ROOT, "cassava-leaf-disease-classification", "test_images"),
]
TRAIN_IMG_DIR = next((d for d in CAND_TRAIN_IMG_DIRS if os.path.isdir(d)), None)
TEST_IMG_DIR = next((d for d in CAND_TEST_IMG_DIRS if os.path.isdir(d)), None)

if TRAIN_IMG_DIR is None or TEST_IMG_DIR is None:
    raise FileNotFoundError(
        "Could not locate train_images/test_images directories under DATA_ROOT. "
        f"Checked train={CAND_TRAIN_IMG_DIRS}, test={CAND_TEST_IMG_DIRS}"
    )

SUB_PATH = "/kaggle/working/submission.csv"

for p in [TRAIN_CSV, SAMPLE_SUB]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Required file not found: {p}")

if os.path.exists(SUB_PATH):
    os.remove(SUB_PATH)

print("DATA_ROOT     :", DATA_ROOT)
print("TRAIN_CSV     :", TRAIN_CSV)
print("SAMPLE_SUB    :", SAMPLE_SUB)
print("TRAIN_IMG_DIR :", TRAIN_IMG_DIR)
print("TEST_IMG_DIR  :", TEST_IMG_DIR)
print("SUB_PATH      :", SUB_PATH)



## === cell 2
import pandas as pd
from PIL import Image

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError(
        f"train.csv missing required columns. Found: {train_df.columns.tolist()}"
    )
if not {"image_id", "label"}.issubset(sample_sub.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns. Found: {sample_sub.columns.tolist()}"
    )

label_counts = train_df["label"].value_counts().sort_index()
majority_label = int(label_counts.idxmax())


def dhash(img_path: str, size: int = 8) -> int:
    """
    Standard dHash: resize to (size+1, size), compare adjacent pixels horizontally -> size*size bits.
    size=8 -> 64 bits, size=16 -> 256 bits.
    """
    with Image.open(img_path) as im:
        im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    diff = arr[:, 1:] > arr[:, :-1]  # (size, size)
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(bool(b))
    return h


def phash(img_path: str, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """
    Perceptual hash (pHash): DCT on resized grayscale image, take top-left low frequencies,
    threshold by median -> hash_size*hash_size bits.
    hash_size=8 -> 64 bits.
    """
    img_size = hash_size * highfreq_factor
    with Image.open(img_path) as im:
        im = im.convert("L").resize((img_size, img_size), Image.BILINEAR)
        pixels = np.asarray(im, dtype=np.float32)

    N = img_size
    x = np.arange(N, dtype=np.float32)
    u = x[:, None]
    alpha = np.sqrt(2.0 / N) * np.ones((N,), dtype=np.float32)
    alpha[0] = np.sqrt(1.0 / N)
    C = alpha[:, None] * np.cos((np.pi / (2.0 * N)) * (2.0 * x[None, :] + 1.0) * u)

    dct = C @ pixels @ C.T
    dctlow = dct[:hash_size, :hash_size]
    med = np.median(dctlow[1:, 1:])  # ignore DC for threshold robustness
    diff = dctlow > med
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(bool(b))
    return h


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


PER_CLASS = 1600

HASH_CONFIG = [
    ("dhash8", "dhash", {"size": 8}, 1.0),
    ("dhash16", "dhash", {"size": 16}, 0.25),
    ("phash8", "phash", {"hash_size": 8, "highfreq_factor": 4}, 0.9),
]

train_df = train_df.copy()
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)

rng = np.random.RandomState(SEED)

prototypes = []
for lbl in sorted(train_df["label"].unique().tolist()):
    ids = train_df.loc[train_df["label"] == lbl, "image_id"].values
    if len(ids) == 0:
        continue

    take = min(PER_CLASS, len(ids))

    head_n = min(250, take)
    head_ids = ids[:head_n]
    remaining = take - head_n
    if remaining > 0:
        pool = ids[head_n:] if len(ids) > head_n else ids
        if len(pool) >= remaining:
            rand_ids = rng.choice(pool, size=remaining, replace=False)
        else:
            rand_ids = rng.choice(ids, size=remaining, replace=False)
        chosen = np.concatenate([head_ids, rand_ids])
    else:
        chosen = head_ids

    for img_id in chosen.tolist():
        p = os.path.join(TRAIN_IMG_DIR, img_id)
        try:
            hs = []
            for _, kind, kwargs, _w in HASH_CONFIG:
                if kind == "dhash":
                    hs.append(dhash(p, **kwargs))
                elif kind == "phash":
                    hs.append(phash(p, **kwargs))
                else:
                    raise ValueError(f"Unknown hash kind: {kind}")
            prototypes.append((tuple(hs), int(lbl)))
        except Exception:
            continue

if len(prototypes) == 0:
    print(
        "WARNING: No prototypes built; falling back to majority-label constant prediction."
    )
    chosen_mode = "majority_fallback"
else:
    chosen_mode = f"hash_ensemble_knn_{len(prototypes)}protos_cfg{[(n,w) for n,_,_,w in HASH_CONFIG]}"

print("Train label distribution:\n", label_counts.to_string())
print("Computed majority_label:", majority_label)
print("Prediction mode        :", chosen_mode)


def bucket_key_from_dhash8(dhash8_val: int, key_bits: int = 32) -> int:
    shift = 64 - key_bits
    if shift <= 0:
        return int(dhash8_val)
    return int((dhash8_val >> shift) & ((1 << key_bits) - 1))


BUCKET_BITS = 20

NEARBY_FLIPS = [
    0,  # no flip
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
]

MAX_CANDS = 12000  # if fewer found, we use all.

bucket_to_proto_idxs = {}
for idx, (hs, _lbl) in enumerate(prototypes):
    dh8 = int(hs[0])  # HASH_CONFIG[0] is dhash8 by construction
    bk = bucket_key_from_dhash8(dh8, key_bits=BUCKET_BITS)
    bucket_to_proto_idxs.setdefault(bk, []).append(idx)

print("Built bucket index with buckets:", len(bucket_to_proto_idxs))

test_image_ids = sample_sub["image_id"].astype(str).values

pred_labels = np.empty(shape=(len(test_image_ids),), dtype=int)
unreadable = 0

weights = np.array([w for _n, _k, _kw, w in HASH_CONFIG], dtype=np.float32)

KNN_K = 11
EPS = 1e-6

for i, img_id in enumerate(test_image_ids):
    test_path = os.path.join(TEST_IMG_DIR, img_id)
    if len(prototypes) == 0:
        pred_labels[i] = majority_label
        continue
    try:
        ths = []
        for _n, kind, kwargs, _w in HASH_CONFIG:
            if kind == "dhash":
                ths.append(dhash(test_path, **kwargs))
            elif kind == "phash":
                ths.append(phash(test_path, **kwargs))
        ths = tuple(ths)

        tbk = bucket_key_from_dhash8(int(ths[0]), key_bits=BUCKET_BITS)
        cand_idxs = []
        for f in NEARBY_FLIPS:
            bk = tbk if f == 0 else (tbk ^ (1 << (f % BUCKET_BITS)))
            lst = bucket_to_proto_idxs.get(bk)
            if lst:
                cand_idxs.extend(lst)
                if len(cand_idxs) >= MAX_CANDS:
                    break

        if len(cand_idxs) == 0:
            cand_iter = range(len(prototypes))
        else:
            seen = set()
            uniq = []
            for idx in cand_idxs:
                if idx not in seen:
                    seen.add(idx)
                    uniq.append(idx)
            cand_iter = uniq[:MAX_CANDS]

        best = []  # list of (d, lbl), sorted ascending by d
        for idx in cand_iter:
            phs, plbl = prototypes[idx]
            d = 0.0
            for j, (a, b) in enumerate(zip(ths, phs)):
                d += float(weights[j]) * float(hamming(int(a), int(b)))
                if len(best) >= KNN_K and d >= best[-1][0]:
                    break

            if len(best) == 0:
                best.append((d, plbl))
            elif len(best) < KNN_K or d < best[-1][0]:
                pos = len(best)
                while pos > 0 and d < best[pos - 1][0]:
                    pos -= 1
                best.insert(pos, (d, plbl))
                if len(best) > KNN_K:
                    best.pop()

                if best[0][0] == 0.0 and len(best) >= KNN_K:
                    break

        if len(best) == 0:
            pred_labels[i] = majority_label
            continue

        votes = np.zeros(5, dtype=np.float64)
        for d, lbl in best:
            votes[int(lbl)] += 1.0 / (float(d) + EPS)

        pred_labels[i] = int(np.argmax(votes))
    except Exception:
        unreadable += 1
        pred_labels[i] = majority_label

sub = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
sub.to_csv(SUB_PATH, index=False)

print("Saved:", SUB_PATH)
print("Unreadable test images (fallback to majority):", unreadable)
print(sub.head())
print("shape:", sub.shape)
print("label dtype:", sub["label"].dtype)

assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["image_id", "label"]
assert sub["label"].between(0, 4).all()
assert os.path.isfile(SUB_PATH)
assert SUB_PATH.endswith(".csv")
