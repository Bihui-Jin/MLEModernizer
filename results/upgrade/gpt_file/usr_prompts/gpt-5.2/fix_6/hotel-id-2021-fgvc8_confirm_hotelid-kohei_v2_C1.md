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

0.00184

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it depends on external wheels (`pekolib*`) that are not present in the provided dataset tree, so the pipeline fails before producing `submission.csv`. I remove those installs and replace them with a minimal, self-contained submission generator that always runs end-to-end on the provided `/kaggle/input/hotel-id-2021-fgvc8/` data and writes a valid `submission.csv` in the required `image,hotel_id` format. To get a non-trivial baseline MAP@5 without changing “modeling” complexity, it predict the 5 most frequent `hotel_id` values from `train.csv` for every test image (a common safe baseline that should score above random and, crucially, yields a valid submission). This moves you from “no score” to a real score, which is the necessary first step toward the target.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.6702), so we should improve the baseline while keeping the “predict 5 hotel_ids per image” core logic intact. The smallest meaningful step is to replace the global top-5 hotel prior with a chain-aware prior: for each test image, infer its `chain` from the test image folder name (0–89) and then predict the top-5 `hotel_id` within that chain from `train.csv`. This uses only provided metadata and file structure (no extra models), preserves evaluation semantics, and should move MAP@5 substantially toward the target compared to a single global prior. We also add safe fallbacks when a chain folder is missing/unknown, ensuring a valid submission is always produced.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.6702), so we need a meaningful lift while keeping your “chain-aware top-5 prior” core logic. The biggest issue is that the notebook likely fails to infer chains for most test images because the hidden test set uses a different folder layout than your existence-check loop, so you silently fall back to the global prior almost everywhere. I replace per-image `os.path.exists` probing with a fast, robust lookup that builds an `image -> chain` map by scanning the actual `test_images` directory structure once, and then uses that map for predictions. This preserves the same prediction strategy (top-5 within inferred chain, else global top-5) but should substantially increase chain coverage and move MAP@5 toward the target.'
- What this solution (achieved 0.00209) has done: 'Your current 0.00209 is far below the 0.6702 target, and the main reason is that your chain inference almost always falls back to the global prior because the test images are not organized into numeric chain subfolders in the released/hidden test set. The smallest meaningful improvement that preserves your “metadata-prior top-5” core logic is to add a timestamp-aware split and a hotel-to-hotel co-occurrence prior: for each `hotel_id`, learn its most common “neighbors” within the same chain from training (using near-in-time images), then predict a chain prior list expanded/re-ranked by these neighbors. This still uses only `train.csv` metadata (no image model), keeps the same submission semantics (5 IDs per image), and typically lifts MAP@5 significantly versus pure frequency priors. We also keep your robust submission writing and add a strict guarantee of exactly 5 unique IDs per row.'
- What this solution (achieved 0.00184) has done: 'Your current MAP@5 (0.00209) is extremely far below the target (0.6702), and the main culprit is that the “chain-aware” prior almost never activates on the hidden test because test images are not stored under numeric chain folders—so you mostly submit the weak global top-5. The smallest meaningful improvement that preserves your “metadata-only priors” core logic is to (1) infer chain from the test image filename prefix using the `sample_submission.csv` structure (common in this dataset) and (2) when chain is unknown, back off to a stronger per-hotel popularity prior (top-5 hotels overall, but with chain-conditioned smoothing) while guaranteeing exactly 5 unique IDs. These changes keep the same modeling approach (frequency/neighbor priors, no image model), but should significantly increase the fraction of rows using a relevant chain prior, moving your score materially toward the target.'

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd

DATA_DIR = "/kaggle/input/hotel-id-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SUB_PATH = "submission.csv"

print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))




## === cell 1
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
neighbor_counts = {}  # (chain, hotel_id) -> dict neighbor_id -> count

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

_chain_prefix_re = re.compile(r"^(\d{1,3})[^0-9]")  # e.g., "12_xxx.jpg" or "7-xxxx.jpg"
_chain_prefix_alt_re = re.compile(r"^(\d{1,3})")  # e.g., "12xxxx.jpg" (fallback)


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


def build_test_image_chain_map(test_img_dir: str):
    image_to_chain = {}

    if not os.path.isdir(test_img_dir):
        return image_to_chain

    candidate_dirs = [test_img_dir]
    nested = os.path.join(test_img_dir, "test_images")
    if os.path.isdir(nested):
        candidate_dirs.append(nested)

    for base in candidate_dirs:
        for root, dirs, files in os.walk(base):
            for fn in files:
                if not fn.lower().endswith(".jpg"):
                    continue
                parent = os.path.basename(root)
                try:
                    ch = int(parent)
                except ValueError:
                    continue
                image_to_chain[fn] = ch

    return image_to_chain


image_to_chain_from_dirs = build_test_image_chain_map(TEST_IMG_DIR)


def infer_chain_for_image(img_name: str):
    ch = image_to_chain_from_dirs.get(img_name)
    if ch is not None:
        return int(ch)
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


images = sample["image"].astype(str).tolist()
inferred_chains = [infer_chain_for_image(img) for img in images]

preds = []
for ch in inferred_chains:
    if ch is not None:
        top5_int = make_top5_for_chain(int(ch))
    else:
        top5_int = [int(x) for x in global_top5]

    seen = set()
    top5 = []
    for hid in top5_int:
        hid = int(hid)
        if hid in seen:
            continue
        seen.add(hid)
        top5.append(str(hid))
        if len(top5) == 5:
            break
    if len(top5) < 5:
        for s in global_top5:
            if s in top5:
                continue
            top5.append(s)
            if len(top5) == 5:
                break
    if len(top5) < 5:
        top5 = (top5 * 5)[:5]

    preds.append(" ".join(top5))

sub = pd.DataFrame({"image": images, "hotel_id": preds})

assert list(sub.columns) == ["image", "hotel_id"]
assert len(sub) == len(sample)
sub.to_csv(SUB_PATH, index=False)

print("Wrote:", SUB_PATH, "rows:", len(sub))
print(sub.head())
print(
    "Chain inference coverage (non-None):",
    sum(c is not None for c in inferred_chains),
    "/",
    len(inferred_chains),
)
print("Unique inferred chains:", len({c for c in inferred_chains if c is not None}))




## === cell 2
with open(SUB_PATH, "r", encoding="utf-8") as f:
    for i in range(6):
        print(f.readline().rstrip("\n"))
