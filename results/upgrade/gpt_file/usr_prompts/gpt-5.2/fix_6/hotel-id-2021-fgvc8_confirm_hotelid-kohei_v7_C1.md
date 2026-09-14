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

0.7201545002946846

# 6. Current score

0.0014

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00232) has done: 'Your notebook currently doesn’t yield a score because it relies on private `/kaggle/input/pekolib*` wheels and a `peko.subs.hotelid.v7` module that are not present in the provided environment, so it never produces a valid `submission.csv` from your actual competition data. The smallest safe fix is to remove that dependency and instead generate a valid submission directly from the provided `sample_submission.csv`, which run end-to-end and create a properly formatted `submission.csv`. This not reach your target MAP@5 (since it’s a baseline), but it unblock scoring and let you iterate further with real model code later while preserving evaluation semantics and submission format.'
- What this solution (achieved 0.00209) has done: 'Your current score is extremely far below the target, and the main issue is that you’re submitting the untouched `sample_submission.csv` (all blank/constant predictions), which yields near-random MAP@5. To move toward the target with minimal core-logic changes, I keep the “no external model” approach but replace the dummy predictions with a simple, legitimate frequency baseline: predict the 5 most common `hotel_id`s from `train.csv` for every test image. This preserves evaluation semantics and produces a valid `submission.csv`, and it should substantially increase MAP@5 versus the current file. I also keep your robust path search and add a similarly robust search for `train.csv`.'
- What this solution (achieved 0.0014) has done: 'Your current baseline predicts the same top-5 hotels for every test image, which severely limits MAP@5. To move toward the target while keeping the same “no model, frequency-based” core logic, I make the predictions chain-aware: infer each test image’s chain from its folder name (as the dataset is organized by chain), then predict the top hotels within that chain (falling back to global top hotels if needed). This is still a simple frequency baseline (no architecture/training changes) but uses legitimate metadata structure to produce more relevant candidate lists. I also ensure we always output exactly 5 unique hotel IDs per row and keep robust path discovery intact.'
- What this solution (achieved 0.0014) has done: 'Your score is far below the target, so we should increase MAP@5 with the smallest change that keeps your “frequency baseline + chain awareness” core logic intact. The biggest weakness is the very expensive and unreliable chain inference loop that searches up to 100 folders per image; instead, we build a one-time lookup from `test_images/**/image.jpg -> chain_id` by scanning the filesystem once, then use O(1) lookup per test row. We also improve the chain-aware ranking slightly (still purely frequency-based) by using a “chain weight + global weight” combined score to choose the top-5, which tends to be more robust when chain information is noisy or sparse. The submission format checks remain, and we still write a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.0014) has done: 'Your current score is far below the target, so we should increase MAP@5 with the smallest change that keeps the same “chain-aware frequency baseline” core logic. The main weakness is that using only `chain` leaves many hotels within a chain and gives weak per-image ranking; we can legitimately add a second metadata signal from the folder structure: infer `hotel_id` directly when test images are stored under `test_images/<hotel_id>/...` (and similarly allow nested layouts). We build a one-time `image -> hotel_id` map by scanning `test_images` recursively (fast enough for ~10k files), and when available we place that inferred hotel first, then fill remaining slots with your existing chain/global frequency ranker. This preserves the overall approach (no model, still frequency-based) while improving relevance for many rows and should move the score substantially toward your target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

CANDIDATE_SAMPLE_PATHS = [
    "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv",
    "/kaggle/data/hotel-id-2021-fgvc8/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]

CANDIDATE_TRAIN_PATHS = [
    "/kaggle/input/hotel-id-2021-fgvc8/train.csv",
    "/kaggle/data/hotel-id-2021-fgvc8/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]

CANDIDATE_TEST_IMAGE_DIRS = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/data/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
]


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


sample_path = find_first_existing(CANDIDATE_SAMPLE_PATHS)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations: "
        + ", ".join(CANDIDATE_SAMPLE_PATHS)
    )

train_path = find_first_existing(CANDIDATE_TRAIN_PATHS)
if train_path is None:
    raise FileNotFoundError(
        "Could not find train.csv in expected locations: "
        + ", ".join(CANDIDATE_TRAIN_PATHS)
    )

test_img_root = find_first_existing(CANDIDATE_TEST_IMAGE_DIRS)
if test_img_root is None:
    raise FileNotFoundError(
        "Could not find test_images directory in expected locations: "
        + ", ".join(CANDIDATE_TEST_IMAGE_DIRS)
    )

sample = pd.read_csv(sample_path)
required_cols = ["image", "hotel_id"]
missing = [c for c in required_cols if c not in sample.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")
sample = sample[required_cols].copy()
sample["image"] = sample["image"].astype(str)


def build_test_image_maps(root_dir: str, known_hotel_ids: set):
    img2chain = {}
    img2hotel = {}

    for dirpath, _, filenames in os.walk(root_dir):
        base = os.path.basename(dirpath)
        base_is_digit = base.isdigit()
        base_is_hotel = base in known_hotel_ids

        chain_id = None
        if base_is_digit:
            chain_id = int(base)
        else:
            parts = dirpath.split(os.sep)
            for p in reversed(parts):
                if p.isdigit():
                    chain_id = int(p)
                    break

        for fn in filenames:
            if not fn.lower().endswith(".jpg"):
                continue
            if chain_id is not None:
                img2chain[fn] = chain_id
            if base_is_hotel:
                img2hotel[fn] = base

    return img2chain, img2hotel


train = pd.read_csv(train_path, usecols=["chain", "hotel_id"])
train["chain"] = train["chain"].fillna(0).astype(int)
train["hotel_id"] = train["hotel_id"].astype(str)

known_hotel_ids = set(train["hotel_id"].unique().tolist())
img2chain, img2hotel = build_test_image_maps(test_img_root, known_hotel_ids)

global_counts = train["hotel_id"].value_counts()
global_top = global_counts.head(
    300
).index.tolist()  # modestly wider pool for fill robustness
if len(global_top) == 0:
    raise ValueError("train.csv seems empty or hotel_id could not be read.")

chain_counts_map = {}
chain_top_map = {}
for chain_id, grp in train.groupby("chain", sort=False):
    vc = grp["hotel_id"].value_counts()
    chain_counts_map[int(chain_id)] = vc
    chain_top_map[int(chain_id)] = vc.head(300).index.tolist()

ALPHA = 3.0  # weight for chain frequency relative to global frequency
BETA = 1.0  # weight for global frequency
K = 5  # MAP@5 submission requirement


def predict_topk_for_chain(chain_id: int):
    cvc = chain_counts_map.get(chain_id, None)
    cand = []
    if cvc is not None:
        cand.extend(chain_top_map.get(chain_id, [])[:300])
    cand.extend(global_top[:300])

    seen = set()
    uniq = []
    for x in cand:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    scored = []
    for hid in uniq:
        chain_cnt = float(cvc.get(hid, 0.0)) if cvc is not None else 0.0
        glob_cnt = float(global_counts.get(hid, 0.0))
        score = ALPHA * chain_cnt + BETA * glob_cnt
        scored.append((score, hid))

    scored.sort(key=lambda t: (-t[0], t[1]))
    out = [hid for _, hid in scored[:K]]

    if len(out) < K:
        for hid in global_top:
            if hid not in out:
                out.append(hid)
            if len(out) == K:
                break
    if len(out) < K:
        out = (out * K)[:K]
    return out


preds = []
missing_chain = 0
has_inferred_hotel = 0

for img in sample["image"].tolist():
    inferred_hid = img2hotel.get(img, None)
    c = img2chain.get(img, 0)
    if img not in img2chain:
        missing_chain += 1

    base = predict_topk_for_chain(int(c))

    if inferred_hid is not None:
        has_inferred_hotel += 1
        out = [inferred_hid] + [x for x in base if x != inferred_hid]
        out = out[:K]
        if len(out) < K:
            for hid in global_top:
                if hid not in out:
                    out.append(hid)
                if len(out) == K:
                    break
    else:
        out = base

    preds.append(" ".join(out))

sub = pd.DataFrame({"image": sample["image"], "hotel_id": preds})
out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(f"Using sample_submission.csv from: {sample_path}")
print(f"Using train.csv from: {train_path}")
print(f"Using test_images root from: {test_img_root}")
print(
    f"Built img2chain size: {len(img2chain)}; missing chain for {missing_chain}/{len(sample)} rows"
)
print(
    f"Built img2hotel size: {len(img2hotel)}; inferred hotel for {has_inferred_hotel}/{len(sample)} rows"
)
print(f"Global fallback top-5: {global_top[:5]}")
print(f"Wrote {out_path} with shape={sub.shape}")
print(sub.head())



## === cell 1
import pandas as pd

sub = pd.read_csv("submission.csv")
assert list(sub.columns) == [
    "image",
    "hotel_id",
], "Submission columns must be exactly: image, hotel_id"
assert sub["image"].notna().all(), "All image values must be present"
assert sub["hotel_id"].notna().all(), "All hotel_id predictions must be present"
assert (
    sub["hotel_id"].astype(str).str.len().gt(0).all()
), "hotel_id strings must be non-empty"

counts = sub["hotel_id"].astype(str).str.split().map(len)
assert counts.min() >= 1, "Each prediction must contain at least 1 hotel_id"
assert (
    counts.min() == 5 and counts.max() == 5
), "Each prediction must contain exactly 5 hotel_ids"
print("submission.csv looks valid.")
print(sub.head())
print(
    "Predictions per row (min/mean/max):",
    int(counts.min()),
    float(counts.mean()),
    int(counts.max()),
)
