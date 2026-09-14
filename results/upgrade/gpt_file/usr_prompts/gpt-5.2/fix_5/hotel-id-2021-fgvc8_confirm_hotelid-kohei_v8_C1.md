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

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it depends on external wheels/modules (`pekolib`) that aren’t present in the provided filesystem, so it won’t reliably run end-to-end or create `submission.csv`. I replace that dependency with a minimal, self-contained baseline that always produces a valid `submission.csv` in the required `image,hotel_id` format using only the provided `train.csv` and `sample_submission.csv`. To move the score upward (vs. a blank/failed run) while keeping logic simple and within time, I predict the global top-5 most frequent `hotel_id` from the training data for every test image. This won’t hit the target MAP@5, but it generate a valid submission and produce a non-zero baseline score, which is the smallest legitimate step toward the target given the current “no submission” state.'
- What this solution (achieved 0.00209) has done: 'Your current baseline predicts the same global top-5 hotels for every image, which is valid but far from the target MAP@5. To move the score upward with minimal logic change and without using images/models, I generate *chain-aware* top-5 candidates by leveraging the `chain` field in `train.csv` and the fact that test images are stored in chain subfolders (the parent directory name). This keeps the approach as “frequency-based top-5” but makes it conditional on the inferred chain, which should materially increase MAP@5 vs the global prior while remaining fast and deterministic. I also keep a safe fallback to the global top-5 when the chain cannot be inferred or has no stats.'
- What this solution (achieved 0.00209) has done: 'Your current approach is valid but likely underperforms because many test images won’t have a numeric parent folder (or won’t match your `fn`-only mapping), causing frequent fallback to the global top-5. I make a minimal, score-relevant fix: build the test image “chain” mapping using the *relative path* (to robustly detect chain folders) and also match both `image` and `test_images/<chain>/<image>` naming patterns. I also add a tiny amount of smoothing by using global priors only when chain has too few unique hotels, otherwise keep chain top-5 unchanged—this preserves the same frequency-based core logic while improving recall. Output format and paths remain the same and it still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.00209) has done: 'Your current score is low because most test images likely don’t get a usable inferred `chain` from the directory structure, so you fall back to a weak global top-5 too often. I keep the same frequency-based “chain-aware top-5” core logic, but make the chain inference more robust by (1) also extracting chain from the immediate parent folder name (common layout), and (2) building the mapping keyed by the exact `image` strings in `sample_submission.csv` (so we match whether Kaggle provides bare filenames or paths). I also make the “chain too small” fallback less aggressive (only fall back when the chain has no candidates), which should increase MAP@5 while still being deterministic and fast. The output remains a valid `submission.csv` with `image,hotel_id` and exactly 5 IDs per row.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

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
        path2 = os.path.join(base, "hotel-id-2021-fgvcvc8", filename)
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


train_csv = find_file("train.csv")
sample_sub_csv = find_file("sample_submission.csv")

print("Using train.csv:", train_csv)
print("Using sample_submission.csv:", sample_sub_csv)

train = pd.read_csv(train_csv)
sub = pd.read_csv(sample_sub_csv)

train["hotel_id"] = train["hotel_id"].astype(str)
train["chain"] = train["chain"].astype(int)

global_top5 = train["hotel_id"].value_counts().head(5).index.tolist()
if len(global_top5) == 0:
    raise RuntimeError("No hotel_id found in training data.")
while len(global_top5) < 5:
    global_top5.append(global_top5[-1])

chain_top5 = {}
vc = train.groupby("chain")["hotel_id"].value_counts()
for chain_id in vc.index.get_level_values(0).unique().tolist():
    top = vc.loc[chain_id].head(5).index.tolist()
    if len(top) == 0:
        continue
    while len(top) < 5:
        top.append(top[-1])
    chain_top5[int(chain_id)] = top

test_images_dir = find_dir("test_images")
print("Using test_images dir:", test_images_dir)

sub_images = sub["image"].astype(str).tolist()
sub_image_set = set(sub_images)
sub_basename_set = set([os.path.basename(x) for x in sub_images])

img2chain = {}
for root, _, files in os.walk(test_images_dir):
    parent = os.path.basename(root)
    chain_part = int(parent) if parent.isdigit() else None

    if chain_part is None:
        rel = os.path.relpath(root, test_images_dir)
        parts = [] if rel == "." else rel.split(os.sep)
        for p in parts:
            if p.isdigit():
                chain_part = int(p)
                break

    if chain_part is None:
        continue

    for fn in files:
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        if fn in sub_image_set:
            img2chain[fn] = chain_part

        if fn in sub_basename_set:
            for s in sub_images:
                if os.path.basename(s) == fn:
                    img2chain[s] = chain_part

print(
    "Inferred chain for",
    len(img2chain),
    "submission image keys (may include duplicates).",
)

sub = sub[["image"]].copy()

preds = []
for img in sub["image"].astype(str).tolist():
    ch = img2chain.get(img, None)
    top5 = chain_top5.get(ch, None) if ch is not None else None

    if top5 is None:
        top5 = global_top5

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
