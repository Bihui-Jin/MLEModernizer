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

0.7406331565210046

# 6. Current score

0.0014

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it depends on private wheel files (`/kaggle/input/pekolib*`) that aren’t present in the provided dataset tree, so it fails before producing a valid `submission.csv`. To get a valid submission end-to-end with minimal, safe changes, I remove those external installs and replace them with a simple, fully self-contained baseline that reads `train.csv`, builds a frequency-based top-5 hotel list, and writes a correctly-formatted `submission.csv`. This won’t reach the target MAP@5, but it unblock scoring and provide a stable baseline that we can incrementally improve afterward. I also make the input path robust by preferring `/kaggle/input/hotel-id-2021-fgvc8/` and falling back to `/kaggle/data/hotel-id-2021-fgvc8/` if needed.'
- What this solution (achieved 0.0014) has done: 'Your current 0.00209 score comes from predicting the same global top-5 hotels for every test image, which is a very weak baseline for MAP@5. With minimal changes and without introducing any new modeling/training loops, we can leverage the provided `chain` metadata by predicting the top-5 hotels *within the most likely chain* for each test image, inferred from the test image’s folder name (the chain subdirectory in `test_images/`). When a test image isn’t in a chain folder or the chain is unknown, we fall back to the global top-5, preserving robustness and submission validity. This keeps the approach “frequency top-k” (same core logic) but makes it conditional, which should substantially increase MAP@5 toward your target.'

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
BASE = next((p for p in BASE_CANDIDATES if os.path.exists(p)), None)
print("Using BASE:", BASE)


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


train_csv = find_first_existing(
    [
        os.path.join(BASE, "train.csv"),
        "/kaggle/input/hotel-id-2021-fgvc8/train.csv",
        "/kaggle/data/hotel-id-2021-fgvc8/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)
sample_sub_csv = find_first_existing(
    [
        os.path.join(BASE, "sample_submission.csv"),
        "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv",
        "/kaggle/data/hotel-id-2021-fgvc8/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

TEST_IMG_ROOT = find_first_existing(
    [
        os.path.join(BASE, "test_images"),
        "/kaggle/input/hotel-id-2021-fgvc8/test_images",
        "/kaggle/data/hotel-id-2021-fgvc8/test_images",
        "/kaggle/input/test_images",
        "/kaggle/data/test_images",
    ]
)

assert train_csv is not None, "train.csv not found in expected locations."
assert (
    sample_sub_csv is not None
), "sample_submission.csv not found in expected locations."
assert TEST_IMG_ROOT is not None, "test_images folder not found in expected locations."

print("train_csv:", train_csv)
print("sample_sub_csv:", sample_sub_csv)
print("TEST_IMG_ROOT:", TEST_IMG_ROOT)

train = pd.read_csv(train_csv)
sample_sub = pd.read_csv(sample_sub_csv)

print("train shape:", train.shape)
print("sample_sub shape:", sample_sub.shape)
print("train columns:", train.columns.tolist())
print("sample_sub columns:", sample_sub.columns.tolist())



## === cell 1
train2 = train.copy()
train2["hotel_id"] = train2["hotel_id"].astype(str)
train2["chain"] = train2["chain"].fillna(0).astype(int)

global_counts = train2["hotel_id"].value_counts()
global_top5 = global_counts.index[:5].tolist()


def pad_to_5(hids, fallback):
    out = []
    for x in hids:
        if x not in out:
            out.append(x)
        if len(out) == 5:
            return out
    for x in fallback:
        if x not in out:
            out.append(x)
        if len(out) == 5:
            return out
    uniq = train2["hotel_id"].unique().tolist()
    for x in uniq:
        if x not in out:
            out.append(x)
        if len(out) == 5:
            return out
    return out


chain_top = {}
for ch, grp in train2.groupby("chain", sort=False):
    top = grp["hotel_id"].value_counts().index[:5].tolist()
    chain_top[int(ch)] = pad_to_5(top, global_top5)


def infer_chain_for_image(img_name: str) -> int:
    p = os.path.join(TEST_IMG_ROOT, img_name)
    if os.path.exists(p):
        return 0
    try:
        for d in os.listdir(TEST_IMG_ROOT):
            dd = os.path.join(TEST_IMG_ROOT, d)
            if not os.path.isdir(dd):
                continue
            if os.path.exists(os.path.join(dd, img_name)):
                if d.isdigit():
                    return int(d)
                else:
                    return 0
    except FileNotFoundError:
        return 0
    return 0


preds = []
missing_chain_hits = 0
for img in sample_sub["image"].astype(str).tolist():
    ch = infer_chain_for_image(img)
    if ch not in chain_top:
        missing_chain_hits += 1
        ch = 0
    top5 = chain_top.get(ch, global_top5)
    preds.append(" ".join(pad_to_5(top5, global_top5)))

sub = sample_sub.copy()
sub["hotel_id"] = preds

assert list(sub.columns) == [
    "image",
    "hotel_id",
], "Submission columns must be exactly: image, hotel_id"
assert len(sub) == len(sample_sub), "Submission row count must match sample_submission."

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(sub.head())
print("Global top-5 used for fallback:", global_top5)
print("Chains in mapping:", len(chain_top))
print("Test images with unknown/unseen chain mapping:", missing_chain_hits)
