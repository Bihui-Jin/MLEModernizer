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

0.00105

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your current notebook never yields a score because it relies on private Kaggle Dataset wheels (`/kaggle/input/pekolib*`) that are not present in the provided environment, so the install/import step fail before writing `submission.csv`. To unblock scoring with minimal change, I remove the external wheel installs and replace the `peko` call with a simple, fully self-contained baseline that still produces a valid `submission.csv` in the required format. Since MAP@5 requires 5 space-delimited hotel IDs per image, the baseline predict the 5 most frequent `hotel_id`s from `train.csv` for every test image (a common safe fallback that typically scores non-trivially above random). This keeps I/O paths Kaggle-compatible and guarantees an end-to-end run and a valid submission file.'
- What this solution (achieved 0.00209) has done: 'Your current baseline predicts the same global top-5 hotels for every test image, which usually yields a very low MAP@5. To move the score toward the 0.745 target with minimal core-logic change, I keep the “no-ML” approach but make predictions image-specific using only train.csv: for each test image, infer its chain folder (the parent directory in `test_images/`) and use the top-5 hotels within that chain from the training metadata, falling back to global top-5 if needed. This leverages the dataset’s provided folder structure without changing any model/training code (there is none) and typically gives a large, legitimate lift in MAP@5. I also ensure the submission includes all test images (including nested directories) and exactly 5 space-delimited IDs per row.'
- What this solution (achieved 0.00105) has done: 'Your current score is far below the target, so we should legitimately increase MAP@5 with the smallest possible change while keeping the same “metadata-only, no-ML” core approach. The main issue is that for this competition the test images are *not* organized into chain-number folders (unlike train), so your chain-folder heuristic almost never triggers and you effectively fall back to the same global top-5 for most images. I keep the same prediction strategy structure (top-5 by group with fallback), but switch the grouping key to something actually available at inference: the test image’s *nearest neighbors in filename space* using the shared image-ID prefix (first 1–4 hex chars) and using the top-5 hotels among training images with the same prefix bucket; if no matches, fall back to global top-5. This remains fully self-contained, fast, and typically improves over pure global-frequency while still preserving your simple baseline semantics and producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd

INPUT_ROOT = "/kaggle/input"
print("Listing /kaggle/input (top-level):")
print("\n".join(sorted(os.listdir(INPUT_ROOT))[:200]))



## === cell 1
import os
import glob
import pandas as pd


CAND_TRAIN = [
    "/kaggle/input/hotel-id-2021-fgvc8/train.csv",
    "/kaggle/input/train.csv",
]
train_path = next((p for p in CAND_TRAIN if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(f"Could not find train.csv in any of: {CAND_TRAIN}")

CAND_TEST_DIRS = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
]
test_dir = next((p for p in CAND_TEST_DIRS if os.path.isdir(p)), None)
if test_dir is None:
    raise FileNotFoundError(f"Could not find test_images/ in any of: {CAND_TEST_DIRS}")

print("Using train_path:", train_path)
print("Using test_dir:", test_dir)

train_df = pd.read_csv(train_path)
required_cols = {"hotel_id", "image"}
missing = required_cols - set(train_df.columns)
if missing:
    raise ValueError(f"train.csv missing required columns: {sorted(missing)}")

global_top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + [global_top5[-1]] * 5)[:5]
global_pred_str = " ".join(global_top5)
print("Global fallback top-5 hotel_ids:", global_pred_str)


def _prefix(s: str, n: int) -> str:
    s = os.path.basename(str(s))
    if s.lower().endswith(".jpg"):
        s = s[:-4]
    return s[:n]


prefix_lengths = [4, 3, 2, 1]
prefix_top5 = {n: {} for n in prefix_lengths}

train_images = train_df["image"].astype(str)
train_hotels = train_df["hotel_id"].astype(str)

for n in prefix_lengths:
    pref = train_images.map(lambda x, nn=n: _prefix(x, nn))
    tmp = pd.DataFrame({"pref": pref, "hotel_id": train_hotels})
    for pfx, g in tmp.groupby("pref", sort=False):
        top = g["hotel_id"].value_counts().head(5).index.tolist()
        if not top:
            continue
        if len(top) < 5:
            top = (top + [top[-1]] * 5)[:5]
        prefix_top5[n][pfx] = " ".join(top)

for n in prefix_lengths:
    print(f"Built prefix_top5[{n}] entries:", len(prefix_top5[n]))

test_paths = sorted(glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True))
if len(test_paths) == 0:
    raise RuntimeError(
        f"No .jpg files found under {test_dir}. Cannot build submission."
    )

images = [os.path.basename(p) for p in test_paths]


def _fix5(s: str) -> str:
    parts = str(s).split()
    if len(parts) == 0:
        parts = global_top5[:]
    if len(parts) < 5:
        parts = parts + [parts[-1]] * (5 - len(parts))
    return " ".join(parts[:5])


preds = []
n_prefix_used = {n: 0 for n in prefix_lengths}
for img in images:
    pred = global_pred_str
    stem = img[:-4] if img.lower().endswith(".jpg") else img
    for n in prefix_lengths:
        key = stem[:n]
        hit = prefix_top5[n].get(key)
        if hit is not None:
            pred = hit
            n_prefix_used[n] += 1
            break
    preds.append(pred)

sub = pd.DataFrame({"image": images, "hotel_id": list(map(_fix5, preds))})

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(sub), "cols:", list(sub.columns))
print("Prefix usage counts:", n_prefix_used)
print(sub.head())
