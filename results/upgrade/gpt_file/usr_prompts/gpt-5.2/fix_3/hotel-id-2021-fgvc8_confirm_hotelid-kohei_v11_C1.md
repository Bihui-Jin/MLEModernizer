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

0.7607266144649304

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your current notebook doesn’t yield a score because it never reliably produces a valid `submission.csv`: it depends on external wheels (`pekolib`) that aren’t present in the provided environment, so the pipeline fails before writing the file. I replace those external-package calls with a minimal, self-contained baseline that (1) reads the provided `sample_submission.csv`, (2) builds a simple, legitimate prediction list from the most frequent `hotel_id` values in `train.csv`, and (3) writes a valid `submission.csv` with the exact required columns and row alignment. This not be a top solution, but it run end-to-end within the time limit and produce a valid submission so you can obtain a real Kaggle score and iterate toward the target afterward.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.7607), so we need a legitimate improvement while keeping changes minimal and preserving the “simple baseline” core logic. The main issue is that predicting the same global top-5 hotels for every test image is extremely weak for MAP@5; a small, safe step up is to use available metadata (`chain`) to make per-image predictions more relevant. We can compute per-chain top-5 hotel IDs from `train.csv`, then for each test image infer its chain from the folder structure in `test_images/<chain>/...` and output the corresponding chain-specific top-5; if chain can’t be inferred, we fall back to the global top-5. This keeps the approach simple (frequency-based), runs fast, and should move the score substantially toward the target without changing the evaluation semantics.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

BASE = "/kaggle/input/hotel-id-2021-fgvc8"
train_csv_path = os.path.join(BASE, "train.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")
test_images_dir = os.path.join(BASE, "test_images")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_path), f"Missing: {sample_path}"
assert os.path.exists(test_images_dir), f"Missing: {test_images_dir}"

print("train.csv:", train_csv_path)
print("sample_submission.csv:", sample_path)
print("test_images_dir:", test_images_dir)



## === cell 1
train = pd.read_csv(train_csv_path)
sample = pd.read_csv(sample_path)

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + ["0"] * 5)[:5]
global_pred_str = " ".join(global_top5)

chain_top5 = (
    train.groupby("chain")["hotel_id"]
    .apply(lambda s: s.value_counts().head(5).index.astype(str).tolist())
    .to_dict()
)

for c, ids in list(chain_top5.items()):
    out = []
    seen = set()
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) == 5:
            break
    if len(out) < 5:
        for x in global_top5:
            if x not in seen:
                out.append(x)
                seen.add(x)
            if len(out) == 5:
                break
    if len(out) < 5:
        out = (out + ["0"] * 5)[:5]
    chain_top5[c] = out

print("Global top5:", global_top5)
print("Example chain keys:", list(chain_top5.keys())[:5])




## === cell 2
def infer_chain_from_fs(image_name: str):
    for entry in os.scandir(test_images_dir):
        if not entry.is_dir():
            continue
        candidate = os.path.join(entry.path, image_name)
        if os.path.exists(candidate):
            try:
                return int(entry.name)
            except ValueError:
                return None
    return None


preds = []
missing_chain = 0
for img in sample["image"].astype(str).tolist():
    ch = infer_chain_from_fs(img)
    if ch is None:
        missing_chain += 1
        preds.append(global_pred_str)
    else:
        ids = chain_top5.get(ch, global_top5)
        preds.append(" ".join(ids))

submission = sample.copy()
submission["hotel_id"] = preds

print(
    "Rows with inferred chain missing (fallback to global):",
    missing_chain,
    "/",
    len(sample),
)

assert submission.shape[0] == sample.shape[0]
assert list(submission.columns) == ["image", "hotel_id"]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())



## === cell 3
sub = pd.read_csv("submission.csv")
print("submission.csv shape:", sub.shape)
print("columns:", sub.columns.tolist())
print("example row:", sub.iloc[0].to_dict())
print("unique prediction strings:", sub["hotel_id"].nunique())
lens = sub["hotel_id"].astype(str).str.split().map(len)
print("min/max #ids per row:", int(lens.min()), int(lens.max()))
assert lens.min() == 5 and lens.max() == 5
