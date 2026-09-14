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

0.7368948387639952

# 6. Current score

0.00184

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your current notebook doesn’t yield a score because it relies on private wheel files (`/kaggle/input/pekolib-deps/*.whl`) that are not present in the provided environment, so it can’t reliably run end-to-end to produce a valid `submission.csv`. To make it run and generate a valid submission, I remove the external installs and replace them with a minimal, deterministic baseline that uses only the provided `train.csv` to build a global “most frequent hotel_id” top-5 list, then writes predictions for every test image from `sample_submission.csv`. This preserves the “predict 5 hotel_ids per image” evaluation semantics and guarantees a correctly formatted CSV. While this baseline won’t reach your target MAP@5, it produce a valid scored submission so you can iterate from there.'
- What this solution (achieved 0.00209) has done: 'Your current score is low because the submission predicts the same global top-5 hotels for every image, which provides almost no retrieval power under MAP@5. To move the score up toward your target while keeping the approach simple and deterministic, I keep the same “frequency-based” core logic but condition predictions on the image’s chain folder (the parent directory name in `test_images/`), using `train.csv` to compute a top-5 per chain and falling back to the global top-5 when needed. This uses only provided files, preserves the “predict 5 hotel_ids per image” semantics, and should substantially increase MAP@5 versus a single global list. The output remains a valid `submission.csv` with exactly 5 space-delimited IDs per row.'
- What this solution (achieved 0.00184) has done: 'Your score is low because the test set does not include `chain` folders (unlike `train_images/`), so `infer_chain_id_from_path()` almost always returns `-1` and you fall back to the same global top-5 for every image. To move toward your target while keeping the same frequency-based core logic, I condition predictions on the **test image filename prefix** (first 2 hex chars), which is a stable, available signal, by learning a top-5 hotel list per prefix from `train.csv`. I keep a robust fallback cascade (prefix → global), and I also remove the expensive `os.walk` which risks timeouts and doesn’t help for this dataset layout. The result still produces a valid `submission.csv` with exactly 5 space-delimited hotel IDs per test image.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

BASE = "/kaggle/input/hotel-id-2021-fgvc8"
train_path = os.path.join(BASE, "train.csv")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")
test_images_dir = os.path.join(BASE, "test_images")

assert os.path.exists(train_path), f"Missing train.csv at {train_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at {sample_sub_path}"
assert os.path.exists(test_images_dir), f"Missing test_images dir at {test_images_dir}"

train = pd.read_csv(train_path)
sample_sub = pd.read_csv(sample_sub_path)

assert "hotel_id" in train.columns, "train.csv must contain hotel_id"
assert "image" in train.columns, "train.csv must contain image"
assert (
    "image" in sample_sub.columns and "hotel_id" in sample_sub.columns
), "sample_submission.csv schema mismatch"



## === cell 1
train["hotel_id"] = train["hotel_id"].astype(str)
train["image"] = train["image"].astype(str)
train["prefix2"] = train["image"].str[:2]

global_top5 = train["hotel_id"].value_counts().head(5).index.tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + [global_top5[-1]] * 5)[:5]

prefix_top5 = {}
vc = train.groupby("prefix2")["hotel_id"].value_counts().groupby(level=0).head(5)
for (p2, hid), cnt in vc.items():
    prefix_top5.setdefault(str(p2), []).append(str(hid))

for p2 in list(prefix_top5.keys()):
    lst = prefix_top5[p2]
    if len(lst) < 5:
        need = 5 - len(lst)
        pad = [x for x in global_top5 if x not in lst][:need]
        lst = (lst + pad)[:5]
        if len(lst) < 5:
            lst = (lst + [lst[-1]] * 5)[:5]
        prefix_top5[p2] = lst




## === cell 2
def infer_prefix2(img_name: str) -> str:
    img_name = str(img_name)
    return img_name[:2] if len(img_name) >= 2 else ""


preds = []
for img in sample_sub["image"].astype(str).tolist():
    p2 = infer_prefix2(img)
    top5 = prefix_top5.get(p2, global_top5)
    preds.append(" ".join(top5))

sub = sample_sub[["image"]].copy()
sub["hotel_id"] = preds

assert len(sub) == len(sample_sub), "Submission row count mismatch"
assert (
    sub["hotel_id"].str.split().map(len).eq(5).all()
), "Each prediction must contain exactly 5 IDs"
sub.to_csv("submission.csv", index=False)

print(sub.head())



## === cell 3
import subprocess

subprocess.run(["ls", "-lha", "submission.csv"], check=True)
subprocess.run(["head", "submission.csv"], check=True)
