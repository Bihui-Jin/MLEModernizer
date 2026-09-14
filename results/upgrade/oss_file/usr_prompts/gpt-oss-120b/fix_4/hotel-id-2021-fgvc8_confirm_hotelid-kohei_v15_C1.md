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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the package‑install and opaque module call with a small self‑contained baseline that loads the training metadata, picks the five most frequent hotel IDs, and writes a valid `submission.csv` containing those IDs for every test image. The script also computes an internal MAP@5 on a hold‑out split of the training data so you can see how far the simple baseline is from the target score, without changing any core modeling logic. This guarantees a correctly formatted CSV and moves the score from “not yielded” toward the target.'
- What this solution (achieved 0.0) has done: 'We avoid the stratify error by removing stratification, and improve predictions by using the hotel‑frequency within each chain (derived from the folder name of each test image). This keeps the original simple‑baseline logic but makes the per‑image list much more relevant, moving the MAP@5 toward the target while still producing a correctly formatted CSV.'
- What this solution (achieved 0.0) has done: 'I keep the original simple frequency‑based logic but ensure every prediction list has exactly five IDs by padding missing slots with the global most‑frequent hotels that are not already present. This modest change can increase the chance that the true hotel appears in the top‑5, thus moving the internal MAP@5 closer to the target while preserving the overall approach and producing a valid submission file.'

# 9. Code solution

## === cell 0
import os, glob, pandas as pd, numpy as np
from collections import defaultdict

print("Root input dirs:", os.listdir("/kaggle/input"))
print("Working dirs:", os.listdir("/kaggle/working"))




## === cell 1
train_path = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
train_df = pd.read_csv(train_path)
print("Train rows:", len(train_df))

global_top5_list = train_df["hotel_id"].value_counts().head(5).astype(str).tolist()
global_top5_str = " ".join(global_top5_list)
print("Global top‑5 frequent hotel IDs:", global_top5_str)

chain_top5 = {}
for chain_id, grp in train_df.groupby("chain"):
    top5 = grp["hotel_id"].value_counts().head(5).astype(str).tolist()
    if top5:  # safety check
        chain_top5[str(chain_id)] = top5




## === cell 2
test_images_dir = "/kaggle/input/hotel-id-2021-fgvc8/test_images"
test_image_paths = glob.glob(
    os.path.join(test_images_dir, "**", "*.jpg"), recursive=True
)
test_images = [os.path.basename(p) for p in test_image_paths]
test_chains = [os.path.basename(os.path.dirname(p)) for p in test_image_paths]


def build_prediction(chain):
    """
    Return a space‑separated string of 5 hotel IDs.
    Use the chain‑specific top‑5 if available; otherwise fall back to global top‑5.
    If the list has fewer than 5 IDs, pad with global IDs not already present.
    """
    chain_list = chain_top5.get(chain, [])
    if not chain_list:
        return global_top5_str

    padded = chain_list.copy()
    for gid in global_top5_list:
        if len(padded) >= 5:
            break
        if gid not in padded:
            padded.append(gid)
    padded = padded[:5]
    return " ".join(padded)


predictions = [build_prediction(chain) for chain in test_chains]

submission_df = pd.DataFrame({"image": test_images, "hotel_id": predictions})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Submission written to:", submission_path)
print(submission_df.head())




## === cell 3
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    train_df.index, test_size=0.1, random_state=42, shuffle=True
)
val_df = train_df.loc[val_idx]


def map5_score(row, pred_str):
    preds = pred_str.split()
    true = str(row["hotel_id"])
    if true in preds:
        rank = preds.index(true) + 1  # 1‑based rank
        return 1.0 / rank
    return 0.0


val_predictions = [build_prediction(str(row["chain"])) for _, row in val_df.iterrows()]
val_scores = [
    map5_score(row, pred) for (_, row), pred in zip(val_df.iterrows(), val_predictions)
]
internal_map5 = np.mean(val_scores)
print(f"Internal MAP@5 on 10% hold‑out: {internal_map5:.5f}")
