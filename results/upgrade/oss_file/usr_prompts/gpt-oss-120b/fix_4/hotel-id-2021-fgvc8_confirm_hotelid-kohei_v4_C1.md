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

0.7110676096657378

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the pip‑install and library call with a simple, self‑contained baseline that reads the training CSV, extracts the five most frequent hotel IDs, and writes those as the prediction for every test image using the provided sample_submission file. This guarantees a valid `submission.csv` with the correct columns and format, allowing the notebook to finish and produce a score (even if modest) toward the target.'
- What this solution (achieved 0.00209) has done: 'I keep the initial environment‑checking cell unchanged and replace the baseline generation cell with a simple lookup‑based approach: map any test image that also appears in the training set to its true hotel ID (as first prediction) and fill the remaining slots with the globally most frequent hotel IDs. This small change adds useful signal without altering the overall pipeline, and it should raise the MAP@5 score considerably toward the target.'
- What this solution (achieved 0.00209) has done: 'I add a lightweight per‑chain heuristic: map each test image to its chain folder, compute the top‑5 most frequent hotel IDs for every chain from the training metadata, and use these chain‑specific predictions (falling back to the global top‑5). This keeps the original simple baseline while giving much more relevant guesses, moving the MAP@5 score toward the target.'

# 9. Code solution

## === cell 0
import os, pathlib, sys

base_path = pathlib.Path("/kaggle/input")
print("Available input folders:")
for p in base_path.iterdir():
    if p.is_dir():
        print(" -", p.name)



## === cell 1
import pandas as pd
from pathlib import Path

train_csv = Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv")
sample_sub = Path("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv")
output_sub = Path("/kaggle/working/submission.csv")
test_images_root = Path("/kaggle/input/hotel-id-2021-fgvc8/test_images")

train_df = pd.read_csv(train_csv)

image_to_hotel = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))

global_top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
print(f"Global top‑5 hotel IDs: {' '.join(global_top5)}")

chain_top5 = {}
for chain_id, grp in train_df.groupby("chain"):
    top5 = grp["hotel_id"].value_counts().head(5).index.astype(str).tolist()
    chain_top5[str(chain_id)] = top5  # store as string keys for easy matching

test_image_to_chain = {}
if test_images_root.exists():
    for chain_dir in test_images_root.iterdir():
        if chain_dir.is_dir():
            chain_name = chain_dir.name  # e.g., "0", "1", …
            for img_path in chain_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }:
                    test_image_to_chain[img_path.name] = chain_name
else:
    print(
        "Warning: test_images root not found; chain‑specific fallback will be unavailable."
    )

sub_df = pd.read_csv(sample_sub)


def build_prediction(img_name: str) -> str:
    """
    Produce a space‑separated list of up to 5 hotel IDs for a test image.
    1. If the image exists in the training set, use its true hotel_id as the first prediction.
    2. Otherwise, try the chain‑specific top‑5 (derived from the folder the image resides in).
    3. Fall back to the global top‑5 if no chain information is available.
    Duplicates are removed while preserving order.
    """
    preds = []

    if img_name in image_to_hotel:
        first = image_to_hotel[img_name]
        preds.append(first)
        preds.extend([hid for hid in global_top5 if hid != first][:4])
    else:
        chain_id = test_image_to_chain.get(img_name)
        if chain_id and chain_id in chain_top5:
            preds = chain_top5[chain_id].copy()
        else:
            preds = global_top5.copy()

    seen = set()
    final_preds = []
    for hid in preds:
        if hid not in seen:
            final_preds.append(hid)
            seen.add(hid)
        if len(final_preds) == 5:
            break
    if len(final_preds) < 5:
        for hid in global_top5:
            if hid not in seen:
                final_preds.append(hid)
                seen.add(hid)
            if len(final_preds) == 5:
                break

    return " ".join(final_preds[:5])


sub_df["hotel_id"] = sub_df["image"].apply(build_prediction)

sub_df.to_csv(output_sub, index=False)
print(f"Submission written to {output_sub}")
print("First few rows of the submission:")
print(sub_df.head())
