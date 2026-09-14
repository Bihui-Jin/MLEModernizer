# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
import concurrent.futures

TRAIN_CSV = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
SAMPLE_SUBMISSION_CSV = "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv"
TRAIN_IMAGES_DIR = "/kaggle/input/hotel-id-2021-fgvc8/train_images"
TEST_IMAGES_DIR = "/kaggle/input/hotel-id-2021-fgvc8/test_images"
OUTPUT_SUBMISSION = "/kaggle/working/submission.csv"

train_df = pd.read_csv(TRAIN_CSV)


def _load_feature(args):
    """Load image, down‑sample to 8×8, flatten RGB into a 192‑dim vector.
    Tries both the chain‑folder path and a fallback path without the chain folder."""
    hotel_id, img_path = args
    if not os.path.isfile(img_path):
        fallback_path = os.path.join(TRAIN_IMAGES_DIR, os.path.basename(img_path))
        if not os.path.isfile(fallback_path):
            return None
        img_path = fallback_path
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((8, 8), Image.BILINEAR)
            feat = np.asarray(img, dtype=np.float32).flatten() / 255.0
        return hotel_id, feat
    except Exception:
        return None


tasks = [
    (
        str(row["hotel_id"]),
        os.path.join(TRAIN_IMAGES_DIR, str(row["chain"]), row["image"]),
    )
    for _, row in train_df.iterrows()
]

max_workers = max(1, os.cpu_count() * 5)
train_features = []
train_hotel_ids = []

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for result in executor.map(_load_feature, tasks):
        if result is None:
            continue
        hotel_id, feat = result
        train_features.append(feat)
        train_hotel_ids.append(hotel_id)

if len(train_features) == 0:
    raise RuntimeError("No training features could be loaded.")
train_features = np.stack(train_features).astype(np.float32)  # shape (N,192)
print(f"Loaded features for {train_features.shape[0]} training images.")

top5_frequent = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
default_top5_str = " ".join(top5_frequent)

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
test_images = sample_sub["image"].tolist()


def predict_for_image(img_name, k_nearest=20):
    """Return a space‑separated string of 5 hotel IDs for a test image."""
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    if not os.path.isfile(img_path):
        return default_top5_str

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((8, 8), Image.BILINEAR)
            feat = np.asarray(img, dtype=np.float32).flatten() / 255.0
    except Exception:
        return default_top5_str

    diffs = train_features - feat  # (N,192) broadcasting
    dists = np.einsum("ij,ij->i", diffs, diffs)

    nearest_idx_unsorted = np.argpartition(dists, k_nearest)[:k_nearest]
    nearest_idx = nearest_idx_unsorted[np.argsort(dists[nearest_idx_unsorted])]

    seen = set()
    top_ids = []
    for idx in nearest_idx:
        hid = train_hotel_ids[idx]
        if hid not in seen:
            seen.add(hid)
            top_ids.append(hid)
        if len(top_ids) == 5:
            break

    if len(top_ids) < 5:
        for hid in top5_frequent:
            if hid not in seen:
                top_ids.append(hid)
                seen.add(hid)
            if len(top_ids) == 5:
                break

    return " ".join(top_ids)


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    predictions = list(executor.map(predict_for_image, test_images))

submission = pd.DataFrame({"image": test_images, "hotel_id": predictions})
submission.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Submission written to {OUTPUT_SUBMISSION}")
print(submission.head())



## === cell 1
import os, pandas as pd

assert os.path.isfile(OUTPUT_SUBMISSION), "submission.csv was not created"
sub = pd.read_csv(OUTPUT_SUBMISSION)
assert list(sub.columns) == ["image", "hotel_id"], "Incorrect column names"
print(f"Submission shape: {sub.shape}")
