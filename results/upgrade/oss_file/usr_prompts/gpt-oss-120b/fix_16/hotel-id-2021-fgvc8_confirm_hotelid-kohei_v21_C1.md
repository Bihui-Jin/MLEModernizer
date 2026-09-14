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
import concurrent.futures
from sklearn.neighbors import NearestNeighbors

TRAIN_CSV = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
SAMPLE_SUBMISSION_CSV = "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv"
TRAIN_IMAGES_DIR = "/kaggle/input/hotel-id-2021-fgvc8/train_images"
TEST_IMAGES_DIR = "/kaggle/input/hotel-id-2021-fgvc8/test_images"
OUTPUT_SUBMISSION = "/kaggle/working/submission.csv"

RESIZE_DIM = 16  # 16×16 → 256‑dim RGB vector
FEATURE_SIZE = RESIZE_DIM * RESIZE_DIM * 3
MAX_WORKERS = max(1, os.cpu_count())  # avoid excessive thread count

train_df = pd.read_csv(TRAIN_CSV)


def _load_train_feature(args):
    """Read a train image, down‑sample and return (hotel_id, feature)."""
    hotel_id, img_path = args
    if not os.path.isfile(img_path):
        fallback_path = os.path.join(TRAIN_IMAGES_DIR, os.path.basename(img_path))
        if not os.path.isfile(fallback_path):
            return None
        img_path = fallback_path
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((RESIZE_DIM, RESIZE_DIM), Image.BILINEAR)
            feat = np.asarray(img, dtype=np.float16).flatten() / 255.0
        return hotel_id, feat
    except Exception:
        return None


tasks = [
    (
        str(row.hotel_id),
        os.path.join(TRAIN_IMAGES_DIR, str(row.chain), row.image),
    )
    for row in train_df.itertuples(index=False)
]

train_features_list = []
train_hotel_ids = []

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for result in executor.map(_load_train_feature, tasks):
        if result is None:
            continue
        hid, feat = result
        train_features_list.append(feat)
        train_hotel_ids.append(hid)

if not train_features_list:
    raise RuntimeError("No training features could be loaded.")

train_features = np.stack(train_features_list).astype(np.float16)  # (N, FEATURE_SIZE)

print(f"Loaded features for {train_features.shape[0]} training images.")

top5_frequent = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
default_top5_str = " ".join(top5_frequent)


def _load_test_feature(img_name):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    if not os.path.isfile(img_path):
        return None
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((RESIZE_DIM, RESIZE_DIM), Image.BILINEAR)
            feat = np.asarray(img, dtype=np.float16).flatten() / 255.0
        return feat
    except Exception:
        return None


sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
test_images = sample_sub["image"].tolist()

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    test_feats_raw = list(executor.map(_load_test_feature, test_images))

valid_mask = np.array([f is not None for f in test_feats_raw])
test_features = np.zeros((len(test_feats_raw), FEATURE_SIZE), dtype=np.float16)
if valid_mask.any():
    test_features[valid_mask] = np.stack([f for f in test_feats_raw if f is not None])

nbrs = NearestNeighbors(
    n_neighbors=2000,
    algorithm="brute",
    metric="euclidean",
    n_jobs=-1,
).fit(train_features)

batch_size = 2000  # larger batch reduces Python loop overhead
predictions = [""] * len(test_images)

for start in range(0, len(test_images), batch_size):
    end = min(start + batch_size, len(test_images))
    batch_feats = test_features[start:end]  # (B, FEATURE_SIZE)

    _, neighbor_idxs = nbrs.kneighbors(batch_feats, return_distance=True)  # (B, 2000)

    for i, global_idx in enumerate(range(start, end)):
        if not valid_mask[global_idx]:
            predictions[global_idx] = default_top5_str
            continue

        nearest_idx = neighbor_idxs[i]
        distinct_ids = []
        seen = set()
        for idx in nearest_idx:
            hid = train_hotel_ids[idx]
            if hid not in seen:
                distinct_ids.append(hid)
                seen.add(hid)
            if len(distinct_ids) == 5:
                break

        if len(distinct_ids) < 5:
            for hid in top5_frequent:
                if hid not in seen:
                    distinct_ids.append(hid)
                    seen.add(hid)
                if len(distinct_ids) == 5:
                    break

        predictions[global_idx] = " ".join(distinct_ids)

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
