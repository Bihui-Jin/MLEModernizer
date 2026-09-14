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

centroid_sum = defaultdict(lambda: np.zeros(3, dtype=np.float64))
centroid_cnt = defaultdict(int)

for _, row in train_df.iterrows():
    hotel_id = str(row["hotel_id"])
    if centroid_cnt[hotel_id] >= 5:
        continue

    image_name = row["image"]
    chain_folder = str(row["chain"])
    img_path = os.path.join(TRAIN_IMAGES_DIR, chain_folder, image_name)

    if not os.path.isfile(img_path):
        continue

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_arr = np.array(img, dtype=np.float32)
            mean_rgb = img_arr.mean(axis=(0, 1))
    except Exception:
        continue

    centroid_sum[hotel_id] += mean_rgb
    centroid_cnt[hotel_id] += 1

hotel_centroids = {hid: (centroid_sum[hid] / centroid_cnt[hid]) for hid in centroid_cnt}
print(f"Computed centroids for {len(hotel_centroids)} hotels.")

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
test_images = sample_sub["image"].tolist()

centroid_ids = list(hotel_centroids.keys())
centroid_vectors = np.stack([hotel_centroids[hid] for hid in centroid_ids]).astype(
    np.float32
)

default_top5_ids = centroid_ids[:5]
default_top5_str = " ".join(default_top5_ids)


def predict_for_image(img_name):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    if not os.path.isfile(img_path):
        return default_top5_str

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_arr = np.array(img, dtype=np.float32)
            mean_rgb = img_arr.mean(axis=(0, 1))
    except Exception:
        return default_top5_str

    diffs = centroid_vectors - mean_rgb  # (N,3) broadcasting
    dists = np.linalg.norm(diffs, axis=1)
    nearest_idx = np.argpartition(dists, 5)[:5]  # unsorted top‑5
    top5_ids = [centroid_ids[i] for i in nearest_idx]
    return " ".join(top5_ids)


with concurrent.futures.ThreadPoolExecutor() as executor:
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
