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

0.00168

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the fragile package‑install approach with a self‑contained baseline that reads the provided CSV files, picks the five most frequent hotel IDs from the training data, and writes those as the prediction for every test image. This guarantees a valid `submission.csv` in the required format, allowing the notebook to finish and produce a score (even if modest), moving us from “no submission” toward the target.'
- What this solution (achieved 0.00153) has done: 'The updates keep the exact algorithm while dramatically cutting image‑processing overhead.  
- Image means are now obtained with Pillow’s fast `ImageStat.Stat` instead of converting each image to a NumPy array, eliminating large temporary buffers.  
- Thread pools are given a larger worker count (`os.cpu_count()*5`) to better saturate I/O without changing any logic.  
- Minor clean‑ups (pre‑computed list conversion, explicit `float32` casting) keep the numerical results identical.'
- What this solution (achieved 0.00142) has done: 'The update expands the training data used for each hotel (now all images) and replaces the single‑mean RGB feature with a simple 8×8 RGB down‑sampled fingerprint (192‑dim vector).  Using richer visual descriptors and full‑hotel statistics gives more discriminative centroids, so nearest‑centroid predictions should move the MAP@5 substantially closer to the target while preserving the original pipeline logic.'
- What this solution (achieved 0.00168) has done: 'I keep the same overall pipeline but improve the default predictions by using the five most frequent hotel IDs from the training set and ensure the top‑5 nearest‑centroid IDs are ordered by increasing distance (so the most relevant appears first). These small adjustments keep the core logic unchanged while making the ranking more sensible and giving a better MAP@5, moving the score toward the target.'
- What this solution (achieved 0.00168) has done: 'The changes add a fallback when loading training images so that images are found even if they are not stored inside their chain‑folder (e.g., missing or chain = 0). This greatly increases the number of computed hotel centroids, giving the nearest‑centroid model many more reference vectors and moving the MAP@5 score noticeably closer to the target while keeping the original pipeline unchanged.'

# 9. Code solution

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

centroid_sum = defaultdict(lambda: np.zeros(192, dtype=np.float64))
centroid_cnt = defaultdict(int)


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
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for result in executor.map(_load_feature, tasks):
        if result is None:
            continue
        hotel_id, feat = result
        centroid_sum[hotel_id] += feat
        centroid_cnt[hotel_id] += 1

hotel_centroids = {hid: (centroid_sum[hid] / centroid_cnt[hid]) for hid in centroid_cnt}
print(f"Computed centroids for {len(hotel_centroids)} hotels.")

top5_frequent = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
default_top5_str = " ".join(top5_frequent)

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
test_images = sample_sub["image"].tolist()

centroid_ids = list(hotel_centroids.keys())
centroid_vectors = np.stack([hotel_centroids[hid] for hid in centroid_ids]).astype(
    np.float32
)


def predict_for_image(img_name):
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

    diffs = centroid_vectors - feat  # (N,192) broadcasting
    dists = np.einsum("ij,ij->i", diffs, diffs)

    nearest_idx_unsorted = np.argpartition(dists, 5)[:5]
    nearest_idx = nearest_idx_unsorted[np.argsort(dists[nearest_idx_unsorted])]

    top5_ids = [centroid_ids[i] for i in nearest_idx]
    return " ".join(top5_ids)


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
