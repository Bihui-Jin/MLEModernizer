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

0.02458

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
- What this solution (achieved 0.02458) has done: 'The changes replace the per‑image brute‑force distance loop with a batched, fully vectorized distance computation that uses pre‑computed train norms and matrix multiplication, dramatically reducing Python‑level overhead while keeping the exact nearest‑neighbor logic and fallback behavior unchanged. Test image features are still loaded in parallel, and the final ranking/prediction steps remain identical.'
- What this solution (achieved 0.01927) has done: 'I increase the neighbor pool from 20 to 100 and select the top‑5 hotels based on how often they appear among these nearest neighbours (using a frequency count coupled with the smallest distance as a tie‑breaker). This modest adjustment keeps the original pipeline intact while giving the model a better chance to rank the correct hotel higher, moving the MAP@5 score closer to the target.'
- What this solution (achieved 0.02458) has done: 'I replace the frequency‑based voting with a simpler distinct‑nearest‑hotel approach: for each test image we look at the 500 closest training images, then select the first five different hotel IDs ordered by distance (filling with the global top‑5 frequent IDs if needed). This keeps the same feature extraction and nearest‑neighbor logic but improves the ranking quality, moving the MAP@5 score closer to the target. I also increase the neighbor pool from 100 to 500 to give more candidates for distinct selection.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
import concurrent.futures
from collections import Counter

TRAIN_CSV = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
SAMPLE_SUBMISSION_CSV = "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv"
TRAIN_IMAGES_DIR = "/kaggle/input/hotel-id-2021-fgvc8/train_images"
TEST_IMAGES_DIR = "/kaggle/input/hotel-id-2021-fgvc8/test_images"
OUTPUT_SUBMISSION = "/kaggle/working/submission.csv"

train_df = pd.read_csv(TRAIN_CSV)


def _load_train_feature(args):
    """Load a training image, down‑sample to 8×8 and return (hotel_id, feature)."""
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
    for result in executor.map(_load_train_feature, tasks):
        if result is None:
            continue
        hid, feat = result
        train_features.append(feat)
        train_hotel_ids.append(hid)

if len(train_features) == 0:
    raise RuntimeError("No training features could be loaded.")
train_features = np.stack(train_features).astype(np.float32)  # (N,192)
train_norms = np.einsum("ij,ij->i", train_features, train_features)  # (N,)

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
            img = img.resize((8, 8), Image.BILINEAR)
            feat = np.asarray(img, dtype=np.float32).flatten() / 255.0
        return feat
    except Exception:
        return None


sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
test_images = sample_sub["image"].tolist()

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    test_feats_list = list(executor.map(_load_test_feature, test_images))

valid_mask = np.array([f is not None for f in test_feats_list])
test_features = np.zeros((len(test_feats_list), 192), dtype=np.float32)
test_features[valid_mask] = np.stack([f for f in test_feats_list if f is not None])

test_norms = np.einsum("ij,ij->i", test_features, test_features)  # (M,)

batch_size = 500  # fits comfortably in memory (~200 MB per batch)
predictions = [""] * len(test_images)  # placeholder list

for start in range(0, len(test_images), batch_size):
    end = min(start + batch_size, len(test_images))
    batch_feats = test_features[start:end]  # (B,192)
    batch_norms = test_norms[start:end]  # (B,)

    dists = (
        batch_norms[:, None]
        + train_norms[None, :]
        - 2.0 * (batch_feats @ train_features.T)
    )

    for i, idx in enumerate(range(start, end)):
        if not valid_mask[idx]:
            predictions[idx] = default_top5_str
            continue

        k_nearest = 500
        nearest_idx_unsorted = np.argpartition(dists[i], k_nearest)[:k_nearest]
        nearest_idx = nearest_idx_unsorted[np.argsort(dists[i, nearest_idx_unsorted])]

        distinct_ids = []
        seen = set()
        for j in nearest_idx:
            hid = train_hotel_ids[j]
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

        predictions[idx] = " ".join(distinct_ids)

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
