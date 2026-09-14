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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import numpy as np
import pandas as pd

try:
    import tensorflow as tf

    _tf_import_success = True
except Exception:
    tf = None
    _tf_import_success = False

USING_TF = False
SEED = 42
np.random.seed(SEED)
if USING_TF and tf is not None:
    tf.random.set_seed(SEED)



## === cell 1
DIR = "../input/hotel-id-2021-fgvc8"
TRAIN_IMG_DIR = os.path.join(DIR, "train_images")
TEST_IMG_DIR = os.path.join(DIR, "test_images")

train_df = pd.read_csv(os.path.join(DIR, "train.csv"))
train_df = train_df.drop_duplicates(subset=["image"])
print("Unique chains:", train_df.chain.nunique())
print("Unique hotels:", train_df.hotel_id.nunique())
print("Training samples:", train_df.shape[0])



## === cell 2
top5_hotels = train_df["hotel_id"].value_counts().nlargest(5).index.astype(str).tolist()
print("Top‑5 frequent hotel IDs:", top5_hotels)

test_paths = []
for root, _, files in os.walk(TEST_IMG_DIR):
    for f in files:
        if f.lower().endswith(".jpg"):
            test_paths.append(os.path.join(root, f))
test_names = [os.path.basename(p) for p in test_paths]
print(f"Found {len(test_names)} test images.")

chain_top5 = (
    train_df.groupby("chain")["hotel_id"]
    .apply(lambda s: s.value_counts().index.astype(str).tolist()[:5])
    .to_dict()
)
print(f"Computed chain‑specific top‑5 lists for {len(chain_top5)} chains.")



## === cell 3
from PIL import Image

IMG_VEC_SIZE = (32, 32)  # 32×32 RGB vectors – fast and memory‑friendly
MAX_IMAGES_PER_HOTEL = 12  # use more training images per hotel for richer centroids


def img_to_vec(path, size=IMG_VEC_SIZE):
    """Convert an image to a normalized flat vector."""
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten()
    except Exception:
        return np.zeros(size[0] * size[1] * 3, dtype=np.float32)


hotel_to_paths = {}
for _, row in train_df.iterrows():
    hotel = str(row.hotel_id)
    img_path = os.path.join(TRAIN_IMG_DIR, str(row.chain), row.image)
    hotel_to_paths.setdefault(hotel, []).append(img_path)

for h in hotel_to_paths:
    hotel_to_paths[h] = hotel_to_paths[h][:MAX_IMAGES_PER_HOTEL]

hotel_ids = list(hotel_to_paths.keys())
centroid_list = []
for h in hotel_ids:
    vecs = [img_to_vec(p) for p in hotel_to_paths[h] if os.path.exists(p)]
    if len(vecs) == 0:
        centroid = np.zeros(IMG_VEC_SIZE[0] * IMG_VEC_SIZE[1] * 3, dtype=np.float32)
    else:
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
    centroid_list.append(centroid.astype(np.float32))

centroids = np.stack(centroid_list)  # shape (num_hotels, feat_dim)

test_vecs = np.stack([img_to_vec(p) for p in test_paths])
test_norms = np.linalg.norm(test_vecs, axis=1, keepdims=True)
test_norms[test_norms == 0] = 1.0
test_vecs = test_vecs / test_norms

predictions = []
batch_sz = 500
for start in range(0, len(test_vecs), batch_sz):
    batch = test_vecs[start : start + batch_sz]  # (B, D)
    sims = centroids @ batch.T  # (num_hotels, B)
    top_idx = np.argsort(-sims, axis=0)[:5, :]  # (5, B)

    for col in range(top_idx.shape[1]):
        nearest = [hotel_ids[i] for i in top_idx[:, col]]

        idx = start + col
        chain_dir = os.path.basename(os.path.dirname(test_paths[idx]))
        chain_id = int(chain_dir) if chain_dir.isdigit() else None
        chain_specific = chain_top5.get(chain_id, []) if chain_id is not None else []

        candidates = chain_specific + nearest + top5_hotels
        seen = set()
        final = []
        for h in candidates:
            if h not in seen:
                final.append(h)
                seen.add(h)
            if len(final) == 5:
                break
        predictions.append(" ".join(final[:5]))



## === cell 4
submission = pd.DataFrame({"image": test_names, "hotel_id": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
