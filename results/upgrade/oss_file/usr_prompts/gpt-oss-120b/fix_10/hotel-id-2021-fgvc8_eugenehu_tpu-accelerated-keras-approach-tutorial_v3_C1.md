# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.0463500884061631

# 6. Current score

0.02686

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the TensorFlow‑based pipeline with a lightweight baseline that avoids the protobuf import error and the stratified split issue. The new version loads the training metadata, finds the five most common hotel IDs, and assigns this same list to every test image, producing a correctly formatted `submission.csv`. This fixes all runtime errors while providing a valid submission that yields a non‑zero MAP@5 score.'
- What this solution (achieved 0.00209) has done: 'I add a small heuristic that uses the folder‑structure chain information (available in the training metadata) to predict the most common hotels for each chain instead of always using the global top‑5 hotels. This keeps the original baseline logic but personalizes predictions per chain, which should increase MAP@5 toward the target. The changes only adjust how test image paths are collected and how the prediction list is built, preserving the overall workflow.'
- What this solution (achieved 0.00209) has done: 'I keep the overall pipeline unchanged and only adjust how the five predictions are assembled. For each test image we now start with the most frequent hotel for its chain (if known), then add the remaining chain‑specific hotels, and finally fill any remaining slots with the global most‑common hotels that are not already in the list. This tiny change adds useful fallback candidates while preserving the original chain‑based logic, and should raise MAP@5 toward the target without altering the core model or data handling.'
- What this solution (achieved 0.02686) has done: 'The script was slowed mainly by repeatedly invoking the model on many tiny batches (up to three images per hotel) and by loading each image individually with Python loops. I replaced the per‑hotel looping with a single large TF Dataset that loads, preprocesses, and batches images efficiently, then computes all embeddings in a few big batches. Hotel centroids are built from these embeddings using index bookkeeping, preserving the exact same normalization and averaging logic. The test‑set embedding computation is also switched to the new fast pipeline, so the overall runtime drops well below the 600 s limit while the predictions remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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




## === cell 3
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




## === cell 4
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, pooling="avg", weights="imagenet"
)
IMG_SIZE = (224, 224)


def _preprocess_tf(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def compute_embeddings(paths, batch_sz=128):
    """Load, preprocess and embed a list of image paths using a tf.data pipeline."""
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_sz).prefetch(tf.data.AUTOTUNE)
    emb_list = []
    for batch in ds:
        emb = base_model(batch, training=False)
        emb_list.append(emb.numpy())
    return np.concatenate(emb_list, axis=0)


hotel_to_paths = {}
for _, row in train_df.iterrows():
    hotel = str(row.hotel_id)
    img_path = os.path.join(TRAIN_IMG_DIR, str(row.chain), row.image)
    hotel_to_paths.setdefault(hotel, []).append(img_path)

for h in hotel_to_paths:
    hotel_to_paths[h] = hotel_to_paths[h][:3]

all_train_paths = []
hotel_id_list = []  # parallel list of hotel ids for each image
hotel_idx_map = {}  # hotel -> list of positions in all_train_paths
for hotel, paths in hotel_to_paths.items():
    start = len(all_train_paths)
    all_train_paths.extend(paths)
    hotel_id_list.extend([hotel] * len(paths))
    hotel_idx_map[hotel] = list(range(start, start + len(paths)))

train_emb = compute_embeddings(all_train_paths, batch_sz=256)

train_emb_norm = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)

hotel_centroids = {}
for hotel, idxs in hotel_idx_map.items():
    if not idxs:  # safety, should not happen
        continue
    emb_vectors = train_emb_norm[idxs]  # (k, D)
    centroid = np.mean(emb_vectors, axis=0)  # mean of unit vectors
    centroid /= np.linalg.norm(centroid)  # re‑normalize
    hotel_centroids[hotel] = centroid

print(f"Computed centroids for {len(hotel_centroids)} hotels.")

hotel_ids = list(hotel_centroids.keys())
centroid_matrix = np.stack([hotel_centroids[h] for h in hotel_ids])  # (N_hotels, D)




## === cell 5
test_emb = compute_embeddings(test_paths, batch_sz=256)  # shape (N_test, D)

norms = np.linalg.norm(test_emb, axis=1, keepdims=True)
norms[norms == 0] = np.nan
test_emb_norm = test_emb / norms

sim_matrix = centroid_matrix @ test_emb_norm.T  # (N_hotels, N_test)

predictions = []
for idx, name in enumerate(test_names):
    if np.isnan(sim_matrix[:, idx]).all():
        parent_dir = os.path.basename(os.path.dirname(test_paths[idx]))
        if parent_dir.isdigit():
            chain_id = int(parent_dir)
            chain_preds = chain_top5.get(chain_id, [])
        else:
            chain_preds = []
        preds = list(chain_preds)
        for h in top5_hotels:
            if len(preds) >= 5:
                break
            if h not in preds:
                preds.append(h)
        while len(preds) < 5:
            preds.append(top5_hotels[len(preds) % len(top5_hotels)])
        predictions.append(" ".join(preds[:5]))
    else:
        sims = sim_matrix[:, idx]
        top_idx = np.argpartition(-sims, 5)[:5]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        nearest_hotels = [hotel_ids[i] for i in top_idx]
        predictions.append(" ".join(nearest_hotels[:5]))

submission = pd.DataFrame({"image": test_names, "hotel_id": predictions})




## === cell 6
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
