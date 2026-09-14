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
- What this solution (achieved 0.02686) has done: 'I wrapped the TensorFlow import in a safe try‑except block and, if it fails, the script skips all embedding‑based logic and falls back to the chain‑specific + global top‑5 heuristic that already produces a reasonable MAP@5 score. This removes the protobuf‑related import error while still generating a correctly formatted `submission.csv`.'
- What this solution (achieved 0.02686) has done: 'The script now avoids the TensorFlow import error by falling back to a lightweight image‑based similarity method when TensorFlow cannot be used. It builds normalized RGB feature vectors (16×16 pixels) for a few training images per hotel, creates hotel centroids, and then matches each test image to the nearest centroids, yielding a more informed top‑5 prediction list that improves MAP@5 toward the target score while still writing a correctly formatted `submission.csv`.'
- What this solution (achieved 0.02686) has done: 'The fix adds a simple post‑processing step to the non‑TensorFlow fallback: after retrieving the five most similar hotels by image vectors, we supplement them with the globally most frequent hotels, deduplicate, and keep the first five. This modest adjustment keeps the core logic unchanged while biasing predictions toward frequent hotels, which improves MAP@5 and moves the score closer to the target. No other parts of the pipeline are altered.'
- What this solution (achieved 0.02686) has done: 'Implemented a chain‑aware fallback: each test image now first tries its own hotel chain’s top‑5 list (when available) before using image‑similarity nearest hotels and the global top‑5. This leverages the already‑computed `chain_top5` without altering the core model or training logic, giving a modest MAP@5 boost toward the target while keeping the same overall pipeline.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    import tensorflow as tf

    USING_TF = True
except Exception:  # catch the AttributeError seen in the environment
    tf = None
    USING_TF = False

SEED = 42
np.random.seed(SEED)
if USING_TF:
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
if USING_TF:
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
    hotel_id_list = []
    hotel_idx_map = {}
    for hotel, paths in hotel_to_paths.items():
        start = len(all_train_paths)
        all_train_paths.extend(paths)
        hotel_id_list.extend([hotel] * len(paths))
        hotel_idx_map[hotel] = list(range(start, start + len(paths)))

    train_emb = compute_embeddings(all_train_paths, batch_sz=256)
    train_emb_norm = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)

    hotel_centroids = {}
    for hotel, idxs in hotel_idx_map.items():
        if not idxs:
            continue
        emb_vectors = train_emb_norm[idxs]
        centroid = np.mean(emb_vectors, axis=0)
        centroid /= np.linalg.norm(centroid)
        hotel_centroids[hotel] = centroid

    print(f"Computed centroids for {len(hotel_centroids)} hotels.")
    hotel_ids = list(hotel_centroids.keys())
    centroid_matrix = np.stack([hotel_centroids[h] for h in hotel_ids])

    test_emb = compute_embeddings(test_paths, batch_sz=256)
    norms = np.linalg.norm(test_emb, axis=1, keepdims=True)
    norms[norms == 0] = np.nan
    test_emb_norm = test_emb / norms

    sim_matrix = centroid_matrix @ test_emb_norm.T  # (N_hotels, N_test)

    predictions = []
    for idx, name in enumerate(test_names):
        sims = sim_matrix[:, idx]
        top_idx = np.argpartition(-sims, 5)[:5]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        nearest_hotels = [hotel_ids[i] for i in top_idx]

        chain_dir = os.path.basename(os.path.dirname(test_paths[idx]))
        chain_id = int(chain_dir) if chain_dir.isdigit() else None
        chain_specific = chain_top5.get(chain_id, []) if chain_id is not None else []

        candidates = chain_specific + nearest_hotels + top5_hotels
        seen = set()
        final = []
        for h in candidates:
            if h not in seen:
                final.append(h)
                seen.add(h)
            if len(final) == 5:
                break
        predictions.append(" ".join(final[:5]))
else:
    from PIL import Image

    def img_to_vec(path, size=(16, 16)):
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
        hotel_to_paths[h] = hotel_to_paths[h][:3]

    hotel_ids = list(hotel_to_paths.keys())
    centroid_list = []
    for h in hotel_ids:
        vecs = [img_to_vec(p) for p in hotel_to_paths[h] if os.path.exists(p)]
        if len(vecs) == 0:
            centroid = np.zeros(16 * 16 * 3, dtype=np.float32)
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
        top_idx = np.argsort(-sims, axis=0)[:5]  # (5, B)
        for col in range(top_idx.shape[1]):
            nearest = [hotel_ids[i] for i in top_idx[:, col]]

            idx = start + col
            chain_dir = os.path.basename(os.path.dirname(test_paths[idx]))
            chain_id = int(chain_dir) if chain_dir.isdigit() else None
            chain_specific = (
                chain_top5.get(chain_id, []) if chain_id is not None else []
            )

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
