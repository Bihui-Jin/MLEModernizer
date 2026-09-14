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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8822907222725899

# 6. Current score

0.29522

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I replace the broken Keras import with TensorFlow‑Keras, add a safe fallback that predicts the most frequent class when the pretrained model file is missing, filter the test directory for image files, and ensure the prediction list matches the test image list before writing a correct `submission.csv`.'
- What this solution (achieved 0.10725) has done: 'The fix adds a safe environment setting before importing TensorFlow to avoid the protobuf `MessageFactory` error, wraps the TensorFlow import in a try/except, and falls back gracefully when TensorFlow isn’t available.  
If the pretrained model file is missing, a lightweight image‑based classifier is built on‑the‑fly: it samples a few training images per class, computes the average green‑channel intensity for each class, and predicts the nearest class for each test image.  
Finally, the script always writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.13789) has done: 'The changes fix the wrong input path, keep TensorFlow optional, and replace the simple green‑channel centroid with a full RGB‑mean centroid per class, which gives much richer class representations while staying within the original fallback‑only logic. This improves prediction quality and brings the validation score nearer the target, and the script now reliably writes a correctly‑named `submission.csv`.'
- What this solution (achieved 0.1932) has done: 'I prevent the TensorFlow import from crashing by skipping it entirely when the library is unavailable, and I improve the fallback classifier: instead of using only RGB channel means, I build full‑image centroids (flattened 32×32 RGB vectors) for each class and classify test images by nearest‑centroid distance. This modest enhancement keeps the original logic while substantially raising accuracy, moving the score closer to the target.'
- What this solution (achieved 0.23879) has done: 'I enhance the fallback classifier by storing all sampled image vectors per class and classifying each test image using the nearest‑vector cosine distance (which is more discriminative than a single centroid). This keeps the original structure, adds only NumPy/Pillow operations, and should raise the validation accuracy toward the target while still writing a proper submission.csv.'
- What this solution (achieved 0.20217) has done: 'I improve the fallback classifier by (1) resizing images directly to the 32×32 centroid size (removing the unnecessary 300‑pixel resize), (2) increasing the per‑class sample limit to use more training images, and (3) storing L2‑normalized vectors for cosine similarity, which yields more discriminative nearest‑vector matching. These tweaks keep the overall architecture unchanged while raising validation accuracy toward the target.'
- What this solution (achieved 0.28737) has done: 'I increase the image resolution used for the simple fallback classifier (64×64 instead of 32×32) and raise the per‑class sample limit so more training images contribute to each class representation. After building the class centroids I also store their L2‑normalized versions and, during prediction, choose the label whose centroid has the smallest cosine distance to the test image vector. These modest adjustments keep the overall fallback logic unchanged while providing richer features that should raise the validation accuracy toward the target.'
- What this solution (achieved 0.16854) has done: 'I keep the overall fallback‑centroid approach but replace the single‑centroid cosine match with a lightweight nearest‑neighbor search over all L2‑normalized training vectors (still using cosine distance). This richer similarity measure usually improves classification accuracy while preserving the original pipeline structure. I also lower the per‑class sample limit to keep the computation fast. The script now write a proper `submission.csv` and should move the validation score much closer to the target.'
- What this solution (achieved 0.287) has done: 'I replace the nearest‑neighbor search with a simpler cosine‑distance‑to‑class‑centroid decision, which is more stable for this pixel‑based fallback and should raise accuracy toward the target. The rest of the pipeline (data loading, image preprocessing, centroid construction) stays unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.13191) has done: 'I enhance the fallback classifier by adding a lightweight k‑nearest‑neighbors search over all L2‑normalized training vectors (using cosine similarity) while keeping the existing centroid logic as a backup. This adds richer similarity information and should raise accuracy toward the target without altering the overall pipeline or requiring new packages.'
- What this solution (achieved 0.29522) has done: 'We replace the noisy k‑NN voting with a direct nearest‑centroid cosine similarity check, which aligns the prediction step more closely with the data representation already built (centroids). This change keeps the overall fallback‑classifier structure intact, removes the dependence on the large global vector matrix, and is expected to raise accuracy toward the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

np.random.seed(42)

ROOT_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
print("Root contents:", os.listdir(ROOT_DIR))

tf = None
load_model = None
print("TensorFlow imports are skipped; fallback predictions will be used.")



## === cell 1
train_path = os.path.join(ROOT_DIR, "train.csv")
train_df = pd.read_csv(train_path)
fallback_label = train_df["label"].mode()[0]  # most frequent label
print("Fallback label (most common class):", fallback_label)



## === cell 2
centroids = {}
centroids_norm = {}  # L2‑normalized centroids (kept for completeness)
class_vectors = {}  # per‑class normalized vectors (unused in final prediction)
global_vectors = []  # all normalized vectors across classes
global_labels = []  # corresponding class labels
print(
    "Building RGB‑image centroids and per‑class normalized vectors from a subset of training images..."
)
train_images_dir = os.path.join(ROOT_DIR, "train_images")
if not os.path.isdir(train_images_dir):
    raise RuntimeError(f"Training images directory not found at {train_images_dir}")

CENTROID_SIZE = (64, 64)  # resolution used for feature extraction
MAX_PER_CLASS = 1000  # limit samples per class to keep memory reasonable

for cls in sorted(train_df["label"].unique()):
    cls_images = train_df[train_df["label"] == cls]["image_id"].tolist()
    np.random.shuffle(cls_images)
    selected = cls_images[:MAX_PER_CLASS]

    vectors = []
    for img_name in selected:
        img_path = os.path.join(train_images_dir, img_name)
        if not os.path.isfile(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(CENTROID_SIZE)  # resize to 64×64
            arr = np.array(img, dtype=np.float32) / 255.0
            vectors.append(arr.flatten())
        except Exception:
            continue

    if vectors:
        vectors_np = np.stack(vectors)  # (N, 64*64*3)
        centroid = np.mean(vectors_np, axis=0)
        centroids[int(cls)] = centroid
        centroids_norm[int(cls)] = centroid / (np.linalg.norm(centroid) + 1e-8)

        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True) + 1e-8
        vectors_norm = vectors_np / norms
        class_vectors[int(cls)] = vectors_norm
        global_vectors.append(vectors_norm)
        global_labels.extend([int(cls)] * vectors_norm.shape[0])
        print(f"Class {cls}: {vectors_np.shape[0]} vectors stored.")
    else:
        print(f"Class {cls}: no valid images found.")

if global_vectors:
    global_vectors = np.vstack(global_vectors).astype(
        np.float32
    )  # shape (total_N, feat)
    global_labels = np.array(global_labels, dtype=np.int32)
    print(f"Aggregated {global_vectors.shape[0]} normalized vectors for k‑NN.")
else:
    print("No training vectors were collected for k‑NN.")

print(f"Computed centroids for {len(centroids)} classes.")
print(f"Stored normalized vectors for {len(class_vectors)} classes.")

if centroids_norm:
    centroid_labels = np.array(sorted(centroids_norm.keys()), dtype=np.int32)
    centroid_matrix = np.stack([centroids_norm[cls] for cls in centroid_labels])
else:
    centroid_labels = np.array([], dtype=np.int32)
    centroid_matrix = np.empty((0, 64 * 64 * 3), dtype=np.float32)



## === cell 3
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
test_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
test_images.sort()  # deterministic order
print(f"Found {len(test_images)} test images.")




## === cell 4
def cosine_distance(a, b):
    """Cosine distance between two 1‑D float32 vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return 1.0 - np.dot(a, b) / denom


preds = []
for image_name in test_images:
    img_path = os.path.join(TEST_DIR, image_name)
    img = Image.open(img_path).convert("RGB")
    img = img.resize(CENTROID_SIZE)  # same size as training
    flat_vec = np.array(img, dtype=np.float32) / 255.0
    flat_vec = flat_vec.flatten()
    flat_vec_norm = flat_vec / (np.linalg.norm(flat_vec) + 1e-8)

    if centroid_matrix.size > 0:
        sims = centroid_matrix @ flat_vec_norm  # cosine similarity with each centroid
        best_idx = int(np.argmax(sims))
        label = int(centroid_labels[best_idx])
    else:
        if global_vectors.size > 0:
            sims = np.dot(global_vectors, flat_vec_norm)  # (total_N,)
            K = 3
            if K < len(sims):
                top_k_idx = np.argpartition(-sims, K - 1)[:K]
            else:
                top_k_idx = np.arange(len(sims))
            top_labels = global_labels[top_k_idx]
            counts = np.bincount(top_labels, minlength=5)
            label = int(np.argmax(counts))
        else:
            label = int(fallback_label)

    preds.append(int(label))

print(f"Generated predictions for {len(preds)} images.")



## === cell 5
submission = pd.DataFrame({"image_id": test_images, "label": preds})
print(submission.head())
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
