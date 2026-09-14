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

0.8541855545482019

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The fix filters the test set using the provided sample submission file (so only real images are predicted), guards against TensorFlow import problems, and safely falls back to a majority‑class baseline while keeping the original logic unchanged. This ensures the generated CSV has the correct length and format for Kaggle submission.'
- What this solution (achieved 0.61099) has done: 'I set the protobuf implementation environment variable before importing TensorFlow to avoid the `MessageFactory` attribute error, and adjust the data and model paths to use the standard Kaggle `/kaggle/input` location. This fixes the runtime crash, allowing the pretrained model to be loaded and used for predictions, which should raise the validation accuracy toward the target score while keeping the original logic intact.'
- What this solution (achieved 0.61099) has done: 'The fix updates the model path to look for the pretrained EfficientNet‑B4 checkpoint in the correct dataset directory (and falls back to the original location if needed). It also adds logic to try each possible location before falling back to the majority‑class baseline, ensuring predictions are generated with the high‑performing model and the submission file is written correctly.'
- What this solution (achieved 0.61099) has done: 'I add a lightweight training fallback that builds and trains a small EfficientNet‑B0 model on the provided training images when the pretrained checkpoint isn’t found. This keeps the original logic (try to load the checkpoint first) but replaces the simple majority‑class baseline with a real model, which should raise the validation accuracy from ~0.61 toward the target while still writing a correctly‑formatted CSV.'
- What this solution (achieved 0.61099) has done: 'The fix expands model loading by recursively searching the input directory for the EfficientNet‑B4 checkpoint, so the pretrained high‑performing model is used when the exact path isn’t matched. If the model is still not found, the lightweight EfficientNet‑B0 fallback is trained for a few more epochs (5 instead of 3) to gain a modest accuracy boost while keeping the original logic. These minimal changes ensure a valid CSV is written and move the score closer to the target.'
- What this solution (achieved 0.18423) has done: 'I replace the failing TensorFlow fallback with a lightweight nearest‑centroid classifier built from the training images. This removes the protobuf crash, keeps the original loading‑attempt logic, and provides a much better-than‑majority prediction (expected accuracy ≈ 0.75‑0.80), moving the score toward the target while still writing a correct submission.csv.'
- What this solution (achieved 0.48057) has done: 'I replace the simple nearest‑centroid fallback with a lightweight k‑Nearest‑Neighbors classifier (using scikit‑learn if available). This keeps the original logic for loading a pretrained model, but when TensorFlow isn’t usable it builds a KNN on 32×32 RGB thumbnails of the training images, which provides a much stronger baseline and pushes the validation accuracy toward the target. The rest of the pipeline and the CSV writing remain unchanged.'
- What this solution (achieved 0.48169) has done: 'The changes batch the test‑image inference when a pretrained EfficientNet model is found, reducing the per‑image Python overhead and avoiding a costly loop of separate `predict` calls. Images are loaded into a pre‑allocated NumPy array in modest‑size batches (default 64) and classified in one GPU/CPU forward pass per batch, preserving exact predictions while staying well within the 600 s limit. The fallback thumbnail‑based classifier logic is left untouched, as it runs only when no pretrained model exists.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import numpy as np
import pandas as pd
from PIL import Image
import multiprocessing  # added for parallel image preprocessing

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None
    load_model = None
    print(f"TensorFlow import failed: {e}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = 300
size = (IMG_SIZE, IMG_SIZE)


def locate(path_parts):
    possible = [
        os.path.join("/", *path_parts),  # absolute root
        os.path.join("/kaggle/input", *path_parts),  # Kaggle path
        os.path.join(*path_parts),  # relative path
    ]
    for p in possible:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Unable to locate {'/'.join(path_parts)}")


SAMPLE_SUB_PATH = locate(
    ["cassava-leaf-disease-classification", "sample_submission.csv"]
)
sample_df = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_df["image_id"].tolist()  # guaranteed correct length and order

possible_model_paths = [
    locate(["cassava-leaf-disease-classification", "Cassava_best_model_effnetb4.h5"]),
    locate(["cassava-challenge-models", "Cassava_best_model_effnetb4.h5"]),
]

best_model = None
for model_path in possible_model_paths:
    if load_model is not None and os.path.exists(model_path):
        try:
            best_model = load_model(model_path)
            print(f"Loaded pretrained model from {model_path}")
            break
        except Exception as e:
            print(f"Failed to load model at {model_path}: {e}")

if best_model is None and load_model is not None:
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.startswith("Cassava_best_model_effnetb4") and f.endswith(".h5"):
                candidate_path = os.path.join(root, f)
                try:
                    best_model = load_model(candidate_path)
                    print(f"Loaded pretrained model from {candidate_path}")
                    break
                except Exception as e:
                    print(f"Failed to load model at {candidate_path}: {e}")
        if best_model is not None:
            break

if best_model is None:
    print(
        "Pretrained model not found – using enhanced lightweight classifier fallback."
    )
    train_csv_path = locate(["cassava-leaf-disease-classification", "train.csv"])
    train_images_dir = locate(["cassava-leaf-disease-classification", "train_images"])
    train_df = pd.read_csv(train_csv_path)

    thumb_size = (128, 128)
    vec_len = thumb_size[0] * thumb_size[1] * 3

    train_image_paths = [
        os.path.join(train_images_dir, img_name) for img_name in train_df["image_id"]
    ]

    def _load_thumb(path):
        try:
            img = Image.open(path).convert("RGB").resize(thumb_size)
            return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        except Exception:
            return np.zeros(vec_len, dtype=np.float32)

    with multiprocessing.Pool(os.cpu_count()) as pool:
        X_train = np.stack(pool.map(_load_thumb, train_image_paths))

    train_label_mode = train_df["label"].mode()[0]
    y_train = train_df["label"].astype(np.int32).values

    nan_mask = np.isnan(X_train).any(axis=1)
    if nan_mask.any():
        X_train[nan_mask] = np.zeros(vec_len, dtype=np.float32)
        y_train[nan_mask] = int(train_label_mode)

    classifier = None
    use_type = None  # "logreg", "knn", "mlp", "centroid"
    try:
        from sklearn.linear_model import LogisticRegression

        logreg = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=500,
            n_jobs=-1,
            C=2.0,
        )
        logreg.fit(X_train, y_train)
        classifier = logreg
        use_type = "logreg"
        print("Logistic Regression classifier trained.")
    except Exception as e:
        print(f"LogisticRegression not available or training failed ({e}); trying KNN.")
        try:
            from sklearn.neighbors import KNeighborsClassifier

            knn = KNeighborsClassifier(
                n_neighbors=5, weights="distance", metric="euclidean"
            )
            knn.fit(X_train, y_train)
            classifier = knn
            use_type = "knn"
            print("KNN classifier trained.")
        except Exception as e2:
            print(f"KNN also unavailable ({e2}); trying MLP.")
            try:
                from sklearn.neural_network import MLPClassifier

                mlp = MLPClassifier(
                    hidden_layer_sizes=(128,),
                    activation="relu",
                    solver="adam",
                    max_iter=40,
                    batch_size=256,
                    random_state=42,
                )
                mlp.fit(X_train, y_train)
                classifier = mlp
                use_type = "mlp"
                print("MLP classifier trained.")
            except Exception as e3:
                print(f"MLP also unavailable ({e3}); falling back to centroids.")
                sums = {c: np.zeros(vec_len, dtype=np.float64) for c in range(5)}
                counts = {c: 0 for c in range(5)}
                for idx, label in enumerate(y_train):
                    sums[label] += X_train[idx]
                    counts[label] += 1
                centroids = np.stack(
                    [
                        sums[c] / counts[c] if counts[c] > 0 else np.zeros(vec_len)
                        for c in range(5)
                    ]
                )
                use_type = "centroid"

    print("Computing test thumbnails for classifier...")
    test_image_paths = [
        os.path.join(
            locate(["cassava-leaf-disease-classification", "test_images"]),
            img_name,
        )
        for img_name in test_images
    ]

    with multiprocessing.Pool(os.cpu_count()) as pool:
        X_test = np.stack(pool.map(_load_thumb, test_image_paths))

    predictions = []
    if use_type in ("logreg", "knn", "mlp"):
        preds = classifier.predict(X_test)
        predictions = preds.astype(int).tolist()
    else:  # centroid
        dists = np.linalg.norm(centroids[None, :, :] - X_test[:, None, :], axis=2)
        predictions = np.argmin(dists, axis=1).astype(int).tolist()

else:
    batch_size = 64
    predictions = []

    def load_batch(image_names):
        batch_arr = np.empty(
            (len(image_names), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32
        )
        for i, name in enumerate(image_names):
            img_path = os.path.join(
                locate(["cassava-leaf-disease-classification", "test_images"]),
                name,
            )
            img = Image.open(img_path).convert("RGB").resize(size)
            batch_arr[i] = np.asarray(img, dtype=np.float32) / 255.0
        return batch_arr

    for start in range(0, len(test_images), batch_size):
        batch_names = test_images[start : start + batch_size]
        batch_data = load_batch(batch_names)
        batch_pred = best_model.predict(batch_data, verbose=0)
        batch_labels = np.argmax(batch_pred, axis=1).astype(int)
        predictions.extend(batch_labels.tolist())



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/108627411.py in <cell line: 0>()
     24 
     25 possible_model_paths = [
---> 26     locate(["cassava-leaf-disease-classification", "Cassava_best_model_effnetb4.h5"]),
     27     locate(["cassava-challenge-models", "Cassava_best_model_effnetb4.h5"]),
     28 ]

/tmp/ipykernel_11/108627411.py in locate(path_parts)
     13         if os.path.exists(p):
     14             return p
---> 15     raise FileNotFoundError(f"Unable to locate {'/'.join(path_parts)}")
     16 
     17 

FileNotFoundError: Unable to locate cassava-leaf-disease-classification/Cassava_best_model_effnetb4.h5

## === cell 2
submission = pd.DataFrame({"image_id": test_images, "label": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2801428327.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": test_images, "label": predictions})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission written to {submission_path} with {len(submission)} rows.")

NameError: name 'predictions' is not defined
