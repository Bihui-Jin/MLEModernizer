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

3.13

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

0.891054699304926

# 6. Current score

0.6136

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The fix adds an environment flag to avoid the protobuf import error, guards model loading so missing `.h5` files no longer crash the notebook, falls back to the most common label from the training set when no models are available, and ensures the `models` variable is always defined. This lets the pipeline run end‑to‑end and creates a valid `submission.csv` file.'
- What this solution (achieved 0.61958) has done: 'I speed up the script by reducing memory and computation load during image preprocessing and model training: the image‑loading function now casts to float16 (halving memory bandwidth), and the RandomForest is built with fewer trees (200 vs 500) which cuts training time while keeping the same classifier type. These changes keep the overall workflow identical and preserve predictions up to negligible floating‑point differences.'
- What this solution (achieved 0.6136) has done: 'I increase the capacity of the lightweight sklearn model by training a larger, class‑balanced RandomForest (500 trees instead of 200). This modest change keeps the overall pipeline identical while giving the model more expressive power and better handling of any class imbalance, which should raise the validation accuracy and move the Kaggle score closer to the target. No other parts of the code are altered.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
from collections import Counter

print("Skipping TensorFlow import; using sklearn fallback.")
TF_AVAILABLE = False




## === cell 1
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"




## === cell 2
sample_csv = pd.read_csv(sample_submission_path)
train_df = pd.read_csv(train_csv_path)
majority_label = train_df["label"].mode().iloc[0]




## === cell 3
models = []
if TF_AVAILABLE:
    models_info = [
        (
            "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5",
            (512, 512),
        ),
        (
            "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5",
            (550, 550),
        ),
    ]

    for path, input_size in models_info:
        if os.path.isfile(path):
            try:
                from tensorflow.keras.models import load_model

                mdl = load_model(path)
                models.append((mdl, input_size))
            except Exception as e:
                print(f"Could not load model {path}: {e}")
        else:
            print(f"Model file not found, skipping: {path}")

    if not models:
        print("No TensorFlow models loaded – will train a lightweight sklearn model.")
else:
    print("TensorFlow not available – will train a lightweight sklearn model.")




## === cell 4
if not models:
    from sklearn.ensemble import RandomForestClassifier
    from PIL import Image
    import concurrent.futures

    IMG_SIZE = (64, 64)  # small enough for fast training

    def _load_and_flatten(img_path):
        """Load an image, resize, normalize and flatten to 1‑D float16 array."""
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(img).astype(np.float16) / np.float16(255.0)
        return arr.flatten()

    train_records = [
        (os.path.join(train_image_dir, row.image_id), row.label)
        for row in train_df.itertuples(index=False)
    ]

    print(
        "Loading and processing training images for sklearn model (parallel, thread‑based)..."
    )
    max_workers = min(os.cpu_count(), 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        flat_arrays = list(
            executor.map(_load_and_flatten, (p for p, _ in train_records))
        )

    if flat_arrays:
        X_train = np.stack(flat_arrays)  # dtype is float16
        y_train = np.array([label for _, label in train_records])

        rf_clf = RandomForestClassifier(
            n_estimators=500,  # more trees than original 200
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",  # address potential class imbalance
        )
        print("Training RandomForest classifier...")
        rf_clf.fit(X_train, y_train)
        rf_clf.set_params(n_jobs=1)
        print("RandomForest training completed.")
    else:
        print("No training images were loaded; falling back to majority label.")
        rf_clf = None
else:
    rf_clf = None




## === cell 5
class_labels = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}

image_predictions = []


def predict_with_sklearn(img_path):
    from PIL import Image

    if rf_clf is None:
        return int(majority_label)
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype(np.float16) / np.float16(255.0)
    flat = arr.flatten().reshape(1, -1)
    return int(rf_clf.predict(flat)[0])


print("Starting inference on test images (parallel)...")
test_image_ids = [
    f
    for f in os.listdir(test_image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]


def _predict_one(image_id):
    img_path = os.path.join(test_image_dir, image_id)
    if models:  # TensorFlow ensemble path (will not be taken here)
        model_predictions = []
        confidence_scores = {}
        for model, input_size in models:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array

            img = load_img(img_path, target_size=input_size)
            img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
            preds = model.predict(img_array, verbose=0)
            pred_class = int(np.argmax(preds, axis=1)[0])
            conf = float(preds[0][pred_class])
            model_predictions.append(pred_class)
            confidence_scores.setdefault(pred_class, []).append(conf)
        votes = Counter(model_predictions)
        most_common = votes.most_common()
        final_pred = most_common[0][0]
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            tie_classes = [cls for cls, cnt in most_common if cnt == most_common[0][1]]
            final_pred = max(
                tie_classes,
                key=lambda cls: sum(confidence_scores[cls])
                / len(confidence_scores[cls]),
            )
    else:
        final_pred = predict_with_sklearn(img_path)
    return {"image_id": image_id, "label": final_pred}


max_inf_workers = min(os.cpu_count(), 8)
with concurrent.futures.ThreadPoolExecutor(max_workers=max_inf_workers) as executor:
    for result in executor.map(_predict_one, test_image_ids):
        image_predictions.append(result)

submission_df = pd.DataFrame(image_predictions)
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 6
submission_df.head()
