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

0.8925657298277425

# 6. Current score

0.13752

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the immediate import/runtime crash by avoiding the `tensorflow.keras` import path that triggers the protobuf `MessageFactory` error in this environment, and instead use the stable `tf.keras` path. I also fix the missing external model files issue by making model loading robust: if the referenced .h5 paths don’t exist, the script fall back to a simple majority-class baseline derived from `train.csv` so it still produces a valid `submission.csv`. This keeps the core “ensemble + majority vote” logic when models are available, while ensuring end-to-end execution and correct submission formatting. Finally, I align predictions to `sample_submission.csv` ordering to avoid any image_id mismatch issues.'
- What this solution (achieved 0.61099) has done: 'I fix the immediate crash happening on `import tensorflow as tf` (protobuf `MessageFactory.GetPrototype` issue) by removing the TensorFlow dependency entirely and switching to a pure-Python/PIL image loader. Since your referenced `.h5` model files are not available in this environment anyway, the script keep the same robust fallback behavior (majority-class prediction from `train.csv`) and still generate a correctly formatted `submission.csv`. I also keep the ordering aligned to `sample_submission.csv` and add small safeguards so the submission is always fully populated and typed as integer labels.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.89256), and the reason is that the script always falls back to a majority-class predictor because TensorFlow/model loading is disabled. The smallest change that can materially improve accuracy without changing your overall “ensemble voting” semantics is to re-enable TensorFlow *when it imports successfully* and actually load the `.h5` models, while keeping your existing safe fallback to majority class if TF or model files aren’t available. I implement a guarded `tf.keras.models.load_model(...)` path and keep your existing PIL preprocessing and majority-vote aggregation intact. I also ensure prediction ordering matches `sample_submission.csv` exactly and that a valid `submission.csv` is always produced.'
- What this solution (achieved 0.13752) has done: 'The immediate blocker is the TensorFlow import crash caused by an incompatible protobuf runtime (`MessageFactory.GetPrototype`), so I make TensorFlow truly optional by removing the eager import and only attempting it inside the model-loading block with a clean fallback. To move the score up toward the target without changing the “simple fallback” semantics, I replace the majority-class fallback with a lightweight image-based classifier (PIL + NumPy) trained from `train.csv`/`train_images` using per-class mean colors; this keeps the overall approach simple and deterministic but should significantly outperform a constant label. I also keep the existing prediction ordering aligned to `sample_submission.csv` and ensure the submission is always fully populated with integer labels. All paths stay the same and the script still produces `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from collections import Counter

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)

_TF_AVAILABLE = False
_tf_import_error = None




## === cell 1
model_path_1 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5"
)
model_path_2 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5"
)
model_path_3 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5"
)
model_path_4 = (
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5"
)

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
sample = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"




## === cell 2
sample_csv = pd.read_csv(sample)
train_df = pd.read_csv(train_csv_path)

majority_label = int(train_df["label"].value_counts().idxmax())

if not os.path.isdir(test_image_dir):
    raise FileNotFoundError(f"test_image_dir not found: {test_image_dir}")

if not _PIL_AVAILABLE:
    raise RuntimeError(
        "PIL is required for this notebook (image IO) but is not available."
    )

sample_csv.head(), majority_label




## === cell 3
models_info = [
    (model_path_1, (550, 550)),
    (model_path_3, (448, 448)),
    (model_path_4, (550, 550)),
]

models = []
load_errors = []

any_model_file_exists = any(os.path.exists(p) for p, _ in models_info)

if any_model_file_exists:
    try:
        import tensorflow as tf  # noqa: F401

        _TF_AVAILABLE = True
    except Exception as e:
        _TF_AVAILABLE = False
        _tf_import_error = repr(e)

if _TF_AVAILABLE:
    import tensorflow as tf

    for path, input_size in models_info:
        if not os.path.exists(path):
            load_errors.append(f"Missing model file: {path}")
            continue
        try:
            m = tf.keras.models.load_model(path, compile=False)
            models.append((m, input_size))
        except Exception as e:
            load_errors.append(f"Failed to load model {path}: {repr(e)}")
else:
    if _tf_import_error is not None:
        load_errors.append(
            f"TensorFlow import failed (will use fallback): {_tf_import_error}"
        )
    for path, _ in models_info:
        if not os.path.exists(path):
            load_errors.append(f"Missing model file: {path}")
        else:
            load_errors.append(
                f"Model file exists but TensorFlow unavailable, cannot load: {path}"
            )

if load_errors:
    print("Model loading notes:")
    for msg in load_errors:
        print(" -", msg)

print(f"TF available: {_TF_AVAILABLE}")
print(f"Loaded {len(models)} models.")




## === cell 4
def load_and_preprocess_image_pil(image_path, target_size):
    """Returns float32 array in [0,1], shape (1,H,W,3). target_size is (H,W)."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize((target_size[1], target_size[0]))  # PIL expects (W,H)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def compute_image_mean_rgb(image_path, target_size=(64, 64)):
    """Small, fast feature: mean RGB after resizing. Returns shape (3,) float32."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize((target_size[1], target_size[0]))  # (W,H)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
    return arr.reshape(-1, 3).mean(axis=0)


def fit_class_color_centroids(
    train_df, train_image_dir, target_size=(64, 64), max_per_class=800
):
    """
    Fallback "training": compute per-class centroid in mean-RGB space.
    This is a minimal, deterministic upgrade over the majority-class baseline
    and keeps the script end-to-end without external model files.
    """
    centroids = {}
    counts = {c: 0 for c in sorted(train_df["label"].unique().tolist())}
    sums = {c: np.zeros((3,), dtype=np.float64) for c in counts.keys()}

    df = train_df.copy()
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    for _, row in df.iterrows():
        label = int(row["label"])
        if counts[label] >= max_per_class:
            continue
        img_path = os.path.join(train_image_dir, row["image_id"])
        if not os.path.exists(img_path):
            continue
        try:
            feat = compute_image_mean_rgb(img_path, target_size=target_size)
        except Exception:
            continue
        sums[label] += feat.astype(np.float64)
        counts[label] += 1

        if all(counts[c] >= max_per_class for c in counts.keys()):
            break

    for c in counts.keys():
        if counts[c] > 0:
            centroids[c] = (sums[c] / counts[c]).astype(np.float32)
        else:
            centroids[c] = None

    return centroids, counts


def predict_by_nearest_centroid(
    image_path, centroids, majority_label, target_size=(64, 64)
):
    """Predict class whose centroid is closest in L2 distance to mean-RGB feature."""
    try:
        feat = compute_image_mean_rgb(image_path, target_size=target_size).astype(
            np.float32
        )
    except Exception:
        return int(majority_label)

    best_c = None
    best_d = None
    for c, cen in centroids.items():
        if cen is None:
            continue
        d = float(np.sum((feat - cen) ** 2))
        if best_d is None or d < best_d:
            best_d = d
            best_c = c

    return int(best_c) if best_c is not None else int(majority_label)




## === cell 5
image_id_to_pred = {}

if len(models) > 0:
    image_ids = sample_csv["image_id"].tolist()
    for image_id in image_ids:
        image_path = os.path.join(test_image_dir, image_id)

        if not os.path.exists(image_path):
            image_id_to_pred[image_id] = majority_label
            continue

        model_predictions = []
        for model, input_size in models:
            img_array = load_and_preprocess_image_pil(image_path, input_size)
            preds = model.predict(img_array, verbose=0)
            model_predictions.append(int(np.argmax(preds, axis=1)[0]))

        final_predicted_class = Counter(model_predictions).most_common(1)[0][0]
        image_id_to_pred[image_id] = int(final_predicted_class)

else:
    if not os.path.isdir(train_image_dir):
        print(
            f"Warning: train_image_dir not found: {train_image_dir}. Using majority-label fallback."
        )
        for image_id in sample_csv["image_id"].tolist():
            image_id_to_pred[image_id] = majority_label
    else:
        centroids, counts = fit_class_color_centroids(
            train_df,
            train_image_dir=train_image_dir,
            target_size=(64, 64),
            max_per_class=800,
        )
        print("Fallback centroids fitted with per-class counts:", counts)

        for image_id in sample_csv["image_id"].tolist():
            image_path = os.path.join(test_image_dir, image_id)
            if not os.path.exists(image_path):
                image_id_to_pred[image_id] = majority_label
            else:
                image_id_to_pred[image_id] = predict_by_nearest_centroid(
                    image_path,
                    centroids=centroids,
                    majority_label=majority_label,
                    target_size=(64, 64),
                )

submission_df = sample_csv.copy()
submission_df["label"] = submission_df["image_id"].map(image_id_to_pred)

if submission_df["label"].isna().any():
    submission_df["label"] = submission_df["label"].fillna(majority_label)
submission_df["label"] = submission_df["label"].astype(int)

submission_df.head()




## === cell 6
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file saved at: {submission_path}")
print("Submission shape:", submission_df.shape)
print("Label distribution:", submission_df["label"].value_counts().to_dict())
