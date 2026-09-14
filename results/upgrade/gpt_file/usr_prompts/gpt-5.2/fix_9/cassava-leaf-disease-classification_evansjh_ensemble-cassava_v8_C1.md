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

0.8899969779389544

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the TensorFlow/protobuf crash by removing the environment override that forces the pure-Python protobuf implementation, which is incompatible with the Kaggle TensorFlow build and causes the `MessageFactory` error. Next, I make model discovery more robust by scanning `/kaggle/input` for any `.h5/.keras` models if the hardcoded paths don’t exist, and I safely continue even if some models fail to load. Finally, I harden inference so you never get an empty vote list (the `IndexError`), by falling back to predicting class `4` (Healthy) if no model produced a prediction for an image, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.11584) has done: 'I fix the TensorFlow/protobuf crash by explicitly forcing the C++ protobuf backend *before* importing TensorFlow (the current environment is ending up on the incompatible pure-Python path, causing `MessageFactory.GetPrototype` to be missing). Then I keep your ensemble/voting logic intact but make inference numerically correct for models that output logits by applying a softmax only when predictions don’t already look like probabilities; this is a minimal calibration change that should move accuracy toward your target without changing the architecture or training. Finally, I keep the robust model discovery/loading and ensure a valid `submission.csv` is always written with the required format and ordering.'
- What this solution (achieved 0.11584) has done: 'I fix the TensorFlow import crash by removing the protobuf environment overrides that force an unavailable C++/pyext protobuf implementation in this runtime, letting TensorFlow use the compatible default protobuf backend. Then I keep your ensemble voting and inference logic intact, only adding a safe fallback to load Keras models even when TensorFlow can’t be imported (so the notebook still produces a valid `submission.csv` instead of failing early). Finally, I preserve your submission formatting and ordering checks so the output CSV is always valid for Kaggle.'
- What this solution (achieved 0.11584) has done: 'The immediate blocker is the TensorFlow/protobuf crash (`MessageFactory` missing `GetPrototype`), which happens when an incompatible protobuf package version is installed for the TensorFlow build in this runtime. The minimal, safe fix is to pin protobuf to a TensorFlow-compatible version at runtime (via `pip`) before importing TensorFlow, then re-import TensorFlow cleanly. After that, I keep your ensemble/voting logic unchanged, but ensure TF is actually available so predictions come from the loaded models instead of the current fallback-to-4 path that drives the very low score. Finally, I keep the same submission formatting and ordering assertions so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.11584) has done: 'Your low score is most consistent with silently using the default `label=4` path for most/all test images because TensorFlow (or model loading) is failing, so the smallest impactful change is to make TensorFlow import/model loading reliable in this Kaggle runtime. I remove the runtime `pip install protobuf==3.20.*` step (it’s risky under Python 3.13 and can break TF), and instead use a safe TensorFlow import with no protobuf backend forcing and a single retry after clearing protobuf modules. Then, keeping your exact ensemble/voting logic, I add a minimal input-shape alignment step: for each model we infer the needed `target_size` from `model.input_shape` when available (your hardcoded 448/512/550 likely mismatches many models and causes inference exceptions that currently get swallowed). These changes should move accuracy sharply upward toward your target by ensuring real model predictions are produced rather than the fallback.'
- What this solution (achieved 0.11584) has done: 'The crash happens before any inference because TensorFlow is importing against an incompatible protobuf API where `MessageFactory.GetPrototype` is missing; as a result your code never reaches model loading and ends up in fallback behavior, yielding the very low score. I fix this by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this matches the protobuf API TensorFlow expects in this runtime) and by adding a single robust retry that clears protobuf-related modules. I also remove the `tf.nn.softmax` dependency inside `_to_probabilities` so it still works if TF is unavailable, without changing your ensemble/voting logic. These changes are minimal, unblock end-to-end execution, and should materially increase accuracy by ensuring real model predictions are produced.'
- What this solution (achieved 0.11584) has done: 'The main blocker is the TensorFlow/protobuf import crash (`MessageFactory` missing `GetPrototype`), which prevents any models from loading and forces the script into the “predict 4 for everything” fallback that drives the very low score. I fix this by removing the protobuf-backend environment override and by adding a safe TF import routine that (a) prefers the default runtime config and (b) retries once after clearing protobuf modules if needed. With TF loading again, the existing ensemble/voting logic actually run and should move accuracy substantially upward toward your target without changing the model architecture/training semantics. I also keep the robust model discovery and ensure a valid `submission.csv` is always written in the exact sample order.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd


def _import_tensorflow_safely():
    """
    FIX (runtime): Import TF without forcing protobuf backend.
    Retry once after clearing protobuf-related modules to avoid stale state.
    """
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model  # noqa: F401
        from tensorflow.keras.preprocessing.image import (
            load_img,
            img_to_array,
        )  # noqa: F401

        return tf, load_model, load_img, img_to_array, True
    except Exception as e1:
        print(f"[WARN] TensorFlow import attempt 1 failed: {type(e1).__name__}: {e1}")

        for k in list(sys.modules.keys()):
            if k.startswith(("google.protobuf", "protobuf", "google.")):
                sys.modules.pop(k, None)

        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow.keras.models import load_model  # noqa: F401
            from tensorflow.keras.preprocessing.image import (
                load_img,
                img_to_array,
            )  # noqa: F401

            return tf, load_model, load_img, img_to_array, True
        except Exception as e2:
            print(
                f"[WARN] TensorFlow import attempt 2 failed; will fall back to default predictions.\n"
                f"  -> {type(e2).__name__}: {e2}"
            )
            return None, None, None, None, False


tf, load_model, load_img, img_to_array, _TF_AVAILABLE = _import_tensorflow_safely()

from collections import Counter

np.random.seed(42)
if _TF_AVAILABLE:
    tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
model_path_5 = (
    "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5"
)
model_path_6 = "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5"

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"

assert os.path.exists(sample), f"Missing sample submission at {sample}"
assert os.path.isdir(test_image_dir), f"Missing test image dir at {test_image_dir}"



## === cell 2
sample_csv = pd.read_csv(sample)
assert "image_id" in sample_csv.columns and "label" in sample_csv.columns
sample_csv.head()




## === cell 3
def _find_candidate_model_files(root="/kaggle/input"):
    exts = (".h5", ".keras")
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                candidates.append(os.path.join(dirpath, fn))
    return candidates


def _safe_load_model(path):
    if not _TF_AVAILABLE or load_model is None:
        return None
    try:
        return load_model(path, compile=False)
    except Exception as e:
        print(f"[WARN] Failed to load model: {path}\n  -> {type(e).__name__}: {e}")
        return None


def _infer_input_size_from_model(m, fallback=(512, 512)):
    """
    Keep existing semantics; pick target_size based on model.input_shape when available.
    """
    try:
        ishape = getattr(m, "input_shape", None)
        if ishape is None:
            return fallback
        if (
            isinstance(ishape, (list, tuple))
            and len(ishape) > 0
            and isinstance(ishape[0], (list, tuple))
        ):
            ishape = ishape[0]
        if isinstance(ishape, (list, tuple)) and len(ishape) >= 3:
            h, w = ishape[1], ishape[2]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return (h, w)
    except Exception:
        pass
    return fallback


models_info_preferred = [
    (model_path_1, (550, 550)),
    (model_path_2, (512, 512)),
    (model_path_3, (448, 448)),
    (model_path_4, (550, 550)),
    (model_path_5, (512, 512)),
    (model_path_6, (512, 512)),
]

existing_preferred = [(p, sz) for (p, sz) in models_info_preferred if os.path.exists(p)]

if len(existing_preferred) == 0:
    print(
        "[WARN] None of the specified model paths exist. Falling back to scanning /kaggle/input for model files."
    )
    found = _find_candidate_model_files("/kaggle/input")
    print(f"[INFO] Found {len(found)} candidate model files.")
    existing_preferred = [(p, (512, 512)) for p in sorted(found)[:6]]

models = []
for path, input_size in existing_preferred:
    m = _safe_load_model(path)
    if m is not None:
        input_size = _infer_input_size_from_model(m, fallback=input_size)
        models.append((m, input_size, path))

print(f"[INFO] Loaded {len(models)} models:")
for _, sz, p in models:
    print(f"  - {p} @ input_size={sz}")

if len(models) == 0:
    print(
        "[WARN] No models could be loaded. Will generate a valid submission using default label=4."
    )



## === cell 4
class_labels = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}



## === cell 5
test_ids = sample_csv["image_id"].astype(str).tolist()

missing = [
    img_id
    for img_id in test_ids
    if not os.path.exists(os.path.join(test_image_dir, img_id))
]
if len(missing) > 0:
    print(
        f"[WARN] {len(missing)} images listed in sample_submission.csv not found under {test_image_dir}. "
        f"Example: {missing[:3]}"
    )




## === cell 6
def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _to_probabilities(preds: np.ndarray) -> np.ndarray:
    """
    Do not depend on tf.nn.softmax; use NumPy softmax when needed.
    Keeps semantics: apply softmax only when outputs don't resemble probabilities.
    """
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)

    row_sums = preds.sum(axis=1)
    looks_like_probs = (
        np.all(np.isfinite(preds))
        and np.all(preds >= -1e-6)
        and np.all(preds <= 1.0 + 1e-6)
        and np.all(np.abs(row_sums - 1.0) < 1e-2)
    )
    if looks_like_probs:
        return preds

    return _softmax_np(preds, axis=-1)


image_predictions = []

fallback_count = 0
predicted_count = 0

for image_id in test_ids:
    img_path = os.path.join(test_image_dir, image_id)
    if not os.path.exists(img_path):
        fallback_count += 1
        image_predictions.append({"image_id": image_id, "label": 4})
        continue

    if len(models) == 0 or (not _TF_AVAILABLE):
        fallback_count += 1
        image_predictions.append({"image_id": image_id, "label": 4})
        continue

    model_predictions = []
    confidence_scores = {}

    for model, input_size, _path in models:
        try:
            img = load_img(img_path, target_size=input_size)
            img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)

            preds = model.predict(img_array, verbose=0)

            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            preds = np.asarray(preds)
            if preds.ndim == 1:
                preds = preds.reshape(1, -1)

            if preds.shape[-1] < 2:
                continue

            preds = _to_probabilities(preds)

            predicted_class = int(np.argmax(preds, axis=1)[0])
            confidence_score = float(preds[0][predicted_class])

            model_predictions.append(predicted_class)
            confidence_scores.setdefault(predicted_class, []).append(confidence_score)
        except Exception:
            continue

    if len(model_predictions) == 0:
        fallback_count += 1
        image_predictions.append({"image_id": image_id, "label": 4})
        continue

    predicted_count += 1
    class_votes = Counter(model_predictions)
    most_common = class_votes.most_common()
    final_predicted_class = most_common[0][0]

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        tied_count = most_common[0][1]
        tied_classes = [cls for cls, count in most_common if count == tied_count]
        final_predicted_class = max(
            tied_classes,
            key=lambda cls: (
                sum(confidence_scores.get(cls, [0.0]))
                / max(len(confidence_scores.get(cls, [])), 1)
            ),
        )

    image_predictions.append(
        {"image_id": image_id, "label": int(final_predicted_class)}
    )

submission_df = pd.DataFrame(image_predictions)

assert (
    submission_df.shape[0] == sample_csv.shape[0]
), "Submission row count must match sample_submission.csv"
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns must be exactly: image_id,label"
assert (
    submission_df["image_id"].tolist() == test_ids
), "Submission must follow sample_submission.csv image_id order"

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"[INFO] Wrote submission to: {submission_path}")
print(f"[INFO] predicted_count={predicted_count}, fallback_count={fallback_count}")
submission_df.head()



## === cell 7
submission_df.tail()
