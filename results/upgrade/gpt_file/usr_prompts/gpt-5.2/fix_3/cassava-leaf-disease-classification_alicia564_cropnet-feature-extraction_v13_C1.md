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

0.9082804472650348

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it runs end-to-end and produces `submission.csv` by (1) removing the protobuf-triggering imports/usage that crash at startup, and (2) fixing the invalid `MODEL_PATH` by automatically locating a real SavedModel directory under the provided Kaggle input (or falling back to a simple baseline if the model truly isn’t available). I also ensure `infer` is always defined so prediction doesn’t fail with `NameError`. These changes keep your core inference logic (SavedModel signature → argmax class) intact while making the notebook robust to the current environment and filesystem layout. Finally, the script always write a properly formatted CSV with `image_id,label` to `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUB_PATH = "/kaggle/working/submission.csv"

MODEL_PATH = "/kaggle/input/cropnet_from_kaggle/tensorflow2/default/1/kaggle/working/cropnet_model_tf"

IMG_SIZE = (224, 224)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

tf.random.set_seed(42)
np.random.seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_and_preprocess_image(img_path: str) -> tf.Tensor:
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
    return img


def _extract_logits_or_probs(pred):
    """
    pred: output of infer(...) which may be a dict of tensors or a tensor.
    Returns: a 2D numpy array shape (batch, num_classes).
    """
    if isinstance(pred, dict):
        keys = list(pred.keys())
        preferred = None
        for k in keys:
            lk = k.lower()
            if "logit" in lk or "prob" in lk or "pred" in lk or "output" in lk:
                preferred = k
                break
        if preferred is None:
            preferred = sorted(keys)[0]
        t = pred[preferred]
    else:
        t = pred

    t = tf.convert_to_tensor(t)
    arr = t.numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def find_savedmodel_dir(start_path: str) -> str | None:
    """
    Returns a directory containing saved_model.pb (or saved_model.pbtxt), searching:
    - start_path itself
    - its parents (a few levels up)
    - the whole /kaggle/input tree as a last resort (bounded search)
    """

    def is_sm_dir(d):
        return (
            os.path.isdir(d)
            and (
                os.path.exists(os.path.join(d, "saved_model.pb"))
                or os.path.exists(os.path.join(d, "saved_model.pbtxt"))
            )
            and os.path.isdir(
                os.path.join(d, "variables")
            )  # typical SavedModel structure
        )

    p = start_path
    for _ in range(6):
        if is_sm_dir(p):
            return p
        parent = os.path.dirname(p.rstrip("/"))
        if parent == p:
            break
        p = parent

    candidates = []
    roots = []
    if start_path.startswith("/kaggle/input/"):
        parts = start_path.split("/")
        if len(parts) >= 4:
            roots.append("/".join(parts[:4]))  # /kaggle/input/<dataset>
    roots.append("/kaggle/input")

    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
                if is_sm_dir(dirpath):
                    candidates.append(dirpath)
                    if len(candidates) >= 5:
                        break
            rel_depth = dirpath[len(root) :].count(os.sep)
            if rel_depth > 8:
                dirnames[:] = []
            if len(candidates) >= 5:
                break
        if candidates:
            break

    if not candidates:
        return None

    def score_path(d):
        ld = d.lower()
        s = 0
        if "cropnet" in ld:
            s += 5
        if "cassava" in ld:
            s += 3
        if "model" in ld:
            s += 1
        return s

    candidates.sort(key=score_path, reverse=True)
    return candidates[0]




## === cell 2
infer = None
loaded = None

resolved_model_dir = find_savedmodel_dir(MODEL_PATH)
if resolved_model_dir is not None:
    print("Resolved SavedModel directory:", resolved_model_dir)
    loaded = tf.saved_model.load(resolved_model_dir)

    if (
        hasattr(loaded, "signatures")
        and isinstance(loaded.signatures, dict)
        and len(loaded.signatures) > 0
    ):
        if "serving_default" in loaded.signatures:
            infer = loaded.signatures["serving_default"]
        else:
            infer = next(iter(loaded.signatures.values()))
    else:
        infer = loaded
else:
    print(
        "Warning: Could not find a SavedModel to load under MODEL_PATH or /kaggle/input."
    )
    print("Falling back to a deterministic baseline (always predicts class 0).")

    def infer(x):
        b = tf.shape(x)[0]
        out = tf.concat(
            [tf.ones([b, 1], tf.float32), tf.zeros([b, 4], tf.float32)], axis=1
        )
        return out


OUTPUT_KEYS = None
try:
    dummy = tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.float32)
    out = infer(dummy)
    if isinstance(out, dict):
        OUTPUT_KEYS = list(out.keys())
except Exception:
    OUTPUT_KEYS = None

print("Using inference callable:", type(infer))
print("Detected output keys:", OUTPUT_KEYS)


def predict_one_image(img_path: str) -> int:
    x = load_and_preprocess_image(img_path)
    pred = infer(x)
    arr = _extract_logits_or_probs(pred)
    return int(np.argmax(arr, axis=1)[0])




## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in sample_sub.columns:
    raise ValueError("sample_submission.csv missing required column: image_id")

pred_labels = []
missing = 0

image_ids = sample_sub["image_id"].astype(str).tolist()
for image_id in image_ids:
    img_path = os.path.join(TEST_IMG_DIR, image_id)
    if not os.path.exists(img_path):
        pred_labels.append(0)
        missing += 1
        continue
    pred_labels.append(predict_one_image(img_path))

if missing:
    print(
        f"Warning: {missing} images listed in sample_submission.csv were not found on disk."
    )

submission_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
submission_df.to_csv(SUB_PATH, index=False)

print("Submission file created:", SUB_PATH)
print(submission_df.head())
print("Rows:", len(submission_df), "Columns:", list(submission_df.columns))
print("Label value counts:\n", submission_df["label"].value_counts().sort_index())
