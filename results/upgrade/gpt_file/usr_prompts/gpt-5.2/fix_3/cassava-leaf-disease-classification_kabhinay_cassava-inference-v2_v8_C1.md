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

2.7

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

0.8856149894227864

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'I fix the early TensorFlow import crash by removing the problematic TF/Keras usage in cell 1 and switching the data pipeline to use `tf.data` + `tf.io.decode_jpeg`, which is stable in Kaggle’s TF environment. I also fix the missing external SavedModel paths by loading models from the standard Kaggle dataset folder if present, and otherwise fall back to a simple, deterministic baseline (majority-class) so the notebook always produces a valid `submission.csv`. To keep the ensemble logic intact, predictions are still combined as `pred_v1 + pred_v2 + pred_v4` and then `argmax`. Finally, I ensure `image_id` ordering and output schema match `sample_submission.csv` exactly.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd


np.random.seed(42)

print("Python OK. Numpy:", np.__version__)
print("Working dir:", os.getcwd())

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise RuntimeError("Could not find cassava dataset folder in expected locations.")

TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)
print("TEST_DIR exists:", os.path.isdir(TEST_DIR), "files:", len(os.listdir(TEST_DIR)))



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

majority_label = int(train_df["label"].value_counts().idxmax())
num_classes = int(train_df["label"].nunique())

print(
    "Train rows:",
    len(train_df),
    "Num classes:",
    num_classes,
    "Majority label:",
    majority_label,
)
print("Sample submission rows:", len(sample_sub), "cols:", list(sample_sub.columns))

test_df = sample_sub[["image_id"]].copy()
assert test_df["image_id"].nunique() == len(
    test_df
), "Duplicate image_ids in sample_submission?"
print("Test rows:", len(test_df))



## === cell 2
tf = None
TF_OK = False
try:
    import tensorflow as tf  # noqa: F401

    TF_OK = True
    try:
        tf.random.set_seed(42)
    except Exception:
        pass
    print("TF version:", tf.__version__)
except Exception as e:
    print(
        "WARNING: TensorFlow import failed; will use fallback baseline. Error:", repr(e)
    )

MODEL_DIRS = {
    "v1": "../input/only-xception-with-cropping/saved-model-11-0.879",
    "v2": "../input/efficientnet-with-cropping/saved-model-06-0.88",
    "v3": "../input/gambler-s-loss-cassava/saved-model-10-0.843",
    "v4": "../input/efficientnet-with-cropping-v2/saved-model-06-0.887",
}


def _find_saved_model_dir(path):
    if path is None:
        return None
    if os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "saved_model.pb"))
        or os.path.exists(os.path.join(path, "saved_model.pbtxt"))
    ):
        return path
    return None


def _load_savedmodel(path):
    obj = tf.saved_model.load(path)
    sigs = getattr(obj, "signatures", {})
    if isinstance(sigs, dict) and len(sigs) > 0:
        if "serving_default" in sigs:
            return sigs["serving_default"]
        return sigs[list(sigs.keys())[0]]
    return obj


loaded_models = {}
if TF_OK:
    for k, p in MODEL_DIRS.items():
        p2 = _find_saved_model_dir(p)
        if p2 is None:
            print("Model", k, "not found at:", p)
            continue
        try:
            loaded_models[k] = _load_savedmodel(p2)
            print("Loaded model", k, "from", p2)
        except Exception as e:
            print("WARNING: could not load model", k, "from", p2, "error:", repr(e))

print("Loaded models keys:", sorted(list(loaded_models.keys())))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
IMG_SIZE = 448
BATCH_SIZE = 16


def make_test_dataset(image_ids):
    paths = [os.path.join(TEST_DIR, x) for x in image_ids]

    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 4
N = len(test_df)
pred_v1 = pred_v2 = pred_v4 = None


def _predict_with_model(model_fn, ds):
    outs = []
    for batch in ds:
        y = model_fn(batch)
        if isinstance(y, dict):
            key = sorted(list(y.keys()))[0]
            y = y[key]
        y = tf.convert_to_tensor(y)
        outs.append(y)
    return tf.concat(outs, axis=0).numpy()


if TF_OK and (
    ("v1" in loaded_models) or ("v2" in loaded_models) or ("v4" in loaded_models)
):
    test_ds = make_test_dataset(test_df["image_id"].tolist())

    if "v1" in loaded_models:
        pred_v1 = _predict_with_model(loaded_models["v1"], test_ds)
        print("pred_v1 shape:", pred_v1.shape)
    if "v2" in loaded_models:
        pred_v2 = _predict_with_model(loaded_models["v2"], test_ds)
        print("pred_v2 shape:", pred_v2.shape)
    if "v4" in loaded_models:
        pred_v4 = _predict_with_model(loaded_models["v4"], test_ds)
        print("pred_v4 shape:", pred_v4.shape)


def _baseline_preds(n, num_classes, majority_label):
    p = np.full((n, num_classes), 1e-6, dtype=np.float32)
    p[:, majority_label] = 1.0
    return p


if pred_v1 is None:
    pred_v1 = _baseline_preds(N, num_classes, majority_label)
if pred_v2 is None:
    pred_v2 = _baseline_preds(N, num_classes, majority_label)
if pred_v4 is None:
    pred_v4 = _baseline_preds(N, num_classes, majority_label)

assert (
    pred_v1.shape[0] == N and pred_v2.shape[0] == N and pred_v4.shape[0] == N
), "Prediction rows mismatch."
assert (
    pred_v1.shape[1] == num_classes
    and pred_v2.shape[1] == num_classes
    and pred_v4.shape[1] == num_classes
), "Prediction class dim mismatch."

print("Final pred shapes:", pred_v1.shape, pred_v2.shape, pred_v4.shape)



## === cell 5
pred_new = pred_v1 + pred_v2 + pred_v4
predicted_class_indices_new = np.argmax(pred_new, axis=1).astype(int)

results_new = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices_new}
)

assert list(results_new.columns) == ["image_id", "label"]
assert len(results_new) == len(sample_sub)
assert (
    results_new["image_id"].tolist() == sample_sub["image_id"].tolist()
), "image_id order mismatch vs sample_submission."

out_path = "/kaggle/working/submission.csv"
results_new.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(results_new.head())
print("Rows:", len(results_new), "Cols:", list(results_new.columns))
print("Label distribution:", results_new["label"].value_counts().to_dict())
