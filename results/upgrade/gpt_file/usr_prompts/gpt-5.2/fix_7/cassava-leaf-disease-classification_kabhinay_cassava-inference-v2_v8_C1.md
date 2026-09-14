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

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the early TensorFlow import crash by removing the problematic TF/Keras usage in cell 1 and switching the data pipeline to use `tf.data` + `tf.io.decode_jpeg`, which is stable in Kaggle’s TF environment. I also fix the missing external SavedModel paths by loading models from the standard Kaggle dataset folder if present, and otherwise fall back to a simple, deterministic baseline (majority-class) so the notebook always produces a valid `submission.csv`. To keep the ensemble logic intact, predictions are still combined as `pred_v1 + pred_v2 + pred_v4` and then `argmax`. Finally, I ensure `image_id` ordering and output schema match `sample_submission.csv` exactly.'
- What this solution (achieved 0.61099) has done: 'I fix the TensorFlow/protobuf crash that prevents model loading by forcing the pure-Python protobuf implementation (a common Kaggle/TF2+protobuf mismatch) before importing TensorFlow, and by falling back cleanly to the baseline if TF still can’t import. I also add a small guard so downstream cells don’t reference `tf` when TensorFlow isn’t available, ensuring the notebook always runs end-to-end. These changes keep your ensemble logic (`pred_v1 + pred_v2 + pred_v4` then `argmax`) intact while enabling the SavedModel inference that should move accuracy up toward the target. Finally, I keep submission formatting and ordering exactly matched to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'The crash happens before inference because TensorFlow’s protobuf bindings are incompatible with the environment, triggering `MessageFactory.GetPrototype` during TF import/model loading. To keep your ensemble logic unchanged while moving the score toward the target, I avoid TensorFlow entirely and instead run deterministic inference using OpenCV to load/resize images and ONNX Runtime to execute the same three pretrained models (v1/v2/v4) as ONNX graphs if present in the Kaggle input folders; otherwise it cleanly falls back to your majority-class baseline. I also add a small, safe model-file discovery layer so it finds the ONNX files regardless of their exact filename, and preserve exact submission ordering/format to match `sample_submission.csv`. This should restore real model predictions (instead of baseline) and materially improve accuracy toward your target.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.8856), so we should improve accuracy by making the smallest changes that restore “real” model inference instead of the majority-class fallback. The main issue is that your `MODEL_DIRS` point to Kaggle datasets that likely aren’t attached, so no ONNX models are found and you silently fall back to baseline. I keep your ensemble logic (`pred_v1 + pred_v2 + pred_v4` then `argmax`) identical, but make model discovery search under `/kaggle/input/*` for any `.onnx` files that look like v1/v2/v4, and add a robust fallback to TensorFlow SavedModel loading if TensorFlow is importable (still without changing the model architecture/training). I also make preprocessing slightly more compatible with common EfficientNet/Xception exports by handling dynamic input sizes and both NCHW/NHWC input layouts, while keeping your 448 default to avoid core-logic changes.'
- What this solution (achieved 0.61099) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by preventing TensorFlow from being imported in this environment (it’s Python 2.7 and TF2/protobuf not be compatible), so the pipeline can proceed using ONNXRuntime when available and otherwise the existing baseline fallback. We also guard all TF-specific code paths behind `TF_OK` without ever triggering a TF import attempt, so cell 2 cannot fail before ONNX discovery/loading. This is a correctness/stability fix (ensures end-to-end execution and a valid `submission.csv`) and also increases score toward the target when ONNX models are present because it restores real model inference instead of crashing/falling back early. Core ensemble logic (`pred_v1 + pred_v2 + pred_v4` then `argmax`) and preprocessing approach remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
import glob
import re

ORT_OK = False
CV2_OK = False
TF_OK = False  # Fixed: do not attempt TF import in this environment

ort = None
cv2 = None
tf = None

try:
    import onnxruntime as ort  # noqa: F401

    ORT_OK = True
    print("onnxruntime version:", ort.__version__)
except Exception as e:
    print(
        "WARNING: onnxruntime import failed; will fallback baseline. Error:",
        repr(e),
    )

try:
    import cv2  # noqa: F401

    CV2_OK = True
    print("cv2 version:", cv2.__version__)
except Exception as e:
    print(
        "WARNING: cv2 import failed; will fallback baseline (or TF if enabled). Error:",
        repr(e),
    )

print(
    "TensorFlow import skipped (TF_OK=False) to avoid protobuf crash in this environment."
)

MODEL_DIRS = {
    "v1": "../input/only-xception-with-cropping/saved-model-11-0.879",
    "v2": "../input/efficientnet-with-cropping/saved-model-06-0.88",
    "v3": "../input/gambler-s-loss-cassava/saved-model-10-0.843",
    "v4": "../input/efficientnet-with-cropping-v2/saved-model-06-0.887",
}


def _list_kaggle_input_dirs():
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return []
    out = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            out.append(p)
    return sorted(out)


def _find_any_onnx_file(base_path):
    """
    Finds an .onnx file either:
    - directly at base_path if base_path is a file
    - anywhere under base_path if base_path is a directory
    Returns a single best candidate or None.
    """
    if base_path is None:
        return None
    if os.path.isfile(base_path) and base_path.lower().endswith(".onnx"):
        return base_path
    if not os.path.isdir(base_path):
        return None

    candidates = []
    candidates.extend(glob.glob(os.path.join(base_path, "*.onnx")))
    candidates.extend(
        glob.glob(os.path.join(base_path, "**", "*.onnx"), recursive=True)
    )

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    if not uniq:
        return None

    preferred = []
    for c in uniq:
        name = os.path.basename(c).lower()
        if any(k in name for k in ["model", "saved", "export", "onnx"]):
            preferred.append(c)
    return preferred[0] if preferred else uniq[0]


def _score_candidate(path, key):
    """
    Prefer files whose path/name suggest the right backbone/version.
    """
    s = (path or "").lower()
    score = 0
    if s.endswith(".onnx"):
        score += 1
    if "best" in s:
        score += 1
    if "fold" in s:
        score += 1

    if key == "v1":
        if "xception" in s:
            score += 5
        if re.search(r"\bv1\b", s):
            score += 2
    if key == "v2":
        if "efficientnet" in s:
            score += 5
        if "b3" in s or "b4" in s or "b5" in s:
            score += 1
        if re.search(r"\bv2\b", s):
            score += 2
    if key == "v4":
        if "efficientnet" in s:
            score += 5
        if "v2" in s:
            score += 3
        if re.search(r"\bv4\b", s):
            score += 2
    return score


def _discover_onnx_for_key(key):
    for p in [MODEL_DIRS.get(key), os.path.dirname(MODEL_DIRS.get(key, ""))]:
        onnx_path = _find_any_onnx_file(p)
        if onnx_path is not None:
            return onnx_path

    best = None
    best_score = -1
    for d in _list_kaggle_input_dirs():
        candidates = glob.glob(os.path.join(d, "**", "*.onnx"), recursive=True)
        for c in candidates:
            sc = _score_candidate(c, key)
            if sc > best_score:
                best_score = sc
                best = c
    return best


def _make_ort_session(onnx_path):
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)


loaded_ort = {}
if ORT_OK:
    for k in ["v1", "v2", "v4"]:
        onnx_path = _discover_onnx_for_key(k)
        if onnx_path is None:
            print(
                "ONNX model",
                k,
                "not found (searched configured paths + /kaggle/input).",
            )
            continue
        try:
            loaded_ort[k] = _make_ort_session(onnx_path)
            print("Loaded ONNX model", k, "from", onnx_path)
        except Exception as e:
            print(
                "WARNING: could not load ONNX model",
                k,
                "from",
                onnx_path,
                "error:",
                repr(e),
            )

print("Loaded ONNX models keys:", sorted(list(loaded_ort.keys())))

tf_models = {}
print("Loaded TF SavedModels keys:", sorted(list(tf_models.keys())))




## === cell 3
IMG_SIZE = 448
BATCH_SIZE = 16


def _baseline_preds(n, num_classes, majority_label):
    p = np.full((n, num_classes), 1e-6, dtype=np.float32)
    p[:, majority_label] = 1.0
    return p


def _softmax(x, axis=1):
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _preprocess_bgr_to_float01_rgb(img_bgr, out_h, out_w):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    x = img_rgb.astype(np.float32) / 255.0
    return x


def _infer_target_hw_from_meta_shape(shape, default_size):
    try:
        if shape is None or not isinstance(shape, (list, tuple)) or len(shape) != 4:
            return default_size, default_size
        if shape[1] == 3 and shape[2] not in (None, "None"):
            h = int(shape[2])
            w = int(shape[3])
            if h > 0 and w > 0:
                return h, w
        if shape[3] == 3 and shape[1] not in (None, "None"):
            h = int(shape[1])
            w = int(shape[2])
            if h > 0 and w > 0:
                return h, w
    except Exception:
        pass
    return default_size, default_size


def _predict_onnx_session(sess, image_ids, default_img_size, batch_size):
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    in_shape = input_meta.shape

    out_h, out_w = _infer_target_hw_from_meta_shape(in_shape, default_img_size)

    outs = []
    n = len(image_ids)

    expect_nchw = False
    if isinstance(in_shape, (list, tuple)) and len(in_shape) == 4:
        if in_shape[1] == 3:
            expect_nchw = True

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_ids = image_ids[start:end]

        batch = np.zeros((len(batch_ids), out_h, out_w, 3), dtype=np.float32)
        for i, img_id in enumerate(batch_ids):
            path = os.path.join(TEST_DIR, img_id)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Failed to read image: %s" % path)
            batch[i] = _preprocess_bgr_to_float01_rgb(img, out_h, out_w)

        feed = batch
        if expect_nchw:
            feed = np.transpose(batch, (0, 3, 1, 2)).astype(np.float32)

        y = sess.run(None, {input_name: feed})
        y = np.asarray(y[0], dtype=np.float32)

        row_sum = float(np.mean(np.sum(y, axis=1)))
        if not (
            np.min(y) >= -1e-4 and np.max(y) <= 1.0 + 1e-4 and abs(row_sum - 1.0) < 1e-2
        ):
            y = _softmax(y, axis=1)

        outs.append(y)

    return np.vstack(outs)


def _tf_get_infer_fn(loaded_obj):
    return None


def _predict_tf_savedmodel(loaded_obj, image_ids, default_img_size, batch_size):
    raise RuntimeError("TensorFlow inference disabled in this environment.")


N = len(test_df)
pred_v1 = pred_v2 = pred_v4 = None
image_ids = test_df["image_id"].tolist()

if ORT_OK and CV2_OK and ("v1" in loaded_ort):
    pred_v1 = _predict_onnx_session(loaded_ort["v1"], image_ids, IMG_SIZE, BATCH_SIZE)
    print("pred_v1 (onnx) shape:", pred_v1.shape)
elif TF_OK and ("v1" in tf_models):
    pred_v1 = _predict_tf_savedmodel(tf_models["v1"], image_ids, IMG_SIZE, BATCH_SIZE)
    print("pred_v1 (tf) shape:", pred_v1.shape)

if ORT_OK and CV2_OK and ("v2" in loaded_ort):
    pred_v2 = _predict_onnx_session(loaded_ort["v2"], image_ids, IMG_SIZE, BATCH_SIZE)
    print("pred_v2 (onnx) shape:", pred_v2.shape)
elif TF_OK and ("v2" in tf_models):
    pred_v2 = _predict_tf_savedmodel(tf_models["v2"], image_ids, IMG_SIZE, BATCH_SIZE)
    print("pred_v2 (tf) shape:", pred_v2.shape)

if ORT_OK and CV2_OK and ("v4" in loaded_ort):
    pred_v4 = _predict_onnx_session(loaded_ort["v4"], image_ids, IMG_SIZE, BATCH_SIZE)
    print("pred_v4 (onnx) shape:", pred_v4.shape)
elif TF_OK and ("v4" in tf_models):
    pred_v4 = _predict_tf_savedmodel(tf_models["v4"], image_ids, IMG_SIZE, BATCH_SIZE)
    print("pred_v4 (tf) shape:", pred_v4.shape)

if pred_v1 is None:
    pred_v1 = _baseline_preds(N, num_classes, majority_label)
    print("pred_v1 fallback: baseline")
if pred_v2 is None:
    pred_v2 = _baseline_preds(N, num_classes, majority_label)
    print("pred_v2 fallback: baseline")
if pred_v4 is None:
    pred_v4 = _baseline_preds(N, num_classes, majority_label)
    print("pred_v4 fallback: baseline")

assert (
    pred_v1.shape[0] == N and pred_v2.shape[0] == N and pred_v4.shape[0] == N
), "Prediction rows mismatch."
assert (
    pred_v1.shape[1] == num_classes
    and pred_v2.shape[1] == num_classes
    and pred_v4.shape[1] == num_classes
), "Prediction class dim mismatch."

print("Final pred shapes:", pred_v1.shape, pred_v2.shape, pred_v4.shape)




## === cell 4
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
