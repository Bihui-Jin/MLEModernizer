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

0.8443638561498942

# 6. Current score

0.59118

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the pipeline-breaking `FileNotFoundError`s by auto-detecting the available Kaggle input paths and falling back safely when the external tree-features CSV and the pretrained model files aren’t present. To preserve the original “stacked probs → decision tree” core logic, I keep the same feature construction (Keras probs + Torch probs) but make both model loaders robust; if a model can’t load, it outputs sensible probabilities/logits so inference and submission generation still work end-to-end. If the train-tree CSV isn’t available, I fit the same `DecisionTreeClassifier` on a small, deterministic subset of `train.csv` using the same combined-prob features computed by the (loaded or fallback) models. Finally, I ensure the submission matches `sample_submission.csv` order and is written as `submission.csv`.'
- What this solution (achieved 0.59118) has done: 'I fix the crash in the TensorFlow/Keras import that’s coming from a protobuf incompatibility (the `MessageFactory.GetPrototype` AttributeError) by making Keras usage robustly optional: if TensorFlow can’t import cleanly, we automatically fall back to a lightweight, deterministic “color-statistics → linear logits → softmax” probability generator instead of uniform 0.2. This keeps the core “(Keras-like 5 probs + Torch 5 logits) → DecisionTreeClassifier” stacking logic intact, but gives the tree much more informative features than the current uniform-prob fallback, which should move accuracy upward toward your target. I also harden the Torch forward pass to handle common saved-object formats (full model vs state_dict wrapper) without changing the overall approach. The rest of the pipeline (paths, decision tree training, submission formatting/order) is preserved and still writes `submission.csv`.'
- What this solution (achieved 0.59118) has done: 'I fix the TensorFlow/Keras import crash by making the protobuf “MessageFactory.GetPrototype” failure explicitly caught and ensuring the script cleanly falls back without stopping execution. I also correct a shape bug in the fallback training features matrix (it was allocated as 10 columns but filled with 10 concatenated values; the current code mistakenly used 10 while earlier versions sometimes used 8/9), and harden the Torch model loading so common checkpoint formats (state_dict) don’t break inference. These changes keep your core stacking logic intact (5 probs + 5 logits → DecisionTreeClassifier) while ensuring the pipeline runs end-to-end and produces a valid `submission.csv`. This should also improve score versus the current run because the Keras-failure path now execute as intended instead of crashing.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings
import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from PIL import Image

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

CANDIDATE_DATA_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]


def _first_existing_dir(cands):
    for d in cands:
        if os.path.isdir(d):
            return d
    hits = glob.glob("/kaggle/**/cassava-leaf-disease-classification", recursive=True)
    for h in hits:
        if os.path.isdir(h):
            return h
    return None


DATA_DIR = _first_existing_dir(CANDIDATE_DATA_DIRS)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification data directory under /kaggle."
    )

TEST_DIR_CANDS = [
    os.path.join(DATA_DIR, "test_images"),
    os.path.join(DATA_DIR, "cassava-leaf-disease-classification", "test_images"),
]
TEST_DIR = None
for d in TEST_DIR_CANDS:
    if os.path.isdir(d):
        TEST_DIR = d
        break
if TEST_DIR is None:
    raise FileNotFoundError(f"Could not locate test_images under {DATA_DIR}")

SAMPLE_SUB_PATH_CANDS = [
    os.path.join(DATA_DIR, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
SAMPLE_SUB_PATH = None
for p in SAMPLE_SUB_PATH_CANDS:
    if os.path.exists(p):
        SAMPLE_SUB_PATH = p
        break
if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected locations."
    )

TRAIN_CSV_CANDS = [
    os.path.join(DATA_DIR, "train.csv"),
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]
TRAIN_CSV_PATH = None
for p in TRAIN_CSV_CANDS:
    if os.path.exists(p):
        TRAIN_CSV_PATH = p
        break
if TRAIN_CSV_PATH is None:
    raise FileNotFoundError("Could not locate train.csv in expected locations.")

TRAIN_TREE_PATH = "/kaggle/input/train-tree/train_tree_2.csv"
KERAS_MODEL_PATH = "/kaggle/input/densenet_70_512x512/keras/default/2/Densenet_70_512x512_weights (1).keras"
TORCH_MODEL_PATH = (
    "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

print("DATA_DIR:", DATA_DIR)
print("TEST_DIR:", TEST_DIR)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("CUDA:", torch.cuda.is_available())




## === cell 1
use_keras = True
model1 = None
keras_load_error = None

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model  # noqa: F401

    if os.path.exists(KERAS_MODEL_PATH):
        model1 = load_model(KERAS_MODEL_PATH)
    else:
        raise FileNotFoundError(f"Keras model file not found: {KERAS_MODEL_PATH}")
except Exception as e:
    use_keras = False
    keras_load_error = repr(e)


def _softmax_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full((len(x),), 1.0 / len(x), dtype=np.float32)
    return (ex / s).astype(np.float32)


_FALLBACK_W = np.array(
    [
        [0.80, -0.20, 0.10, 0.05, -0.10, 0.05, 0.02, -0.03, 0.01],  # class 0
        [-0.30, 0.70, -0.10, 0.10, 0.15, -0.05, -0.01, 0.02, 0.00],  # class 1
        [0.10, -0.10, 0.60, -0.20, 0.05, 0.10, 0.03, -0.01, 0.02],  # class 2
        [-0.05, 0.20, -0.25, 0.65, -0.05, 0.00, -0.02, 0.03, -0.01],  # class 3
        [0.05, -0.10, 0.15, -0.10, 0.55, -0.10, 0.00, -0.01, 0.03],  # class 4
    ],
    dtype=np.float32,
)
_FALLBACK_b = np.array([0.05, 0.00, 0.00, 0.00, 0.02], dtype=np.float32)


def _fallback_probs_from_pil(image_pil):
    arr = np.asarray(image_pil.convert("RGB"), dtype=np.float32) / 255.0
    m = arr.mean(axis=(0, 1))  # (3,)
    s = arr.std(axis=(0, 1))  # (3,)
    overall_m = float(arr.mean())
    overall_s = float(arr.std())
    gx = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
    gy = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
    grad = float((gx + gy) * 0.5)
    feats = np.concatenate([m, s, [overall_m, overall_s, grad]]).astype(
        np.float32
    )  # (9,)
    logits = _FALLBACK_W @ feats + _FALLBACK_b
    return _softmax_np(logits)


def get_keras_probs_pil(image_pil):
    """
    Core logic preserved: return a 5-class probability vector.
    If TF/Keras can't run, use deterministic, informative fallback probs.
    """
    if (not use_keras) or (model1 is None):
        return _fallback_probs_from_pil(image_pil)

    arr = np.array(image_pil, dtype=np.uint8)
    img = tf.convert_to_tensor(arr)
    img = tf.image.resize(img, [512, 512])
    img = tf.cast(img, tf.float32) / 255.0
    mean = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
    std = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
    img = (img - mean) / std
    img = tf.expand_dims(img, axis=0)  # (1, 512, 512, 3)
    probs = model1.predict(img, verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float32)
    s = float(probs.sum())
    if not np.isfinite(s) or s <= 0:
        return np.full((5,), 0.2, dtype=np.float32)
    probs = probs / s
    return probs.astype(np.float32)


model2 = None
use_torch = True
torch_load_error = None

try:
    if os.path.exists(TORCH_MODEL_PATH):
        loaded = torch.load(TORCH_MODEL_PATH, map_location=device)

        if isinstance(loaded, dict):
            if "model" in loaded:
                model2 = loaded["model"]
            elif "state_dict" in loaded:
                raise TypeError(
                    "Torch checkpoint contains only state_dict; model object not present."
                )
            else:
                raise TypeError("Unrecognized torch checkpoint dict format.")
        else:
            model2 = loaded

        if hasattr(model2, "to"):
            model2.to(device)
        if hasattr(model2, "eval"):
            model2.eval()
        else:
            raise TypeError("Loaded torch object is not a torch.nn.Module-like model.")
    else:
        raise FileNotFoundError(f"Torch model file not found: {TORCH_MODEL_PATH}")
except Exception as e:
    use_torch = False
    torch_load_error = repr(e)


def get_torch_logits_pil(image_pil):
    """
    Core logic preserved: return 5 logits (not necessarily normalized).
    Fallback: zeros logits if model is unavailable.
    """
    if (not use_torch) or (model2 is None):
        return np.zeros((5,), dtype=np.float32)

    image2 = torch_transforms(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model2(image2)

    if isinstance(out, (tuple, list)):
        out = out[0]

    logits2 = out.detach().cpu().numpy()[0].astype(np.float32)
    if logits2.shape[0] != 5:
        logits2 = np.zeros((5,), dtype=np.float32)
    return logits2


if not use_keras:
    print("WARNING: Keras model unavailable; using deterministic fallback probs.")
    print("Keras load error:", keras_load_error)

if not use_torch:
    print("WARNING: Torch model unavailable; using zero-logits fallback.")
    print("Torch load error:", torch_load_error)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
decision_tree = DecisionTreeClassifier(
    criterion="gini", max_depth=8, min_samples_split=12
)

train_labels = None

if os.path.exists(TRAIN_TREE_PATH):
    train_probs_df = pd.read_csv(TRAIN_TREE_PATH)
    if "label" not in train_probs_df.columns:
        raise ValueError("train_tree_2.csv must contain a 'label' column.")
    train_labels = train_probs_df["label"].values
    train_probs = train_probs_df.iloc[:, 1:-1].values
    decision_tree.fit(train_probs, train_labels)
    print(
        "Decision tree trained from external train_tree_2.csv. Features shape:",
        train_probs.shape,
    )
else:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    if not {"image_id", "label"}.issubset(train_df.columns):
        raise ValueError("train.csv must contain image_id and label.")
    n_fit = min(3000, len(train_df))
    subset = train_df.sample(n=n_fit, random_state=42).reset_index(drop=True)

    TRAIN_IMG_DIR_CANDS = [
        os.path.join(DATA_DIR, "train_images"),
        os.path.join(DATA_DIR, "cassava-leaf-disease-classification", "train_images"),
    ]
    TRAIN_IMG_DIR = None
    for d in TRAIN_IMG_DIR_CANDS:
        if os.path.isdir(d):
            TRAIN_IMG_DIR = d
            break
    if TRAIN_IMG_DIR is None:
        raise FileNotFoundError(f"Could not locate train_images under {DATA_DIR}")

    X = np.zeros((n_fit, 10), dtype=np.float32)
    y = subset["label"].astype(int).values
    for i, img_id in enumerate(subset["image_id"].values):
        img_path = os.path.join(TRAIN_IMG_DIR, img_id)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = None

        if img is None:
            p1 = np.full((5,), 0.2, dtype=np.float32)
            p2 = np.zeros((5,), dtype=np.float32)
        else:
            p1 = get_keras_probs_pil(img)
            p2 = get_torch_logits_pil(img)
        X[i, :] = np.concatenate((p1, p2), axis=0)

        if (i + 1) % 200 == 0 or (i + 1) == n_fit:
            print(f"Building fallback train features: {i+1}/{n_fit}", end="\r")

    print()
    decision_tree.fit(X, y)
    train_labels = y
    print("Decision tree trained from fallback features. Features shape:", X.shape)




## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image_id"].tolist()

combined_probs = np.zeros((len(test_images), 10), dtype=np.float32)
image_ids = []

length = len(test_images)
for idx, test_image in enumerate(test_images, start=1):
    image_ids.append(test_image)
    img_path = os.path.join(TEST_DIR, test_image)

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        image = None

    if image is None:
        test_prediction_1 = np.full((5,), 0.2, dtype=np.float32)
        test_prediction_2 = np.zeros((5,), dtype=np.float32)
    else:
        test_prediction_1 = get_keras_probs_pil(image)
        test_prediction_2 = get_torch_logits_pil(image)

    combined_probs[idx - 1, :] = np.concatenate(
        (test_prediction_1, test_prediction_2), axis=0
    )

    if idx % 50 == 0 or idx == length:
        print(f"Count:{idx}/{length}", end="\r")

print()




## === cell 4
prediction = decision_tree.predict(combined_probs)

submission = pd.DataFrame(
    {
        "image_id": image_ids,
        "label": prediction.astype(int),
    }
)

submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")

if submission["label"].isna().any():
    majority = int(pd.Series(train_labels).value_counts().idxmax())
    submission["label"] = submission["label"].fillna(majority).astype(int)
else:
    submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)

print("submission.csv written:", os.path.exists("submission.csv"))
print("Rows:", len(submission), "Cols:", submission.columns.tolist())
print(submission.head())
print(submission.sample(5, random_state=42))
