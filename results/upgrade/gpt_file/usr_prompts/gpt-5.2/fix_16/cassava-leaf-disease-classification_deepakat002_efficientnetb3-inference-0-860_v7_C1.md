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

0.8652160773647628

# 6. Current score

0.45404

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the model-loading crash by switching from standalone `keras` to `tensorflow.keras` and forcing protobuf to use the pure-Python implementation, which avoids the `MessageFactory.GetPrototype` error seen in many Kaggle images. I also fix the preprocessing bug (PIL image not converted to RGB/float and missing EfficientNet-style normalization), which currently makes predictions unreliable and can change tensor shapes. Finally, I ensure predictions are produced for every test image in a deterministic order that matches `sample_submission.csv`, so `image_id` and `label` lengths always match and the submission format is valid (`submission.csv`). These changes keep the core “load pretrained model → predict → argmax → write submission” logic intact while making it run end-to-end and improving expected accuracy.'
- What this solution (achieved 0.05531) has done: 'I fix the model-loading crash by avoiding `load_model()` (which triggers the protobuf `MessageFactory.GetPrototype` issue in this environment) and instead rebuilding the same EfficientNetB3 classifier in code, then loading weights from the provided `.hdf5`. I keep the core logic identical (pretrained EfficientNetB3 → predict on test → argmax → write submission) and keep your preprocessing, only switching it to the official `efficientnet.preprocess_input` to match training normalization and improve accuracy toward the target. I also make the file/path handling more robust (fallback paths, existence checks) while keeping the same Kaggle input locations and producing a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.05531) has done: 'We fix the protobuf `MessageFactory.GetPrototype` crash that prevents TensorFlow/Keras from importing/initializing by forcing the pure-Python protobuf implementation *before* any TensorFlow import, and by clearing any already-imported `google.protobuf` modules to ensure the setting takes effect in this notebook runtime. This should make the model build + `load_weights()` run so the rest of your existing inference logic works unchanged and produces `submission.csv`. I also keep your EfficientNetB3 architecture and preprocessing intact, only adding a small, safe fallback for `preprocess_input` import to avoid version-specific import errors. No training logic or modeling approach is changed—this is purely to unblock execution and restore expected accuracy.'
- What this solution (achieved 0.05531) has done: 'We fix the root-cause crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* any TensorFlow import and restarting protobuf-related modules so the setting actually takes effect in this runtime. Then we keep your core inference logic unchanged (rebuild EfficientNetB3 → load provided weights → preprocess → predict → argmax → write `submission.csv`). I also add a small compatibility fallback for the EfficientNet `preprocess_input` import and guard against missing model files so the notebook always reaches CSV creation. These changes should both unblock execution and bring the score back up toward the target by ensuring the model and preprocessing match what the weights expect.'
- What this solution (achieved 0.05531) has done: 'We fix the root cause of the `MessageFactory.GetPrototype` crash by ensuring protobuf is forced to the pure-Python implementation *before* TensorFlow is imported, and by clearing both `google.protobuf` and `tensorflow` from `sys.modules` so the setting actually takes effect in this runtime. This unblocks the TensorFlow/Keras import, model construction, and `load_weights()` so `model` is defined and downstream inference runs. We keep your core logic intact (EfficientNetB3 → load provided weights → preprocess with EfficientNet `preprocess_input` → predict → argmax → write `submission.csv`). The only other changes are small robustness guards (explicitly disable GPU to avoid slow/fragile CUDA init in Kaggle CPU images, and a clearer check for the weights path) without changing evaluation semantics.'
- What this solution (achieved 0.05531) has done: 'You’re still hitting the protobuf `MessageFactory.GetPrototype` crash during the TensorFlow import/build, so the main fix is to make the “force pure-Python protobuf” setting actually take effect before TensorFlow initializes by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` at the very start and clearing any already-imported protobuf modules. Then, to ensure the notebook always produces a valid `submission.csv`, I keep your exact model/preprocess/predict→argmax pipeline but add a safe fallback path: if TensorFlow cannot be imported in this environment, it write a deterministic baseline submission (valid format) instead of crashing (this is score-worse but guarantees “Not yielded” becomes a valid submission). No training logic, architecture, loss, or evaluation semantics are changed when TensorFlow successfully loads; the fallback only activates if the environment cannot run TF due to protobuf issues.'
- What this solution (achieved 0.05531) has done: 'We fix the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation and additionally pinning a safe protobuf Python API version before TensorFlow import, then clearing any already-loaded protobuf/tensorflow modules so the setting actually takes effect. This is a minimal, execution-unblocking change that preserves your core pipeline (rebuild EfficientNetB3 → load weights → preprocess → predict → argmax → write submission) and should restore the intended accuracy instead of falling back to all-zeros predictions. We also make `preprocess_pil` robust so it can still run (with a safe identity fallback) if TensorFlow import fails, ensuring the notebook always writes a valid `submission.csv`. No model architecture, inference logic, or submission formatting is changed.'
- What this solution (achieved 0.40396) has done: 'We fix the hard crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by avoiding TensorFlow/Keras entirely in this environment and switching inference to a lightweight, deterministic classical image classifier that uses only PIL + numpy/pandas (all available). This keeps the overall “load data → preprocess images → predict labels → write submission.csv” core pipeline intact while making it run end-to-end reliably. To improve accuracy far beyond the current near-random output, we train a simple multiclass classifier on `train.csv` using downscaled pixel features (no external ML libraries), then predict the test set in the exact `sample_submission.csv` order. The submission format and file name remain correct (`submission.csv`, columns `image_id,label`).'
- What this solution (achieved 0.19806) has done: 'Your current score (0.40396) is far below the target (0.8652), so we should improve accuracy with minimal changes while keeping your same core pipeline: downscale image → grayscale features → softmax regression trained with SGD → argmax submission. The biggest likely issue is optimization instability/underfitting from a too-large learning rate and missing a bias term in feature standardization; we keep the exact model but make training numerically steadier and more expressive by (1) adding a constant “intercept” feature (doesn’t change architecture category; still linear softmax), (2) switching to a safe learning-rate schedule (same SGD loop, just smaller steps), and (3) using class-balanced sample weights to reduce majority-class dominance (same loss, just weighted). These are small, legitimate adjustments that typically increase accuracy materially for this kind of linear classifier without changing the overall approach. The script still run end-to-end and write a valid `submission.csv` in the required format.'
- What this solution (achieved 0.53176) has done: 'Your current score (0.198) is far below the target (0.865), so we should improve accuracy with the smallest changes that don’t alter the core “downscale → linear softmax regression (SGD) → argmax submission” pipeline. The biggest win for this kind of simple model is adding a tiny amount of cheap, information-preserving features without changing the model type: we augment the grayscale pixels with a few global color statistics (mean/std per RGB channel) and a simple edge-strength statistic, then standardize them using the same training-set moments. To reduce sensitivity to class weighting hurting pure accuracy, we soften the class balancing (square-root weighting) rather than fully inverse-frequency, keeping the same weighted cross-entropy objective and SGD loop. Finally, we keep determinism and submission alignment unchanged, and still write `submission.csv` in the exact required format.'
- What this solution (achieved 0.60762) has done: 'We keep your exact “downscale → linear softmax (SGD) → argmax submission” pipeline, but make two minimal, score-relevant adjustments that typically lift accuracy without changing the model type. First, we switch training from class-weighted cross-entropy to unweighted cross-entropy because the Kaggle metric is plain accuracy (not balanced accuracy), and weighting often hurts overall accuracy by over-predicting minority classes. Second, we add a tiny amount of L2 regularization to the bias term as well (while keeping the same ridge concept) to reduce overconfident bias shifts that can hurt test calibration for a linear model; this is a negligible change to the same objective family and keeps inference identical.'
- What this solution (achieved 0.51345) has done: 'Your current score (0.60762) is far below the target (0.8652), so we should improve accuracy with the smallest changes that keep the exact same pipeline: handcrafted image features → linear softmax regression trained with SGD → argmax submission. The most likely accuracy limiter here is underfitting from too-low image resolution and overly-strong regularization, so we only (1) increase the downscale resolution moderately (more signal, same feature type) and (2) slightly reduce ridge penalties to let the same linear model fit better. We keep the same loss (unweighted softmax cross-entropy), same training loop, same prediction/argmax logic, and the same submission alignment to `sample_submission.csv`. All paths remain unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.46039) has done: 'We keep your exact pipeline (downscale image → handcrafted features → linear softmax regression trained with the same SGD loop → argmax submission), but make a couple of small, score-relevant tweaks aimed at reducing underfitting and improving stability. Specifically, we moderately increase the input resolution again (more signal with the same feature type), and slightly reduce L2 regularization so the same linear model can fit the training distribution better. We also add a tiny amount of label smoothing (still cross-entropy on the same softmax outputs) which often improves generalization for noisy multi-class labels without changing the model/loop structure. All paths, ordering (aligned to `sample_submission.csv`), and the `submission.csv` format remain unchanged.'
- What this solution (achieved 0.45404) has done: 'We keep your exact pipeline (downscale image → handcrafted features → linear softmax regression with the same SGD loop → argmax → submission) and only make small changes aimed at reducing underfitting, since your current score (0.46039) is far below the target (0.8652). Concretely, we (1) moderately increase feature resolution to preserve more leaf texture signal, (2) slightly reduce L2 regularization to let the same linear model fit better, and (3) set label smoothing to 0 because this task’s metric is plain accuracy and smoothing often lowers peak accuracy for simple linear models. Everything else (paths, ordering aligned to `sample_submission.csv`, training loop structure, prediction semantics, and `submission.csv` format) stays the same.'

# 9. Code solution

## === cell 0
import os
import sys
import importlib

import numpy as np
import pandas as pd

from PIL import Image

np.random.seed(123)



## === cell 1
BASE_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{BASE_DIR}/train.csv"
TRAIN_DIR = f"{BASE_DIR}/train_images/"
TEST_DIR = f"{BASE_DIR}/test_images/"
SAMPLE_SUB_PATH = f"{BASE_DIR}/sample_submission.csv"

if not os.path.exists(SAMPLE_SUB_PATH):
    alt_base = "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
    if os.path.exists(os.path.join(alt_base, "sample_submission.csv")):
        BASE_DIR = alt_base
        TRAIN_CSV = f"{BASE_DIR}/train.csv"
        TRAIN_DIR = f"{BASE_DIR}/train_images/"
        TEST_DIR = f"{BASE_DIR}/test_images/"
        SAMPLE_SUB_PATH = f"{BASE_DIR}/sample_submission.csv"

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found at {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_DIR), f"test_images dir not found at {TEST_DIR}"
assert os.path.exists(TRAIN_CSV), f"train.csv not found at {TRAIN_CSV}"
assert os.path.exists(TRAIN_DIR), f"train_images dir not found at {TRAIN_DIR}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image_id"].tolist()

train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(train_df.columns)

print("BASE_DIR:", BASE_DIR)
print("Train rows:", len(train_df), "Test rows:", len(test_images))



## === cell 2
FEAT_SIZE = 80  # was 64

NUM_CLASSES = 5

RIDGE = 5.0e-3  # was 8.0e-3
RIDGE_BIAS = 1.5e-4  # was 2.5e-4

EPOCHS = 18
LR0 = 0.12
BATCH = 512

EXTRA_FEATS = 7  # RGB mean(3) + RGB std(3) + edge_strength(1)

LABEL_SMOOTH = 0.0  # was 0.04


def img_to_feat(path: str, feat_size: int = FEAT_SIZE) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((feat_size, feat_size))
        arr = np.asarray(img, dtype=np.float32)  # (H,W,3) in [0,255]

    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    x = gray.reshape(-1) / 255.0  # [0,1]

    rgb = arr / 255.0
    rgb_mean = rgb.reshape(-1, 3).mean(axis=0)  # (3,)
    rgb_std = rgb.reshape(-1, 3).std(axis=0)  # (3,)

    dx = np.mean(np.abs(gray[:, 1:] - gray[:, :-1])) / 255.0
    dy = np.mean(np.abs(gray[1:, :] - gray[:-1, :])) / 255.0
    edge_strength = np.array([(dx + dy) * 0.5], dtype=np.float32)

    extra = np.concatenate(
        [rgb_mean.astype(np.float32), rgb_std.astype(np.float32), edge_strength], axis=0
    )
    return np.concatenate([x.astype(np.float32), extra], axis=0)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    oh[np.arange(len(y)), y.astype(int)] = 1.0
    return oh


train_paths = [os.path.join(TRAIN_DIR, fn) for fn in train_df["image_id"].tolist()]
train_labels = train_df["label"].astype(int).to_numpy()

_missing_train = [p for p in train_paths[:200] if not os.path.exists(p)]
assert not _missing_train, f"Some train images are missing (e.g. {_missing_train[:3]})"

print(
    "Feature extraction: downscale to",
    FEAT_SIZE,
    "x",
    FEAT_SIZE,
    "grayscale +",
    EXTRA_FEATS,
    "extra feats",
)



## === cell 3
X_train = np.zeros(
    (len(train_paths), FEAT_SIZE * FEAT_SIZE + EXTRA_FEATS), dtype=np.float32
)
for i, p in enumerate(train_paths):
    X_train[i] = img_to_feat(p, FEAT_SIZE)
    if (i + 1) % 4000 == 0:
        print("Processed train:", i + 1, "/", len(train_paths))

y_train = train_labels
Y_train = one_hot(y_train, NUM_CLASSES)

if LABEL_SMOOTH > 0.0:
    Y_train = (1.0 - LABEL_SMOOTH) * Y_train + (LABEL_SMOOTH / NUM_CLASSES)

mu = X_train.mean(axis=0, keepdims=True)
sd = X_train.std(axis=0, keepdims=True) + 1e-6
X_train = (X_train - mu) / sd

X_train = np.concatenate(
    [X_train, np.ones((X_train.shape[0], 1), dtype=np.float32)], axis=1
)

class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)

sample_w = np.ones((len(y_train),), dtype=np.float32)

print("Train feature matrix:", X_train.shape)
print("Class counts:", class_counts.tolist())
print("Using unweighted cross-entropy for accuracy metric alignment.")
print("Label smoothing:", LABEL_SMOOTH)



## === cell 4
D = X_train.shape[1]
W = np.zeros((D, NUM_CLASSES), dtype=np.float32)
b = np.zeros((1, NUM_CLASSES), dtype=np.float32)

n = X_train.shape[0]
indices = np.arange(n)

for epoch in range(EPOCHS):
    np.random.shuffle(indices)
    total_loss = 0.0
    correct = 0
    seen = 0

    lr = LR0 / (1.0 + 0.15 * epoch)

    for start in range(0, n, BATCH):
        batch_idx = indices[start : start + BATCH]
        Xb = X_train[batch_idx]
        Yb = Y_train[batch_idx]
        wb = sample_w[batch_idx].reshape(-1, 1)

        logits = Xb @ W + b
        P = softmax(logits)

        ce = -np.sum(wb * (Yb * np.log(P + 1e-9))) / (np.sum(wb) + 1e-9)
        loss = ce + 0.5 * RIDGE * np.sum(W * W) + 0.5 * RIDGE_BIAS * np.sum(b * b)
        total_loss += loss * len(batch_idx)

        pred = np.argmax(P, axis=1)
        true = y_train[batch_idx]
        correct += int(np.sum(pred == true))
        seen += len(batch_idx)

        dlogits = (wb * (P - Yb)) / (np.sum(wb) + 1e-9)
        dW = Xb.T @ dlogits + RIDGE * W
        db = np.sum(dlogits, axis=0, keepdims=True) + RIDGE_BIAS * b

        W -= lr * dW
        b -= lr * db

    avg_loss = total_loss / n
    acc = correct / seen
    print(
        f"Epoch {epoch+1}/{EPOCHS} - lr={lr:.4f} - loss={avg_loss:.4f} - train_acc={acc:.4f}"
    )



## === cell 5
missing_test = [
    img_id
    for img_id in test_images
    if not os.path.exists(os.path.join(TEST_DIR, img_id))
]
if missing_test:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test images, e.g. {missing_test[:5]}"
    )

X_test = np.zeros(
    (len(test_images), FEAT_SIZE * FEAT_SIZE + EXTRA_FEATS), dtype=np.float32
)
for i, image_id in enumerate(test_images):
    X_test[i] = img_to_feat(os.path.join(TEST_DIR, image_id), FEAT_SIZE)
    if (i + 1) % 600 == 0:
        print("Processed test:", i + 1, "/", len(test_images))

X_test = (X_test - mu) / sd

X_test = np.concatenate(
    [X_test, np.ones((X_test.shape[0], 1), dtype=np.float32)], axis=1
)

logits_test = X_test @ W + b
probs_test = softmax(logits_test)
preds = np.argmax(probs_test, axis=1).astype(np.int64)

sub = pd.DataFrame({"image_id": test_images, "label": preds})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with", len(sub), "rows")
print("submission.csv columns:", list(sub.columns))
print("Unique predicted labels:", np.unique(preds))
