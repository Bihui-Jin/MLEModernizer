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

0.8830462375339981

# 6. Current score

0.42564

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I remove the protobuf environment override that’s causing TensorFlow to crash under Python 3.13, and I add a safe fallback model so the notebook can always run end-to-end even if no external `.h5` assets are available. I also fix the empty-ensemble edge case that caused `most_common` to be empty (leading to an IndexError) by ensuring we always have at least one loaded model and by adding a deterministic fallback label. Finally, I make `display()` safe outside notebooks and ensure the submission is written as `/kaggle/working/submission.csv` with the exact required columns.'
- What this solution (achieved 0.61099) has done: 'The crash happens before your code runs because TensorFlow’s import path triggers a protobuf `MessageFactory.GetPrototype` attribute error under this Python 3.13 environment. The minimal, score-improving fix is to avoid importing TensorFlow at all (and thus avoid protobuf), and instead run a deterministic, legitimate baseline: predict the most frequent training label for every test image (this is a standard majority-class baseline and should substantially improve accuracy over the current near-random output). This keeps the pipeline end-to-end, uses only the provided competition files, and writes a correctly formatted `/kaggle/working/submission.csv`. I’m also keeping your model-discovery/loading code present but safely gated so it won’t execute when TensorFlow isn’t available.'
- What this solution (achieved 0.43759) has done: 'Your current 0.61099 is a majority-class baseline; to move toward the 0.883 target without changing the “no-TensorFlow” constraint, the smallest legitimate improvement is to replace the constant prediction with a lightweight classical image pipeline that doesn’t require external packages. I keep your end-to-end flow and submission writing identical, but add a simple per-image feature extractor (grayscale downsample + edge magnitude) and train a fast multinomial logistic regression implemented in NumPy on the provided `train_images/`. To keep runtime within limits, the training uses a fixed number of randomly sampled training images (seeded for determinism) and predicts all test images. This should materially increase accuracy versus majority-class while staying far from a full deep learning rewrite.'
- What this solution (achieved 0.42564) has done: 'To move your score up toward the 0.883 target without changing the “no-TensorFlow” core approach, I make two minimal, legitimate improvements to the existing NumPy logistic-regression pipeline: (1) use more training samples (still bounded for runtime) and (2) add a tiny amount of deterministic data augmentation (horizontal flip) by duplicating features and labels, which strengthens generalization without changing the model, loss, or training loop structure. I also slightly increase feature resolution (64→72) to capture more detail while keeping the same feature-extraction logic. These changes are small but typically give a meaningful accuracy bump over the current 0.43759 and should move you closer to the target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np

from collections import Counter
from PIL import Image

TF_AVAILABLE = False
tf = None
load_model = None

print("TF_AVAILABLE:", TF_AVAILABLE)



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
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
sample = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"



## === cell 2
sample_csv = pd.read_csv(sample)
print(sample_csv.head())
print("sample_csv shape:", sample_csv.shape)

train_df = pd.read_csv(train_csv_path)
print(train_df.head())
print("train_df shape:", train_df.shape)

majority_label = int(train_df["label"].value_counts().idxmax())
label_dist = train_df["label"].value_counts().sort_index()
print("Train label distribution:\n", label_dist.to_string())
print("Majority label:", majority_label)




## === cell 3
def discover_models_under_kaggle_input(root="/kaggle/input", max_models=2):
    h5_paths = []
    savedmodel_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".h5", ".hdf5")):
                h5_paths.append(os.path.join(dirpath, fn))
        if "saved_model.pb" in filenames:
            savedmodel_dirs.append(dirpath)

    h5_paths = sorted(h5_paths)
    savedmodel_dirs = sorted(savedmodel_dirs)

    candidates = h5_paths + savedmodel_dirs
    return candidates[:max_models], {
        "h5_found": len(h5_paths),
        "savedmodel_found": len(savedmodel_dirs),
    }


def try_load_model(path):
    try:
        m = load_model(path, compile=False)
        return m, None
    except Exception as e:
        return None, e


preferred_paths = [
    model_path_5,
    model_path_6,
    model_path_4,
    model_path_1,
    model_path_2,
    model_path_3,
]
existing_preferred = [p for p in preferred_paths if os.path.exists(p)]

if len(existing_preferred) < 2:
    discovered, stats = discover_models_under_kaggle_input(
        "/kaggle/input", max_models=10
    )
    print("Preferred existing:", existing_preferred)
    print("Discovered models stats:", stats)
    for p in discovered:
        if p not in existing_preferred:
            existing_preferred.append(p)

print("Model candidates (first 10):")
for p in existing_preferred[:10]:
    print(" -", p)




## === cell 4
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def extract_features_from_path(img_path, size=64):
    """
    Deterministic, fast features:
    - grayscale resized to (size,size) then flattened
    - add simple edge magnitude summary (gx, gy) and intensity stats
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        n = size * size + 6
        return np.zeros((n,), dtype=np.float32)

    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32
    )

    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge_mean = float((gx.mean() + gy.mean()) * 0.5)
    edge_std = float((gx.std() + gy.std()) * 0.5)

    mean = float(gray.mean())
    std = float(gray.std())
    q25 = float(np.quantile(gray, 0.25))
    q75 = float(np.quantile(gray, 0.75))

    flat = gray.reshape(-1)
    feats = np.concatenate(
        [flat, np.array([mean, std, q25, q75, edge_mean, edge_std], dtype=np.float32)],
        axis=0,
    )
    return feats


def build_dataset(df, img_dir, size=64, hflip=False):
    X = np.zeros((len(df), size * size + 6), dtype=np.float32)
    y = None
    if "label" in df.columns:
        y = df["label"].to_numpy(dtype=np.int64)
    for i, image_id in enumerate(df["image_id"].tolist()):
        img_path = os.path.join(img_dir, image_id)
        if not hflip:
            X[i] = extract_features_from_path(img_path, size=size)
        else:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                img = img.resize((size, size), resample=Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                gray = (
                    0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
                ).astype(np.float32)

                gx = np.abs(gray[:, 1:] - gray[:, :-1])
                gy = np.abs(gray[1:, :] - gray[:-1, :])
                edge_mean = float((gx.mean() + gy.mean()) * 0.5)
                edge_std = float((gx.std() + gy.std()) * 0.5)

                mean = float(gray.mean())
                std = float(gray.std())
                q25 = float(np.quantile(gray, 0.25))
                q75 = float(np.quantile(gray, 0.75))

                flat = gray.reshape(-1)
                X[i] = np.concatenate(
                    [
                        flat,
                        np.array(
                            [mean, std, q25, q75, edge_mean, edge_std],
                            dtype=np.float32,
                        ),
                    ],
                    axis=0,
                )
            except Exception:
                X[i] = np.zeros((size * size + 6,), dtype=np.float32)

    return X, y


def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True)
    sig = np.where(sig < 1e-6, 1.0, sig)
    return mu.astype(np.float32), sig.astype(np.float32)


def standardize_apply(X, mu, sig):
    return ((X - mu) / sig).astype(np.float32)


def train_multinomial_logreg(
    X, y, n_classes=5, lr=0.15, reg=1e-3, epochs=60, batch_size=256, seed=123
):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros((n_classes,), dtype=np.float32)

    y_onehot = np.eye(n_classes, dtype=np.float32)[y]

    for ep in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            j = idx[start : start + batch_size]
            Xb = X[j]
            yb = y_onehot[j]

            logits = Xb @ W + b
            P = softmax(logits)

            grad_logits = (P - yb) / Xb.shape[0]
            gW = Xb.T @ grad_logits + reg * W
            gb = grad_logits.sum(axis=0)

            W -= lr * gW
            b -= lr * gb

    return W, b


def predict_logreg(X, W, b):
    logits = X @ W + b
    return np.argmax(logits, axis=1).astype(np.int64)


SEED = 123

MAX_TRAIN_SAMPLES = 14000

FEAT_SIZE = 72

train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
if len(train_df_shuf) > MAX_TRAIN_SAMPLES:
    train_used = train_df_shuf.iloc[:MAX_TRAIN_SAMPLES].copy()
else:
    train_used = train_df_shuf

print("Using train samples:", len(train_used), "of", len(train_df))

X_train, y_train = build_dataset(
    train_used, train_image_dir, size=FEAT_SIZE, hflip=False
)

X_train_flip, y_train_flip = build_dataset(
    train_used, train_image_dir, size=FEAT_SIZE, hflip=True
)
X_train = np.concatenate([X_train, X_train_flip], axis=0)
y_train = np.concatenate([y_train, y_train_flip], axis=0)

mu, sig = standardize_fit(X_train)
X_train = standardize_apply(X_train, mu, sig)

W, b = train_multinomial_logreg(
    X_train,
    y_train,
    n_classes=5,
    lr=0.15,
    reg=1e-3,
    epochs=60,
    batch_size=256,
    seed=SEED,
)

X_test, _ = build_dataset(sample_csv, test_image_dir, size=FEAT_SIZE, hflip=False)
X_test = standardize_apply(X_test, mu, sig)
test_pred = predict_logreg(X_test, W, b)

test_pred = np.clip(test_pred, 0, 4).astype(int)



## === cell 5
image_predictions = []
test_ids = sample_csv["image_id"].tolist()

if TF_AVAILABLE:
    raise RuntimeError(
        "TF_AVAILABLE unexpectedly True in this environment; this script expects TF import to fail. "
        "If TF becomes available, re-enable the original model inference code path."
    )
else:
    for image_id, pred in zip(test_ids, test_pred):
        image_predictions.append({"image_id": image_id, "label": int(pred)})

submission_df = pd.DataFrame(image_predictions)
submission_df = submission_df[["image_id", "label"]]
assert submission_df.shape[0] == sample_csv.shape[0], (
    submission_df.shape,
    sample_csv.shape,
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission_df.shape)



## === cell 6
print(submission_df.head())
print(submission_df["label"].value_counts().sort_index())
print("Submission columns:", submission_df.columns.tolist())
print("Submission preview saved at /kaggle/working/submission.csv")
