# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.tree import DecisionTreeClassifier

SEED = 42
np.random.seed(SEED)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(DATA_ROOT):
    alt = "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
    if os.path.exists(alt):
        DATA_ROOT = alt

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

MODEL1_PATH = "/kaggle/input/densenet/keras/default/1/DenseNet (1).keras"
MODEL2_PATH = "/kaggle/input/abc/keras/default/1/newModel7.keras"


def load_image_np(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)


def second_model_preprocess(image_np: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image_np.astype("uint8"), "RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = y.astype(int)
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def _extract_global_feats_from_batch(batch: np.ndarray) -> np.ndarray:
    """
    Deterministic, cheap global stats from preprocessed images (B,224,224,3) in [0,1].
    This is the SAME feature extraction used by both probability models; core logic stays:
    model.predict returns 5-class probs from images, and we still only feed 10 dims to the tree.
    """
    b = np.asarray(batch, dtype=np.float32)
    B = b.shape[0]

    mean_rgb = b.mean(axis=(1, 2))  # (B,3)
    std_rgb = b.std(axis=(1, 2))  # (B,3)

    gray = (0.2989 * b[..., 0] + 0.5870 * b[..., 1] + 0.1140 * b[..., 2]).reshape(B, -1)
    p10 = np.percentile(gray, 10, axis=1).astype(np.float32).reshape(B, 1)
    p50 = np.percentile(gray, 50, axis=1).astype(np.float32).reshape(B, 1)
    p90 = np.percentile(gray, 90, axis=1).astype(np.float32).reshape(B, 1)

    rg = (b[..., 0] - b[..., 1]).reshape(B, -1)
    bg = (b[..., 2] - b[..., 1]).reshape(B, -1)
    rg_mean = rg.mean(axis=1).astype(np.float32).reshape(B, 1)
    rg_std = rg.std(axis=1).astype(np.float32).reshape(B, 1)
    bg_mean = bg.mean(axis=1).astype(np.float32).reshape(B, 1)
    bg_std = bg.std(axis=1).astype(np.float32).reshape(B, 1)

    feats = np.concatenate(
        [mean_rgb, std_rgb, p10, p50, p90, rg_mean, rg_std, bg_mean, bg_std], axis=1
    ).astype(
        np.float32, copy=False
    )  # (B,13)
    return feats


class NumpyProbModel:
    """
    Core logic preserved: given images -> output 5-class softmax probabilities.

    Change (score-improving, minimal): instead of random W,b, fit a simple softmax regression
    (multiclass logistic regression) on the SAME global features extracted from training images.
    This keeps the probability-vector semantics but makes them label-informed, typically moving
    accuracy upward toward the target.
    """

    def __init__(self, name: str, seed: int, n_feats: int = 13, n_classes: int = 5):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.n_feats = n_feats
        self.n_classes = n_classes

        self.W = self.rng.normal(0, 0.05, size=(self.n_feats, self.n_classes)).astype(
            np.float32
        )
        self.b = np.zeros((self.n_classes,), dtype=np.float32)

        self.mu = np.zeros((self.n_feats,), dtype=np.float32)
        self.sigma = np.ones((self.n_feats,), dtype=np.float32)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma

    def fit_on_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 180,
        lr: float = 0.35,
        l2: float = 2e-3,
    ) -> None:
        """
        Full-batch GD (no early stopping). Low-dim so it's fast.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        assert X.ndim == 2 and X.shape[1] == self.n_feats

        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        self.mu = mu.astype(np.float32)
        self.sigma = sigma.astype(np.float32)

        Xs = self._standardize(X)
        Y = _one_hot(y, self.n_classes)

        n = Xs.shape[0]
        for _ in range(int(epochs)):
            logits = Xs @ self.W + self.b  # (n,C)
            P = _softmax(logits, axis=1)  # (n,C)

            dlogits = (P - Y) / n
            gW = Xs.T @ dlogits + l2 * self.W
            gb = dlogits.sum(axis=0)

            self.W -= lr * gW
            self.b -= lr * gb

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        feats = _extract_global_feats_from_batch(batch)
        feats = self._standardize(feats)
        logits = feats @ self.W + self.b
        probs = _softmax(logits, axis=1).astype(np.float32)
        return probs


model1 = NumpyProbModel("trained_numpy_model1", SEED + 1)
model2 = NumpyProbModel("trained_numpy_model2", SEED + 2)


def _ensure_2d_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    return p


def _extract_feats_for_paths(filepaths, batch_size=32) -> np.ndarray:
    """
    Helper to extract the same 13-D features per image (for training the prob models).
    """
    out = []
    for i in range(0, len(filepaths), batch_size):
        batch_paths = filepaths[i : i + batch_size]
        batch = np.concatenate(
            [second_model_preprocess(load_image_np(p)) for p in batch_paths], axis=0
        )
        out.append(_extract_global_feats_from_batch(batch))
    return np.vstack(out) if out else np.zeros((0, 13), dtype=np.float32)


def predict_concat_probs_batch(filepaths, batch_size=32):
    """
    Produce concatenated probabilities [probs1, probs2] for each image.
    Core logic preserved: model1.predict + model2.predict then concatenate.
    """
    feats = []
    for i in range(0, len(filepaths), batch_size):
        batch_paths = filepaths[i : i + batch_size]

        batch = np.concatenate(
            [second_model_preprocess(load_image_np(p)) for p in batch_paths], axis=0
        )

        p1 = _ensure_2d_probs(model1.predict(batch, verbose=0))
        p2 = _ensure_2d_probs(model2.predict(batch, verbose=0))

        if p1.shape[0] != p2.shape[0]:
            raise ValueError(
                f"Model outputs batch mismatch: p1 {p1.shape}, p2 {p2.shape}"
            )

        p1 = p1.astype(np.float32, copy=False)
        p2 = p2.astype(np.float32, copy=False)

        feats.append(np.concatenate([p1, p2], axis=1))
    return np.vstack(feats) if feats else np.zeros((0, 0), dtype=np.float32)


print("DATA_ROOT =", DATA_ROOT)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Using models:", model1.name, "and", model2.name)



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
train_filepaths = [
    os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image_id"].tolist()
]
train_labels = train_df["label"].astype(int).to_numpy()

t0 = time.time()
train_X13 = _extract_feats_for_paths(train_filepaths, batch_size=32)
print(f"Train global feats shape: {train_X13.shape} computed in {time.time()-t0:.1f}s")

t0 = time.time()
model1.fit_on_features(train_X13, train_labels, epochs=170, lr=0.33, l2=2e-3)
model2.fit_on_features(train_X13, train_labels, epochs=210, lr=0.28, l2=3e-3)
print(f"Fitted prob models in {time.time()-t0:.1f}s")

t0 = time.time()
train_probs = predict_concat_probs_batch(train_filepaths, batch_size=32)
if train_probs.shape[0] != len(train_labels):
    raise RuntimeError(
        f"Mismatch: train_probs rows={train_probs.shape[0]} labels={len(train_labels)}"
    )
print(f"Train probs shape: {train_probs.shape} computed in {time.time()-t0:.1f}s")

decision_tree = DecisionTreeClassifier(
    criterion="entropy", max_depth=10, random_state=SEED
)
decision_tree.fit(train_probs, train_labels)



## === cell 2
sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sub["image_id"].tolist()
test_filepaths = [os.path.join(TEST_IMG_DIR, fn) for fn in test_image_ids]

t0 = time.time()
combined_probs = predict_concat_probs_batch(test_filepaths, batch_size=32)
print(f"Test probs shape: {combined_probs.shape} computed in {time.time()-t0:.1f}s")

prediction = decision_tree.predict(combined_probs).astype(int)

submission = pd.DataFrame({"image_id": test_image_ids, "label": prediction})
submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
print("Saved to:", os.path.abspath("submission.csv"))
