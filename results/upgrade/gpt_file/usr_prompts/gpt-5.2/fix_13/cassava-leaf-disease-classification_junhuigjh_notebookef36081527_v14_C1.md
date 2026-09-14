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

0.8108189785433666

# 6. Current score

0.35202

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60725) has done: 'I fix the crash happening during TensorFlow/Keras model loading (the `MessageFactory.GetPrototype` protobuf incompatibility) so cell 1/2 can see `predict_concat_probs_batch` and run. The safest minimal change in this environment is to avoid loading the external `.keras` models (which triggers the protobuf error) and instead keep the same “two-model probabilities → concatenate → DecisionTree” core logic by replacing them with lightweight deterministic TF models that output 5-class softmax probabilities. I also add a small fallback to automatically locate the dataset root under `/kaggle/input` to avoid path mismatches. This ensure the notebook runs end-to-end and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.61024) has done: 'We fix the `MessageFactory.GetPrototype` crash by avoiding TensorFlow import/model-loading entirely (it happens at import time in this environment), while preserving the core “two models → probability vectors → concatenate → DecisionTree” pipeline. Concretely, we replace the two TF/Keras probability models with lightweight, deterministic NumPy probability “models” that output 5-class softmax probabilities from the same 224×224 preprocessed images. This keeps the same feature semantics (two 5-dim probability vectors concatenated into 10 features) so the DecisionTree training/inference code remains unchanged, and it restores end-to-end execution and submission generation. As a small quality nudge toward the target accuracy without changing the overall approach, we also switch the tree criterion to `entropy` (still a DecisionTree with the same depth/seed), which often fits probabilistic features slightly better.'
- What this solution (achieved 0.59865) has done: 'To move your score upward toward the 0.81 target while keeping the exact same overall pipeline (“two 5-class prob models → concat (10 features) → DecisionTree”), I make the two NumPy probability models slightly more expressive but still deterministic and lightweight. Concretely, I (1) add a few extra global image statistics (brightness percentiles + simple channel-difference color cues) to the features used inside each probability model, and (2) increase the DecisionTree depth a bit to better exploit the slightly richer 10-D probability features without changing the training approach. These are minimal, safe changes that typically improve separability for cassava classes while preserving evaluation semantics and producing the same submission format.'
- What this solution (achieved 0.59417) has done: 'We keep the exact same pipeline (two 5-class probability “models” → concatenate to 10 features → DecisionTree) but make the probability vectors more class-informative without adding new modeling approaches. The smallest high-impact fix is to make each NumPy probability model *trained* (linear softmax regression) on the training set features instead of using random weights; this preserves the same semantics (5-class probs) and keeps inference identical, but should lift accuracy substantially toward your 0.81 target. We also add a tiny L2-regularized, deterministic gradient-descent fit inside each model (no early stopping, fixed epochs), and keep the DecisionTree unchanged aside from using the now-better probability features. This stays within runtime by computing global image statistics once per image and fitting on those low-dimensional features.'
- What this solution (achieved 0.59417) has done: 'The timeout is dominated by repeatedly decoding + resizing all images to 224×224 float32 and writing huge memmaps, plus slow per-batch percentile feature extraction and redundant passes over the full training set (preprocess → extract feats → predict probs). I keep the exact same models, training loops, features, and decision tree, but eliminate the expensive preprocessed-image memmaps by caching only the 13 global features (and then the concatenated 10-dim probs) in compact memmaps. I also speed up the hot feature extractor by replacing `np.percentile` with an equivalent `np.partition`-based computation (same definition for linear interpolation on discrete ranks here) and reduce Python overhead by processing in larger batches and avoiding extra array copies. All paths remain unchanged, determinism is preserved, and the resulting probabilities/features are numerically equivalent up to negligible float differences.'
- What this solution (achieved 0.35202) has done: 'Your current score (0.59417) is far below the target (0.81082), so we should increase accuracy while keeping the same pipeline (“two 5-class prob models → concat (10) → DecisionTree”). The most impactful minimal change is to address severe class imbalance in cassava by training the two softmax-regression probability models with per-class weighted cross-entropy (no change to architecture, features, or training loop structure; just reweighting the same loss). To better match accuracy, we also pass the same class weights into the DecisionTree via `class_weight`, which often improves minority-class correctness without changing the modeling approach. All I/O paths and submission format stay identical, and runtime remains similar.'

# 9. Code solution

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


def _percentiles_partition(x2d: np.ndarray, qs=(10, 50, 90)) -> np.ndarray:
    """
    x2d: (B,N) float32
    returns: (B,len(qs)) float32 percentiles with linear interpolation on ranks (like numpy default).
    """
    x = np.asarray(x2d, dtype=np.float32, order="C")
    B, N = x.shape
    out = np.empty((B, len(qs)), dtype=np.float32)

    for qi, q in enumerate(qs):
        h = (q / 100.0) * (N - 1)
        lo = int(np.floor(h))
        hi = int(np.ceil(h))
        if lo == hi:
            kth = lo
            part = np.partition(x, kth=kth, axis=1)
            out[:, qi] = part[:, kth]
        else:
            part = np.partition(x, kth=(lo, hi), axis=1)
            v_lo = part[:, lo]
            v_hi = part[:, hi]
            w = np.float32(h - lo)
            out[:, qi] = (1.0 - w) * v_lo + w * v_hi
    return out


def _extract_global_feats_from_batch(batch: np.ndarray) -> np.ndarray:
    """
    Deterministic, cheap global stats from preprocessed images (B,224,224,3) in [0,1].
    """
    b = np.asarray(batch, dtype=np.float32)
    B = b.shape[0]

    mean_rgb = b.mean(axis=(1, 2))  # (B,3)
    std_rgb = b.std(axis=(1, 2))  # (B,3)

    gray = (0.2989 * b[..., 0] + 0.5870 * b[..., 1] + 0.1140 * b[..., 2]).reshape(B, -1)
    p = _percentiles_partition(gray, qs=(10, 50, 90))  # (B,3)
    p10 = p[:, 0:1]
    p50 = p[:, 1:2]
    p90 = p[:, 2:3]

    rg = (b[..., 0] - b[..., 1]).reshape(B, -1)
    bg = (b[..., 2] - b[..., 1]).reshape(B, -1)
    rg_mean = rg.mean(axis=1, dtype=np.float32).reshape(B, 1)
    rg_std = rg.std(axis=1).astype(np.float32).reshape(B, 1)
    bg_mean = bg.mean(axis=1, dtype=np.float32).reshape(B, 1)
    bg_std = bg.std(axis=1).astype(np.float32).reshape(B, 1)

    feats = np.concatenate(
        [mean_rgb, std_rgb, p10, p50, p90, rg_mean, rg_std, bg_mean, bg_std], axis=1
    ).astype(
        np.float32, copy=False
    )  # (B,13)
    return feats


class NumpyProbModel:
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
        class_weight: np.ndarray | None = None,
    ) -> None:
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

        if class_weight is None:
            sample_w = np.ones((Xs.shape[0], 1), dtype=np.float32)
        else:
            cw = np.asarray(class_weight, dtype=np.float32).reshape(-1)
            if cw.shape[0] != self.n_classes:
                raise ValueError(
                    f"class_weight must have shape ({self.n_classes},), got {cw.shape}"
                )
            sample_w = cw[y].astype(np.float32).reshape(-1, 1)

        n = Xs.shape[0]
        for _ in range(int(epochs)):
            logits = Xs @ self.W + self.b  # (n,C)
            P = _softmax(logits, axis=1)  # (n,C)

            dlogits = (P - Y) * sample_w
            dlogits = dlogits / np.maximum(sample_w.sum(), 1e-12)

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


def _build_or_load_feats13_memmap(
    filepaths, memmap_path: str, batch_size: int = 128
) -> np.memmap:
    n = len(filepaths)
    shape = (n, 13)

    if os.path.exists(memmap_path):
        return np.memmap(memmap_path, mode="r", dtype=np.float32, shape=shape)

    os.makedirs(os.path.dirname(memmap_path) or ".", exist_ok=True)
    mm = np.memmap(memmap_path, mode="w+", dtype=np.float32, shape=shape)

    for i in range(0, n, batch_size):
        batch_paths = filepaths[i : i + batch_size]
        bs = len(batch_paths)
        batch = np.empty((bs, 224, 224, 3), dtype=np.float32)
        for j, p in enumerate(batch_paths):
            batch[j] = second_model_preprocess(load_image_np(p))[0]
        mm[i : i + bs] = _extract_global_feats_from_batch(batch)
    mm.flush()
    return np.memmap(memmap_path, mode="r", dtype=np.float32, shape=shape)


def _predict_concat_probs_from_feats13(X13: np.ndarray) -> np.ndarray:
    X13 = np.asarray(X13, dtype=np.float32)
    Xs1 = model1._standardize(X13)
    Xs2 = model2._standardize(X13)
    p1 = _softmax(Xs1 @ model1.W + model1.b, axis=1).astype(np.float32, copy=False)
    p2 = _softmax(Xs2 @ model2.W + model2.b, axis=1).astype(np.float32, copy=False)
    return np.concatenate([p1, p2], axis=1).astype(np.float32, copy=False)


def _build_or_load_probs10_memmap(X13: np.ndarray, memmap_path: str) -> np.memmap:
    n = X13.shape[0]
    shape = (n, 10)
    if os.path.exists(memmap_path):
        return np.memmap(memmap_path, mode="r", dtype=np.float32, shape=shape)

    os.makedirs(os.path.dirname(memmap_path) or ".", exist_ok=True)
    mm = np.memmap(memmap_path, mode="w+", dtype=np.float32, shape=shape)
    mm[:] = _predict_concat_probs_from_feats13(X13)
    mm.flush()
    return np.memmap(memmap_path, mode="r", dtype=np.float32, shape=shape)


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

num_classes = 5
counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
class_weight_vec = (counts.sum() / (num_classes * np.maximum(counts, 1.0))).astype(
    np.float32
)

t0 = time.time()
train_feats_path = "/kaggle/working/_train_feats13_float32.dat"
train_X13 = _build_or_load_feats13_memmap(
    train_filepaths, train_feats_path, batch_size=128
)
print(f"Train feats13 memmap shape: {train_X13.shape} ready in {time.time()-t0:.1f}s")

t0 = time.time()
model1.fit_on_features(
    train_X13, train_labels, epochs=170, lr=0.33, l2=2e-3, class_weight=class_weight_vec
)
model2.fit_on_features(
    train_X13, train_labels, epochs=210, lr=0.28, l2=3e-3, class_weight=class_weight_vec
)
print(f"Fitted prob models in {time.time()-t0:.1f}s")

t0 = time.time()
train_probs_path = "/kaggle/working/_train_probs10_float32.dat"
train_probs = _build_or_load_probs10_memmap(train_X13, train_probs_path)
if train_probs.shape[0] != len(train_labels):
    raise RuntimeError(
        f"Mismatch: train_probs rows={train_probs.shape[0]} labels={len(train_labels)}"
    )
print(f"Train probs shape: {train_probs.shape} ready in {time.time()-t0:.1f}s")

decision_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=10,
    random_state=SEED,
    class_weight={i: float(class_weight_vec[i]) for i in range(num_classes)},
)
decision_tree.fit(train_probs, train_labels)



## === cell 2
sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sub["image_id"].tolist()
test_filepaths = [os.path.join(TEST_IMG_DIR, fn) for fn in test_image_ids]

t0 = time.time()
test_feats_path = "/kaggle/working/_test_feats13_float32.dat"
test_X13 = _build_or_load_feats13_memmap(
    test_filepaths, test_feats_path, batch_size=128
)
print(f"Test feats13 memmap shape: {test_X13.shape} ready in {time.time()-t0:.1f}s")

t0 = time.time()
test_probs_path = "/kaggle/working/_test_probs10_float32.dat"
combined_probs = _build_or_load_probs10_memmap(test_X13, test_probs_path)
print(f"Test probs shape: {combined_probs.shape} ready in {time.time()-t0:.1f}s")

prediction = decision_tree.predict(combined_probs).astype(int)

submission = pd.DataFrame({"image_id": test_image_ids, "label": prediction})
submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
print("Saved to:", os.path.abspath("submission.csv"))
