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

0.8661226956784527

# 6. Current score

0.36921

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix execution by (1) addressing the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, and (2) removing the dependency on the missing `/kaggle/input/f-models` models by switching to a minimal, local baseline that still produces a valid `submission.csv`. To keep changes minimal and stable, the baseline predict the majority class from `train.csv`, which is score-reasonable for Cassava and avoids training time. I also make the data path resolution robust to either `/kaggle/input/...` or `/kaggle/data/...` layouts while keeping the existing intended paths. Finally, I ensure the submission file is written with the exact required columns and row alignment.'
- What this solution (achieved 0.41741) has done: 'The crash happens before any training/prediction because importing TensorFlow triggers a protobuf incompatibility (`MessageFactory.GetPrototype` missing). Since this baseline doesn’t actually use TensorFlow/Keras, the smallest and most stable fix is to remove the TensorFlow/Keras imports entirely so the pipeline runs end-to-end and writes `submission.csv`. To move the score toward your target without changing the “no-training baseline” approach, I replace the pure majority-class guess with a slightly better prior-based predictor: sample labels according to the empirical class distribution from `train.csv` (seeded for determinism). This keeps the same simple semantics (no image usage, no model training) but should improve expected accuracy vs always predicting the single majority class. The script still validate paths and submission format and write a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current solution samples labels from the global class prior, which has expected accuracy equal to the sum of squared class frequencies and is typically low for Cassava—explaining the 0.417 score. To move toward the 0.866 target with minimal change and without introducing training loops or image modeling, I keep the “no-training baseline” approach but switch to a deterministic, higher-accuracy heuristic: always predict the majority class from `train.csv`. This preserves the same overall semantics (no image use, no model), is stable/deterministic, and should substantially increase accuracy on this dataset relative to random prior sampling. I also keep the same path checks and ensure the submission format stays identical.'
- What this solution (achieved 0.59081) has done: 'To move your score up toward 0.866 with minimal disruption, the smallest legitimate step is to keep your current “no heavy training loop” structure but switch from a constant-majority guess to a lightweight image-based classifier using the provided `train_images/` and `test_images/`. This preserves the simple single-pass pipeline (no TensorFlow, no deep model), stays within the package constraints, and should substantially improve accuracy because Cassava classes have strong color/texture cues that a basic linear model can pick up. Concretely, we extract a compact RGB color histogram feature per image (fast), train a multinomial logistic regression (implemented in NumPy with a fixed number of iterations), and predict labels for the test set in the exact sample_submission order. The output remains a valid `submission.csv` with identical schema and row alignment.'
- What this solution (achieved 0.37145) has done: 'Your current linear softmax setup is OK, but its optimization is likely under-trained/unstable (fixed large LR, no shuffling, no class balancing), which can cap accuracy around ~0.59. To move the score upward toward 0.866 with minimal changes and the same core model/loss, I (1) add sample-weighting for class imbalance (still the same softmax regression, just reweighted cross-entropy), (2) switch to minibatch SGD with deterministic shuffling and a conservative learning-rate decay to improve convergence, and (3) slightly enrich the existing histogram feature by concatenating a coarse “grid color histogram” (same histogram logic, just computed on 2x2 tiles) while keeping runtime within limits. The submission writing and row alignment remain identical.'
- What this solution (achieved 0.35613) has done: 'Your current softmax regression is likely not converging enough in 6 epochs for this feature space; the smallest safe way to move accuracy upward toward the 0.866 target (without changing the model, loss, or features) is to train longer with a slightly more conservative step size. I keep the same histogram+grid features and the same linear softmax cross-entropy objective, but increase `iters` and reduce `lr` so optimization is more stable and reaches a better solution within the same semantics. I also make `num_classes` fixed to 5 (as per competition) to avoid any edge-case mismatch if a class is missing in a local split/run, while leaving the rest unchanged. Submission writing, order, and schema remain identical.'
- What this solution (achieved 0.37294) has done: 'Your current score (0.35613) is far below the target (0.86612), so we should improve accuracy while keeping the same core approach (RGB hist + 2x2 grid hist features, linear softmax regression, cross-entropy). The biggest issue is optimization stability: the gradient scaling currently depends on `wb.mean()` and batch size in a way that can distort step sizes, especially with class weights, preventing good convergence. I change the gradient to the standard *weighted mean* cross-entropy gradient (divide by `sum(weights)`), keep the same class-weighting idea, and slightly tune learning-rate/epochs conservatively to converge better without changing the model/feature semantics. I also add a tiny numerical-stability improvement in `softmax` (float32-safe) while keeping predictions and submission format identical.'
- What this solution (achieved 0.36921) has done: 'Your current score is far below the target, so we should improve accuracy while keeping the same feature extraction (RGB hist + 2x2 grid hist) and the same linear softmax regression with cross-entropy. The biggest likely issue is that plain SGD on full 18k samples is sensitive to feature scale and step size; with histograms, L2-normalization alone can still leave some dimensions dominating. I add a very small, standard stabilization: clip feature standardization to a reasonable range and add gradient-norm clipping (doesn’t change the model/loss, just prevents occasional oversized updates), plus a tiny adjustment to the learning-rate decay to converge a bit better. The submission format and ordering remain identical, and it still run end-to-end and write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.36921) has done: 'Your current score is far below the target, so we should improve accuracy while keeping the same core approach (RGB hist + 2x2 grid hist features, linear softmax regression, weighted cross-entropy, SGD). The largest likely accuracy bug is feature extraction: the per-tile histogram is normalized independently, which discards “how much of the image is this color” information across tiles and makes tile features artificially comparable; instead we should compute *counts* for all tiles and then L1-normalize the *entire concatenated vector once* (same features, same dims, just correct global normalization). I also make the histogram binning a bit more stable by using `np.bincount` (still identical histogram semantics) and keep everything else (training loop, loss, standardization, submission alignment) unchanged. This is a minimal change that typically gives a noticeable accuracy lift for linear models on histogram features, moving you closer to the target.'
- What this solution (achieved 0.36958) has done: 'Your score is far below the target, so we should improve accuracy while keeping your exact core approach (RGB+2x2 grid hist features, linear softmax regression, weighted cross-entropy, SGD). The biggest low-risk win is to fix a likely train/test mismatch: your features are extracted in the `sample_submission` order, but Kaggle scoring uses the platform’s internal test order by `image_id`; if there’s any ordering discrepancy, accuracy collapses. I (1) build predictions for the actual `test_images/` filenames, (2) merge them back onto `sample_submission.csv` by `image_id` to guarantee correct alignment, and (3) add one tiny, standard optimization stabilization that does not change the model/loss/features: scale the learning rate by `batch_size/denom` so the effective step size is consistent across batches with varying sample weights (this is equivalent to using the *mean* weighted gradient). These are minimal changes intended to move the score upward toward your target without changing the model family.'
- What this solution (achieved 0.36958) has done: 'Your current 0.36958 is far below the 0.86612 target, and the biggest likely reason (given the very high reported train-acc you should be seeing) is a train/test preprocessing mismatch: you resize with PIL in RGB but do not enforce consistent numeric scaling beyond histogram L1, and the optimizer is still quite sensitive to the per-feature standardization computed on all-zero rows if any images are missing/failed loads. I make two minimal, score-directed fixes while preserving your exact core approach (same histogram+grid features, same linear softmax regression, same loss/training loop): (1) make feature extraction robust by explicitly validating image decode and retrying with a safe fallback path to avoid silent zero-feature rows, and (2) add a tiny, standard “train-time feature sanity” guard that removes any all-zero feature rows from training (these act like label noise and can destroy generalization). These changes don’t alter the model family or objective, but they prevent a common failure mode that can collapse test accuracy. Submission alignment via merge on `image_id` is kept as-is to ensure correctness.'
- What this solution (achieved 0.36921) has done: 'Your current score is far below the target, so we should improve accuracy while keeping the exact same core approach (RGB/global + 2x2 grid hist features, linear softmax regression, weighted cross-entropy SGD). The biggest issue I see is an error in the optimizer update: you already divide the gradient by the (weighted) batch denominator, but then you additionally multiply the step by `batch_size/denom` again (`step_scale`), effectively shrinking/warping the learning rate depending on weights and harming convergence. I remove that extra scaling (use a standard mean weighted gradient update), keep everything else the same, and add one minimal stability fix: compute softmax in float64 then cast back to float32 to avoid overflow/underflow affecting gradients. This should move your score upward toward the target without changing the model family, loss, feature extraction, or overall training loop semantics, and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from PIL import Image

import pandas as pd
import numpy as np

np.random.seed(42)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(DATA_DIR):
    alt = "/kaggle/data/cassava-leaf-disease-classification"
    if os.path.exists(alt):
        DATA_DIR = alt

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir at {TEST_IMG_DIR}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert {"image_id", "label"}.issubset(
    sample_sub.columns
), "sample_submission.csv must have image_id and label columns"

train_df = pd.read_csv(TRAIN_CSV_PATH)
assert {"image_id", "label"}.issubset(
    train_df.columns
), "train.csv must have image_id and label columns"

print("DATA_DIR:", DATA_DIR)
print("sample rows:", len(sample_sub), "train rows:", len(train_df))




## === cell 1
def _hist_counts(arr_uint8, bins=16):
    q = (arr_uint8.astype(np.int32) * bins) // 256
    feat = []
    for c in range(3):
        bc = np.bincount(q[..., c].ravel(), minlength=bins).astype(np.float32)
        feat.append(bc)
    return np.concatenate(feat, axis=0)


def image_to_feature(img_path, size=(96, 96), bins=16, grid=(2, 2)):
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize(size, resample=Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
            raise ValueError("bad image array")
    except Exception:
        return None

    feats = [_hist_counts(arr, bins=bins)]

    gh, gw = grid
    H, W, _ = arr.shape
    hs = np.linspace(0, H, gh + 1, dtype=int)
    ws = np.linspace(0, W, gw + 1, dtype=int)
    for i in range(gh):
        for j in range(gw):
            tile = arr[hs[i] : hs[i + 1], ws[j] : ws[j + 1], :]
            feats.append(_hist_counts(tile, bins=bins))

    feat = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    feat /= feat.sum() + 1e-8
    return feat


def build_feature_matrix(image_ids, img_dir, size=(96, 96), bins=16, grid=(2, 2)):
    feat_dim = (1 + grid[0] * grid[1]) * 3 * bins
    X = np.zeros((len(image_ids), feat_dim), dtype=np.float32)
    missing = 0
    failed = 0
    for i, img_id in enumerate(image_ids):
        p = os.path.join(img_dir, img_id)
        if not os.path.exists(p):
            missing += 1
            continue
        feat = image_to_feature(p, size=size, bins=bins, grid=grid)
        if feat is None:
            failed += 1
            continue
        X[i] = feat
    if missing:
        print(
            f"Warning: {missing} images missing in {img_dir}. Their features are zeros."
        )
    if failed:
        print(
            f"Warning: {failed} images failed to decode in {img_dir}. Their features are zeros."
        )
    return X


def softmax(z):
    z64 = z.astype(np.float64, copy=False)
    z64 = z64 - np.max(z64, axis=1, keepdims=True)
    ez = np.exp(z64)
    p = ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)
    return p.astype(np.float32, copy=False)


def train_softmax_regression(
    X,
    y,
    num_classes,
    lr=0.07,
    l2=1e-3,
    iters=45,
    batch_size=1024,
    lr_decay=0.985,
    seed=42,
    grad_clip_norm=5.0,
):
    rng = np.random.RandomState(seed)

    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    n, d = Xb.shape
    W = np.zeros((d, num_classes), dtype=np.float32)

    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    class_w = (counts.sum() / (num_classes * (counts + 1e-6))).astype(np.float32)
    sample_w = class_w[y].astype(np.float32)

    for epoch in range(iters):
        perm = rng.permutation(n)
        Xp = Xb[perm]
        yp = y[perm]
        wp = sample_w[perm]

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            xb = Xp[start:end]
            yb = yp[start:end]
            wb = wp[start:end]

            logits = xb @ W
            P = softmax(logits)

            y_onehot = np.zeros((end - start, num_classes), dtype=np.float32)
            y_onehot[np.arange(end - start), yb] = 1.0

            wcol = wb.reshape(-1, 1)
            denom = float(np.sum(wb) + 1e-12)

            grad = (xb.T @ ((P - y_onehot) * wcol)) / denom

            reg = l2 * W
            reg[-1, :] = 0.0
            grad += reg

            if grad_clip_norm is not None and grad_clip_norm > 0:
                gn = float(np.sqrt(np.sum(grad * grad)) + 1e-12)
                if gn > grad_clip_norm:
                    grad *= grad_clip_norm / gn

            W -= lr * grad

        lr *= lr_decay

        logits_full = Xb @ W
        pred_full = np.argmax(logits_full, axis=1)
        acc_full = (pred_full == y).mean()
        print(f"epoch {epoch+1:>2d}/{iters} train-acc={acc_full:.4f} lr={lr:.5f}")

    return W


def predict_softmax_regression(X, W):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    logits = Xb @ W
    P = softmax(logits)
    return np.argmax(P, axis=1).astype(int)


train_image_ids = train_df["image_id"].values
y = train_df["label"].astype(int).values
num_classes = 5

print("Extracting train features...")
X_train = build_feature_matrix(
    train_image_ids, TRAIN_IMG_DIR, size=(96, 96), bins=16, grid=(2, 2)
)

row_sums = X_train.sum(axis=1)
keep = row_sums > 0
dropped = int((~keep).sum())
if dropped > 0:
    print(
        f"Dropping {dropped} training rows with all-zero features (missing/failed images)."
    )
X_train = X_train[keep]
y = y[keep]

mu = X_train.mean(axis=0, keepdims=True)
sigma = X_train.std(axis=0, keepdims=True) + 1e-6

X_train_std = (X_train - mu) / sigma
X_train_std = np.clip(X_train_std, -5.0, 5.0).astype(np.float32, copy=False)

print("Training linear softmax classifier...")
W = train_softmax_regression(
    X_train_std,
    y,
    num_classes=num_classes,
    lr=0.07,
    l2=1e-3,
    iters=45,
    batch_size=1024,
    lr_decay=0.985,
    seed=42,
    grad_clip_norm=5.0,
)

test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")])
assert len(test_files) > 0, f"No .jpg files found in {TEST_IMG_DIR}"

print("Extracting test features...")
X_test = build_feature_matrix(
    test_files, TEST_IMG_DIR, size=(96, 96), bins=16, grid=(2, 2)
)

X_test_std = (X_test - mu) / sigma
X_test_std = np.clip(X_test_std, -5.0, 5.0).astype(np.float32, copy=False)

preds = predict_softmax_regression(X_test_std, W)
preds = np.clip(preds, 0, 4).astype(int)

pred_df = pd.DataFrame({"image_id": test_files, "label": preds})

my_submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
assert len(my_submission) == len(
    sample_sub
), "Submission row count must match sample_submission.csv"
assert (
    my_submission["label"].isnull().sum() == 0
), "Missing predictions for some test image_ids"
my_submission["label"] = my_submission["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(my_submission.head())
print("Rows:", len(my_submission))
print("Unique predicted labels:", sorted(my_submission["label"].unique().tolist()))
