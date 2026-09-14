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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7456140350877198

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'Your notebook currently can’t yield a score because it fails before creating a valid submission: the weights path points to a dataset that isn’t present in the provided file tree, and even if it ran, the CSV writing risks wrong row order and label decoding issues due to Tensor/NumPy mixing. I make the smallest changes to (1) robustly locate and load the weight file if it exists in the competition input directory, otherwise fall back to a “healthy-only” submission (valid CSV so you can obtain a score), and (2) ensure predictions are converted to NumPy consistently and written in the exact sample_submission order. These changes preserve your model and thresholds exactly and only touch loading + submission generation correctness. This should move you from “no score” to a valid scored submission, which is the necessary first step toward the target 0.7456.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf version by pinning protobuf to a TF‑compatible release at runtime (a minimal environment fix that unblocks execution). Then I keep your exact model/threshold logic, but make sure the label list is never empty (fallback to `healthy`) so the submission matches the competition’s expected format and typically improves F1 versus blank labels. Finally, I make data paths robust (`/kaggle/input/...`) and keep the submission row order exactly as `sample_submission.csv` to avoid silent misalignment.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.24507) is far below the target (0.7456), so we should improve performance without changing your model architecture or training approach. The biggest issue is that you’re using `per_image_standardization`, but Xception was trained/expected to use `tf.keras.applications.xception.preprocess_input`; switching to the correct preprocessing is a minimal, evaluation-semantic-aligned change that typically yields a large F1 lift. I also make the prediction loop batched for stability/consistency (same model outputs, just fewer Python/Tensor overhead side-effects) and ensure the submission is written in exactly the sample order (unchanged). Thresholds and label decoding remain exactly as you defined.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target (0.245 vs 0.746), so we should improve prediction quality while keeping your exact model and overall inference approach intact. The biggest likely issue is class/threshold misalignment: `training_class` is built from a `np.unique` over tokens, which sorts labels alphabetically and can mismatch the class order used when thresholds (and weights) were originally created. I keep the same model and thresholds, but rebuild `training_class` in a stable “first-seen” order from `train.csv` (common practice for multilabel pipelines) and add a safety check that the number of thresholds matches the number of classes. This is a minimal change that often produces a large F1 jump if your current submission is effectively decoding the wrong classes.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.245) is far below the target (0.746), so we should improve multi-label decision quality while keeping your exact model, weights, and inference flow intact. The biggest likely bottleneck now is using a fixed per-class threshold list that may not be well-calibrated for the hidden test distribution; since the competition metric is mean F1, a small threshold calibration on a held-out validation split of the provided training set is a minimal change that directly targets the metric. I keep your architecture, weights loading, preprocessing, and label decoding, but add an offline step to select per-class thresholds (same “pred>=thr” rule) that maximize mean F1 on the validation split. If weights are missing, the code still writes a valid `submission.csv` fallback.'
- What this solution (achieved 0.24507) has done: 'Your score (0.245) is far below the target (0.746), so we should improve prediction quality with minimal changes while preserving your model and inference semantics. The most likely remaining bottleneck is suboptimal threshold calibration: your current grid is coarse and capped to 0.50, which can prevent selecting better per-class thresholds for mean F1. I keep the exact same model, preprocessing, and “pred>=thr” rule, but (1) expand the threshold search grid to include higher values and finer steps, and (2) use a slightly larger validation sample cap to reduce noise in threshold selection. This should move the score upward toward the target without changing architecture, training, or loss.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.245) is far below the target (0.746), so we should increase performance with minimal, metric-aligned changes while preserving your exact model and inference semantics. The biggest likely issue left is that threshold calibration is optimizing plain macro-F1 across all 6 classes, but the competition evaluates mean F1 across images (sample-wise), so we should calibrate thresholds directly against a sample-wise F1 objective. I keep your exact architecture, weights loading, preprocessing, and `pred >= thr` decision rule, and only swap the calibration scoring function to a sample-wise mean F1 (with the same “empty -> healthy” fallback as your submission). This should move the score upward toward the target without changing training or the model.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.245) is far below the target (0.746), so we should improve prediction quality with minimal, metric-aligned changes while keeping your model and inference logic intact. The biggest likely issue now is that the validation threshold calibration is performed on a random split that can be label-imbalanced; switching to an iterative stratified split for multilabel data is a small change that typically yields thresholds that generalize better and increases mean sample-F1. I keep the exact model, preprocessing, and `pred >= thr` decision rule, but change only the calibration split construction and make calibration deterministic and safe for rare classes. The submission writing stays in the exact `sample_submission.csv` row order and still falls back to all-healthy if weights are missing.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd


def _ensure_tf_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version detected: {pb_ver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib
        import google.protobuf  # noqa: F401

        importlib.reload(google.protobuf)


_ensure_tf_protobuf_compat()

print("number of training image")
train_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
test_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
print(len([name for name in os.listdir(train_img_dir)]))
print(len([name for name in os.listdir(test_img_dir)]))

training_csv = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")

_seen = set()
training_class_list = []
for lbls in training_csv["labels"].astype(str).tolist():
    for tok in lbls.split():
        if tok not in _seen:
            _seen.add(tok)
            training_class_list.append(tok)
training_class = np.array(training_class_list, dtype=object)

print("\nnumber of class")
print(training_class)
print(len(training_class))




## === cell 1
def predict2strings(pred, threshold):
    pred = np.asarray(pred).reshape(-1)
    t = np.asarray(threshold).reshape(-1)
    s = " ".join(training_class[np.where(pred >= t)[0]])
    return s if s.strip() else "healthy"


def predict2n_hot(pred, threshold):
    pred = np.asarray(pred)
    t = np.asarray(threshold)
    return np.where(pred >= t, np.ones(pred.shape), np.zeros(pred.shape))


def n_hot2string(n_hot):
    n_hot = np.asarray(n_hot).reshape(-1)
    s = " ".join(training_class[np.where(n_hot >= 1)[0]])
    return s if s.strip() else "healthy"




## === cell 2
import tensorflow as tf

model_input = tf.keras.layers.Input(shape=(299, 299, 3))
xception_layer = tf.keras.applications.Xception(
    include_top=False,
    weights=None,
    input_shape=(299, 299, 3),
    input_tensor=model_input,
    pooling="max",
).output
fc_1 = tf.keras.layers.Dense(1024)(xception_layer)
fc_2 = tf.keras.layers.Dense(512)(fc_1)
model_output = tf.keras.layers.Dense(len(training_class), activation="sigmoid")(fc_2)
model = tf.keras.Model(inputs=model_input, outputs=model_output)




## === cell 3
def _find_weights_path():
    p0 = "../input/cs5489-project-train-baseline/my_model/model_1"
    if tf.io.gfile.exists(p0) or tf.io.gfile.exists(p0 + ".index"):
        return p0

    for root, _, files in os.walk("/kaggle/input"):
        if "model_1.index" in files:
            return os.path.join(root, "model_1")
        if "model_1" in files:
            return os.path.join(root, "model_1")
    return None


weights_path = _find_weights_path()
if weights_path is not None:
    model.load_weights(weights_path)
    print(f"Loaded weights from: {weights_path}")
else:
    print(
        "WARNING: model weights not found under /kaggle/input; will write a valid fallback submission."
    )



## === cell 4
threshold = [
    0.5469251871109009,
    0.4226226806640625,
    0.9167647957801819,
    0.1412835568189621,
    0.22426636517047882,
    0.4077064096927643,
]

if len(threshold) != len(training_class):
    raise ValueError(
        f"threshold length ({len(threshold)}) != number of classes ({len(training_class)}). "
        "This indicates class-order/class-count mismatch; fix before submitting."
    )

train_img_path = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
test_img_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"

_xception_preprocess = tf.keras.applications.xception.preprocess_input


def _load_and_preprocess_batch(filepaths):
    imgs = []
    for fp in filepaths:
        img = tf.keras.preprocessing.image.load_img(fp, target_size=(299, 299))
        img = tf.keras.preprocessing.image.img_to_array(img)
        imgs.append(img)
    x = np.stack(imgs, axis=0).astype(np.float32)
    x = _xception_preprocess(x)
    return x


def _labels_to_multihot(labels_strs):
    y = np.zeros((len(labels_strs), len(training_class)), dtype=np.float32)
    class_to_idx = {c: i for i, c in enumerate(training_class.tolist())}
    for i, s in enumerate(labels_strs):
        for tok in str(s).split():
            if tok in class_to_idx:
                y[i, class_to_idx[tok]] = 1.0
    return y


def _mean_sample_f1(y_true, y_pred_bin, eps=1e-9):
    y_true = (np.asarray(y_true) > 0.5).astype(np.float32)
    y_pred_bin = (np.asarray(y_pred_bin) > 0.5).astype(np.float32)

    tp = (y_true * y_pred_bin).sum(axis=1)
    fp = ((1.0 - y_true) * y_pred_bin).sum(axis=1)
    fn = (y_true * (1.0 - y_pred_bin)).sum(axis=1)

    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return float(np.mean(f1))


def _iterative_stratified_val_indices(y, val_size, seed=123):
    rng = np.random.RandomState(seed)
    n, c = y.shape
    desired = np.maximum(1.0, y.sum(axis=0) * (val_size / float(n)))
    val_idx = []
    remaining = set(range(n))

    while len(val_idx) < val_size:
        need = desired.copy()
        for i in val_idx:
            need -= y[i]
        need = np.maximum(0.0, need)

        if need.sum() <= 0:
            cand = rng.choice(list(remaining))
            val_idx.append(int(cand))
            remaining.remove(int(cand))
            continue

        lbl = int(np.argmax(need))
        candidates = [i for i in remaining if y[i, lbl] > 0.5]
        if not candidates:
            cand = rng.choice(list(remaining))
            val_idx.append(int(cand))
            remaining.remove(int(cand))
            continue

        scores = []
        for i in candidates:
            scores.append(float((y[i] * need).sum()))
        best = candidates[int(np.argmax(scores))]
        val_idx.append(int(best))
        remaining.remove(int(best))

    return np.array(val_idx, dtype=np.int64)


def _calibrate_thresholds_on_val(
    base_thresholds,
    val_frac=0.15,
    seed=123,
    batch_size=32,
    grid=None,
    max_val_samples=4000,
):
    if grid is None:
        grid = np.array(
            [
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
            ],
            dtype=np.float32,
        )

    rng = np.random.RandomState(seed)
    idx_all = np.arange(len(training_csv))

    y_all = _labels_to_multihot(training_csv["labels"].tolist())

    n_val = int(round(len(idx_all) * val_frac))
    n_val = max(1, n_val)
    if n_val > max_val_samples:
        n_val = max_val_samples

    val_idx = _iterative_stratified_val_indices(y_all, val_size=n_val, seed=seed)

    rng.shuffle(val_idx)

    val_df = training_csv.iloc[val_idx].reset_index(drop=True)
    y_val = y_all[val_idx]

    probs = np.zeros((len(val_df), len(training_class)), dtype=np.float32)
    images = val_df["image"].tolist()
    for i in range(0, len(images), batch_size):
        batch_names = images[i : i + batch_size]
        batch_files = [os.path.join(train_img_path, f) for f in batch_names]
        x = _load_and_preprocess_batch(batch_files)
        p = model(x, training=False).numpy().astype(np.float32)
        probs[i : i + p.shape[0], :] = p

    base_t = np.asarray(base_thresholds, dtype=np.float32).reshape(-1)
    best_t = base_t.copy()

    current_pred = (probs >= best_t[None, :]).astype(np.float32)
    current_score = _mean_sample_f1(y_val, current_pred)

    for c in range(len(training_class)):
        best_score_c = current_score
        best_thr_c = best_t[c]
        for thr in grid:
            tmp_t = best_t.copy()
            tmp_t[c] = thr
            tmp_pred = (probs >= tmp_t[None, :]).astype(np.float32)
            sc = _mean_sample_f1(y_val, tmp_pred)
            if sc > best_score_c + 1e-12:
                best_score_c = sc
                best_thr_c = float(thr)
        best_t[c] = best_thr_c
        current_score = best_score_c

    print("Threshold calibration done.")
    print("Base thresholds:", base_t.tolist())
    print("Calibrated thresholds:", best_t.tolist())
    print("Val mean sample-F1 (approx.):", current_score)
    return best_t.tolist()


def write_csv_kaggle_tags(batch_size=32):
    sample = pd.read_csv(
        "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
    )

    if weights_path is None:
        sub = sample.copy()
        sub["labels"] = "healthy"
        sub.to_csv("submission.csv", index=False)
        return

    calibrated_threshold = _calibrate_thresholds_on_val(
        threshold, val_frac=0.15, seed=123, batch_size=batch_size
    )

    images = sample["image"].tolist()
    labels_out = []

    for i in range(0, len(images), batch_size):
        batch_names = images[i : i + batch_size]
        batch_files = [os.path.join(test_img_path, f) for f in batch_names]
        x = _load_and_preprocess_batch(batch_files)
        preds = model(x, training=False).numpy()  # (B, C)
        for j in range(preds.shape[0]):
            labels_out.append(predict2strings(preds[j], threshold=calibrated_threshold))

    sub = pd.DataFrame({"image": sample["image"], "labels": labels_out})
    sub.to_csv("submission.csv", index=False)




## === cell 5
write_csv_kaggle_tags()
print(pd.read_csv("submission.csv").head())
print("Wrote submission.csv with rows:", len(pd.read_csv("submission.csv")))
