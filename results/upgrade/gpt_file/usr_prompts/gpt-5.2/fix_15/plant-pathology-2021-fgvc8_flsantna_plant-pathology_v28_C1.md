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

0.7845060018467225

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I fix the test image listing bug that incorrectly includes the nested `test_images/` directory, which causes `tf.io.read_file` to try to read a directory and crash. I make the file collection robust by filtering to actual image files and by auto-resolving the correct `test_images` path if the provided one is missing or points to an unexpected structure. I also keep the SavedModel search/wrapping logic intact but make it safe in Kaggle by avoiding the protobuf downgrade (which can break TF 2.18) and by ensuring inference input/output handling is stable. These changes are score-neutral except that they allow the notebook to run end-to-end and generate a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'I fix the immediate TensorFlow/protobuf runtime crash by removing the protobuf-import probe that triggers the `MessageFactory.GetPrototype` incompatibility in this Kaggle environment. Then I keep your overall inference pipeline intact but add a minimal, metric-aligned improvement: tune the multi-label threshold on a small validation split using the same mean F1 logic (without changing the model), which should move the score substantially upward from the current overly-high fixed threshold. Finally, I make the preprocessing scale robust by trying both `[0..255]` and `[0..1]` input scaling (picked by validation F1), and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf crash happening at import time by avoiding the incompatible protobuf message-factory path and by forcing TensorFlow to use the Python protobuf implementation (a common Kaggle TF2.18 workaround). Then I keep your inference logic intact but make it robust to different SavedModel output shapes by applying a sigmoid only when outputs look like logits (to improve mean-F1 thresholding without changing the model). Finally, I ensure the test/train directory resolution and submission writing stay the same, so the notebook runs end-to-end and produces a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf import crash by removing the environment forcing of the Python protobuf implementation, which is incompatible with this TF 2.18 runtime and triggers the `MessageFactory.GetPrototype` error. Then I keep your SavedModel inference and threshold/scale tuning logic intact, only adding small robustness guards so the notebook always finds the correct data directories and always produces a valid `submission.csv`. These changes should both unblock execution and improve score substantially versus the current run (which is failing before inference), because the tuned threshold/scale and sigmoid-on-logits behavior actually be applied. Finally, I ensure the submission has the exact required columns and space-delimited labels.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow import crash by removing the incompatible protobuf environment override so TF 2.18 can import cleanly in this Kaggle runtime. Then I renumber the cells and ensure TensorFlow is imported before any functions that reference `tf`, which also fixes the downstream `NameError` chain that prevented submission generation. Finally, I keep your SavedModel inference + threshold/scale tuning logic intact, only making the script run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.24507) has done: 'I remove the protobuf environment override that forces the C++ protobuf implementation, because it causes TensorFlow 2.18 to fail importing in this Kaggle environment. Then I renumber cells and ensure TensorFlow is successfully imported before any functions that reference `tf`, which fixes the downstream `NameError` chain and allows the pipeline to run end-to-end. I keep your SavedModel inference + threshold/scale tuning logic intact, only adding a small safety fallback if TF still cannot import so a valid `submission.csv` is always produced. These changes are primarily correctness/stability fixes; the tuned threshold/scale (already in your logic) is what should move score upward once the script actually runs.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running at all by forcing protobuf to use the C++ implementation *before* importing TensorFlow (this is the stable setting for TF 2.18 on Kaggle). Then I keep your SavedModel inference, sigmoid-if-logits behavior, and threshold/scale tuning intact, only adding small robustness so the model output is converted safely to `numpy` and class lists are correctly aligned. Finally, I ensure the pipeline always writes a valid `submission.csv` with the exact required columns and space-delimited labels.'
- What this solution (achieved 0.24507) has done: 'Your score gap is large (0.245 → target 0.7845), and the most likely cause is label-set mismatch: your `dataset_labels` is derived from `train.csv` dummies (12 classes including `complex`), but the competition’s required submission label-set is the fixed 6 classes, so predicting extra labels (especially `complex`) can heavily hurt mean-F1. I keep your SavedModel inference, sigmoid-if-logits, and threshold/scale tuning logic intact, but constrain both tuning and final label mapping to the official 6 labels in the exact order used by the competition. Additionally, I enforce the “at least one label” rule by selecting the max-probability class when nothing passes the threshold (instead of defaulting to `healthy`), which is a minimal post-processing change aligned with mean-F1.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.245) is far below the target (0.7845), so we should push performance up with minimal risk while keeping your SavedModel + threshold/scale tuning approach intact. The biggest likely remaining issue is that your validation tuning uses macro-F1 but the competition metric is *sample-wise* mean F1 (per image, then averaged), so the chosen threshold can be badly miscalibrated for the leaderboard. I change only the tuning metric to match Kaggle’s mean F1 (sample-wise) and slightly densify the threshold grid around typical operating points; inference, model, preprocessing, and submission formatting stay the same. This should move the tuned threshold toward a much better operating point and raise the public score toward your target without changing core modeling logic.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf message-factory path by forcing the pure-Python protobuf implementation *before* importing TensorFlow (a common TF 2.18 + protobuf 6 workaround on Kaggle). Then I keep your SavedModel-based inference and your validation-based threshold/scale tuning logic intact, only adding a small safety fallback so the script still produces a valid `submission.csv` even if TF/model loading fails for any reason. Finally, I ensure test/train directory resolution and submission formatting remain correct (space-delimited labels, correct columns), so the notebook runs end-to-end and can achieve a much higher mean-F1 once TF loads successfully.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
import numpy as np
import pandas as pd

TF_AVAILABLE = True
try:
    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print(
        "WARNING: TensorFlow failed to import. Will write a safe baseline submission. Error:\n",
        repr(e),
    )

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conv2de01/epoch-1"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")

OFFICIAL_LABELS = [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
    "frog_eye_leaf_spot",
    "powdery_mildew",
]
dataset_labels = OFFICIAL_LABELS[:]  # keep variable name used elsewhere

print("Using official Num classes:", len(dataset_labels))
print("Classes:", dataset_labels)




## === cell 2
def _is_savedmodel_dir(p: str) -> bool:
    return os.path.isdir(p) and (
        os.path.exists(os.path.join(p, "saved_model.pb"))
        or os.path.exists(os.path.join(p, "saved_model.pbtxt"))
    )


def _find_any_savedmodel(start_path: str) -> str:
    """
    Search within ../input for a directory containing saved_model.pb when the provided model_dir
    is not valid in this Kaggle environment. Preserves the same "use a SavedModel" inference logic.
    """
    if start_path and _is_savedmodel_dir(start_path):
        return start_path

    candidates_roots = []
    try:
        parent = os.path.abspath(os.path.join(start_path, os.pardir))
        if os.path.isdir(parent):
            candidates_roots.append(parent)
    except Exception:
        pass
    if os.path.isdir("../input"):
        candidates_roots.append("../input")

    seen = set()
    for root in candidates_roots:
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        for dirpath, dirnames, filenames in os.walk(root):
            if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
                return dirpath
    return ""


def _wrap_savedmodel_as_layer(savedmodel_path: str):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available; cannot wrap SavedModel.")
    return tf.keras.layers.TFSMLayer(savedmodel_path, call_endpoint="serving_default")


def _call_infer(infer_layer, images_batch):
    """
    Try common SavedModel input patterns (tensor first, then dict with common keys).
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available; cannot run inference.")
    try:
        out = infer_layer(images_batch, training=False)
        return out
    except Exception:
        pass

    for key in ("inputs", "input_1", "image", "images", "x"):
        try:
            out = infer_layer({key: images_batch}, training=False)
            return out
        except Exception:
            continue

    raise RuntimeError(
        "Could not call inference layer with tensor or common dict keys."
    )


def _resolve_test_dir(p: str) -> str:
    """
    Resolve to a directory that actually contains image files (handles nested test_images/test_images).
    """
    candidates = []
    if p:
        candidates.append(p)
        candidates.append(os.path.join(p, "test_images"))
    candidates.extend(
        [
            "../input/plant-pathology-2021-fgvc8/test_images",
            "../input/plant-pathology-2021-fgvc8/test_images/test_images",
            "../input/test_images",
            "../input/test_images/test_images",
        ]
    )

    def has_img(dirpath: str) -> bool:
        if not os.path.isdir(dirpath):
            return False
        for fn in os.listdir(dirpath):
            full = os.path.join(dirpath, fn)
            if os.path.isfile(full) and fn.lower().endswith((".jpg", ".jpeg", ".png")):
                return True
        return False

    for c in candidates:
        if has_img(c):
            return c

    return p


def _list_image_files(dirpath: str):
    """
    Filter out directories and non-image files.
    """
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"test_dir not found or not a directory: {dirpath}")

    files = []
    for fn in os.listdir(dirpath):
        full = os.path.join(dirpath, fn)
        if os.path.isfile(full) and fn.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(fn)
    files.sort()
    return files


def _resolve_train_dir() -> str:
    """
    Resolve train_images directory robustly across Kaggle path layouts.
    """
    candidates = [
        "../input/plant-pathology-2021-fgvc8/train_images",
        "../input/plant-pathology-2021-fgvc8/train_images/train_images",
        "../input/train_images",
        "../input/train_images/train_images",
    ]
    for c in candidates:
        if os.path.isdir(c):
            for fn in os.listdir(c):
                if fn.lower().endswith(".jpg") and os.path.isfile(os.path.join(c, fn)):
                    return c
    raise FileNotFoundError(
        "Could not resolve train_images directory from known candidates."
    )


def _extract_pred_tensor(pred_out):
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not available; cannot extract prediction tensor."
        )
    if isinstance(pred_out, dict):
        if "outputs" in pred_out:
            t = pred_out["outputs"]
        elif "predictions" in pred_out:
            t = pred_out["predictions"]
        else:
            t = next(iter(pred_out.values()))
    else:
        t = pred_out
    t = tf.convert_to_tensor(t)
    if len(t.shape) == 1:
        t = tf.expand_dims(t, axis=0)
    return t


def _maybe_apply_sigmoid(pred):
    """
    Metric-aligned improvement (kept): apply sigmoid if outputs look like logits (outside [0,1]).
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available; cannot apply sigmoid logic.")
    pred = tf.convert_to_tensor(pred)
    min_v = tf.reduce_min(pred)
    max_v = tf.reduce_max(pred)
    if (min_v < -1e-3) or (max_v > 1.0 + 1e-3):
        return tf.math.sigmoid(pred)
    return pred




## === cell 3
def _load_and_preprocess_image(img_path: str):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available; cannot load images.")
    input_img = tf.io.read_file(img_path)
    image = tf.io.decode_image(
        contents=input_img,
        channels=3,
        dtype=tf.dtypes.float32,
        expand_animations=False,
    )
    image = tf.image.resize(image, [image_dims[0], image_dims[1]])
    return image  # float32 in [0,1]


def _mean_f1_samples(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Competition metric is mean F1-score computed per sample (image), then averaged.
    """
    eps = 1e-9
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    tp = (y_true & y_pred).sum(axis=1)
    fp = ((1 - y_true) & y_pred).sum(axis=1)
    fn = (y_true & (1 - y_pred)).sum(axis=1)

    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


def _tune_threshold_and_scale(
    infer_layer, train_df: pd.DataFrame, train_img_dir: str, n_val: int = 512
):
    """
    Select threshold and whether to scale by 255 using a small validation subset.
    Also ensures y_true columns match the official label set and the model output dimension.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available; cannot tune threshold/scale.")

    n_val = min(n_val, len(train_df))
    val_df = train_df.sample(n=n_val, random_state=SEED).reset_index(drop=True)

    preds_01 = []
    preds_255 = []
    batch_size = 16

    for start in range(0, n_val, batch_size):
        batch = val_df.iloc[start : start + batch_size]
        imgs = []
        for fn in batch["image"].tolist():
            img_path = os.path.join(train_img_dir, fn)
            imgs.append(_load_and_preprocess_image(img_path))
        x = tf.stack(imgs, axis=0)  # [0,1]

        out_01 = _extract_pred_tensor(_call_infer(infer_layer, x))
        out_01 = _maybe_apply_sigmoid(out_01)
        out_255 = _extract_pred_tensor(_call_infer(infer_layer, x * 255.0))
        out_255 = _maybe_apply_sigmoid(out_255)

        preds_01.append(out_01.numpy())
        preds_255.append(out_255.numpy())

    p01 = np.concatenate(preds_01, axis=0)
    p255 = np.concatenate(preds_255, axis=0)

    used_labels = dataset_labels[:]  # the official 6
    y_true = (
        val_df["labels"]
        .str.get_dummies(sep=" ")
        .reindex(columns=used_labels, fill_value=0)
        .values.astype(np.int32)
    )

    thresholds = (
        [0.05, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20]
        + [0.22, 0.24, 0.25, 0.26, 0.28, 0.30]
        + [0.32, 0.34, 0.35, 0.36, 0.38, 0.40]
        + [0.42, 0.45, 0.48, 0.50]
        + [0.55, 0.60, 0.65, 0.70]
    )

    if int(p01.shape[1]) < len(used_labels):
        print(
            f"Warning: model outputs only {int(p01.shape[1])} dims (< {len(used_labels)} official labels). "
            "Will tune on the available dims and map those dims onto the first labels."
        )
        used_labels = used_labels[: int(p01.shape[1])]
        y_true = y_true[:, : len(used_labels)]

    best = {"f1": -1.0, "thr": 0.5, "scale": "255", "labels": used_labels}
    for scale_name, p in (("01", p01), ("255", p255)):
        p = p[:, : len(used_labels)]
        for thr in thresholds:
            y_pred = (p > thr).astype(np.int32)
            f1 = _mean_f1_samples(y_true, y_pred)
            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "thr": float(thr),
                    "scale": scale_name,
                    "labels": used_labels,
                }

    print(
        f"Validation tuning on n={n_val}: best_mean_sample_f1={best['f1']:.5f} thr={best['thr']} scale={best['scale']} labels={best['labels']}"
    )
    return best["thr"], best["scale"], best["labels"]




## === cell 4
if __name__ == "__main__":
    test_dir_resolved = _resolve_test_dir(test_dir)
    print("Resolved test_dir:", test_dir_resolved)

    images_path_list = _list_image_files(test_dir_resolved)
    if len(images_path_list) == 0:
        raise RuntimeError(f"No image files found under: {test_dir_resolved}")

    infer_layer = None
    thr, scale_mode, used_labels = 0.7, "255", dataset_labels

    if TF_AVAILABLE:
        try:
            resolved_model_dir = _find_any_savedmodel(model_dir)
            if resolved_model_dir:
                print("Using SavedModel at:", resolved_model_dir)
                infer_layer = _wrap_savedmodel_as_layer(resolved_model_dir)

                train_img_dir = _resolve_train_dir()
                thr, scale_mode, used_labels = _tune_threshold_and_scale(
                    infer_layer, data_set, train_img_dir, n_val=512
                )
            else:
                print(
                    "Warning: No SavedModel found under the expected paths. "
                    "Will generate a valid submission with a safe baseline ('healthy')."
                )
        except Exception as e:
            infer_layer = None
            print(
                "WARNING: Model loading/tuning failed; will write baseline submission. Error:\n",
                repr(e),
            )
    else:
        print(
            "Warning: TensorFlow unavailable. Will generate a valid submission with a safe baseline ('healthy')."
        )

    def test_on_sub(index):
        img_path = os.path.join(test_dir_resolved, images_path_list[index])
        image = _load_and_preprocess_image(img_path)
        name_jpg = images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(image, axis=0)

    values = []

    for idx in range(len(images_path_list)):
        name = images_path_list[idx]
        if infer_layer is None:
            values.append([name, "healthy"])
            continue

        name, images = test_on_sub(index=idx)
        images_in = images * 255.0 if scale_mode == "255" else images

        pred_out = _call_infer(infer_layer, images_in)
        test_values = _extract_pred_tensor(pred_out)
        test_values = _maybe_apply_sigmoid(test_values)

        test_np = test_values.numpy()
        n_use = min(int(test_np.shape[1]), len(used_labels))
        probs = test_np[0, :n_use]
        label_names = used_labels[:n_use]

        index_values = [i for i, v in enumerate(probs) if v > thr]
        if len(index_values) == 0:
            index_values = [int(np.argmax(probs))]

        classes_img = " ".join([str(label_names[i]) for i in index_values])
        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape)
    print("Saved to:", out_path)
    print(csv_pd.head())
