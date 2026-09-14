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

0.6613428307611109

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the protobuf auto-downgrade logic that is breaking TensorFlow 2.18 in this environment and instead rely on the already-installed compatible versions. Then I fix the invalid `model_dir` path by automatically locating a real SavedModel/Keras model within `../input/` (or fall back to a simple, valid baseline submission if no model is present), so the notebook always produces `submission.csv`. Finally, I make the prediction-to-label mapping robust to output shape/dtype and ensure the submission columns and row order match `sample_submission.csv`, which avoids silent format/ordering issues that can ruin the score.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow/Keras, which is the standard workaround for the `MessageFactory.GetPrototype` error in some Kaggle images. Then I keep your inference logic intact but make the label threshold selection more robust by auto-calibrating a per-model threshold using the training label cardinality (average number of labels per image), which should improve Mean F1 substantially versus a fixed 0.6 when the model’s output calibration differs. Finally, I ensure the test image list matches `sample_submission.csv` ordering and always write a valid `submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.24507) has done: 'You’re hitting a TensorFlow/protobuf ABI mismatch (`MessageFactory.GetPrototype`) before the code can even load data or write a submission. I fix this by removing the protobuf “python implementation” override (which is incompatible in this TF 2.18 Kaggle image) and instead force the upb C++ implementation before importing TensorFlow. Then I keep your inference and threshold-calibration logic intact, but add a safe fallback so that if the model outputs logits (not probabilities) we apply a sigmoid before thresholding, which should materially improve Mean F1 versus thresholding raw logits. Finally, I keep the submission ordering aligned to `sample_submission.csv` and always write `submission.csv` with the required columns.'
- What this solution (achieved 0.24507) has done: 'You’re crashing before any submission can be produced due to the well-known TF/protobuf `MessageFactory.GetPrototype` incompatibility; the minimal reliable fix in this Kaggle TF 2.18 environment is to force the pure-Python protobuf implementation *before* importing TensorFlow/Keras. After that, I keep your model discovery, inference, and threshold calibration logic intact, but add a small guard so prediction vectors are always correctly sliced/padded to exactly `len(dataset_labels)` (avoids silent misalignment that can tank mean F1). Finally, I ensure the test image list is taken strictly from `sample_submission.csv` order (as you intended) and always write a valid `submission.csv` with `image,labels`.'
- What this solution (achieved 0.24507) has done: 'We fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf implementation (it’s what triggers the `MessageFactory.GetPrototype` issue in this TF 2.18 Kaggle image) and instead force the default C++/upb implementation before importing TensorFlow. Then we keep your inference + threshold-calibration logic intact, but ensure the test image list is always complete and ordered exactly like `sample_submission.csv` (fallback to sample list if directory listing is inconsistent). Finally, we make the label output formatting strictly match the competition requirement (`image,labels` with no extra spaces), and always write a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'We fix the TensorFlow/protobuf crash by forcing protobuf to use the upb/C++ implementation (not the pure-Python one) before importing TensorFlow/Keras, which avoids the `MessageFactory.GetPrototype` error in this TF 2.18 environment. Then we keep your inference and threshold-calibration logic intact, but make sure we never filter out submission rows based on files present on disk (Kaggle’s hidden test set requires predicting for all rows in `sample_submission.csv`). Finally, we ensure the submission is written with the exact required columns and ordering, always producing a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'We fix the TensorFlow/protobuf crash by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow/Keras (this is the most reliable workaround for the `MessageFactory.GetPrototype` error in Kaggle-style images). Then we keep your inference + threshold calibration logic intact, but remove the “predict healthy for non-visible test images” fallback (it catastrophically harms the hidden test score, since most test images aren’t locally visible); instead we always attempt to read the image by filename and only fall back to `healthy` if the file truly can’t be read. Finally, we make the label post-processing slightly safer by guaranteeing at least one label (use `healthy`) if nothing crosses the threshold, and we still write `submission.csv` in exactly the required format and order.'
- What this solution (achieved 0.24507) has done: 'We fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf setting (it triggers the `MessageFactory.GetPrototype` failure in this TF 2.18 Kaggle image) and explicitly forcing the default C++/upb implementation before importing TensorFlow. Then we keep your existing inference, label mapping, and threshold-calibration logic intact, but add a tiny robustness guard so `decode_image` always returns a supported dtype/shape and so we don’t silently skip calibration when only a few images are readable. Finally, we ensure the submission is always produced as `submission.csv` with exactly `image,labels` columns and in the exact `sample_submission.csv` order (score-critical for Kaggle).'
- What this solution (achieved 0.24507) has done: 'The crash happens before any of your logic runs because TensorFlow 2.18 in this environment is importing an incompatible protobuf runtime; the minimal fix is to force protobuf to use the pure-Python implementation *before* importing TensorFlow/Keras. I also fix the cell numbering (your notebook starts at cell 0) so it matches the required “cell 1..N” format, but I not change your model discovery, inference, sigmoid-guard, or threshold calibration logic. Finally, I add a tiny safety guard to disable XLA (often implicated alongside protobuf issues) to make imports more stable without changing model semantics.'
- What this solution (achieved 0.24507) has done: 'The crash happens before any of your logic runs because forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` is incompatible with this Kaggle TensorFlow 2.18 image and triggers the `MessageFactory.GetPrototype` error. I remove that override and instead force protobuf to use the default C++/upb implementation (or simply not override it), which resolves the import/runtime issue while keeping your model discovery, inference, and threshold-calibration logic unchanged. I also renumber the cells to start at `cell 1` to match the required format, without changing any core modeling behavior. The rest of the code remains the same and still always write a valid `submission.csv` with the exact `image,labels` columns in `sample_submission.csv` order.'
- What this solution (achieved 0.24507) has done: 'We fix the immediate crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the default C++/upb implementation (and not the pure-Python one) before importing TensorFlow, which is the compatible setting for TF 2.18 in this environment. Then we keep your model discovery/inference/threshold-calibration logic intact, but remove the `if __name__ == "__main__":` gate so Kaggle notebooks actually execute the inference cell and always write `submission.csv`. Finally, we add a tiny safety fix to ensure the resolved test directory exists (choose the first valid candidate) and keep submission ordering exactly matching `sample_submission.csv` to avoid silent score drops.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced protobuf implementation override (it’s the root cause of the `MessageFactory.GetPrototype` error in this TF 2.18 environment) and leaving the environment to use the default protobuf runtime. Then I keep your inference and threshold-calibration logic the same, but make the probability conversion safer by applying sigmoid based on per-sample output shape and value range (avoids thresholding raw logits, which can severely hurt Mean F1). Finally, I keep the submission strictly aligned to `sample_submission.csv` order and always write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.24507) has done: 'The crash happens immediately when importing TensorFlow due to a protobuf/TensorFlow incompatibility (`MessageFactory.GetPrototype`). The minimal reliable fix in this TF 2.18 Kaggle image is to force protobuf to use the C++/upb implementation (not the pure-Python one) before importing TensorFlow/Keras. I also renumber the cells to start at `cell 1` (your current notebook starts at `cell 0`) while keeping your model discovery, inference flow, sigmoid-guard, threshold calibration, and submission-writing logic unchanged. This should run end-to-end and (by actually executing inference rather than crashing) move the score materially toward the target versus the current failing/degenerate behavior.'

# 9. Code solution

## === cell 0
import os
import sys
import glob

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "upb"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import pandas as pd
import numpy as np
import tensorflow as tf
import keras

print("Python:", sys.version)
print("TF:", tf.__version__)
print("Keras:", keras.__version__)

output_dir = "./"

_test_dir_candidates = [
    "../input/plant-pathology-2021-fgvc8/test_images/",
    "../input/test_images/",
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images/",
    "/kaggle/input/test_images/",
]
test_dir = next(
    (d for d in _test_dir_candidates if os.path.isdir(d)), _test_dir_candidates[0]
)

model_dir = "../input/model/epoch-10"
image_dims = (300, 300, 3)

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].astype(str)
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print(
    "Test dir exists:",
    os.path.isdir(test_dir),
    "Test dir:",
    test_dir,
    "Num test images (visible in notebook env):",
    len(os.listdir(test_dir)) if os.path.isdir(test_dir) else 0,
)

avg_labels_per_image = (df_labels.str.split().map(len)).mean()
print("Avg labels per image (train):", float(avg_labels_per_image))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _looks_like_savedmodel_dir(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "saved_model.pb")) or os.path.isfile(
        os.path.join(d, "saved_model.pbtxt")
    )


def _find_existing_model_path(preferred: str) -> str:
    if preferred and os.path.isdir(preferred) and _looks_like_savedmodel_dir(preferred):
        return preferred

    candidates = []

    for root in ["../input", "/kaggle/input"]:
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, "**", "saved_model.pb"), recursive=True):
            candidates.append(os.path.dirname(p))
        for p in glob.glob(
            os.path.join(root, "**", "saved_model.pbtxt"), recursive=True
        ):
            candidates.append(os.path.dirname(p))

    for root in ["../input", "/kaggle/input"]:
        if not os.path.isdir(root):
            continue
        candidates += glob.glob(os.path.join(root, "**", "*.keras"), recursive=True)
        candidates += glob.glob(os.path.join(root, "**", "*.h5"), recursive=True)

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    if not uniq:
        return ""

    for c in uniq:
        if "epoch" in os.path.basename(c) or "model" in c.lower():
            return c

    return uniq[0]


def build_inference_model_any(model_path: str):
    if not model_path:
        return None

    if os.path.isfile(model_path) and (
        model_path.endswith(".keras") or model_path.endswith(".h5")
    ):
        m = keras.models.load_model(model_path, compile=False)
        return m

    if os.path.isdir(model_path) and _looks_like_savedmodel_dir(model_path):
        endpoints_to_try = ["serving_default", "serve", "call"]
        last_err = None
        for ep in endpoints_to_try:
            try:
                layer = keras.layers.TFSMLayer(model_path, call_endpoint=ep)
                inp = keras.Input(shape=image_dims, dtype=tf.float32, name="image")
                out = layer(inp)
                if isinstance(out, dict):
                    out = list(out.values())[0]
                return keras.Model(inp, out)
            except Exception as err:
                last_err = err

        sm = tf.saved_model.load(model_path)
        sig_keys = list(getattr(sm, "signatures", {}).keys())
        if not sig_keys:
            raise RuntimeError(
                f"No signatures found in SavedModel at {model_path}. Last endpoint error: {last_err!r}"
            )
        ep = sig_keys[0]
        layer = keras.layers.TFSMLayer(model_path, call_endpoint=ep)
        inp = keras.Input(shape=image_dims, dtype=tf.float32, name="image")
        out = layer(inp)
        if isinstance(out, dict):
            out = list(out.values())[0]
        return keras.Model(inp, out)

    return None


resolved_model_path = _find_existing_model_path(model_dir)
print("Requested model_dir:", model_dir)
print(
    "Resolved model path:",
    resolved_model_path if resolved_model_path else "(none found)",
)

model = build_inference_model_any(resolved_model_path)
if model is None:
    print(
        "WARNING: No loadable model found under ../input. Will write a valid baseline submission (all 'healthy')."
    )
else:
    try:
        model.summary()
    except Exception:
        pass

sample_sub = pd.read_csv(sample_sub_path)
images_path_list = sample_sub["image"].astype(str).tolist()


def load_and_preprocess(img_name: str):
    input_img = tf.io.read_file(os.path.join(test_dir, img_name))
    image = tf.io.decode_image(contents=input_img, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([None, None, 3])
    tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
    return tf.expand_dims(tensor_image, axis=0)


fixed_threshold = 0.6  # original
threshold = fixed_threshold


def _as_2d_logits(x):
    if isinstance(x, dict):
        x = list(x.values())[0]
    x = tf.convert_to_tensor(x)
    if x.shape.rank is None:
        x = tf.reshape(x, [tf.shape(x)[0], -1])
    elif x.shape.rank == 1:
        x = tf.expand_dims(x, 0)
    elif x.shape.rank > 2:
        x = tf.reshape(x, [tf.shape(x)[0], -1])
    return x


def _to_probabilities(x2d: tf.Tensor) -> tf.Tensor:
    x2d = tf.cast(x2d, tf.float32)
    x_min = tf.reduce_min(x2d)
    x_max = tf.reduce_max(x2d)
    return tf.cond(
        tf.logical_or(x_min < 0.0, x_max > 1.0),
        lambda: tf.sigmoid(x2d),
        lambda: x2d,
    )


def _predvec_to_num_classes(pred_vec_1d: np.ndarray, num_classes: int) -> np.ndarray:
    pred_vec_1d = np.asarray(pred_vec_1d, dtype=np.float32).reshape(-1)
    if pred_vec_1d.shape[0] >= num_classes:
        return pred_vec_1d[:num_classes]
    pad = np.zeros((num_classes - pred_vec_1d.shape[0],), dtype=np.float32)
    return np.concatenate([pred_vec_1d, pad], axis=0)


if model is not None and os.path.isdir(test_dir):
    visible_images = os.listdir(test_dir)
    if len(visible_images) > 0:
        calib_n = min(256, len(visible_images))
        preds_collect = []
        for name in visible_images[:calib_n]:
            try:
                images = load_and_preprocess(name)
                tv = _as_2d_logits(model(images, training=False))
                tv = _to_probabilities(tv)
                pv = tv[0].numpy()
                pv = _predvec_to_num_classes(pv, len(dataset_labels))
                preds_collect.append(pv)
            except Exception:
                continue

        if len(preds_collect) >= 8:
            P = np.vstack(preds_collect)
            target_k = float(avg_labels_per_image)
            target_frac = np.clip(
                target_k / max(1.0, float(len(dataset_labels))), 1e-6, 1 - 1e-6
            )
            flat = P.reshape(-1)
            t = float(np.quantile(flat, 1.0 - target_frac))
            threshold = t
            print(
                f"Calibrated threshold from train cardinality: {threshold:.6f} (fixed was {fixed_threshold})"
            )
        else:
            print(
                "Too few readable images for calibration; using fixed threshold:",
                fixed_threshold,
            )
    else:
        print(
            "No visible test images found for calibration; using fixed threshold:",
            fixed_threshold,
        )

values = []
for name in images_path_list:
    if model is None:
        classes_img = "healthy"
    else:
        try:
            images = load_and_preprocess(name)
            tv = _as_2d_logits(model(images, training=False))
            tv = _to_probabilities(tv)
            preds = tv[0].numpy()
            preds = (
                _predvec_to_num_classes(preds, len(dataset_labels))
                .astype(float)
                .tolist()
            )

            index_values = [j for j, v in enumerate(preds) if v > threshold]
            classes_img_list = [dataset_labels[j] for j in index_values]
            classes_img = (
                "healthy" if len(classes_img_list) == 0 else " ".join(classes_img_list)
            )
        except Exception:
            classes_img = "healthy"

    values.append([name, str(classes_img).strip()])

csv_pd = pd.DataFrame(values, columns=["image", "labels"])

out = sample_sub[["image"]].copy()
out = out.merge(csv_pd, on="image", how="left")
out["labels"] = out["labels"].fillna("healthy").astype(str).str.strip()
out.loc[out["labels"].eq(""), "labels"] = "healthy"

out_path = os.path.join(output_dir, "submission.csv")
out.to_csv(out_path, index=False)
print("Wrote submission.csv with shape:", out.shape, "to", out_path)
print(out.head())
