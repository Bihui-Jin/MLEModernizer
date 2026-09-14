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

0.8637050468419462

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the runtime crash caused by importing `tensorflow_hub` (it triggers a protobuf incompatibility in this Kaggle environment), since your script doesn’t actually use TF-Hub. Then I make model loading robust by using `compile=False` and providing a safe `custom_objects` dict so `.h5` models saved with TF-Hub/Keras layers can still be deserialized. Finally, I ensure `sample_sub` is always defined (so cell 29 can run) and add a fallback that creates a valid `submission.csv` even if the external model files aren’t present, so you always get a valid CSV output end-to-end.'
- What this solution (achieved 0.05531) has done: 'We fix the immediate runtime crash by avoiding TensorFlow import-time protobuf issues: set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing `tensorflow` and remove any `tensorflow_hub` dependency (your code doesn’t need it for inference). Then we correct the biggest scoring bug: the models were trained with their own preprocessing, but inference currently feeds raw 0–255 RGB, which makes predictions near-random; we apply the appropriate `tf.keras.applications.*.preprocess_input` for each model before calling it (same ensemble logic, just correct inputs). Finally, we make image loading robust (RGB conversion + file existence guard) and still always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.05531) has done: 'The crash happens before any modeling because TensorFlow import triggers a protobuf API mismatch (`MessageFactory.GetPrototype`) in this environment. The minimal robust fix is to ensure the runtime uses a compatible protobuf implementation by forcing the pure-Python protobuf backend and (critically) using TensorFlow only after that environment variable is set; additionally, we avoid importing TensorFlow at module import time if it fails, so the notebook can still produce a valid submission. Since your current score is extremely low versus the target, the biggest legitimate accuracy fix is to keep the same ensemble logic but ensure inference preprocessing matches the models’ expected inputs (ResNet50/VGG19 preprocess + correct image size), and to fail loudly if models are missing instead of silently predicting a constant label for most images. The patch below makes TensorFlow import stable, loads the two `.h5` models, applies the correct preprocessing, and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.05531) has done: 'You’re crashing on TensorFlow import due to an old protobuf runtime in this environment; the minimal reliable fix is to force the pure-Python protobuf backend *and* ensure we import `protobuf` before `tensorflow`, plus clear conflicting protobuf env vars that can trigger the C++ implementation. Once TF imports, we keep your exact ensemble logic, but make the image tensors explicitly `float32` and use `np.expand_dims` instead of `tf.reshape` to avoid any shape/type edge cases and keep preprocessing consistent. Finally, we keep the fallback path but make it explicit and always write `/kaggle/working/submission.csv` with the required columns and row order from `sample_submission.csv`, so you always get a valid submission.'
- What this solution (achieved 0.05531) has done: 'We fix the root runtime issue preventing TensorFlow from importing by upgrading the `protobuf` package to a TensorFlow-compatible version *inside the notebook session* before importing TF (this is the direct cause of the `MessageFactory.GetPrototype` error). Then we keep your exact ensemble/inference logic the same, but remove the now-unnecessary protobuf “python backend” forcing that can actually keep you on the problematic codepath. Finally, we make the model paths and data paths robust (without changing semantics) and still always write `/kaggle/working/submission.csv` with the required columns and row order.'
- What this solution (achieved 0.05531) has done: 'I fix the immediate crash in cell 1 by removing the invalid import of `google.protobuf.__version__` (protobuf exposes the version as `google.protobuf.__version__`, not as a submodule). I also make the protobuf install step non-fatal/offline-safe so the notebook still runs in Kaggle even if pip install is blocked, while keeping your TensorFlow/model inference logic unchanged. Finally, I keep the same preprocessing + ensemble, but ensure the script always reaches the CSV write step and produces `/kaggle/working/submission.csv` in the required format.'
- What this solution (achieved 0.61099) has done: 'Your current score suggests the submission is dominated by the fallback constant label (or heavily erroring per-image), which yields near-random accuracy on this 5-class task. The smallest score-improving change is to stop doing per-image `.predict()` (slow and more error-prone) and instead run a batched `tf.data` inference pipeline that deterministically loads/decodes images from `sample_submission.csv`, applies the exact same ResNet50/VGG19 preprocessing, and ensemblesthe same way. To avoid the catastrophic “all fallback” case when external `.h5` files aren’t available, we keep your architecture/inference semantics but switch the fallback to a simple, legitimate prior: predict the most frequent class from `train.csv` (still no leakage from test labels), which should move the score substantially toward your target. The script still always writes `/kaggle/working/submission.csv` in the required format.'
- What this solution (achieved 0.61099) has done: 'We keep your same ensemble (ResNet50 + VGG19 with the same weights) but fix a likely score killer: the current inference path converts each TF batch to NumPy (`batch_imgs.numpy()`), which forces eager CPU copies and can silently slow/derail throughput; we instead keep everything in TensorFlow and feed tensors directly to the models. We also ensure the resize matches Keras application expectations by resizing to 256 then center-cropping to 224 (a common training-time pattern that improves accuracy without changing the model), while still using the same preprocess_input functions and no change to the model. Finally, we make the dataset deterministic (fixed ordering, `deterministic=True`) so predictions align perfectly with `sample_submission.csv` order and avoid any rare misalignment that would tank accuracy.'
- What this solution (achieved 0.61099) has done: 'Your current gap to the target is large (0.61099 → 0.8637), so we should improve accuracy with minimal, low-risk changes that don’t alter your ensemble or models. The most likely score-killer here is a subtle input mismatch: your pipeline center-crops using `resize_with_crop_or_pad`, which can pad (black borders) rather than true center-crop, and you also never explicitly set `training=False` on `preprocess_input` inputs’ dtype/range beyond float32. I switch the crop step to a true center-crop via `tf.image.central_crop` after resizing (a common expected eval-time transform), keep your same preprocessors/alphas/models, and add a safety check that the number/order of predictions exactly matches `sample_submission.csv` to prevent any accidental misalignment. These are minimal changes focused on improving accuracy without changing the core approach.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.8637), so we should increase accuracy with minimal, low-risk changes that keep your exact ensemble/models intact. The most likely remaining “silent score killer” is that the two base models were trained on different input scaling (ResNet50 expects `preprocess_input` on 0–255, while VGG19 commonly expects that too, but if either model was trained on [0,1] or [-1,1] scaled inputs, the current preprocessing can underperform); to improve without changing architecture, we ensemble *two* inference variants per model (raw Keras preprocess and simple /255 scaling) and average their logits/probs, which is a calibration-only inference tweak. We also add `tf.io.decode_jpeg(..., dct_method="INTEGER_FAST")` for speed/stability and ensure all paths exist up-front to avoid any accidental per-sample failure modes. The submission writing and row order remain exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'We keep your exact ensemble/models and tf.data batched inference, but remove the “double-preprocess averaging” (/255 variant) which is very likely miscalibrating at least one of the two models and dragging accuracy down. We also switch from resizing→central_crop→resize (which changes scale twice) to a single deterministic resize to 224×224 to better match typical Keras application inference, without changing architecture or training. Finally, we add a tiny test-time augmentation that averages predictions over the original and horizontally flipped image (same logits, same argmax), which often gives a modest accuracy lift with minimal risk and compute overhead. The submission format/path remains identical and still falls back safely if TF/models are unavailable.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.8637), so we should increase accuracy with minimal, low-risk changes that keep your exact ensemble/models intact. The biggest likely issue is an input mismatch: the test images are being resized to 224×224 directly, but these cassava models are commonly trained with a “resize a bit bigger then center-crop to 224” eval pipeline; switching to that preserves your core logic while improving signal. I also make the image decode more robust by cropping to 3 channels after decoding (handles odd JPEGs) and ensure we never crash on a single bad image by using a safe TF `ignore_errors()` path (still deterministic row order because we build paths from sample submission). Submission writing stays identical and always produces `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, so we should improve accuracy with minimal, low-risk fixes that keep your same ensemble and models. The biggest likely silent score-killer is that `ignore_errors()` can drop samples during decoding/reading, which shifts predictions out of alignment with `sample_submission.csv` (tanking accuracy); we instead make decoding “safe” and always return a valid image tensor so row order/count is preserved. Next, we use the model’s own expected preprocessing if the `.h5` already includes it (common in Kaggle), by feeding the same `batch_imgs` into both models and only applying the application preprocess as a fallback when needed, avoiding double-preprocessing. Finally, we keep your TTA and alphas identical, and still write `/kaggle/working/submission.csv` with exact sample order.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
OUT_PATH = "/kaggle/working/submission.csv"

print("Python:", sys.version)
print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TRAIN_CSV_PATH exists:", os.path.exists(TRAIN_CSV_PATH))




## === cell 1
def _pip_install(pkg: str):
    print(f"Installing (best-effort): {pkg}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        return True
    except Exception as e:
        print(f"WARNING: pip install failed for {pkg}: {type(e).__name__}: {e}")
        return False


_pip_install("protobuf==3.20.3")

try:
    import google.protobuf  # noqa: F401

    print("protobuf version:", getattr(google.protobuf, "__version__", "unknown"))
except Exception as e:
    print(f"WARNING: protobuf import failed: {type(e).__name__}: {e}")



## === cell 2
tf = None
keras = None

try:
    import tensorflow as tf  # noqa: E402
    from tensorflow import keras  # noqa: E402

    tf.random.set_seed(42)
    np.random.seed(42)
    print("TF version:", tf.__version__)
except Exception as e:
    print("WARNING: TensorFlow failed to import; will fall back to prior predictions.")
    print(f"TF import error: {type(e).__name__}: {e}")



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert {"image_id", "label"}.issubset(sample_sub.columns)
print("sample_sub shape:", sample_sub.shape)
print(sample_sub.head())

fallback_label = 0
if os.path.exists(TRAIN_CSV_PATH):
    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH, usecols=["label"])
        fallback_label = int(train_df["label"].value_counts().idxmax())
    except Exception as e:
        print(
            f"WARNING: failed to compute fallback_label from train.csv: {type(e).__name__}: {e}"
        )
print("fallback_label (train majority class if available):", fallback_label)




## === cell 4
def safe_load_model(path):
    if tf is None:
        return None
    if not os.path.exists(path):
        print(f"WARNING: Model file not found: {path}")
        return None
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"WARNING: Failed to load model at {path}: {type(e).__name__}: {e}")
        return None


MODEL_DIR_CANDIDATES = [
    "../input/f-models",
    "/kaggle/input/f-models",
]
MODEL_DIR = None
for d in MODEL_DIR_CANDIDATES:
    if os.path.exists(d):
        MODEL_DIR = d
        break
if MODEL_DIR is None:
    MODEL_DIR = MODEL_DIR_CANDIDATES[0]

model1 = safe_load_model(os.path.join(MODEL_DIR, "ResNet50_f.h5"))
model2 = safe_load_model(os.path.join(MODEL_DIR, "VGG19_f.h5"))
model3 = safe_load_model(
    os.path.join(MODEL_DIR, "MobileNetV3L_f.h5")
)  # parity (unused)

norm_constant = 0.87 + 0.91
alpha_1 = 0.91 / norm_constant
alpha_2 = 0.87 / norm_constant

print(
    "Loaded models:",
    {
        "model1": model1 is not None,
        "model2": model2 is not None,
        "model3": model3 is not None,
    },
)
print("alphas:", alpha_1, alpha_2)



## === cell 5
if tf is not None:
    from tensorflow.keras.applications.resnet50 import (
        preprocess_input as resnet50_preprocess,
    )
    from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess

    target_size = (224, 224)
    resize_size = (256, 256)
    batch_size = 32

    image_ids = sample_sub["image_id"].astype(str).values
    paths = np.array([os.path.join(TEST_IMG_DIR, x) for x in image_ids], dtype=object)

    exists_mask = np.array([os.path.exists(p) for p in paths], dtype=bool)
    n_missing = int((~exists_mask).sum())
    if n_missing:
        print(
            f"WARNING: missing test images: {n_missing} (will fallback for those rows)"
        )

    def _safe_load_decode_resize_crop(path):
        path = tf.convert_to_tensor(path, dtype=tf.string)

        def _read_decode():
            img_bytes = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, resize_size, method="bilinear", antialias=True)
            img = tf.image.central_crop(
                img, float(target_size[0]) / float(resize_size[0])
            )
            img = tf.image.resize(img, target_size, method="bilinear", antialias=True)
            return img

        def _zeros():
            return tf.zeros([target_size[0], target_size[1], 3], dtype=tf.float32)

        img = tf.cond(tf.io.gfile.exists(path), _read_decode, _zeros)
        return img

    preds = None
    used_fallback = 0
    missing_or_error = 0

    if (model1 is None) or (model2 is None):
        preds = np.full(shape=(len(image_ids),), fill_value=fallback_label, dtype=int)
        used_fallback = len(image_ids)
        missing_or_error = len(image_ids)
        print("Models unavailable; using fallback for all rows.")
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

        options = tf.data.Options()
        options.experimental_deterministic = True
        ds = ds.with_options(options)

        ds = (
            ds.map(
                _safe_load_decode_resize_crop,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        def _needs_app_preprocess(m):
            try:
                dummy = tf.zeros(
                    [1, target_size[0], target_size[1], 3], dtype=tf.float32
                )
                y = m(dummy, training=False)
                _ = tf.reduce_sum(y)
                return False
            except Exception:
                return True

        model1_use_app_pre = _needs_app_preprocess(model1)
        model2_use_app_pre = _needs_app_preprocess(model2)
        print(
            "model1_use_app_preprocess:",
            model1_use_app_pre,
            "| model2_use_app_preprocess:",
            model2_use_app_pre,
        )

        all_preds = []
        for batch_imgs in ds:
            flipped = tf.image.flip_left_right(batch_imgs)

            if model1_use_app_pre:
                x1 = resnet50_preprocess(batch_imgs)
                x1_f = resnet50_preprocess(flipped)
            else:
                x1 = batch_imgs
                x1_f = flipped

            if model2_use_app_pre:
                x2 = vgg19_preprocess(batch_imgs)
                x2_f = vgg19_preprocess(flipped)
            else:
                x2 = batch_imgs
                x2_f = flipped

            p1 = model1(x1, training=False)
            p2 = model2(x2, training=False)
            p1_f = model1(x1_f, training=False)
            p2_f = model2(x2_f, training=False)

            p = (alpha_1 * (p1 + p1_f) * 0.5) + (alpha_2 * (p2 + p2_f) * 0.5)
            all_preds.append(
                tf.argmax(p, axis=1, output_type=tf.int32).numpy().astype(int)
            )

        preds = np.concatenate(all_preds, axis=0)

        if preds.shape[0] != len(image_ids):
            print(
                f"WARNING: Prediction length mismatch. got {preds.shape[0]} preds for {len(image_ids)} images; "
                f"will force alignment by padding/cropping with fallback_label."
            )
            pad_n = len(image_ids) - preds.shape[0]
            if pad_n > 0:
                preds = np.concatenate(
                    [preds, np.full((pad_n,), fallback_label, dtype=int)], axis=0
                )
            else:
                preds = preds[: len(image_ids)]

        missing_idx = np.where(~exists_mask)[0]
        if missing_idx.size > 0:
            preds[missing_idx] = fallback_label
            used_fallback = int(missing_idx.size)
            missing_or_error = int(missing_idx.size)

        print(
            f"Inference done. missing_files={missing_or_error} fallback_rows={used_fallback}"
        )
else:
    preds = np.full(shape=(len(sample_sub),), fill_value=fallback_label, dtype=int)
    used_fallback = len(sample_sub)
    missing_or_error = len(sample_sub)
    print("TF unavailable; using fallback for all rows.")

my_submission = pd.DataFrame(
    {
        "image_id": sample_sub.image_id.astype(str).values,
        "label": np.asarray(preds, dtype=int),
    }
)
my_submission["label"] = my_submission["label"].astype(int)
my_submission.to_csv(OUT_PATH, index=False)

print(
    f"Wrote submission: {OUT_PATH} shape={my_submission.shape} errors={missing_or_error} fallback_rows={used_fallback}"
)
print(my_submission.head())
print(
    "label value counts:\n",
    my_submission["label"].value_counts(dropna=False).sort_index(),
)
