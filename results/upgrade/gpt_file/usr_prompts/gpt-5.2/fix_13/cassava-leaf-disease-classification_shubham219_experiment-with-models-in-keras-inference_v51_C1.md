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

0.6668177697189483

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.06839) has done: 'Main bottlenecks are (1) the expensive Python-based protobuf implementation forced via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, (2) rebuilding and re-reading the entire `tf.data` pipeline from disk for each TTA pass, and (3) non-vectorized existence checks over ~2.6k files. I switch protobuf back to the default C++ implementation (keeping behavior identical), build the base decoded dataset once and reuse it across TTA passes (so JPEG decode/resize happens only once), and replace the Python loop file check with a vectorized `tf.io.gfile.glob`-based set membership check. TTA randomness/semantics and the model call pattern remain the same (still one non-TTA pass + `TTA_PASSES` augmented passes), but runtime drops significantly because disk decode/resize is no longer repeated.'
- What this solution (achieved 0.15845) has done: 'The timeout is dominated by expensive image decoding/resizing repeated across 4 full prediction passes (1 base + 3 TTA), plus a per-element TTA pipeline that uses random ops with a fixed seed (so it doesn’t actually create different augmentations across passes) while still paying the full augmentation cost each time. I keep identical model/prediction semantics, but remove the redundant TTA loop by computing the TTA-transformed dataset exactly once and reusing that single deterministic prediction for all TTA passes (mathematically identical given the fixed seed). I also eliminate an O(N) set-diff filesystem check that scans the whole directory and replace it with a direct existence check on just the referenced files (same correctness, less overhead). Finally, I keep the same batching/prefetching logic but avoid `repeat()` and extra iterator construction that can add overhead and unpredictability.'
- What this solution (achieved 0.15546) has done: 'The immediate crash happens before any of your cells run because TensorFlow is failing to import due to an incompatible `protobuf` version (the `MessageFactory.GetPrototype` AttributeError). To make the notebook run end-to-end in this Kaggle environment without changing your modeling/prediction logic, I pin `protobuf` to a TensorFlow-compatible version at runtime (a standard Kaggle fix) and then re-import TensorFlow cleanly. I also keep your existing dataset/prediction/TTA semantics unchanged so score changes only come from actually being able to load and run the provided `.h5` model (which should move accuracy much closer to your target than the current broken/fallback behavior). Finally, the script still write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.16218) has done: 'Your score (0.15546) is far below the target (0.6668), so we should improve accuracy with minimal semantic changes. The biggest likely issue is a preprocessing mismatch: your `_decode_and_resize` produces float32 in `[0,255]` (or EfficientNet-preprocessed for fallback), but most Cassava `.h5` models expect `tf.keras.applications.*.preprocess_input` (often scaling to `[-1,1]` or similar). To stay minimally invasive and preserve core logic, I add an automatic “try a small set of common preprocessings” probe on a tiny batch of test images, select the preprocessing that yields the most confident predictions (lowest mean entropy), and then run the exact same prediction/TTA pipeline using that chosen preprocessing. This does not change the model, architecture, TTA logic, or loss; it only fixes likely input normalization, which should move accuracy much closer to your target.'
- What this solution (achieved 0.07773) has done: 'Your current score is far below the target, so we should improve accuracy with minimal semantic risk. The most likely remaining issue is that your “entropy probe” selects preprocessing based on *test* images and uses a heuristic (mean entropy) that can choose a confidently-wrong normalization; instead, we can select preprocessing using a small stratified validation split from `train.csv` (true accuracy), which preserves the same model and inference pipeline but fixes input normalization selection. I keep your model loading, decoding/resize, TTA, and prediction aggregation logic intact; the only behavioral change is how `chosen_preprocess_fn` is chosen (now by validation accuracy, with entropy as a tie-breaker). This should move the score substantially toward your target while keeping changes minimal and runtime under the limit.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import glob
import numpy as np
import pandas as pd


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version detected: {pb_ver}")
    except Exception as e:
        print("Adjusting protobuf to a TensorFlow-compatible version due to:", repr(e))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
        if "google.protobuf" in sys.modules:
            del sys.modules["google.protobuf"]


_ensure_compatible_protobuf()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf
from tensorflow.keras.models import load_model

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TF version:", tf.__version__)

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
preferred_weight_path = "../input/model-v14/model_v0.25.h5"

candidate_paths = []
if os.path.exists(preferred_weight_path):
    candidate_paths = [preferred_weight_path]
else:
    likely_paths = [
        "../input",
        "/kaggle/input",
    ]
    for base in likely_paths:
        if os.path.isdir(base):
            candidate_paths += sorted(glob.glob(os.path.join(base, "*", "*.h5")))
            candidate_paths += sorted(glob.glob(os.path.join(base, "*", "*", "*.h5")))

if candidate_paths:
    weight_path = candidate_paths[0]
    print("Loading model from:", weight_path)
    my_model = load_model(weight_path, compile=False)

    in_shape = my_model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    if (
        in_shape is None
        or len(in_shape) != 4
        or in_shape[1] is None
        or in_shape[2] is None
    ):
        MODEL_INPUT_SIZE = (300, 300)
    else:
        MODEL_INPUT_SIZE = (int(in_shape[1]), int(in_shape[2]))
    NEED_PREPROCESS = False
    print("Model input size:", MODEL_INPUT_SIZE)
else:
    print(
        "No .h5 model found under /kaggle/input or ../input. Using EfficientNetB3(ImageNet) fallback."
    )
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as effnet_preprocess,
    )

    MODEL_INPUT_SIZE = (300, 300)
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3),
        pooling="avg",
    )
    x = layers.Dense(5, activation="softmax")(base.output)
    my_model = Model(inputs=base.input, outputs=x)
    NEED_PREPROCESS = True

    _ = my_model(
        tf.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3), dtype=tf.float32)
    )
    print("Fallback model built:", my_model.name)




## === cell 2
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "../input/cassava-leaf-disease-classification/sample_submission.csv"
    )

sample_sub = pd.read_csv(sample_sub_path)

test_img_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_img_dir):
    test_img_dir = "../input/cassava-leaf-disease-classification/test_images"

df_test = pd.DataFrame({"image_id": sample_sub["image_id"].astype(str).values})
df_test["path"] = test_img_dir.rstrip("/") + "/" + df_test["image_id"].values

paths_np = df_test["path"].to_numpy()

missing = [p for p in paths_np if not tf.io.gfile.exists(p)]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} test images listed in sample_submission.csv were not found in {test_img_dir}. "
        f"Example missing: {missing[0]}"
    )

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as effnet_preprocess,
)
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as inc_preprocess,
)
from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobv2_preprocess,
)


def _identity_preprocess(x):
    return x


def _scale_0_1(x):
    return x / 255.0


def _scale_minus1_1(x):
    return (x / 127.5) - 1.0


_preprocess_candidates = [
    ("identity_[0,255]", _identity_preprocess),
    ("scale_[0,1]", _scale_0_1),
    ("scale_[-1,1]", _scale_minus1_1),
    ("effnet_preprocess", effnet_preprocess),
    ("inceptionv3_preprocess", inc_preprocess),
    ("xception_preprocess", xcep_preprocess),
    ("mobilenetv2_preprocess", mobv2_preprocess),
]

if NEED_PREPROCESS:
    chosen_preprocess_name, chosen_preprocess_fn = (
        "effnet_preprocess",
        effnet_preprocess,
    )
else:
    chosen_preprocess_name, chosen_preprocess_fn = (
        "identity_[0,255]",
        _identity_preprocess,
    )


@tf.function(reduce_retracing=True)
def _decode_and_resize_raw(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, MODEL_INPUT_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _mean_entropy(probs_np, eps=1e-12):
    probs_np = np.asarray(probs_np, dtype=np.float64)
    probs_np = np.clip(probs_np, eps, 1.0)
    ent = -np.sum(probs_np * np.log(probs_np), axis=-1)
    return float(np.mean(ent))


def _select_preprocess_if_needed():
    global chosen_preprocess_name, chosen_preprocess_fn

    if NEED_PREPROCESS:
        print("Preprocess fixed (fallback EfficientNet):", chosen_preprocess_name)
        return

    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    if not os.path.exists(train_csv_path):
        train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
    train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    if not os.path.isdir(train_img_dir):
        train_img_dir = "../input/cassava-leaf-disease-classification/train_images"

    df_train = pd.read_csv(train_csv_path)
    df_train["image_id"] = df_train["image_id"].astype(str)
    df_train["path"] = train_img_dir.rstrip("/") + "/" + df_train["image_id"].values

    rng = np.random.RandomState(SEED)
    val_per_class = 200  # 5*200=1000 images; still fast, much more reliable signal
    val_idx = []
    for c in sorted(df_train["label"].unique()):
        idx_c = df_train.index[df_train["label"].values == c].to_numpy()
        if len(idx_c) == 0:
            continue
        rng.shuffle(idx_c)
        take = min(val_per_class, len(idx_c))
        val_idx.append(idx_c[:take])
    val_idx = (
        np.concatenate(val_idx) if len(val_idx) else df_train.index.to_numpy()[:512]
    )
    df_val = df_train.loc[val_idx].reset_index(drop=True)

    val_paths = df_val["path"].to_numpy()
    val_y = df_val["label"].to_numpy().astype(np.int64)

    val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
    val_ds = (
        val_ds.map(_decode_and_resize_raw, num_parallel_calls=AUTOTUNE)
        .batch(64)
        .prefetch(AUTOTUNE)
    )
    val_imgs = np.concatenate([b.numpy() for b in val_ds], axis=0)

    best = None  # (acc, entropy, name, fn)
    for name, fn in _preprocess_candidates:
        try:
            inp = fn(val_imgs.copy())
            preds = my_model.predict(inp, verbose=0)
            pred_lab = np.argmax(preds, axis=-1).astype(np.int64)
            acc = float(np.mean(pred_lab == val_y))
            ent = _mean_entropy(preds)
            if DEBUG:
                print(
                    f"Val preprocess={name:>22s} acc={acc:.4f} mean_entropy={ent:.6f}"
                )
            if (best is None) or (acc > best[0]) or (acc == best[0] and ent < best[1]):
                best = (acc, ent, name, fn)
        except Exception as e:
            if DEBUG:
                print(f"Val preprocess {name} failed:", repr(e))
            continue

    if best is None:
        chosen_preprocess_name, chosen_preprocess_fn = (
            "identity_[0,255]",
            _identity_preprocess,
        )
        print(
            "Chosen preprocessing (fallback due to probe failure):",
            chosen_preprocess_name,
        )
    else:
        chosen_preprocess_name, chosen_preprocess_fn = (best[2], best[3])
        print(
            "Chosen preprocessing (by highest val accuracy; entropy tiebreak):",
            chosen_preprocess_name,
            "| val_acc=%.4f" % best[0],
            "| val_mean_entropy=%.6f" % best[1],
            "| n_val=%d" % len(df_val),
        )


_select_preprocess_if_needed()


@tf.function(reduce_retracing=True)
def _decode_and_resize(path):
    img = _decode_and_resize_raw(path)
    img = chosen_preprocess_fn(img)
    return img


@tf.function(reduce_retracing=True)
def _apply_tta(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    h = MODEL_INPUT_SIZE[0]
    w = MODEL_INPUT_SIZE[1]
    pad_h = tf.cast(tf.round(0.05 * tf.cast(h, tf.float32)), tf.int32)
    pad_w = tf.cast(tf.round(0.05 * tf.cast(w, tf.float32)), tf.int32)
    img = tf.image.resize_with_crop_or_pad(img, h + 2 * pad_h, w + 2 * pad_w)
    img = tf.image.random_crop(img, size=[h, w, 3], seed=SEED)

    zoom = tf.random.uniform([], minval=0.90, maxval=1.0, seed=SEED)
    crop_h = tf.cast(tf.round(zoom * tf.cast(h, tf.float32)), tf.int32)
    crop_w = tf.cast(tf.round(zoom * tf.cast(w, tf.int32)), tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3], seed=SEED)
    img = tf.image.resize(img, MODEL_INPUT_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    angle = tf.random.uniform([], minval=-10.0, maxval=10.0, seed=SEED) * (
        np.pi / 180.0
    )
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    tx = 0.0
    ty = 0.0
    transform = tf.stack([cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0])[tf.newaxis, :]
    try:
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=img[tf.newaxis, ...],
            transforms=transform,
            output_shape=tf.constant([h, w], dtype=tf.int32),
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0,
        )[0]
    except Exception:
        pass

    return img


def make_base_test_ds():
    ds = tf.data.Dataset.from_tensor_slices(paths_np)
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_base(base_ds, batch_size=64, tta=False):
    ds = base_ds
    if tta:
        ds = ds.map(_apply_tta, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 3
def _find_best_label_permutation():
    if NEED_PREPROCESS:
        return np.arange(5, dtype=np.int64)

    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    if not os.path.exists(train_csv_path):
        train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
    train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    if not os.path.isdir(train_img_dir):
        train_img_dir = "../input/cassava-leaf-disease-classification/train_images"

    df_train = pd.read_csv(train_csv_path)
    df_train["image_id"] = df_train["image_id"].astype(str)
    df_train["path"] = train_img_dir.rstrip("/") + "/" + df_train["image_id"].values

    rng = np.random.RandomState(SEED + 1)
    per_class = 120  # 600 images
    idxs = []
    for c in range(5):
        idx_c = df_train.index[df_train["label"].values == c].to_numpy()
        if len(idx_c) == 0:
            continue
        rng.shuffle(idx_c)
        idxs.append(idx_c[: min(per_class, len(idx_c))])
    idxs = np.concatenate(idxs) if len(idxs) else df_train.index.to_numpy()[:512]
    df_val = df_train.loc[idxs].reset_index(drop=True)

    val_paths = df_val["path"].to_numpy()
    val_y = df_val["label"].to_numpy().astype(np.int64)

    val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
    val_ds = (
        val_ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)
        .batch(64)
        .prefetch(AUTOTUNE)
    )
    preds = my_model.predict(val_ds, verbose=0)
    preds = np.asarray(preds, dtype=np.float64)

    base_pred = np.argmax(preds, axis=-1).astype(np.int64)
    base_acc = float(np.mean(base_pred == val_y))

    best_perm = np.arange(5, dtype=np.int64)
    best_acc = base_acc

    for i in range(5):
        for j in range(i + 1, 5):
            perm = np.arange(5, dtype=np.int64)
            perm[i], perm[j] = perm[j], perm[i]
            pred_swapped = perm[base_pred]
            acc = float(np.mean(pred_swapped == val_y))
            if acc > best_acc:
                best_acc = acc
                best_perm = perm.copy()

    print(
        f"Val permutation probe: base_acc={base_acc:.4f}, best_acc={best_acc:.4f}, best_perm={best_perm.tolist()}, n_val={len(df_val)}"
    )
    return best_perm


label_perm = _find_best_label_permutation()




## === cell 4
BATCH_SIZE = 128
TTA_PASSES = 3

base_test_ds = make_base_test_ds()

n_test = len(df_test)
steps = int(np.ceil(n_test / BATCH_SIZE))

test_ds = make_test_ds_from_base(base_test_ds, batch_size=BATCH_SIZE, tta=False)
pred_sum = my_model.predict(test_ds, verbose=1)
pred_sum = np.asarray(pred_sum, dtype=np.float64)

tta_ds_once = make_test_ds_from_base(base_test_ds, batch_size=BATCH_SIZE, tta=True)
pred_tta = my_model.predict(tta_ds_once, verbose=0)
pred_tta = np.asarray(pred_tta, dtype=np.float64)
pred_sum += pred_tta * TTA_PASSES

pred_mean = pred_sum / (TTA_PASSES + 1)
pred_test_labels = np.argmax(pred_mean, axis=-1).astype(np.int64)

pred_test_labels = label_perm[pred_test_labels].astype(int)

final_csv = pd.DataFrame(
    {"image_id": df_test["image_id"].values, "label": pred_test_labels}
)

if len(final_csv) != len(sample_sub):
    raise ValueError(
        f"Submission length mismatch: got {len(final_csv)} rows, expected {len(sample_sub)}"
    )
if list(final_csv.columns) != ["image_id", "label"]:
    raise ValueError(f"Submission columns incorrect: {final_csv.columns.tolist()}")

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print("Used preprocessing:", chosen_preprocess_name)
print("Used label_perm:", label_perm.tolist())
print(final_csv.head())




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2344634319.py in <cell line: 0>()
     11 pred_sum = np.asarray(pred_sum, dtype=np.float64)
     12 
---> 13 tta_ds_once = make_test_ds_from_base(base_test_ds, batch_size=BATCH_SIZE, tta=True)
     14 pred_tta = my_model.predict(tta_ds_once, verbose=0)
     15 pred_tta = np.asarray(pred_tta, dtype=np.float64)

/tmp/ipykernel_10/1682963448.py in make_test_ds_from_base(base_ds, batch_size, tta)
    234     ds = base_ds
    235     if tta:
--> 236         ds = ds.map(_apply_tta, num_parallel_calls=AUTOTUNE)
    237     ds = ds.batch(batch_size, drop_remainder=False)
    238     ds = ds.prefetch(AUTOTUNE)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filefse1rjtg.py in tf___apply_tta(img)
     17                 zoom = ag__.converted_call(ag__.ld(tf).random.uniform, ([],), dict(minval=0.9, maxval=1.0, seed=ag__.ld(SEED)), fscope)
     18                 crop_h = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).round, (ag__.ld(zoom) * ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h), ag__.ld(tf).float32), None, fscope),), None, fscope), ag__.ld(tf).int32), None, fscope)
---> 19                 crop_w = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).round, (ag__.ld(zoom) * ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(w), ag__.ld(tf).int32), None, fscope),), None, fscope), ag__.ld(tf).int32), None, fscope)
     20                 img = ag__.converted_call(ag__.ld(tf).image.random_crop, (ag__.ld(img),), dict(size=[ag__.ld(crop_h), ag__.ld(crop_w), 3], seed=ag__.ld(SEED)), fscope)
     21                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), ag__.ld(MODEL_INPUT_SIZE)), dict(method=ag__.ld(tf).image.ResizeMethod.BILINEAR), fscope)

TypeError: in user code:

    File "/tmp/ipykernel_10/1682963448.py", line 198, in _apply_tta  *
        crop_w = tf.cast(tf.round(zoom * tf.cast(w, tf.int32)), tf.int32)

    TypeError: Input 'y' of 'Mul' Op has type int32 that does not match type float32 of argument 'x'.


## === cell 5
sub = pd.read_csv("submission.csv")
assert sub.shape[0] == sample_sub.shape[0]
assert sub.columns.tolist() == ["image_id", "label"]
assert sub["image_id"].iloc[0] == sample_sub["image_id"].iloc[0]
print("submission.csv OK:", sub.shape)
print(sub["label"].value_counts().sort_index())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/2932655558.py in <cell line: 0>()
----> 1 sub = pd.read_csv("submission.csv")
      2 assert sub.shape[0] == sample_sub.shape[0]
      3 assert sub.columns.tolist() == ["image_id", "label"]
      4 assert sub["image_id"].iloc[0] == sample_sub["image_id"].iloc[0]
      5 print("submission.csv OK:", sub.shape)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
