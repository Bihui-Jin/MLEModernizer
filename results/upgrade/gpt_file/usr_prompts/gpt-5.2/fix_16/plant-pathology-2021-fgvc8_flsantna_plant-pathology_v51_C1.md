# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf

keras = tf.keras

print("TF:", tf.__version__)
print("Keras:", keras.__version__)

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    _cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _cpu))
    tf.config.threading.set_inter_op_parallelism_threads(min(8, _cpu))
except Exception:
    pass



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e15/epoch-15"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = train_df["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")

dataset_labels = list(one_hot.columns)
if len(dataset_labels) != 6:
    raise ValueError(
        f"Expected 6 classes for this model head, but found {len(dataset_labels)} in train.csv: {dataset_labels}"
    )



## === cell 2
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D,
    Conv2D,
    InputLayer,
)
from tensorflow import concat

_SAVEDMODEL_CACHE = {}
_WEIGHTFILE_CACHE = {}


def _find_savedmodel_dir(base_path: str) -> str:
    base_path = os.path.abspath(base_path)
    if base_path in _SAVEDMODEL_CACHE:
        return _SAVEDMODEL_CACHE[base_path]

    def _is_savedmodel_dir(p: str) -> bool:
        return os.path.isdir(p) and (
            os.path.exists(os.path.join(p, "saved_model.pb"))
            or os.path.exists(os.path.join(p, "saved_model.pbtxt"))
        )

    if _is_savedmodel_dir(base_path):
        _SAVEDMODEL_CACHE[base_path] = base_path
        return base_path

    p2 = os.path.join(base_path, "saved_model")
    if _is_savedmodel_dir(p2):
        _SAVEDMODEL_CACHE[base_path] = p2
        return p2

    if os.path.isdir(base_path):
        try:
            for name in os.listdir(base_path):
                cand = os.path.join(base_path, name)
                if _is_savedmodel_dir(cand):
                    _SAVEDMODEL_CACHE[base_path] = cand
                    return cand
                cand2 = os.path.join(cand, "saved_model")
                if _is_savedmodel_dir(cand2):
                    _SAVEDMODEL_CACHE[base_path] = cand2
                    return cand2
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not find a TensorFlow SavedModel under: {base_path}"
    )


def _find_weight_file(base_path: str):
    """
    Keep behavior: discover a weight file under model_dir; this avoids running with random weights.
    """
    base_path = os.path.abspath(base_path)
    if base_path in _WEIGHTFILE_CACHE:
        return _WEIGHTFILE_CACHE[base_path]

    if os.path.isfile(base_path) and base_path.endswith((".h5", ".weights.h5")):
        _WEIGHTFILE_CACHE[base_path] = base_path
        return base_path
    if not os.path.isdir(base_path):
        _WEIGHTFILE_CACHE[base_path] = None
        return None

    preferred = [
        "weights.h5",
        "model.weights.h5",
        "best.h5",
        "best.weights.h5",
        "final.h5",
        "final.weights.h5",
        "checkpoint.h5",
        "ckpt.h5",
    ]
    for fn in preferred:
        p = os.path.join(base_path, fn)
        if os.path.exists(p):
            _WEIGHTFILE_CACHE[base_path] = p
            return p

    cand = []
    try:
        for fn in os.listdir(base_path):
            if fn.endswith((".h5", ".weights.h5")):
                p = os.path.join(base_path, fn)
                if os.path.isfile(p):
                    cand.append(p)
    except Exception:
        cand = []

    if cand:
        cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        _WEIGHTFILE_CACHE[base_path] = cand[0]
        return cand[0]

    _WEIGHTFILE_CACHE[base_path] = None
    return None


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))

        backbone_layer = None
        backbone_errors = []
        self.backbone_is_keras_app = False

        for p in [efficientB7, model_dir]:
            try:
                sm_path = _find_savedmodel_dir(p)
                try:
                    backbone_layer = keras.layers.TFSMLayer(
                        sm_path, call_endpoint="serving_default"
                    )
                except Exception:
                    backbone_layer = keras.layers.TFSMLayer(
                        sm_path, call_endpoint="predict"
                    )
                break
            except Exception as e:
                backbone_errors.append((p, f"SavedModel not found: {e}"))

        if backbone_layer is None:
            try:
                backbone_layer = keras.applications.EfficientNetB7(
                    include_top=False,
                    weights="imagenet",
                    input_shape=image_dims,
                )
                self.backbone_is_keras_app = True
            except Exception as e:
                msgs = "\n".join([f"- {p}: {err}" for p, err in backbone_errors])
                raise FileNotFoundError(
                    "No valid SavedModel found for backbone, and failed to create EfficientNetB7.\n"
                    + "SavedModel attempts:\n"
                    + msgs
                    + f"\nEfficientNetB7 error: {e}"
                )

        try:
            backbone_layer.trainable = False
        except Exception:
            pass

        self.model.add(backbone_layer)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding="same"))

        self.model_pred1_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred1_3 = BatchNormalization()
        self.model_pred1_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred1_5 = GlobalMaxPool2D()
        self.model_pred1_6 = Dense(units=1, activation="sigmoid")

        self.model_pred2_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred2_3 = BatchNormalization()
        self.model_pred2_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred2_5 = GlobalMaxPool2D()
        self.model_pred2_6 = Dense(units=1, activation="sigmoid")

        self.model_pred3_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred3_3 = BatchNormalization()
        self.model_pred3_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred3_5 = GlobalMaxPool2D()
        self.model_pred3_6 = Dense(units=1, activation="sigmoid")

        self.model_pred4_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred4_3 = BatchNormalization()
        self.model_pred4_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred4_5 = GlobalMaxPool2D()
        self.model_pred4_6 = Dense(units=1, activation="sigmoid")

        self.model_pred5_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred5_3 = BatchNormalization()
        self.model_pred5_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred5_5 = GlobalMaxPool2D()
        self.model_pred5_6 = Dense(units=1, activation="sigmoid")

        self.model_pred6_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred6_3 = BatchNormalization()
        self.model_pred6_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred6_5 = GlobalMaxPool2D()
        self.model_pred6_6 = Dense(units=1, activation="sigmoid")

    def _ensure_4d_backbone_output(self, y):
        if isinstance(y, dict):
            for k in ("outputs", "output", "features", "feature", "logits", "output_0"):
                if k in y:
                    y = y[k]
                    break
            else:
                vals = list(y.values())
                tensor_vals = [v for v in vals if tf.is_tensor(v)]
                y = tensor_vals[0] if tensor_vals else vals[0]
        if isinstance(y, (tuple, list)):
            y = y[0]
        if y.shape.rank != 4:
            raise ValueError(f"Backbone output must be 4D (NHWC), got shape: {y.shape}")
        return y

    def call(self, x, **kwargs):
        y = self.model(x)
        y = self._ensure_4d_backbone_output(y)

        pred1 = y[:, :, :, :400]
        pred2 = y[:, :, :, 400:800]
        pred3 = y[:, :, :, 800:1200]
        pred4 = y[:, :, :, 1200:1600]
        pred5 = y[:, :, :, 1600:2000]
        pred6 = y[:, :, :, 2000:2400]

        pred1 = self.model_pred1_2(pred1)
        pred1 = self.model_pred1_3(pred1)
        pred1 = self.model_pred1_4(pred1)
        pred1 = self.model_pred1_5(pred1)
        pred1 = self.model_pred1_6(pred1)

        pred2 = self.model_pred2_2(pred2)
        pred2 = self.model_pred2_3(pred2)
        pred2 = self.model_pred2_4(pred2)
        pred2 = self.model_pred2_5(pred2)
        pred2 = self.model_pred2_6(pred2)

        pred3 = self.model_pred3_2(pred3)
        pred3 = self.model_pred3_3(pred3)
        pred3 = self.model_pred3_4(pred3)
        pred3 = self.model_pred3_5(pred3)
        pred3 = self.model_pred3_6(pred3)

        pred4 = self.model_pred4_2(pred4)
        pred4 = self.model_pred4_3(pred4)
        pred4 = self.model_pred4_4(pred4)
        pred4 = self.model_pred4_5(pred4)
        pred4 = self.model_pred4_6(pred4)

        pred5 = self.model_pred5_2(pred5)
        pred5 = self.model_pred5_3(pred5)
        pred5 = self.model_pred5_4(pred5)
        pred5 = self.model_pred5_5(pred5)
        pred5 = self.model_pred5_6(pred5)

        pred6 = self.model_pred6_2(pred6)
        pred6 = self.model_pred6_3(pred6)
        pred6 = self.model_pred6_4(pred6)
        pred6 = self.model_pred6_5(pred6)
        pred6 = self.model_pred6_6(pred6)

        return concat([pred1, pred2, pred3, pred4, pred5, pred6], axis=1)

    def create_model(self):
        return self.model




## === cell 3
if __name__ == "__main__":
    model = MultiLabel()

    dummy = tf.zeros([1, image_dims[0], image_dims[1], image_dims[2]], dtype=tf.float32)
    if getattr(model, "backbone_is_keras_app", False):
        _ = model(keras.applications.efficientnet.preprocess_input(dummy * 255.0))
    else:
        _ = model(dummy * 255.0)

    weights_loaded = False
    load_errors = []

    weight_path = _find_weight_file(model_dir)
    if weight_path is not None and os.path.exists(weight_path):
        try:
            model.load_weights(weight_path)
            weights_loaded = True
            print("Loaded weights:", weight_path)
        except Exception as e:
            load_errors.append((weight_path, str(e)))

    if not weights_loaded:
        ckpt = os.path.join(model_dir, "checkpoint")
        if os.path.exists(ckpt):
            try:
                model.load_weights(os.path.join(model_dir, "ckpt"))
                weights_loaded = True
                print("Loaded checkpoint weights from:", model_dir)
            except Exception as e:
                load_errors.append((model_dir, str(e)))

    if not weights_loaded:
        msg = (
            "\n".join([f"- {p}: {err}" for p, err in load_errors])
            if load_errors
            else "(no weight files found)"
        )
        print(
            "WARNING: Failed to load provided weights; using current initialized weights (score will be poor).\nTried:\n"
            + msg
        )

    images_path_list = sorted(os.listdir(test_dir))
    classes = dataset_labels

    test_root = tf.constant(test_dir)
    train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
    train_root = tf.constant(train_img_dir)

    @tf.function(reduce_retracing=True)
    def _load_and_preprocess_tf(filename, root_dir_tf):
        img_path = tf.strings.join([root_dir_tf, filename])
        img_bytes = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(
            img_bytes, channels=3
        )  # faster than tf.io.decode_image for JPEGs
        image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
        image = tf.image.resize(
            image,
            [image_dims[0], image_dims[1]],
            method="bilinear",
            antialias=False,
        )
        image = image * 255.0
        if getattr(model, "backbone_is_keras_app", False):
            image = keras.applications.efficientnet.preprocess_input(image)
        image.set_shape([image_dims[0], image_dims[1], image_dims[2]])
        return filename, image

    @tf.function(
        reduce_retracing=True,
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(
                shape=[None, image_dims[0], image_dims[1], image_dims[2]],
                dtype=tf.float32,
            )
        ],
    )
    def infer(batch_images):
        return model(batch_images, training=False)

    train_df_local = train_df.copy()
    train_df_local["image"] = train_df_local["image"].astype(str)
    train_labels_mat = (
        train_df_local["labels"]
        .str.get_dummies(sep=" ")
        .reindex(columns=classes, fill_value=0)
        .values.astype(np.int32)
    )

    calib_n = 512
    calib_idx = np.random.choice(
        len(train_df_local), size=min(calib_n, len(train_df_local)), replace=False
    )
    calib_images = train_df_local.iloc[calib_idx]["image"].tolist()
    y_true = train_labels_mat[calib_idx]  # (N,6)

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    try:
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    try:
        options.threading.max_intra_op_parallelism = min(8, os.cpu_count() or 8)
    except Exception:
        pass

    calib_bs = 256
    calib_ds = tf.data.Dataset.from_tensor_slices(tf.constant(calib_images))
    calib_ds = calib_ds.with_options(options)
    calib_ds = calib_ds.map(
        lambda fn: _load_and_preprocess_tf(fn, train_root),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    calib_ds = calib_ds.cache()
    calib_ds = calib_ds.batch(calib_bs, drop_remainder=False)
    calib_ds = calib_ds.prefetch(tf.data.AUTOTUNE)

    n_calib = len(calib_images)
    y_pred = np.empty((n_calib, len(classes)), dtype=np.float32)
    o = 0
    for _, batch_images in calib_ds:
        p = infer(batch_images).numpy()
        bs = p.shape[0]
        y_pred[o : o + bs] = p
        o += bs

    def _best_f1_thresholds(y_true_bin, y_prob, grid):
        eps = 1e-9
        y_true_bin = np.asarray(y_true_bin, dtype=np.int32)
        y_prob = np.asarray(y_prob, dtype=np.float32)
        grid = np.asarray(grid, dtype=np.float32)
        N, C = y_true_bin.shape
        G = grid.shape[0]

        best_t = np.empty((C,), dtype=np.float32)
        for c in range(C):
            p = y_prob[:, c]
            yt = y_true_bin[:, c].astype(np.int32)
            total_pos = yt.sum(dtype=np.int64)

            pred_pos = p[:, None] > grid[None, :]
            tp = (pred_pos & (yt[:, None] == 1)).sum(axis=0, dtype=np.int64)
            fp = (pred_pos & (yt[:, None] == 0)).sum(axis=0, dtype=np.int64)
            fn = (total_pos - tp).astype(np.int64)

            denom = (2 * tp + fp + fn).astype(np.float64) + eps
            f1 = (2.0 * tp.astype(np.float64)) / denom
            best_t[c] = grid[int(np.argmax(f1))].astype(np.float32)
        return best_t

    grid = np.linspace(0.2, 0.85, 14, dtype=np.float32)
    thresholds = _best_f1_thresholds(y_true, y_pred, grid)
    print("Calibrated thresholds:", dict(zip(classes, thresholds.tolist())))

    filenames = tf.constant(images_path_list)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.with_options(options)
    ds = ds.map(
        lambda fn: _load_and_preprocess_tf(fn, test_root),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    batch_size = 512
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    n_test = len(images_path_list)
    all_preds = np.empty((n_test, len(classes)), dtype=np.float32)
    all_names = np.empty((n_test,), dtype=object)

    o = 0
    for batch_names, batch_images in ds:
        p = infer(batch_images).numpy()
        bs = p.shape[0]
        all_preds[o : o + bs] = p
        all_names[o : o + bs] = batch_names.numpy().astype(str)
        o += bs

    mask = all_preds > thresholds[None, :]

    label_strings = np.array(classes, dtype=object)
    map_size = 1 << len(classes)  # 64
    bitmask_to_label = np.empty((map_size,), dtype=object)
    for m in range(map_size):
        if m == 0:
            bitmask_to_label[m] = "healthy"
        else:
            idxs = [i for i in range(len(classes)) if (m >> i) & 1]
            bitmask_to_label[m] = " ".join(label_strings[idxs].tolist())

    bits = (
        (mask.astype(np.uint8) * (1 << np.arange(mask.shape[1], dtype=np.uint8))).sum(
            axis=1
        )
    ).astype(np.int64)
    labels_out = bitmask_to_label[bits]

    csv_pd = pd.DataFrame({"image": all_names, "labels": labels_out})
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(csv_pd))
