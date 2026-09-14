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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import random
import numpy as np
import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    tf.config.optimizer.set_jit(False)  # keep deterministic & avoid compile overhead
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e16/epoch-16"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels), "Classes:", dataset_labels)



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

try:
    from tensorflow.keras.layers import TFSMLayer  # type: ignore
except Exception:
    TFSMLayer = None


_SAVEDMODEL_DIR_CACHE = {}
_WEIGHTS_PATH_CACHE = {}


def _find_savedmodel_dir(root_dir: str) -> str:
    """
    Find directory that directly contains saved_model.pb (or pbtxt).
    """
    root_dir = os.path.abspath(root_dir)
    if not tf.io.gfile.exists(root_dir):
        raise FileNotFoundError(f"Path does not exist: {root_dir}")

    root_files = (
        set(tf.io.gfile.listdir(root_dir)) if tf.io.gfile.isdir(root_dir) else set()
    )
    if "saved_model.pb" in root_files or "saved_model.pbtxt" in root_files:
        return root_dir

    for cur, _, files in os.walk(root_dir):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            return cur
    raise FileNotFoundError(
        f"Could not find saved_model.pb or saved_model.pbtxt under: {root_dir}"
    )


def _resolve_backbone_dir(preferred_path: str) -> str:
    """
    Try to locate a SavedModel directory for the backbone. If not found, we will fall back
    to a built-in EfficientNetB7 backbone (handled in MultiLabel.__init__).
    """
    key = os.path.abspath(preferred_path) if preferred_path else preferred_path
    if key in _SAVEDMODEL_DIR_CACHE:
        return _SAVEDMODEL_DIR_CACHE[key]

    candidates = [
        preferred_path,
        os.path.dirname(preferred_path),
        model_dir,
        os.path.dirname(model_dir),
        "../input",
        "/kaggle/input",
    ]

    seen = set()
    last_err = None
    for c in candidates:
        if not c:
            continue
        c_abs = os.path.abspath(c)
        if c_abs in seen:
            continue
        seen.add(c_abs)
        try:
            if tf.io.gfile.exists(c_abs):
                resolved = _find_savedmodel_dir(c_abs)
                _SAVEDMODEL_DIR_CACHE[key] = resolved
                return resolved
        except Exception as e:
            last_err = e

    raise FileNotFoundError(
        f"Could not locate a SavedModel for the backbone. Tried: {candidates}. "
        f"Last error: {repr(last_err)}"
    )


def _resolve_weights_path(path: str) -> str:
    """
    model.load_weights() accepts a file prefix or directory depending on format.
    If a directory is passed, try to find a plausible weights file inside.
    """
    p_abs = os.path.abspath(path)
    if p_abs in _WEIGHTS_PATH_CACHE:
        return _WEIGHTS_PATH_CACHE[p_abs]

    if tf.io.gfile.exists(path) and not tf.io.gfile.isdir(path):
        _WEIGHTS_PATH_CACHE[p_abs] = path
        return path

    if tf.io.gfile.isdir(path):
        ckpt_file = os.path.join(path, "checkpoint")
        if tf.io.gfile.exists(ckpt_file):
            ckpt_state = tf.train.get_checkpoint_state(path)
            if ckpt_state and ckpt_state.model_checkpoint_path:
                resolved = (
                    ckpt_state.model_checkpoint_path
                    if os.path.isabs(ckpt_state.model_checkpoint_path)
                    else os.path.join(path, ckpt_state.model_checkpoint_path)
                )
                _WEIGHTS_PATH_CACHE[p_abs] = resolved
                return resolved

        candidates = []
        for fname in tf.io.gfile.listdir(path):
            fpath = os.path.join(path, fname)
            if fname.endswith((".h5", ".keras", ".ckpt")) or fname.endswith(".index"):
                candidates.append(fpath)

        index_candidates = [c for c in candidates if c.endswith(".index")]
        if index_candidates:
            idx = sorted(index_candidates)[-1]
            resolved = idx[: -len(".index")]
            _WEIGHTS_PATH_CACHE[p_abs] = resolved
            return resolved

        if candidates:
            resolved = sorted(candidates)[-1]
            _WEIGHTS_PATH_CACHE[p_abs] = resolved
            return resolved

    _WEIGHTS_PATH_CACHE[p_abs] = path
    return path


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._backbone_mode = "tf_savedmodel"
        self.model_backbone = None

        if TFSMLayer is not None:
            try:
                sm_dir = _resolve_backbone_dir(efficientB7)
                self.model_backbone = TFSMLayer(sm_dir, call_endpoint="serving_default")
                self._backbone_mode = "tf_savedmodel"
                print("Backbone: TFSMLayer from:", sm_dir)
            except Exception as e:
                print(
                    "Backbone SavedModel not found/usable; falling back to tf.keras EfficientNetB7. Reason:",
                    repr(e),
                )

        if self.model_backbone is None:
            self._backbone_mode = "keras_efficientnetb7"
            self.model_backbone = tf.keras.applications.EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
            )
            self.model_backbone.trainable = False
            print(
                "Backbone: tf.keras.applications.EfficientNetB7 (include_top=False, imagenet)"
            )

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)

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

    def _unwrap_backbone_output(self, y):
        if isinstance(y, dict):
            for k in ("output_0", "outputs", "predictions", "logits"):
                if k in y:
                    return y[k]
            return next(iter(y.values()))
        return y

    def call(self, x, **kwargs):
        y = self.model(x)
        y = self._unwrap_backbone_output(y)

        if len(y.shape) == 2:
            y = tf.reshape(y, [-1, 1, 1, tf.shape(y)[-1]])

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
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    weights_path = _resolve_weights_path(model_dir)
    try:
        if tf.io.gfile.exists(weights_path) or tf.io.gfile.exists(
            weights_path + ".index"
        ):
            model.load_weights(weights_path)
            print("Loaded weights from:", weights_path)
        else:
            print(
                "Weights not found at:",
                weights_path,
                "-> proceeding without loading custom weights.",
            )
    except Exception as e:
        print(
            "Could not load weights from:",
            weights_path,
            "-> proceeding without loading. Reason:",
            repr(e),
        )

    images_path_list = sorted([e.name for e in os.scandir(test_dir) if e.is_file()])
    classes = dataset_labels

    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 128

    test_dir_abs = os.path.abspath(test_dir)
    filepaths = [os.path.join(test_dir_abs, n) for n in images_path_list]

    def _load_and_preprocess(fpath):
        img_bytes = tf.io.read_file(fpath)
        is_png = tf.strings.regex_full_match(tf.strings.lower(fpath), ".*\\.png$")
        img = tf.cond(
            is_png,
            lambda: tf.io.decode_png(img_bytes, channels=3),
            lambda: tf.io.decode_jpeg(img_bytes, channels=3),
        )  # uint8 [H,W,3]
        img.set_shape([None, None, 3])
        img = tf.image.resize(
            img,
            [image_dims[0], image_dims[1]],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )
        img = tf.cast(img, tf.float32) * 255.0
        img.set_shape([image_dims[0], image_dims[1], image_dims[2]])
        fname = tf.strings.split(fpath, os.sep)[-1]
        return fname, img

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(filepaths))

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    thr = tf.constant(0.7, dtype=tf.float32)
    classes_tf = tf.constant(classes, dtype=tf.string)

    @tf.function(reduce_retracing=True)
    def _infer_and_format(batch_images):
        preds = model(batch_images, training=False)  # same as model.call
        preds = tf.reshape(preds, [tf.shape(preds)[0], -1])  # (B, 6)

        mask = preds > thr  # (B, 6) bool
        any_pos = tf.reduce_any(mask, axis=1)  # (B,)
        argmax_idx = tf.argmax(preds, axis=1, output_type=tf.int32)  # (B,)

        ragged_idx = tf.ragged.boolean_mask(
            tf.tile(
                tf.range(tf.shape(preds)[1])[tf.newaxis, :], [tf.shape(preds)[0], 1]
            ),
            mask,
        )  # Ragged [B, (k_i)]
        ragged_labels = tf.gather(classes_tf, ragged_idx)  # Ragged [B, (k_i)]
        labels_pos = tf.strings.reduce_join(ragged_labels, axis=1, separator=" ")

        labels_neg = tf.gather(classes_tf, argmax_idx)  # (B,)
        labels = tf.where(any_pos, labels_pos, labels_neg)  # (B,)
        return labels

    out_images_chunks = []
    out_labels_chunks = []

    for fnames, imgs in ds:
        labels_tf_out = _infer_and_format(imgs)
        out_images_chunks.append(fnames.numpy())
        out_labels_chunks.append(labels_tf_out.numpy())

    out_images_np = np.concatenate(out_images_chunks, axis=0)
    out_labels_np = np.concatenate(out_labels_chunks, axis=0)

    out_images = [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
        for x in out_images_np.tolist()
    ]
    out_labels = [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
        for x in out_labels_np.tolist()
    ]

    csv_pd = pd.DataFrame({"image": out_images, "labels": out_labels})
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)

    print("Wrote:", out_path, "rows:", len(csv_pd))
    print("Head:\n", csv_pd.head())
    print("Unique label strings (sample):", csv_pd["labels"].head(10).tolist())
