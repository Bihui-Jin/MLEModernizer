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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    warnings.warn(f"Could not enable op determinism (continuing): {e}")



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/epoch-1"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
dataset_labels = list(dataset_labels)


def _find_saved_model_dir(path_hint: str) -> str:
    """
    Robustly locate a TF SavedModel directory.
    - Accept direct SavedModel dir (contains saved_model.pb).
    - If hint is wrong, search within hint and its parent.
    - As a last resort, do a shallow search under /kaggle/input (limited).
    Returns "" if not found (caller can decide a fallback).
    """

    def is_saved_model_dir(p: str) -> bool:
        return os.path.isdir(p) and os.path.isfile(os.path.join(p, "saved_model.pb"))

    if is_saved_model_dir(path_hint):
        return path_hint

    if os.path.isdir(path_hint):
        for root, _, files in os.walk(path_hint):
            if "saved_model.pb" in files:
                return root

    parent = os.path.dirname(path_hint.rstrip("/"))
    if os.path.isdir(parent) and parent != path_hint:
        for root, _, files in os.walk(parent):
            if "saved_model.pb" in files:
                return root

    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        max_depth = 1
        for root, dirs, files in os.walk(kaggle_input):
            depth = root[len(kaggle_input) :].count(os.sep)
            if depth > max_depth:
                dirs[:] = []
                continue
            if "saved_model.pb" in files:
                return root

    return ""


def _resolve_weights_path(path_hint: str) -> str:
    """
    `load_weights` needs a concrete weights file (or a valid checkpoint prefix).
    If a directory is given, pick the latest checkpoint or newest likely weights file.
    If not found, return empty string so caller can skip loading safely.
    """
    if os.path.isfile(path_hint):
        return path_hint

    if os.path.isdir(path_hint):
        ckpt = tf.train.latest_checkpoint(path_hint)
        if ckpt is not None:
            return ckpt

        exts = (".h5", ".keras", ".ckpt")
        files = []
        for fn in os.listdir(path_hint):
            fp = os.path.join(path_hint, fn)
            if os.path.isfile(fp) and fn.lower().endswith(exts):
                files.append(fp)
        if files:
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return files[0]

    return ""


efficientB7_resolved = _find_saved_model_dir(efficientB7)
model_dir_resolved = _resolve_weights_path(model_dir)



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


class SavedModelBackbone(layers.Layer):
    """
    Wrap a TF SavedModel as a Keras Layer using tf.saved_model.load().
    """

    def __init__(
        self, saved_model_dir: str, call_endpoint: str = "serving_default", **kwargs
    ):
        super().__init__(**kwargs)
        self.saved_model_dir = saved_model_dir
        self.call_endpoint = call_endpoint
        self._loaded = None
        self._fn = None

    def build(self, input_shape):
        self._loaded = tf.saved_model.load(self.saved_model_dir)
        if (
            hasattr(self._loaded, "signatures")
            and self.call_endpoint in self._loaded.signatures
        ):
            self._fn = self._loaded.signatures[self.call_endpoint]
        else:
            self._fn = None
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self._fn is None:
            out = self._loaded(inputs, training=training)
        else:
            try:
                out = self._fn(inputs=inputs)
            except TypeError:
                out = self._fn(inputs)

        if isinstance(out, dict):
            for k in ("output_0", "outputs", "predictions", "features"):
                if k in out:
                    return out[k]
            return next(iter(out.values()))
        return out


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))

        if efficientB7_resolved:
            self.model_backbone = SavedModelBackbone(
                efficientB7_resolved, call_endpoint="serving_default"
            )
            self.model.add(self.model_backbone)
        else:
            backbone = tf.keras.applications.EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
                pooling=None,
            )
            self.model.add(backbone)

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

    def call(self, x, **kwargs):
        y = self.model(x)

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
    AUTOTUNE = tf.data.AUTOTUNE

    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception:
        pass

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    if model_dir_resolved and (
        os.path.exists(model_dir_resolved)
        or tf.train.checkpoint_exists(model_dir_resolved)
    ):
        model.load_weights(model_dir_resolved)
    else:
        warnings.warn(
            f"Could not resolve weights at '{model_dir}'. Running with randomly initialized head/backbone; "
            f"submission will be valid but score may be low."
        )

    images_path_list = sorted(os.listdir(test_dir))
    n_images = len(images_path_list)

    threshold = 0.7

    classes = dataset_labels[:6]
    if len(classes) < 6:
        classes = [
            "complex",
            "frog_eye_leaf_spot",
            "healthy",
            "powdery_mildew",
            "rust",
            "scab",
        ]

    classes_tf = tf.constant(classes, dtype=tf.string)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.parallel_batch = True

    def _load_and_preprocess(img_name):
        img_path = tf.strings.join([test_dir, img_name])
        img_bytes = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(img_bytes, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        return img_name, image

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, *image_dims], dtype=tf.float32)]
    )
    def _predict_and_labels(batch_images):
        batch_preds = model(batch_images, training=False)
        batch_preds = tf.reshape(batch_preds, [tf.shape(batch_preds)[0], -1])  # (B, 6)

        mask = batch_preds > threshold  # (B, 6)
        any_pos = tf.reduce_any(mask, axis=1)  # (B,)
        tokens = tf.where(
            mask,
            classes_tf[None, :] + tf.constant(" ", tf.string),
            tf.constant("", tf.string),
        )
        joined = tf.strings.reduce_join(tokens, axis=1)  # (B,)
        joined = tf.strings.strip(joined)
        healthy = tf.fill(tf.shape(joined), tf.constant("healthy", tf.string))
        return tf.where(any_pos, joined, healthy)

    batch_size = 128
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(images_path_list))
    ds = ds.with_options(options)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    out_images = []
    out_labels = []

    for batch_names, batch_images in ds:
        batch_labels = _predict_and_labels(batch_images)
        out_images.extend(batch_names.numpy().astype("U"))
        out_labels.extend(batch_labels.numpy().astype("U"))

    if len(out_labels) != n_images:
        raise RuntimeError(
            f"Unexpected number of predictions: got {len(out_labels)} expected {n_images}"
        )

    csv_pd = pd.DataFrame({"image": out_images, "labels": out_labels})
    csv_pd.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
