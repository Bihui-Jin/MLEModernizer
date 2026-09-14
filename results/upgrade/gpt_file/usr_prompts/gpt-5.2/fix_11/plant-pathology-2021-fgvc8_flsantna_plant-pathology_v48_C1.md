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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/effb7-e12/epoch-12"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

data_set = (
    pd.read_csv("../input/plant-pathology-2021-fgvcvc8/train.csv")
    if False
    else pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()


def _find_savedmodel_dir(base_path: str) -> str:
    """Return a directory that contains saved_model.pb(.txt); try common Kaggle layouts without deep scans."""

    def is_savedmodel_dir(p: str) -> bool:
        return (
            p
            and os.path.isdir(p)
            and (
                os.path.exists(os.path.join(p, "saved_model.pb"))
                or os.path.exists(os.path.join(p, "saved_model.pbtxt"))
            )
        )

    if is_savedmodel_dir(base_path):
        return base_path

    if base_path and os.path.isdir(base_path):
        try:
            for name in sorted(os.listdir(base_path)):
                cand = os.path.join(base_path, name)
                if is_savedmodel_dir(cand):
                    return cand
        except Exception:
            pass

    fallback_root = "../input/efficientb7"
    if is_savedmodel_dir(fallback_root):
        return fallback_root
    if os.path.isdir(fallback_root):
        try:
            for name in sorted(os.listdir(fallback_root)):
                cand = os.path.join(fallback_root, name)
                if is_savedmodel_dir(cand):
                    return cand
        except Exception:
            pass

    return base_path


efficientB7 = _find_savedmodel_dir(efficientB7)
print("Resolved EfficientNetB7 SavedModel path candidate:", efficientB7)
has_savedmodel = os.path.exists(
    os.path.join(efficientB7, "saved_model.pb")
) or os.path.exists(os.path.join(efficientB7, "saved_model.pbtxt"))
print("EfficientNetB7 SavedModel exists:", has_savedmodel)



## === cell 1
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
    from keras.layers import TFSMLayer  # Keras 3
except Exception:
    TFSMLayer = None


def _make_backbone_layer(backbone_path: str, image_dims_local):
    """
    Minimal, robust backbone loader:
    1) If a TensorFlow SavedModel is present, use Keras TFSMLayer (original intent).
    2) Else try TF Hub KerasLayer (common for EfficientNet assets).
    3) Else fall back to tf.keras.applications.EfficientNetB7(include_top=False).
    """
    if backbone_path and (
        os.path.exists(os.path.join(backbone_path, "saved_model.pb"))
        or os.path.exists(os.path.join(backbone_path, "saved_model.pbtxt"))
    ):
        if TFSMLayer is None:
            raise RuntimeError(
                "Found a SavedModel for the backbone, but keras.layers.TFSMLayer is unavailable."
            )
        call_endpoint = "serving_default"
        try:
            return TFSMLayer(backbone_path, call_endpoint=call_endpoint)
        except Exception:
            call_endpoint = "serve"
            return TFSMLayer(backbone_path, call_endpoint=call_endpoint)

    try:
        import tensorflow_hub as hub  # noqa: F401

        return hub.KerasLayer(backbone_path, trainable=False)
    except Exception:
        pass

    backbone = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=image_dims_local,
    )
    return backbone


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_backbone = _make_backbone_layer(efficientB7, image_dims)

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


def _resolve_ckpt_prefix(path: str) -> str:
    """
    Find a real TF checkpoint prefix to restore.
    Accepts:
      - a checkpoint prefix (ends with -####)
      - a .index filename
      - a directory containing checkpoints
    Returns a usable prefix, or None if not found.
    """
    if not path or not tf.io.gfile.exists(path):
        return None

    if not tf.io.gfile.isdir(path):
        if path.endswith(".index"):
            return path[:-6]
        if tf.io.gfile.exists(path + ".index"):
            return path
        return None

    ckpt = tf.train.latest_checkpoint(path)
    if ckpt is not None:
        return ckpt

    try:
        for f in sorted(tf.io.gfile.listdir(path)):
            if f.endswith(".index"):
                return os.path.join(path, f[:-6])
    except Exception:
        pass
    return None


def _autofind_checkpoint_under_kaggle_input(preferred_path: str) -> str:
    return _resolve_ckpt_prefix(preferred_path)




## === cell 2
if __name__ == "__main__":
    if not os.path.exists(test_dir):
        alt_test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
        if os.path.exists(alt_test_dir):
            test_dir = alt_test_dir

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    resolved_ckpt = _autofind_checkpoint_under_kaggle_input(model_dir)
    print("Resolved checkpoint prefix/path:", resolved_ckpt)

    if resolved_ckpt is not None:
        try:
            ckpt = tf.train.Checkpoint(model=model)
            status = ckpt.restore(resolved_ckpt)
            status.expect_partial()
            print("Restored checkpoint OK (expect_partial).")
        except Exception as e:
            print(
                "WARNING: Failed to restore weights from checkpoint; continuing with backbone weights only. Error:",
                repr(e),
            )
    else:
        print(
            "WARNING: No TensorFlow checkpoint found; continuing with backbone weights only."
        )

    images_path_list = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )
    if len(images_path_list) == 0:
        raise FileNotFoundError(f"No .jpg images found under test_dir='{test_dir}'")

    classes = dataset_labels

    autotune = tf.data.AUTOTUNE

    @tf.function
    def _load_and_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])
        img = img * 255.0  # keep original scaling
        return img

    @tf.function
    def _predict_batch(batch_imgs):
        return model(batch_imgs, training=False)

    batch_size = 32

    full_paths = [os.path.join(test_dir, f) for f in images_path_list]
    path_ds = tf.data.Dataset.from_tensor_slices(full_paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    path_ds = path_ds.with_options(options)

    img_ds = (
        path_ds.map(
            _load_and_preprocess, num_parallel_calls=autotune, deterministic=True
        )
        .cache()
        .batch(batch_size, drop_remainder=False)
        .prefetch(autotune)
    )

    thr = 0.7
    values = []

    start = 0
    for batch_imgs in img_ds:
        batch_out = _predict_batch(batch_imgs).numpy()  # (B, 6)
        mask = batch_out > thr
        for j in range(batch_out.shape[0]):
            idxs = np.flatnonzero(mask[j])
            if idxs.size == 0:
                labels_str = "healthy"
            else:
                labels_str = " ".join(classes[i] for i in idxs.tolist())
            values.append([images_path_list[start + j], labels_str])
        start += batch_out.shape[0]

    csv_pd = pd.DataFrame(values, columns=["image", "labels"], index=None)

    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape, "to:", out_path)
    print(csv_pd.head())
