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
import glob
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("TF_USE_LEGACY_KERAS", None)

import tensorflow as tf

print("TensorFlow:", tf.__version__)



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e10/epoch-10"
efficientB7_root = "../input/efficientb7/"

image_dims = (300, 300, 3)

train_csv_a = "../input/plant-pathology-2021-fgvc8/train.csv"
train_csv_b = "../input/plant-pathology-2021-fgvcvc8/train.csv"  # fallback typo-safe
data_set = (
    pd.read_csv(train_csv_a)
    if os.path.exists(train_csv_a)
    else pd.read_csv(train_csv_b)
)

df_labels = data_set["labels"].fillna("")
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()


def _find_savedmodel_dir(root_dir: str):
    candidates = []
    if root_dir and os.path.isdir(root_dir):
        candidates.append(root_dir)
        candidates.extend(glob.glob(os.path.join(root_dir, "*")))
        candidates.extend(glob.glob(os.path.join(root_dir, "*", "*")))
        candidates.extend(glob.glob(os.path.join(root_dir, "*", "*", "*")))

    for d in candidates:
        if os.path.isdir(d) and (
            os.path.exists(os.path.join(d, "saved_model.pb"))
            or os.path.exists(os.path.join(d, "saved_model.pbtxt"))
        ):
            return d
    return None


efficientB7 = _find_savedmodel_dir(efficientB7_root)
print("Found efficientB7 SavedModel:", efficientB7)



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


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_backbone = tf.keras.applications.EfficientNetB7(
            include_top=False, weights="imagenet", input_shape=image_dims
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


def _safe_restore_checkpoint(model: tf.keras.Model, path: str):
    """
    Restore variables via TF Checkpoint API for directory/prefix checkpoints.
    This is important to avoid accidentally predicting with random weights.
    """
    if not path or not os.path.exists(path):
        print("Weights path not found, running with current weights:", path)
        return False

    ckpt_prefix = None
    if os.path.isdir(path):
        ckpt_prefix = tf.train.latest_checkpoint(path)
        if ckpt_prefix is None:
            candidates = []
            candidates += glob.glob(os.path.join(path, "ckpt*"))
            candidates += glob.glob(os.path.join(path, "*.ckpt"))
            candidates += glob.glob(os.path.join(path, "*.index"))
            for c in sorted(candidates):
                if c.endswith(".index"):
                    c = c[: -len(".index")]
                ckpt_prefix = c
                break
    else:
        if path.endswith(".h5") or path.endswith(".weights.h5"):
            model.load_weights(path)
            print("Loaded Keras weights:", path)
            return True
        ckpt_prefix = path[:-6] if path.endswith(".index") else path

    if ckpt_prefix is None:
        print("No checkpoint found under:", path)
        return False

    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(ckpt_prefix)
    status.expect_partial()
    print("Restored checkpoint:", ckpt_prefix)
    return True




## === cell 3
if __name__ == "__main__":
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"test_dir not found: {test_dir}")

    test_dir = os.path.normpath(test_dir)
    nested = os.path.join(test_dir, "test_images")
    if os.path.isdir(nested) and os.path.basename(test_dir) != "test_images":
        test_dir = nested
    test_dir = test_dir.rstrip(os.sep) + os.sep

    try:
        tf.keras.utils.set_random_seed(123)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            min(8, os.cpu_count() or 8)
        )
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    _safe_restore_checkpoint(model, model_dir)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images_path_list = sorted(
        [
            e.name
            for e in os.scandir(test_dir)
            if e.is_file() and e.name.lower().endswith(exts)
        ]
    )
    if len(images_path_list) == 0:
        raise RuntimeError(f"No image files found under: {test_dir}")

    classes = dataset_labels

    AUTOTUNE = tf.data.AUTOTUNE

    batch_size = 64

    file_paths = [os.path.join(test_dir, n) for n in images_path_list]
    file_paths_tf = tf.constant(file_paths)

    def _load_and_preprocess(path):
        raw = tf.io.read_file(path)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])
        img = img * 255.0
        name = tf.strings.regex_replace(path, r"^.*/", "")
        return name, img

    ds = tf.data.Dataset.from_tensor_slices(file_paths_tf)

    options = tf.data.Options()
    options.experimental_deterministic = False
    try:
        options.threading.private_threadpool_size = min(32, (os.cpu_count() or 8))
        options.threading.max_intra_op_parallelism = 1
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    @tf.function(reduce_retracing=True, jit_compile=True)
    def _predict_batch(x):
        return model(x, training=False)

    import numpy as np

    threshold = 0.7
    classes_arr = np.asarray(classes, dtype=object)

    out_images = []
    out_labels = []

    for batch_names, batch_imgs in ds:
        preds_np = _predict_batch(batch_imgs).numpy()  # [B, 6]
        mask = preds_np > threshold  # [B, 6] bool
        batch_names_np = batch_names.numpy()

        for name_b, row_mask in zip(batch_names_np, mask):
            idx = np.flatnonzero(row_mask)
            if idx.size == 0:
                lbl = "healthy"
            else:
                lbl = " ".join(classes_arr[idx].tolist()).strip()
            out_images.append(name_b.decode("utf-8"))
            out_labels.append(lbl)

    csv_pd = pd.DataFrame({"image": out_images, "labels": out_labels})
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(csv_pd))
    print(csv_pd.head())
