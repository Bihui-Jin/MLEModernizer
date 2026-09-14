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

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e2/epoch-2"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
dataset_labels = list(dataset_labels)




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


def _resolve_savedmodel_dir(path: str) -> str:
    """
    TFSMLayer expects a directory containing saved_model.pb / saved_model.pbtxt.
    Kaggle datasets sometimes nest the exported model one level deeper.
    """
    path = os.path.abspath(path)
    if os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "saved_model.pb"))
        or os.path.exists(os.path.join(path, "saved_model.pbtxt"))
    ):
        return path

    if os.path.isdir(path):
        try:
            for d in os.listdir(path):
                cand = os.path.join(path, d)
                if os.path.isdir(cand) and (
                    os.path.exists(os.path.join(cand, "saved_model.pb"))
                    or os.path.exists(os.path.join(cand, "saved_model.pbtxt"))
                ):
                    return cand
        except Exception:
            pass

        for root, dirs, files in os.walk(path):
            if "saved_model.pb" in files or "saved_model.pbtxt" in files:
                return root

    return path


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        backbone_dir = _resolve_savedmodel_dir(efficientB7)
        has_savedmodel = os.path.isdir(backbone_dir) and (
            os.path.exists(os.path.join(backbone_dir, "saved_model.pb"))
            or os.path.exists(os.path.join(backbone_dir, "saved_model.pbtxt"))
        )

        self._backbone_is_savedmodel = bool(has_savedmodel)
        if self._backbone_is_savedmodel:
            self.model_backbone = keras.layers.TFSMLayer(
                backbone_dir,
                call_endpoint="serving_default",
            )
        else:
            self.model_backbone = tf.keras.applications.EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
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

        if isinstance(y, dict):
            for k in ("output_0", "outputs", "predictions", "logits"):
                if k in y:
                    y = y[k]
                    break
            else:
                y = y[sorted(y.keys())[0]]

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
    test_dir = os.path.abspath(test_dir)
    candidates = [test_dir, os.path.join(test_dir, "test_images")]
    test_root = next((c for c in candidates if os.path.isdir(c)), test_dir)

    sample_sub = pd.read_csv(
        "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
    )
    images_path_list = sample_sub["image"].astype(str).tolist()

    base_dir1 = test_root
    base_dir2 = os.path.join(test_root, "test_images")
    probe = images_path_list[0] if images_path_list else None

    if (
        probe
        and os.path.exists(os.path.join(base_dir2, probe))
        and not os.path.exists(os.path.join(base_dir1, probe))
    ):
        base_dir = base_dir2
    else:
        base_dir = base_dir1

    if not base_dir.endswith(os.sep):
        base_dir = base_dir + os.sep

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    loaded = False
    try:
        if os.path.isdir(model_dir):
            ckpt = tf.train.latest_checkpoint(model_dir)
            if ckpt is None:
                model.load_weights(model_dir)
            else:
                model.load_weights(ckpt)
        else:
            model.load_weights(model_dir)
        loaded = True
    except Exception as e:
        print(
            f"WARNING: Could not load weights from {model_dir}. Proceeding without them. Error: {e}"
        )

    classes = list(dataset_labels)
    threshold = 0.7

    AUTOTUNE = tf.data.AUTOTUNE
    base_dir_tf = tf.constant(base_dir, dtype=tf.string)

    def _load_and_preprocess(name):
        file_path = tf.strings.join([base_dir_tf, name])
        raw = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(raw, channels=3)  # uint8
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # [0,1] float32
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])
        img = img * 255.0
        return name, img

    @tf.function(reduce_retracing=True)
    def _infer(batch_images):
        return model(batch_images, training=False)

    n_out_expected = 6  # model produces 6 heads by construction
    if len(classes) != n_out_expected:
        if len(classes) > n_out_expected:
            classes_use = classes[:n_out_expected]
        else:
            classes_use = classes + ["unknown"] * (n_out_expected - len(classes))
    else:
        classes_use = classes

    batch_size = 32

    options = tf.data.Options()
    options.experimental_deterministic = True

    ds = tf.data.Dataset.from_tensor_slices(
        tf.constant(images_path_list, dtype=tf.string)
    )
    ds = ds.with_options(options)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # safe: evaluation-only; avoids any accidental re-reads/recomputes
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    values = []
    for batch_names, batch_imgs in ds:
        preds = _infer(batch_imgs)
        preds_np = preds.numpy()  # shape [B, 6]

        mask = preds_np > threshold
        bn = batch_names.numpy()

        for name_b, row_mask in zip(bn, mask):
            idxs = np.flatnonzero(row_mask)
            if idxs.size == 0:
                classes_img = "healthy"
            else:
                classes_img = " ".join(classes_use[j] for j in idxs).strip()
            values.append([name_b.decode("utf-8"), classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_pd = sample_sub[["image"]].merge(csv_pd, on="image", how="left")
    csv_pd["labels"] = csv_pd["labels"].fillna("healthy")

    sub_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(sub_path, index=False)
    print(
        "Wrote:",
        os.path.abspath(sub_path),
        "rows:",
        len(csv_pd),
        "expected_rows:",
        len(sample_sub),
        "base_dir:",
        base_dir,
        "weights_loaded:",
        loaded,
    )
