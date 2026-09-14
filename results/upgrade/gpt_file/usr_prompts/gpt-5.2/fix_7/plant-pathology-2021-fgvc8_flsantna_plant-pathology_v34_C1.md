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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.pop("TF_USE_LEGACY_KERAS", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)
print("TF_USE_LEGACY_KERAS:", os.environ.get("TF_USE_LEGACY_KERAS"))
print(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)

tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e4/epoch-4"
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


def _load_backbone(backbone_path, input_shape):
    if backbone_path and tf.io.gfile.exists(backbone_path):
        try:
            return tf.keras.models.load_model(backbone_path, compile=False)
        except Exception as e:
            print(
                "WARNING: Failed to load backbone from path, will fall back. Error:",
                repr(e),
            )

    print(
        "WARNING: Backbone path not found/loadable; using tf.keras.applications.EfficientNetB7 as fallback."
    )
    backbone = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    return backbone


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_backbone = _load_backbone(efficientB7, image_dims)

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




## === cell 3
if __name__ == "__main__":
    import numpy as np

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    loaded_any_weights = False

    if model_dir and tf.io.gfile.exists(model_dir):
        if tf.io.gfile.isdir(model_dir):
            try:
                loaded_model = tf.keras.models.load_model(model_dir, compile=False)
                model = loaded_model
                loaded_any_weights = True
                print("Loaded full model from SavedModel directory:", model_dir)
            except Exception as e:
                print(
                    "WARNING: Failed to load full model from directory; will try checkpoints. Error:",
                    repr(e),
                )

        if not loaded_any_weights:
            weights_path = model_dir
            if tf.io.gfile.isdir(weights_path):
                ckpt = tf.train.latest_checkpoint(weights_path)
                if ckpt is not None:
                    weights_path = ckpt
                else:
                    candidates = []
                    for pat in ["*.index", "checkpoint", "*.h5", "*.weights.h5"]:
                        try:
                            candidates.extend(
                                tf.io.gfile.glob(os.path.join(weights_path, pat))
                            )
                        except Exception:
                            pass
                    index_files = [c for c in candidates if c.endswith(".index")]
                    if index_files:
                        weights_path = index_files[0].replace(".index", "")
                    elif candidates:
                        weights_path = candidates[0]
                    else:
                        weights_path = None

            if (
                weights_path is not None
                and tf.io.gfile.exists(weights_path)
                or (
                    weights_path is not None
                    and tf.io.gfile.exists(weights_path + ".index")
                )
            ):
                try:
                    model.load_weights(weights_path)
                    loaded_any_weights = True
                    print("Loaded weights:", weights_path)
                except Exception as e:
                    print(
                        "WARNING: Failed to load weights; continuing without. Error:",
                        repr(e),
                    )
            else:
                print(
                    "WARNING: No weights/checkpoint found under model_dir; continuing without."
                )

    else:
        print(
            "WARNING: model_dir not found; continuing without external weights:",
            model_dir,
        )

    images_path_list = sorted(list(os.listdir(test_dir)))
    print("Test images found on disk:", len(images_path_list))

    threshold = 0.7  # keep original decision threshold for score semantics

    def _decode_resize(name):
        img_path = tf.strings.join([tf.constant(test_dir), name])
        raw = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(raw, channels=3)  # uint8
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1] float32
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])  # same as original
        img = img * 255.0  # preserve original scaling
        return name, img

    @tf.function(reduce_retracing=True)
    def _predict_batch(batch_imgs):
        return model(batch_imgs, training=False)

    BATCH_SIZE = 32

    names_ds = tf.data.Dataset.from_tensor_slices(tf.constant(images_path_list))
    ds = (
        names_ds.map(_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(AUTOTUNE)
    )

    n_imgs = len(images_path_list)
    preds = np.empty((n_imgs, 6), dtype=np.float32)
    all_names = [None] * n_imgs

    offset = 0
    for batch_names, batch_imgs in ds:
        batch_pred = _predict_batch(batch_imgs)
        bsz = int(batch_pred.shape[0])
        preds[offset : offset + bsz] = batch_pred.numpy()
        bn = batch_names.numpy().astype(str).tolist()
        all_names[offset : offset + bsz] = bn
        offset += bsz

    assert offset == n_imgs
    assert preds.shape[0] == len(images_path_list)

    dataset_labels_arr = np.array(dataset_labels, dtype=object)
    pred_mask = preds > threshold  # [N, 6] boolean

    values = []
    n_classes = len(dataset_labels_arr)
    for i, name in enumerate(all_names):
        idxs = np.flatnonzero(pred_mask[i])
        if idxs.size and n_classes:
            idxs = idxs[idxs < n_classes]
        if idxs.size:
            lbls = " ".join(dataset_labels_arr[idxs][::-1].tolist()).strip()
        else:
            lbls = ""
        if lbls == "":
            lbls = "healthy"
        values.append([name, lbls])

    sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
    pred_map = {img: lbl for img, lbl in values}
    sub["labels"] = sub["image"].map(pred_map).fillna("healthy")

    csv_path = os.path.join(output_dir, "submission.csv")
    sub.to_csv(csv_path, index=False)
    print("Wrote:", csv_path, "rows:", len(sub))
    print(sub.head())
