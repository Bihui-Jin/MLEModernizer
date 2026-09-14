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
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D,
    Conv2D,
    InputLayer,
)



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e9/epoch-9"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
NUM_CLASSES = len(dataset_labels)

print("NUM_CLASSES:", NUM_CLASSES)
print("Labels:", dataset_labels)



## === cell 2
from tensorflow import concat


def _find_saved_model_dir(path: str):
    """Return a directory containing a valid TF SavedModel (has saved_model.pb or saved_model.pbtxt)."""
    if not path or not os.path.exists(path):
        return None

    def is_savedmodel_dir(d):
        return os.path.isdir(d) and (
            os.path.exists(os.path.join(d, "saved_model.pb"))
            or os.path.exists(os.path.join(d, "saved_model.pbtxt"))
        )

    if is_savedmodel_dir(path):
        return path

    candidates = []
    for root, dirs, files in os.walk(path):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            candidates.append(root)
        rel_depth = os.path.relpath(root, path).count(os.sep)
        if rel_depth >= 2:
            dirs[:] = []

    if candidates:
        candidates = sorted(candidates, key=lambda p: (len(p), -os.path.getmtime(p)))
        return candidates[0]
    return None


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.using_savedmodel = False
        savedmodel_dir = _find_saved_model_dir(efficientB7)

        self.model_backbone = None
        if savedmodel_dir is not None:
            try:
                self.model_backbone = keras.layers.TFSMLayer(
                    savedmodel_dir, call_endpoint="serving_default"
                )
                self.using_savedmodel = True
            except Exception:
                try:
                    self.model_backbone = keras.layers.TFSMLayer(
                        savedmodel_dir, call_endpoint="serve"
                    )
                    self.using_savedmodel = True
                except Exception:
                    self.model_backbone = None
                    self.using_savedmodel = False

        if self.model_backbone is None:
            backbone = keras.applications.EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
                pooling=None,
            )
            backbone.trainable = False
            self.model_backbone = backbone
            self.using_savedmodel = False

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(
            Conv2D(filters=400 * NUM_CLASSES, kernel_size=(1, 1), padding="same")
        )

        self.pred_conv1 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.pred_bn = BatchNormalization()
        self.pred_conv2 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.pred_pool = GlobalMaxPool2D()
        self.pred_dense = Dense(units=1, activation="sigmoid")

    def _apply_pred_head(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn(x)
        x = self.pred_conv2(x)
        x = self.pred_pool(x)
        x = self.pred_dense(x)
        return x

    def call(self, x, training=False, **kwargs):
        y = self.model(x, training=training)

        if isinstance(y, dict):
            if "outputs" in y:
                y = y["outputs"]
            else:
                y = next(iter(y.values()))

        preds = []
        for i in range(NUM_CLASSES):
            sl = y[:, :, :, i * 400 : (i + 1) * 400]
            preds.append(self._apply_pred_head(sl))
        return concat(preds, axis=1)

    def create_model(self):
        return self.model




## === cell 3
if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    weights_path = model_dir
    if os.path.isdir(weights_path):
        candidates = []
        candidates += glob.glob(os.path.join(weights_path, "*.weights.h5"))
        candidates += glob.glob(os.path.join(weights_path, "*.h5"))
        candidates += glob.glob(os.path.join(weights_path, "ckpt*"))
        candidates += glob.glob(os.path.join(weights_path, "*"))
        candidates = [c for c in candidates if os.path.isfile(c)]
        if len(candidates) > 0:
            weights_path = max(candidates, key=os.path.getmtime)
        else:
            weights_path = None

    if weights_path is not None and os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("Loaded weights:", weights_path)
        except Exception:
            ckpt_prefix = weights_path
            if ckpt_prefix.endswith(".index"):
                ckpt_prefix = ckpt_prefix[: -len(".index")]
            model.load_weights(ckpt_prefix)
            print("Loaded checkpoint weights:", ckpt_prefix)
    else:
        print("No external weights found; using backbone default weights (if any).")

    images_path_list = sorted(
        [p for p in os.listdir(test_dir) if p.lower().endswith(".jpg")]
    )

    def test_on_sub(index):
        img_path = os.path.join(test_dir, images_path_list[index])
        input_img = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(input_img, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # [0,1]
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = os.path.basename(img_path)
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    values = []
    classes = dataset_labels

    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)

        images = images * 255.0

        test_values = model(images, training=False)  # shape (1, NUM_CLASSES)
        test_values_np = test_values.numpy()[0]

        index_values = [j for j, v in enumerate(test_values_np) if v > 0.7]

        classes_img = ""
        for j in index_values:
            classes_img = str(classes[j]) + " " + classes_img
        classes_img = classes_img.strip()

        if classes_img == "":
            classes_img = "healthy"

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape, "to", out_path)
    print(csv_pd.head())
