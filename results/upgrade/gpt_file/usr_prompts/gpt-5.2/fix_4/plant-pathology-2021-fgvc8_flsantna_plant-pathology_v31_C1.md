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

# 5. Target score

0.7671283471837491

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    BUGFIX: Robustly locate a TF SavedModel directory.
    - Accept direct SavedModel dir (contains saved_model.pb).
    - If hint is wrong, search within hint, its parent, and /kaggle/input.
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
    if os.path.isdir(parent):
        for root, _, files in os.walk(parent):
            if "saved_model.pb" in files:
                return root

    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            depth = root[len(kaggle_input) :].count(os.sep)
            if depth > 3:
                dirs[:] = []
                continue
            if "saved_model.pb" in files:
                return root

    raise FileNotFoundError(
        f"Could not locate a TensorFlow SavedModel directory from hint: {path_hint}"
    )


def _resolve_weights_path(path_hint: str) -> str:
    """
    BUGFIX: `load_weights` needs a concrete weights file (or a valid checkpoint prefix).
    If a directory is given, pick the latest checkpoint or newest likely weights file.
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

    return path_hint


efficientB7 = _find_saved_model_dir(efficientB7)
model_dir = _resolve_weights_path(model_dir)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3092226634.py in <cell line: 0>()
     84 
     85 # Resolve paths
---> 86 efficientB7 = _find_saved_model_dir(efficientB7)
     87 model_dir = _resolve_weights_path(model_dir)
     88 

/tmp/ipykernel_11/3092226634.py in _find_saved_model_dir(path_hint)
     52                 return root
     53 
---> 54     raise FileNotFoundError(
     55         f"Could not locate a TensorFlow SavedModel directory from hint: {path_hint}"
     56     )

FileNotFoundError: Could not locate a TensorFlow SavedModel directory from hint: ../input/efficientb7/effb7

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
    BUGFIX: TFSMLayer is not available with legacy Keras in this environment.
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

        self.model_backbone = SavedModelBackbone(
            efficientB7, call_endpoint="serving_default"
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




## === cell 3
if __name__ == "__main__":
    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    model.load_weights(model_dir)

    images_path_list = sorted(list(os.listdir(test_dir)))

    def test_on_sub(index):
        img_path = os.path.join(test_dir, images_path_list[index])
        input_img = tf.io.read_file(img_path)

        image = tf.io.decode_jpeg(input_img, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    threshold = 0.7
    values = []
    classes = dataset_labels

    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)

        test_values = model(images, training=False)
        test_values = tf.reshape(test_values, [-1]).numpy()

        index_values = [i for i, v in enumerate(test_values) if v > threshold]

        if len(index_values) == 0:
            classes_img = "healthy"
        else:
            classes_img = " ".join([str(classes[i]) for i in index_values])

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_pd.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3591155831.py in <cell line: 0>()
      1 if __name__ == "__main__":
----> 2     model = MultiLabel()
      3     model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])
      4 
      5     # Keep core logic: load provided weights path/prefix

/tmp/ipykernel_11/3114905148.py in __init__(self, *args, **kwargs)
     71         self.model = Sequential()
     72         self.model.add(InputLayer(input_shape=image_dims))
---> 73         self.model.add(self.model_backbone)
     74         self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
     75         self.model.add(BatchNormalization(momentum=0.7))

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/base.py in _method_wrapper(self, *args, **kwargs)
    202     self._self_setattr_tracking = False  # pylint: disable=protected-access
    203     try:
--> 204       result = method(self, *args, **kwargs)
    205     finally:
    206       self._self_setattr_tracking = previous_value  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/tmp/ipykernel_11/3114905148.py in build(self, input_shape)
     28     def build(self, input_shape):
     29         # Load once at build time
---> 30         self._loaded = tf.saved_model.load(self.saved_model_dir)
     31         if (
     32             hasattr(self._loaded, "signatures")

OSError: SavedModel file does not exist at: ../input/efficientb7/effb7/{saved_model.pbtxt|saved_model.pb}
