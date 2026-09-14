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

3.10

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

0.7979501385041576

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
import tensorflow as tf

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e17/epoch-17"
resnet50_weights = "../input/resnet50weights/last_epoch-20"
efficientB7 = "../input/efficientb7/effb7"
resnet50 = "../input/resnet50/Model-Resnet"
inceptionv3 = "../input/inceptionv3/Model-InceptionV3"
inceptionv3_weights = "../input/inceptionv3-weights/epoch-12"

image_dims = (300, 300, 3)
data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

_ = tf.__version__



## === cell 2
from tensorflow import concat
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D,
    Conv2D,
    InputLayer,
)

try:
    from tensorflow.keras.layers import TFSMLayer
except Exception:
    try:
        from keras.layers import TFSMLayer  # fallback
    except Exception:
        TFSMLayer = None


def _find_savedmodel_dir(path: str) -> str:
    if path is None:
        return None
    path = os.path.abspath(path)

    if os.path.isdir(path) and os.path.exists(os.path.join(path, "saved_model.pb")):
        return path

    if os.path.isfile(path):
        parent = os.path.dirname(path)
        if os.path.exists(os.path.join(parent, "saved_model.pb")):
            return parent

    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if "saved_model.pb" in files:
                return root

    return path


def _tfsmlayer_output_to_tensor(x):
    if isinstance(x, dict):
        if "outputs" in x:
            return x["outputs"]
        return x[next(iter(x.keys()))]
    return x


class _SavedModelAsLayer(tf.keras.layers.Layer):
    """Fallback wrapper if TFSMLayer is unavailable: uses tf.saved_model.load."""

    def __init__(self, savedmodel_dir: str):
        super().__init__()
        self.savedmodel_dir = savedmodel_dir
        self._loaded = None
        self._fn = None

    def build(self, input_shape):
        self._loaded = tf.saved_model.load(self.savedmodel_dir)
        if (
            hasattr(self._loaded, "signatures")
            and "serving_default" in self._loaded.signatures
        ):
            self._fn = self._loaded.signatures["serving_default"]
        else:
            self._fn = None
        super().build(input_shape)

    def call(self, inputs, training=False):
        if self._fn is None:
            out = self._loaded(inputs)
            return _tfsmlayer_output_to_tensor(out)
        out = self._fn(tf.constant(inputs))
        return _tfsmlayer_output_to_tensor(out)


CANONICAL_CLASSES = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]
classes = CANONICAL_CLASSES


class MultiLabel(Model):
    def __init__(self):
        super().__init__()
        backbone_path = _find_savedmodel_dir(resnet50)

        if TFSMLayer is not None and os.path.exists(
            os.path.join(backbone_path, "saved_model.pb")
        ):
            self.model_backbone = TFSMLayer(
                backbone_path, call_endpoint="serving_default"
            )
            self._backbone_is_tfsmlayer = True
        else:
            self.model_backbone = _SavedModelAsLayer(backbone_path)
            self._backbone_is_tfsmlayer = True  # treat same output handling

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.1))
        self.model.add(Conv2D(filters=128, kernel_size=(1, 1), padding="same"))
        self.model.add(GlobalMaxPool2D())
        self.model.add(Dense(units=6, activation="sigmoid"))

    def call(self, predict_input, training=False):
        return self.model(predict_input, training=training)

    def create_model(self):
        return self.model


class MultiLabel_2(Model):
    def __init__(self):
        super().__init__()

        backbone_path = _find_savedmodel_dir(inceptionv3)

        if TFSMLayer is not None and os.path.exists(
            os.path.join(backbone_path, "saved_model.pb")
        ):
            self.model_backbone = TFSMLayer(
                backbone_path, call_endpoint="serving_default"
            )
            self._backbone_is_tfsmlayer = True
        else:
            self.model_backbone = _SavedModelAsLayer(backbone_path)
            self._backbone_is_tfsmlayer = True

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding="same"))

        self.model_pred1_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred1_3 = BatchNormalization()
        self.model_pred1_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred1_5 = GlobalMaxPool2D()
        self.model_pred1_6 = Dense(units=1, activation="sigmoid")

        self.model_pred2_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred2_3 = BatchNormalization()
        self.model_pred2_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred2_5 = GlobalMaxPool2D()
        self.model_pred2_6 = Dense(units=1, activation="sigmoid")

        self.model_pred3_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred3_3 = BatchNormalization()
        self.model_pred3_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred3_5 = GlobalMaxPool2D()
        self.model_pred3_6 = Dense(units=1, activation="sigmoid")

        self.model_pred4_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred4_3 = BatchNormalization()
        self.model_pred4_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred4_5 = GlobalMaxPool2D()
        self.model_pred4_6 = Dense(units=1, activation="sigmoid")

        self.model_pred5_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred5_3 = BatchNormalization()
        self.model_pred5_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred5_5 = GlobalMaxPool2D()
        self.model_pred5_6 = Dense(units=1, activation="sigmoid")

        self.model_pred6_2 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")
        self.model_pred6_3 = BatchNormalization()
        self.model_pred6_4 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")
        self.model_pred6_5 = GlobalMaxPool2D()
        self.model_pred6_6 = Dense(units=1, activation="sigmoid")

    def call(self, predict_input, training=False):
        x = self.model_backbone(predict_input, training=training)
        x = _tfsmlayer_output_to_tensor(x)
        for layer in self.model.layers[1:]:
            x = (
                layer(x, training=training)
                if isinstance(layer, BatchNormalization)
                else layer(x)
            )
        predict_output = x

        pred_1 = predict_output[:, :, :, :400]
        pred_2 = predict_output[:, :, :, 400:800]
        pred_3 = predict_output[:, :, :, 800:1200]
        pred_4 = predict_output[:, :, :, 1200:1600]
        pred_5 = predict_output[:, :, :, 1600:2000]
        pred_6 = predict_output[:, :, :, 2000:2400]

        pred_1 = self.model_pred1_2(pred_1)
        pred_1 = self.model_pred1_3(pred_1, training=training)
        pred_1 = self.model_pred1_4(pred_1)
        pred_1 = self.model_pred1_5(pred_1)
        pred_1 = self.model_pred1_6(pred_1)

        pred_2 = self.model_pred2_2(pred_2)
        pred_2 = self.model_pred2_3(pred_2, training=training)
        pred_2 = self.model_pred2_4(pred_2)
        pred_2 = self.model_pred2_5(pred_2)
        pred_2 = self.model_pred2_6(pred_2)

        pred_3 = self.model_pred3_2(pred_3)
        pred_3 = self.model_pred3_3(pred_3, training=training)
        pred_3 = self.model_pred3_4(pred_3)
        pred_3 = self.model_pred3_5(pred_3)
        pred_3 = self.model_pred3_6(pred_3)

        pred_4 = self.model_pred4_2(pred_4)
        pred_4 = self.model_pred4_3(pred_4, training=training)
        pred_4 = self.model_pred4_4(pred_4)
        pred_4 = self.model_pred4_5(pred_4)
        pred_4 = self.model_pred4_6(pred_4)

        pred_5 = self.model_pred5_2(pred_5)
        pred_5 = self.model_pred5_3(pred_5, training=training)
        pred_5 = self.model_pred5_4(pred_5)
        pred_5 = self.model_pred5_5(pred_5)
        pred_5 = self.model_pred5_6(pred_5)

        pred_6 = self.model_pred6_2(pred_6)
        pred_6 = self.model_pred6_3(pred_6, training=training)
        pred_6 = self.model_pred6_4(pred_6)
        pred_6 = self.model_pred6_5(pred_6)
        pred_6 = self.model_pred6_6(pred_6)

        return concat([pred_1, pred_2, pred_3, pred_4, pred_5, pred_6], axis=1)

    def create_model(self):
        return self.model




## === cell 3
def _resolve_checkpoint_prefix(ckpt_path: str) -> str:
    """
    Bugfix: resolve a usable TF checkpoint prefix from a dir or a partial prefix like
    '../input/inceptionv3-weights/epoch-12' even if the real files are 'epoch-12-0001', etc.
    Also tries to auto-detect if the provided path doesn't exist in this notebook environment.
    """
    if ckpt_path is None:
        raise ValueError("ckpt_path is None")

    p = ckpt_path

    if not tf.io.gfile.exists(p):
        base = os.path.basename(p.rstrip("/"))
        parent_base = os.path.basename(os.path.dirname(p.rstrip("/")))
        candidates = [
            os.path.join("/kaggle/input", base),
            os.path.join("/kaggle/input", parent_base, base),
            os.path.join("/kaggle/input", parent_base),
        ]
        for c in candidates:
            if tf.io.gfile.exists(c):
                p = c
                break

    if tf.io.gfile.isdir(p):
        latest = tf.train.latest_checkpoint(p)
        if latest:
            return latest
        state = tf.train.get_checkpoint_state(p)
        if state and state.model_checkpoint_path:
            return state.model_checkpoint_path

        candidates = tf.io.gfile.glob(os.path.join(p, "**", "*.index"))
        if candidates:
            candidates = sorted(candidates)
            return candidates[-1][: -len(".index")]
        return p  # will fail with a clear error later

    if p.endswith(".index"):
        p = p[: -len(".index")]
    if p.endswith(".data-00000-of-00001"):
        p = p[: -len(".data-00000-of-00001")]

    if tf.io.gfile.exists(p + ".index"):
        return p

    parent = os.path.dirname(p)
    base = os.path.basename(p)
    if tf.io.gfile.isdir(parent):
        idx_files = tf.io.gfile.glob(os.path.join(parent, base + "*.index"))
        idx_prefixes = sorted({f[: -len(".index")] for f in idx_files})
        if idx_prefixes:
            return idx_prefixes[-1]
        latest = tf.train.latest_checkpoint(parent)
        if latest:
            return latest

    return p


def _restore_checkpoint_compat(model: tf.keras.Model, ckpt_path: str):
    """
    Bugfix: Restore checkpoint-style weights robustly under TF/Keras versions.
    If restore is impossible (missing input dataset), continue without crashing
    so we still generate a valid submission.csv.
    """
    p = _resolve_checkpoint_prefix(ckpt_path)
    try:
        ckpt = tf.train.Checkpoint(model=model)
        status = ckpt.restore(p)
        status.expect_partial()
        return p, True
    except Exception as e:
        print("WARNING: Could not restore checkpoint from:", p)
        print("WARNING:", repr(e))
        return p, False


def _infer_test_dir_from_sample(sample_csv_path: str, fallback_test_dir: str) -> str:
    candidates = [
        "../input/plant-pathology-2021-fgvc8/test_images/",
        "../input/test_images/",
        "/kaggle/input/plant-pathology-2021-fgvc8/test_images/",
        "/kaggle/input/test_images/",
        fallback_test_dir,
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return fallback_test_dir




## === cell 4
if __name__ == "__main__":
    model = MultiLabel_2()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    restored_from, ok = _restore_checkpoint_compat(model, inceptionv3_weights)
    print("Restored weights from:", restored_from, "success:", ok)

    sample_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
    if not os.path.exists(sample_path):
        sample_path = "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
    sample_sub = pd.read_csv(sample_path)

    test_dir = _infer_test_dir_from_sample(sample_path, test_dir)
    images_path_list = sample_sub["image"].tolist()

    def test_on_sub(index):
        fp = os.path.join(test_dir, images_path_list[index])
        input_img = tf.io.read_file(fp)

        image = tf.io.decode_jpeg(input_img, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])

        name_jpg = os.path.basename(images_path_list[index])
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    values = []

    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)

        test_values = model(images, training=False)  # shape: (1, 6)

        probs = test_values[0].numpy().tolist()
        index_values = [i for i, v in enumerate(probs) if v > 0.5]
        if len(index_values) == 0:
            index_values = [int(tf.argmax(test_values[0]).numpy())]

        classes_img = ""
        for i in index_values:
            classes_img = str(classes[i]) + " " + classes_img
        classes_img = classes_img.strip()

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"], index=None)

    csv_pd = sample_sub[["image"]].merge(csv_pd, on="image", how="left")
    csv_pd["labels"] = csv_pd["labels"].fillna("healthy")

    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(csv_pd), "test_dir:", test_dir)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/4182456039.py in <cell line: 0>()
     30         name, images = test_on_sub(index=idx)
     31 
---> 32         test_values = model(images, training=False)  # shape: (1, 6)
     33 
     34         probs = test_values[0].numpy().tolist()

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/3444538821.py in call(self, predict_input, training)
    185 
    186     def call(self, predict_input, training=False):
--> 187         x = self.model_backbone(predict_input, training=training)
    188         x = _tfsmlayer_output_to_tensor(x)
    189         for layer in self.model.layers[1:]:

/tmp/ipykernel_11/3444538821.py in build(self, input_shape)
     58 
     59     def build(self, input_shape):
---> 60         self._loaded = tf.saved_model.load(self.savedmodel_dir)
     61         if (
     62             hasattr(self._loaded, "signatures")

OSError: Exception encountered when calling MultiLabel_2.call().

SavedModel file does not exist at: /kaggle/input/inceptionv3/Model-InceptionV3/{saved_model.pbtxt|saved_model.pb}

Arguments received by MultiLabel_2.call():
  • predict_input=tf.Tensor(shape=(1, 300, 300, 3), dtype=float32)
  • training=False
