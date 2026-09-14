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
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

print("TF version:", tf.__version__)



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
print("Num classes:", len(dataset_labels), dataset_labels)



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

TFSMLayer = tf.keras.layers.TFSMLayer


def _looks_like_savedmodel_dir(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    return os.path.isdir(path) and (
        os.path.isfile(os.path.join(path, "saved_model.pb"))
        or os.path.isfile(os.path.join(path, "saved_model.pbtxt"))
    )


def _find_savedmodel_dir(base_path: str, max_depth: int = 4) -> str:
    """
    Searches recursively (bounded by max_depth) for a directory containing saved_model.pb.
    """
    if _looks_like_savedmodel_dir(base_path):
        return base_path

    if not os.path.isdir(base_path):
        raise OSError(f"SavedModel not found (not a directory): {base_path}")

    base_path = os.path.abspath(base_path)

    def _depth(path: str) -> int:
        rel = os.path.relpath(path, base_path)
        if rel == ".":
            return 0
        return rel.count(os.sep) + 1

    for root, dirs, files in os.walk(base_path):
        d = _depth(root)
        if d > max_depth:
            dirs[:] = []
            continue
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            return root

    raise OSError(
        f"SavedModel not found. Checked recursively: {base_path} (depth<={max_depth})."
    )


def _candidate_backbone_roots():
    roots = [model_dir, efficientB7]
    roots += [
        os.path.dirname(model_dir),
        os.path.dirname(efficientB7),
        "../input",
        "/kaggle/input",
    ]
    seen = set()
    out = []
    for r in roots:
        if not r:
            continue
        rr = os.path.abspath(r)
        if rr not in seen:
            seen.add(rr)
            out.append(r)
    return out


def _resolve_backbone_savedmodel_path():
    last_err = None
    for p in _candidate_backbone_roots():
        try:
            return _find_savedmodel_dir(p, max_depth=6)
        except OSError as e:
            last_err = e
            continue
    try:
        return _find_savedmodel_dir("/kaggle/input", max_depth=8)
    except OSError as e:
        last_err = e
    return None, last_err


def _resolve_weights_path(weights_root: str):
    """
    model.load_weights() needs a weights file/prefix. Try to find a likely artifact.
    Returns a path suitable for tf.keras.Model.load_weights, or None if not found.
    """
    if not weights_root:
        return None
    if os.path.isfile(weights_root):
        return weights_root
    if not os.path.isdir(weights_root):
        return None

    ckpt_index = os.path.join(weights_root, "checkpoint")
    if os.path.isfile(ckpt_index):
        try:
            with open(ckpt_index, "r", encoding="utf-8") as f:
                txt = f.read()
            import re

            m = re.search(r'model_checkpoint_path:\s*"([^"]+)"', txt)
            if m:
                prefix = m.group(1)
                cand = os.path.join(weights_root, prefix)
                return cand
        except Exception:
            pass

    files = sorted(os.listdir(weights_root))
    for fn in files:
        if fn.endswith(".index"):
            return os.path.join(weights_root, fn[:-6])
    for fn in files:
        if fn.endswith(".weights.h5") or fn.endswith(".h5"):
            return os.path.join(weights_root, fn)
    return None


def _resolve_backbone_h5_path(base_path: str):
    if not base_path:
        return None
    if os.path.isfile(base_path) and (
        base_path.endswith(".h5") or base_path.endswith(".keras")
    ):
        return base_path
    if os.path.isdir(base_path):
        files = sorted(os.listdir(base_path))
        for fn in files:
            if fn.endswith(".h5") or fn.endswith(".keras"):
                return os.path.join(base_path, fn)
    return None


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_backbone = None
        self._backbone_kind = None

        savedmodel_path, savedmodel_err = _resolve_backbone_savedmodel_path()
        if savedmodel_path is not None:
            print("Using backbone SavedModel:", savedmodel_path)
            self._backbone_path = savedmodel_path
            self._backbone_endpoints = ["serving_default", "call", "predict"]
            last_exc = None
            for ep in self._backbone_endpoints:
                try:
                    self.model_backbone = TFSMLayer(savedmodel_path, call_endpoint=ep)
                    print("Backbone endpoint:", ep)
                    last_exc = None
                    self._backbone_kind = "tfsmlayer"
                    break
                except Exception as e:
                    last_exc = e
            if self.model_backbone is None:
                print(
                    f"WARNING: Found SavedModel but failed to create TFSMLayer: {last_exc}"
                )

        if self.model_backbone is None:
            h5_path = _resolve_backbone_h5_path(
                efficientB7
            ) or _resolve_backbone_h5_path(model_dir)
            if h5_path is not None:
                try:
                    print("Using backbone Keras model file:", h5_path)
                    self.model_backbone = tf.keras.models.load_model(
                        h5_path, compile=False
                    )
                    self._backbone_kind = "keras_model_file"
                except Exception as e:
                    print(
                        f"WARNING: Failed to load backbone .h5/.keras model ({h5_path}): {e}"
                    )
                    self.model_backbone = None

        if self.model_backbone is None:
            if savedmodel_err is not None:
                print(f"WARNING: SavedModel backbone not found: {savedmodel_err}")
            print(
                "Falling back to tf.keras.applications.EfficientNetB7 backbone (imagenet weights)."
            )
            self.model_backbone = tf.keras.applications.EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
            )
            self._backbone_kind = "keras_applications"

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
            if "outputs" in y:
                y = y["outputs"]
            elif "output_0" in y:
                y = y["output_0"]
            else:
                y = next(iter(y.values()))
        elif isinstance(y, (list, tuple)):
            y = y[0]

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
    if weights_path is not None:
        try:
            model.load_weights(weights_path)
            print(f"Loaded weights from: {weights_path}")
        except Exception as e:
            print(f"WARNING: Failed to load weights from {weights_path}: {e}")
    else:
        print(
            f"WARNING: No loadable weights found under model_dir={model_dir}. Proceeding without loading."
        )

    if not os.path.isdir(test_dir):
        raise OSError(f"test_dir not found: {test_dir}")

    images_path_list = sorted(list(os.listdir(test_dir)))
    print("Num test images:", len(images_path_list))

    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    def test_on_sub(index):
        fp = os.path.join(test_dir, images_path_list[index])
        input_img = tf.io.read_file(fp)
        image = tf.io.decode_image(
            contents=input_img,
            channels=3,
            dtype=tf.dtypes.float32,
            expand_animations=False,
        )  # float32 in [0,1]
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = os.path.basename(images_path_list[index])
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    values = []
    classes = dataset_labels

    thr = 0.7

    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)

        images = preprocess_input(images * 255.0)

        test_values = model.call(images)
        test_values = tf.convert_to_tensor(test_values)

        index_values = [
            i for i, v in enumerate(test_values[0].numpy().tolist()) if v > thr
        ]

        if len(index_values) == 0:
            classes_img = "healthy"
        else:
            classes_img = " ".join([str(classes[i]) for i in index_values]).strip()

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_pd = csv_pd.sort_values("image").reset_index(drop=True)

    sub_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(sub_path, index=False)
    print(f"Wrote submission: {sub_path} with shape {csv_pd.shape}")
    print(csv_pd.head())
