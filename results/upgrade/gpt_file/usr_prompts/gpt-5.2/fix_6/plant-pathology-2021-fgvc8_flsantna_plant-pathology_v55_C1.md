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

0.3558448753462597

# 6. Current score

0.31198

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.31198) has done: 'I fix the TensorFlow import crash by removing the protobuf-related environment override that breaks TF’s internal message factory in this Kaggle runtime. Then I fix the dataset input bug where `os.listdir(test_dir)` includes the nested `test_images` directory, causing `tf.io.read_file` to try reading a directory; the fix is to filter for actual image files (or just follow `sample_submission.csv` ordering). Finally, I keep your model/thresholding logic intact while ensuring the submission is written as `submission.csv` with the required `image,labels` columns and correct row alignment.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)



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

image_dims = (300, 300, 3)
data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels), dataset_labels)

if len(dataset_labels) != 6:
    warnings.warn(
        f"Expected 6 classes but found {len(dataset_labels)} in train.csv; "
        f"will use the first 6 for mapping."
    )
dataset_labels = dataset_labels[:6]



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

try:
    from keras.layers import TFSMLayer  # Keras 3 style
except Exception:
    TFSMLayer = None


def _find_savedmodel_dir(root_path: str) -> str:
    """
    Find a valid SavedModel directory under root_path.
    """
    root_path = os.path.abspath(root_path)

    def is_savedmodel_dir(p: str) -> bool:
        return os.path.isdir(p) and (
            os.path.isfile(os.path.join(p, "saved_model.pb"))
            or os.path.isfile(os.path.join(p, "saved_model.pbtxt"))
        )

    if is_savedmodel_dir(root_path):
        return root_path

    candidates = []
    for pb in glob.glob(
        os.path.join(root_path, "**", "saved_model.pb"), recursive=True
    ):
        candidates.append(os.path.dirname(pb))
    for pbtxt in glob.glob(
        os.path.join(root_path, "**", "saved_model.pbtxt"), recursive=True
    ):
        candidates.append(os.path.dirname(pbtxt))

    candidates = sorted(set(candidates))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find a TensorFlow SavedModel under: {root_path}"
    )


def _find_weights_path(weights_path: str) -> str:
    """
    Resolve TF checkpoint prefix or an H5/Keras file from a provided path.
    """
    weights_path = os.path.abspath(weights_path)

    if os.path.isfile(weights_path):
        return weights_path

    if os.path.isfile(weights_path + ".index"):
        return weights_path

    if os.path.isdir(weights_path):
        idx_files = glob.glob(os.path.join(weights_path, "*.index"))
        if idx_files:
            return idx_files[0][:-6]  # remove ".index"

        h5_files = glob.glob(os.path.join(weights_path, "*.h5")) + glob.glob(
            os.path.join(weights_path, "*.keras")
        )
        if h5_files:
            return h5_files[0]

    parent = os.path.dirname(weights_path)
    base = os.path.basename(weights_path)
    idx_match = glob.glob(os.path.join(parent, base + "*.index"))
    if idx_match:
        return idx_match[0][:-6]

    raise FileNotFoundError(f"Could not resolve weights path from: {weights_path}")


def _build_backbone_layer(backbone_path: str, image_dims):
    """
    Try loading backbone from a SavedModel via TFSMLayer; otherwise fall back to ImageNet ResNet50.
    """
    if TFSMLayer is not None:
        try:
            sm_dir = _find_savedmodel_dir(backbone_path)
            return TFSMLayer(sm_dir, call_endpoint="serving_default")
        except Exception as e:
            warnings.warn(
                f"Could not load SavedModel backbone from '{backbone_path}': {e}"
            )

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=image_dims,
        pooling=None,
    )
    backbone.trainable = False
    return backbone


class MultiLabel(Model):
    def __init__(self, backbone_path: str):
        super().__init__()

        self.model_backbone = _build_backbone_layer(backbone_path, image_dims)

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
        predict_output = self.model(predict_input, training=training)

        if isinstance(predict_output, dict):
            k = sorted(predict_output.keys())[0]
            predict_output = predict_output[k]

        return predict_output

    def create_model(self):
        return self.model




## === cell 3
if __name__ == "__main__":
    tf.random.set_seed(123)

    model = MultiLabel(backbone_path=resnet50)
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    try:
        weights_path = _find_weights_path(resnet50_weights)
        print("Using weights:", weights_path)
        model.load_weights(weights_path)
    except Exception as e:
        warnings.warn(f"Could not load weights from '{resnet50_weights}': {e}")

    sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
    sample_df = pd.read_csv(sample_sub_path)
    images_path_list = sample_df["image"].astype(str).tolist()

    images_path_list = [
        n
        for n in images_path_list
        if tf.io.gfile.exists(os.path.join(test_dir, n))
        and tf.io.gfile.isdir(os.path.join(test_dir, n)) is False
    ]

    img_names_tf = tf.constant(images_path_list)

    def _load_and_preprocess(name):
        path = tf.strings.join([test_dir, name])
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(
            contents=raw,
            channels=3,
            dtype=tf.dtypes.float32,
            expand_animations=False,
        )
        image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        return name, image

    ds = tf.data.Dataset.from_tensor_slices(img_names_tf)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    BATCH_SIZE = 32
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    threshold = 0.7  # keep original behavior

    @tf.function(reduce_retracing=True)
    def _predict_batch(x):
        y = model.call(x, training=False)
        if isinstance(y, dict):
            k = tf.sort(tf.constant(list(y.keys())))[0]
            y = y[k]
        y = tf.convert_to_tensor(y)
        y = tf.reshape(y, [tf.shape(y)[0], -1])
        return y

    values = []
    for names_b, images_b in ds:
        probs_b = _predict_batch(images_b)  # [B, 6]
        mask_b = probs_b > threshold

        idxs = tf.ragged.boolean_mask(
            tf.tile(
                tf.range(tf.shape(mask_b)[1])[tf.newaxis, :], [tf.shape(mask_b)[0], 1]
            ),
            mask_b,
        )

        names_py = names_b.numpy()
        idxs_py = idxs.to_list()

        for n_bytes, ind_list in zip(names_py, idxs_py):
            name = (
                n_bytes.decode("utf-8")
                if isinstance(n_bytes, (bytes, bytearray))
                else str(n_bytes)
            )
            classes_img = ""
            for i in ind_list:
                if i < len(dataset_labels):
                    classes_img = str(dataset_labels[i]) + " " + classes_img
            classes_img = classes_img.strip()
            if classes_img == "":
                classes_img = "healthy"
            values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"], index=None)

    if len(csv_pd) == len(sample_df):
        csv_pd = sample_df[["image"]].merge(csv_pd, on="image", how="left")
        csv_pd["labels"] = csv_pd["labels"].fillna("healthy")

    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print(csv_pd.head())
