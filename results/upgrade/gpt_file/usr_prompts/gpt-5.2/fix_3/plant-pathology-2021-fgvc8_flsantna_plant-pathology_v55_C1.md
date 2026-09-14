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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import tensorflow as tf

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

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
    from tensorflow.keras.layers import TFSMLayer  # fallback


def _find_savedmodel_dir(root_path: str) -> str:
    """
    FIX: The provided backbone path may point to a folder that doesn't directly contain saved_model.pb.
    We search for a valid SavedModel directory under root_path and fall back to root_path if it is valid.
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
    FIX: model.load_weights expects a checkpoint prefix or an H5 file.
    If a directory is provided, try to locate a plausible checkpoint inside.
    """
    weights_path = os.path.abspath(weights_path)

    if os.path.isfile(weights_path):
        return weights_path

    if os.path.isfile(weights_path + ".index"):
        return weights_path

    if os.path.isdir(weights_path):
        idx_files = glob.glob(os.path.join(weights_path, "*.index"))
        if idx_files:
            prefix = idx_files[0][:-6]  # remove ".index"
            return prefix

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


class MultiLabel(Model):
    def __init__(self, backbone_path: str):
        super().__init__()

        backbone_path = _find_savedmodel_dir(backbone_path)
        self.model_backbone = TFSMLayer(backbone_path, call_endpoint="serving_default")

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
    backbone_path = _find_savedmodel_dir(resnet50)
    weights_path = _find_weights_path(resnet50_weights)

    print("Using backbone:", backbone_path)
    print("Using weights:", weights_path)

    model = MultiLabel(backbone_path=backbone_path)
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])
    model.load_weights(weights_path)

    images_path_list = sorted(list(os.listdir(test_dir)))

    def test_on_sub(index):
        input_img = tf.io.read_file(test_dir + images_path_list[index])
        image = tf.io.decode_image(
            contents=input_img,
            channels=3,
            dtype=tf.dtypes.float32,
            expand_animations=False,
        )
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    threshold = 0.7  # keep original behavior

    values = []
    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)
        test_values = model.call(images, training=False)

        test_values = tf.convert_to_tensor(test_values)
        test_values = tf.reshape(test_values, [test_values.shape[0], -1])

        index_values = [
            i for i, v in enumerate(test_values[0].numpy().tolist()) if v > threshold
        ]

        classes = dataset_labels
        classes_img = ""
        for i in index_values:
            classes_img = str(classes[i]) + " " + classes_img

        classes_img = classes_img.strip()
        if classes_img == "":
            classes_img = "healthy"

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"], index=None)
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print(csv_pd.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1983688944.py in <cell line: 0>()
      1 if __name__ == "__main__":
      2     # Resolve backbone and weights robustly (bugfix; score-neutral).
----> 3     backbone_path = _find_savedmodel_dir(resnet50)
      4     weights_path = _find_weights_path(resnet50_weights)
      5 

/tmp/ipykernel_11/2472255852.py in _find_savedmodel_dir(root_path)
     46         return candidates[0]
     47 
---> 48     raise FileNotFoundError(
     49         f"Could not find a TensorFlow SavedModel under: {root_path}"
     50     )

FileNotFoundError: Could not find a TensorFlow SavedModel under: /kaggle/input/resnet50/Model-Resnet
