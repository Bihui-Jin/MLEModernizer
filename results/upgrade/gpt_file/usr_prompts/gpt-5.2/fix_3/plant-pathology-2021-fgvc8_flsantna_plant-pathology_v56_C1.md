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

0.3914655058699362

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

dataset_labels = list(dataset_labels)
print("Classes:", dataset_labels, "n_classes=", len(dataset_labels))



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


def _find_savedmodel_dir(path: str) -> str:
    """Return a directory that contains a SavedModel (saved_model.pb / pbtxt), or raise."""
    if path and tf.io.gfile.exists(path):
        if tf.io.gfile.isdir(path):
            entries = tf.io.gfile.listdir(path)
            if "saved_model.pb" in entries or "saved_model.pbtxt" in entries:
                return path
            for sub in entries:
                subp = os.path.join(path, sub)
                if tf.io.gfile.isdir(subp):
                    sub_entries = tf.io.gfile.listdir(subp)
                    if (
                        "saved_model.pb" in sub_entries
                        or "saved_model.pbtxt" in sub_entries
                    ):
                        return subp
    raise OSError(f"SavedModel not found under: {path}")


def _load_backbone_layer(path: str):
    """
    BUGFIX: Original code assumes a SavedModel exists at `resnet50` and that TFSMLayer is available.
    In many Kaggle runtimes, the asset path differs or only `tf.keras.models.load_model()` works.
    We keep identical semantics: backbone is a callable model/layer producing feature maps.
    """
    try:
        from keras.layers import TFSMLayer  # type: ignore

        sm_dir = _find_savedmodel_dir(path)
        return TFSMLayer(sm_dir, call_endpoint="serving_default")
    except Exception as e1:
        try:
            sm_dir = _find_savedmodel_dir(path)
            loaded = tf.keras.models.load_model(sm_dir, compile=False)
            return loaded
        except Exception as e2:
            raise OSError(
                f"Failed to load backbone from {path}. "
                f"TFSMLayer error: {repr(e1)}; load_model error: {repr(e2)}"
            )


class MultiLabel(Model):
    def __init__(self):
        super().__init__()

        self.model_backbone = _load_backbone_layer(resnet50)

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

    def call(self, predict_input):
        return self.model(predict_input)

    def create_model(self):
        return self.model




## === cell 3
if __name__ == "__main__":
    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    model.load_weights(resnet50_weights).expect_partial()

    sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
    sub_df = pd.read_csv(sample_sub_path)
    images_path_list = sub_df["image"].tolist()

    def test_on_sub(image_name: str):
        img_path = os.path.join(test_dir, image_name)
        input_img = tf.io.read_file(img_path)
        image = tf.io.decode_image(
            contents=input_img, channels=3, dtype=tf.dtypes.float32
        )
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        return image_name, tf.expand_dims(tensor_image, axis=0)

    values = []
    classes = dataset_labels

    for name in images_path_list:
        _, images = test_on_sub(image_name=name)
        test_values = model.call(images)

        preds = test_values[0].numpy().tolist()
        index_values = [j for j, v in enumerate(preds) if v > 0.6]

        if len(index_values) == 0:
            classes_img = "healthy"
        else:
            classes_img = " ".join([str(classes[j]) for j in index_values])

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(csv_pd))

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2714510998.py in _load_backbone_layer(path)
     42 
---> 43         sm_dir = _find_savedmodel_dir(path)
     44         return TFSMLayer(sm_dir, call_endpoint="serving_default")

/tmp/ipykernel_11/2714510998.py in _find_savedmodel_dir(path)
     29                         return subp
---> 30     raise OSError(f"SavedModel not found under: {path}")
     31 

OSError: SavedModel not found under: ../input/resnet50/Model-Resnet

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2714510998.py in _load_backbone_layer(path)
     46         try:
---> 47             sm_dir = _find_savedmodel_dir(path)
     48             loaded = tf.keras.models.load_model(sm_dir, compile=False)

/tmp/ipykernel_11/2714510998.py in _find_savedmodel_dir(path)
     29                         return subp
---> 30     raise OSError(f"SavedModel not found under: {path}")
     31 

OSError: SavedModel not found under: ../input/resnet50/Model-Resnet

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/644545780.py in <cell line: 0>()
      1 if __name__ == "__main__":
      2     # Build model so weights can load
----> 3     model = MultiLabel()
      4     model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])
      5 

/tmp/ipykernel_11/2714510998.py in __init__(self)
     60 
     61         # Robust backbone loading
---> 62         self.model_backbone = _load_backbone_layer(resnet50)
     63 
     64         self.model = Sequential()

/tmp/ipykernel_11/2714510998.py in _load_backbone_layer(path)
     49             return loaded
     50         except Exception as e2:
---> 51             raise OSError(
     52                 f"Failed to load backbone from {path}. "
     53                 f"TFSMLayer error: {repr(e1)}; load_model error: {repr(e2)}"

OSError: Failed to load backbone from ../input/resnet50/Model-Resnet. TFSMLayer error: OSError('SavedModel not found under: ../input/resnet50/Model-Resnet'); load_model error: OSError('SavedModel not found under: ../input/resnet50/Model-Resnet')
