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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.7885318559556815

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
import subprocess

import os
import pandas as pd
import tensorflow as tf
import keras

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/model-effb7-01/epoch-5"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("Example classes:", dataset_labels[:10])




## === cell 2
def _find_savedmodel_dir(path: str) -> str:
    """
    Fix: model_dir currently points to a directory that may not directly contain saved_model.pb.
    This searches for a valid SavedModel directory under `path` with minimal assumptions.
    """
    path = os.path.expanduser(path)

    def is_savedmodel_dir(d: str) -> bool:
        return os.path.isdir(d) and (
            os.path.isfile(os.path.join(d, "saved_model.pb"))
            or os.path.isfile(os.path.join(d, "saved_model.pbtxt"))
        )

    if is_savedmodel_dir(path):
        return path

    for sub in ["saved_model", "SavedModel", "export", "model", "1"]:
        cand = os.path.join(path, sub)
        if is_savedmodel_dir(cand):
            return cand

    for root, dirs, files in os.walk(path):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            return root
        rel_depth = os.path.relpath(root, path).count(os.sep)
        if rel_depth >= 5:
            dirs[:] = []

    raise OSError(
        f"SavedModel file does not exist under: {path}. "
        f"Expected a directory containing saved_model.pb (or pbtxt)."
    )




## === cell 3
if __name__ == "__main__":
    resolved_model_dir = _find_savedmodel_dir(model_dir)
    print("Resolved model dir:", resolved_model_dir)

    sm_layer = keras.layers.TFSMLayer(
        resolved_model_dir, call_endpoint="serving_default"
    )

    dummy = tf.zeros((1, image_dims[0], image_dims[1], image_dims[2]), dtype=tf.float32)
    dummy_out = sm_layer(dummy)
    if isinstance(dummy_out, dict):
        out_key = sorted(dummy_out.keys())[0]
        print("SavedModel output is dict. Using key:", out_key)

        def predict_fn(x):
            return sm_layer(x)[out_key]

    else:
        print("SavedModel output is tensor.")

        def predict_fn(x):
            return sm_layer(x)

    images_path_list = sorted(list(os.listdir(test_dir)))

    def test_on_sub(index):
        input_img = tf.io.read_file(os.path.join(test_dir, images_path_list[index]))
        image = tf.io.decode_image(
            contents=input_img, channels=3, dtype=tf.dtypes.float32
        )
        image.set_shape([None, None, 3])
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    values = []
    threshold = 0.7  # keep core semantics

    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)

        images = images * 255.0

        test_values = predict_fn(images)
        test_values = tf.convert_to_tensor(test_values)

        index_values = [
            i for i, v in enumerate(test_values[0].numpy().tolist()) if v > threshold
        ]

        classes_img_list = [dataset_labels[i] for i in index_values]

        if len(classes_img_list) == 0:
            classes_img_list = ["healthy"]

        classes_img = " ".join(classes_img_list).strip()
        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(csv_path, index=False)
    print("Wrote:", csv_path, "rows:", len(csv_pd))
    print(csv_pd.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_56/365613459.py in <cell line: 0>()
      1 if __name__ == "__main__":
----> 2     resolved_model_dir = _find_savedmodel_dir(model_dir)
      3     print("Resolved model dir:", resolved_model_dir)
      4 
      5     sm_layer = keras.layers.TFSMLayer(

/tmp/ipykernel_56/2497350129.py in _find_savedmodel_dir(path)
     32             dirs[:] = []
     33 
---> 34     raise OSError(
     35         f"SavedModel file does not exist under: {path}. "
     36         f"Expected a directory containing saved_model.pb (or pbtxt)."

OSError: SavedModel file does not exist under: ../input/model-effb7-01/epoch-5. Expected a directory containing saved_model.pb (or pbtxt).
