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

0.6434533702677737

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


def _ensure_tf_compatible_protobuf():
    """
    Bug fix: the original notebook tried to pin protobuf to 3.20.3, which is for older TF.
    In this Kaggle environment TF==2.18.0 is compatible with protobuf>=4, and protobuf==6.33.0 is installed.
    Downgrading protobuf can break TF 2.18 at runtime. So we only downgrade if TF is old (<2.13).
    """
    try:
        import tensorflow as _tf
        from importlib.metadata import version

        tf_ver = version("tensorflow")
        pb_ver = version("protobuf")
    except Exception:
        return

    def _major(v):
        try:
            return int(str(v).split(".")[0])
        except Exception:
            return None

    tf_major = _major(tf_ver)
    tf_minor = None
    try:
        tf_minor = int(str(tf_ver).split(".")[1])
    except Exception:
        pass

    if (
        tf_major is not None
        and tf_minor is not None
        and (tf_major < 2 or (tf_major == 2 and tf_minor < 13))
    ):
        if pb_ver is None or (_major(pb_ver) is not None and _major(pb_ver) >= 4):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf") or m == "protobuf":
                    sys.modules.pop(m, None)


_ensure_tf_compatible_protobuf()

import os
import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"

CANDIDATE_TEST_DIRS = [
    "../input/plant-pathology-2021-fgvc8/test_images/",
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images/",
    "../input/test_images/",
    "/kaggle/input/test_images/",
]
test_dir = None
for d in CANDIDATE_TEST_DIRS:
    if os.path.isdir(d):
        test_dir = d
        break
if test_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images dir. Tried: {CANDIDATE_TEST_DIRS}"
    )

image_dims = (300, 300, 3)

CANDIDATE_TRAIN_CSVS = [
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
    "../input/train.csv",
    "/kaggle/input/train.csv",
]
train_csv_path = None
for p in CANDIDATE_TRAIN_CSVS:
    if os.path.isfile(p):
        train_csv_path = p
        break
if train_csv_path is None:
    raise FileNotFoundError(f"Could not find train.csv. Tried: {CANDIDATE_TRAIN_CSVS}")

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Using train.csv:", train_csv_path)
print("Using test_dir:", test_dir)
print("Num classes:", len(dataset_labels))
print("Num test images (dir listing):", len(os.listdir(test_dir)))




## === cell 2
if __name__ == "__main__":
    try:
        import keras
        from keras import layers as keras_layers
    except Exception:
        keras = tf.keras
        keras_layers = tf.keras.layers

    def find_savedmodel_dir(search_roots):
        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
                    return dirpath
        return None

    candidate_roots = [
        "../input/model-complete-with-aug",
        "/kaggle/input/model-complete-with-aug",
        "../input",
        "/kaggle/input",
    ]
    savedmodel_path = find_savedmodel_dir(candidate_roots)
    if savedmodel_path is None:
        raise FileNotFoundError(
            "Could not locate a TensorFlow SavedModel directory under /kaggle/input. "
            f"Searched roots: {candidate_roots}"
        )

    print("Using SavedModel directory:", savedmodel_path)

    endpoints_to_try = ["serving_default", "call", "predict"]
    call_endpoint = None
    last_err = None
    _layer = None

    for ep in endpoints_to_try:
        try:
            _layer = keras_layers.TFSMLayer(savedmodel_path, call_endpoint=ep)
            call_endpoint = ep
            break
        except Exception as e:
            last_err = e

    if call_endpoint is None:
        sm = tf.saved_model.load(savedmodel_path)
        sigs = list(getattr(sm, "signatures", {}).keys())
        if len(sigs) == 0:
            raise RuntimeError(
                f"Could not find a callable endpoint in SavedModel at {savedmodel_path}. Last error: {last_err}"
            )
        call_endpoint = sigs[0]
        _layer = keras_layers.TFSMLayer(savedmodel_path, call_endpoint=call_endpoint)

    print("Using TFSMLayer endpoint:", call_endpoint)

    inp = keras.Input(shape=image_dims, name="image")
    out = _layer(inp)
    model = keras.Model(inp, out)

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

    threshold = 0.7  # preserve original logic

    values = []
    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)
        test_values = model(images, training=False)

        test_values = tf.convert_to_tensor(test_values)
        if len(test_values.shape) > 1:
            scores = test_values[0]
        else:
            scores = test_values

        index_values = [
            i for i, v in enumerate(scores.numpy().tolist()) if v > threshold
        ]

        if len(index_values) == 0:
            classes_img = "healthy"
        else:
            classes_img = " ".join([dataset_labels[i] for i in index_values])

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape, "to:", out_path)
    print(csv_pd.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2726286710.py in <cell line: 0>()
     27     savedmodel_path = find_savedmodel_dir(candidate_roots)
     28     if savedmodel_path is None:
---> 29         raise FileNotFoundError(
     30             "Could not locate a TensorFlow SavedModel directory under /kaggle/input. "
     31             f"Searched roots: {candidate_roots}"

FileNotFoundError: Could not locate a TensorFlow SavedModel directory under /kaggle/input. Searched roots: ['../input/model-complete-with-aug', '/kaggle/input/model-complete-with-aug', '../input', '/kaggle/input']
