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

0.7200738688827324

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess


def _ensure_compatible_protobuf():
    try:
        from google.protobuf import __version__ as pb_ver  # noqa: F401

        major = int(str(pb_ver).split(".")[0])
        if major < 3:
            raise RuntimeError(f"Too old protobuf version detected: {pb_ver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf>=5.0.0"]
        )


_ensure_compatible_protobuf()



## === cell 1
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/model-onlye1/epoch-1"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("Example classes:", dataset_labels[:10])




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def _is_savedmodel_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and (
            os.path.exists(os.path.join(path, "saved_model.pb"))
            or os.path.exists(os.path.join(path, "saved_model.pbtxt"))
        )
        and os.path.isdir(
            os.path.join(path, "variables")
        )  # typical SavedModel structure
    )


def _find_savedmodel_dir(root: str) -> str:
    """
    BUGFIX: The provided model_dir may not be the exact SavedModel export root (could be nested).
    Search root and its subdirectories for a folder containing saved_model.pb and variables/.
    """
    root = os.path.abspath(root)
    if _is_savedmodel_dir(root):
        return root

    for dirpath, dirnames, filenames in os.walk(root):
        if ("saved_model.pb" in filenames) or ("saved_model.pbtxt" in filenames):
            if os.path.isdir(os.path.join(dirpath, "variables")):
                return dirpath

    raise FileNotFoundError(
        f"Could not find a TensorFlow SavedModel under: {root}. "
        f"Expected a directory containing saved_model.pb (or pbtxt) and variables/."
    )


def load_inference_model(savedmodel_dir: str):
    savedmodel_dir = _find_savedmodel_dir(savedmodel_dir)

    layer = keras.layers.TFSMLayer(savedmodel_dir, call_endpoint="serving_default")
    inp = keras.Input(shape=image_dims, dtype=tf.float32, name="image")
    out = layer(inp)
    if isinstance(out, dict):
        key = sorted(list(out.keys()))[0]
        out = out[key]
    return keras.Model(inputs=inp, outputs=out)


model = load_inference_model(model_dir)
model.trainable = False

_ = model(tf.zeros((1,) + image_dims, dtype=tf.float32), training=False)
print("Model loaded for inference.")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/3146098468.py in <cell line: 0>()
     47 
     48 
---> 49 model = load_inference_model(model_dir)
     50 model.trainable = False
     51 

/tmp/ipykernel_56/3146098468.py in load_inference_model(savedmodel_dir)
     34 
     35 def load_inference_model(savedmodel_dir: str):
---> 36     savedmodel_dir = _find_savedmodel_dir(savedmodel_dir)
     37 
     38     # TFSMLayer returns a layer; wrap it into a Keras Model for convenient calling.

/tmp/ipykernel_56/3146098468.py in _find_savedmodel_dir(root)
     27                 return dirpath
     28 
---> 29     raise FileNotFoundError(
     30         f"Could not find a TensorFlow SavedModel under: {root}. "
     31         f"Expected a directory containing saved_model.pb (or pbtxt) and variables/."

FileNotFoundError: Could not find a TensorFlow SavedModel under: /kaggle/input/model-onlye1/epoch-1. Expected a directory containing saved_model.pb (or pbtxt) and variables/.

## === cell 3
images_path_list = sorted(list(os.listdir(test_dir)))


def load_and_preprocess_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(
        img_bytes, channels=3, dtype=tf.float32, expand_animations=False
    )
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    img = tf.ensure_shape(img, image_dims)
    img = img * 255.0
    return img


def predict_single(image_tensor, threshold=0.7):
    batch = tf.expand_dims(image_tensor, axis=0)
    pred = model(batch, training=False)

    pred = tf.convert_to_tensor(pred)
    pred_np = pred.numpy()
    if pred_np.ndim == 1:
        pred_np = pred_np.reshape(1, -1)

    scores = pred_np[0]
    idx = np.where(scores > threshold)[0].tolist()

    if len(idx) == 0:
        idx = [int(np.argmax(scores))]

    return " ".join([dataset_labels[i] for i in idx])




## === cell 4
values = []
threshold = (
    0.7  # kept same as original code to preserve core logic/evaluation semantics
)

for i, fname in enumerate(images_path_list):
    img_path = os.path.join(test_dir, fname)
    img = load_and_preprocess_image(img_path)
    labels_str = predict_single(img, threshold=threshold)

    values.append([fname, labels_str])

    if (i + 1) % 500 == 0 or (i + 1) == len(images_path_list):
        print(f"Processed {i+1}/{len(images_path_list)}")

csv_pd = pd.DataFrame(values, columns=["image", "labels"])
csv_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(csv_path, index=False)
print("Wrote submission to:", csv_path)
print(csv_pd.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2945178170.py in <cell line: 0>()
      7     img_path = os.path.join(test_dir, fname)
      8     img = load_and_preprocess_image(img_path)
----> 9     labels_str = predict_single(img, threshold=threshold)
     10 
     11     values.append([fname, labels_str])

/tmp/ipykernel_56/3701992523.py in predict_single(image_tensor, threshold)
     15 def predict_single(image_tensor, threshold=0.7):
     16     batch = tf.expand_dims(image_tensor, axis=0)
---> 17     pred = model(batch, training=False)
     18 
     19     pred = tf.convert_to_tensor(pred)

NameError: name 'model' is not defined
