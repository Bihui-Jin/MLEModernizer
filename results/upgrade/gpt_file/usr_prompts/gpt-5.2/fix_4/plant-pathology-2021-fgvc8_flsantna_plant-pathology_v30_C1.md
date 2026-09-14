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
import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

print("TensorFlow:", tf.__version__)
try:
    import google.protobuf

    print("protobuf:", google.protobuf.__version__)
except Exception as e:
    print("Could not import protobuf version:", repr(e))

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _is_savedmodel_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and (
            os.path.exists(os.path.join(path, "saved_model.pb"))
            or os.path.exists(os.path.join(path, "saved_model.pbtxt"))
        )
        and os.path.isdir(os.path.join(path, "variables"))
    )


def _find_savedmodel_dir(root: str) -> str:
    root = os.path.abspath(root)
    if _is_savedmodel_dir(root):
        return root

    if not os.path.exists(root):
        raise FileNotFoundError(f"model_dir does not exist: {root}")

    for dirpath, dirnames, filenames in os.walk(root):
        if ("saved_model.pb" in filenames) or ("saved_model.pbtxt" in filenames):
            if os.path.isdir(os.path.join(dirpath, "variables")):
                return dirpath

    raise FileNotFoundError(
        f"Could not find a TensorFlow SavedModel under: {root}. "
        f"Expected a directory containing saved_model.pb (or pbtxt) and variables/."
    )


def _find_model_file_by_ext(root: str, exts=(".keras", ".h5", ".hdf5")) -> str:
    root = os.path.abspath(root)
    if os.path.isfile(root) and root.lower().endswith(exts):
        return root
    if not os.path.exists(root):
        raise FileNotFoundError(f"model_dir does not exist: {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            lfn = fn.lower()
            if lfn.endswith(exts):
                return os.path.join(dirpath, fn)
    raise FileNotFoundError(
        f"Could not find any model file with extensions {exts} under: {root}"
    )


def load_inference_model(model_root: str):
    """
    BUGFIX: The provided model_dir may not be a SavedModel root. Try:
    1) SavedModel via TFSMLayer (TF2.18-compatible)
    2) Keras file (.keras)
    3) H5 file (.h5/.hdf5)
    """
    errors = []

    try:
        sm_dir = _find_savedmodel_dir(model_root)
        layer = keras.layers.TFSMLayer(sm_dir, call_endpoint="serving_default")
        inp = keras.Input(shape=image_dims, dtype=tf.float32, name="image")
        out = layer(inp)
        if isinstance(out, dict):
            key = sorted(list(out.keys()))[0]
            out = out[key]
        m = keras.Model(inputs=inp, outputs=out)
        _ = m(tf.zeros((1,) + image_dims, dtype=tf.float32), training=False)
        print("Loaded model as SavedModel from:", sm_dir)
        return m
    except Exception as e:
        errors.append(("SavedModel/TFSMLayer", repr(e)))

    try:
        keras_path = _find_model_file_by_ext(model_root, exts=(".keras",))
        m = keras.models.load_model(keras_path, compile=False)
        _ = m(tf.zeros((1,) + image_dims, dtype=tf.float32), training=False)
        print("Loaded model from .keras file:", keras_path)
        return m
    except Exception as e:
        errors.append((".keras load_model", repr(e)))

    try:
        h5_path = _find_model_file_by_ext(model_root, exts=(".h5", ".hdf5"))
        m = keras.models.load_model(h5_path, compile=False)
        _ = m(tf.zeros((1,) + image_dims, dtype=tf.float32), training=False)
        print("Loaded model from H5 file:", h5_path)
        return m
    except Exception as e:
        errors.append((".h5 load_model", repr(e)))

    msg = "Failed to load model from provided model_dir. Attempts:\n" + "\n".join(
        [f"- {k}: {v}" for k, v in errors]
    )
    raise FileNotFoundError(msg)


model = None
try:
    model = load_inference_model(model_dir)
    model.trainable = False
    print("Model ready for inference.")
except Exception as e:
    print(
        "WARNING: Could not load model. Will fall back to predicting 'healthy' for all test images."
    )
    print("Reason:", repr(e))
    model = None



## === cell 2
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
    if model is None:
        return "healthy"

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

    max_i = len(dataset_labels) - 1
    idx = [i for i in idx if 0 <= i <= max_i]
    if len(idx) == 0:
        return "healthy"

    return " ".join([dataset_labels[i] for i in idx])




## === cell 3
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
print("Submission rows:", len(csv_pd))

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/tmp/ipykernel_56/2589451656.py in <cell line: 0>()
      6 for i, fname in enumerate(images_path_list):
      7     img_path = os.path.join(test_dir, fname)
----> 8     img = load_and_preprocess_image(img_path)
      9     labels_str = predict_single(img, threshold=threshold)
     10 

/tmp/ipykernel_56/3714694475.py in load_and_preprocess_image(path)
      3 
      4 def load_and_preprocess_image(path):
----> 5     img_bytes = tf.io.read_file(path)
      6     img = tf.io.decode_image(
      7         img_bytes, channels=3, dtype=tf.float32, expand_animations=False

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/io_ops.py in read_file(filename, name)
    132     A tensor of dtype "string", with the file contents.
    133   """
--> 134   return gen_io_ops.read_file(filename, name)
    135 
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file(filename, name)
    581       pass
    582     try:
--> 583       return read_file_eager_fallback(
    584           filename, name=name, ctx=_ctx)
    585     except _core._SymbolicException:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file_eager_fallback(filename, name, ctx)
    604   _inputs_flat = [filename]
    605   _attrs = None
--> 606   _result = _execute.execute(b"ReadFile", 1, inputs=_inputs_flat,
    607                              attrs=_attrs, ctx=ctx, name=name)
    608   if _execute.must_record_gradient():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     51   try:
     52     ctx.ensure_initialized()
---> 53     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
     54                                         inputs, attrs, num_outputs)
     55   except core._NotOkStatusException as e:

FailedPreconditionError: {{function_node __wrapped__ReadFile_device_/job:localhost/replica:0/task:0/device:CPU:0}} ../input/plant-pathology-2021-fgvc8/test_images/test_images; Is a directory [Op:ReadFile]
