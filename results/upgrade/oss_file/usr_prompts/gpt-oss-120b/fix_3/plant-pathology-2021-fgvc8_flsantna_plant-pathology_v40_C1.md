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

0.7927423822714695

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import tensorflow as tf




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
num_classes = len(dataset_labels)




## === cell 2
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import GlobalMaxPool2D, Dense, InputLayer


class MultiLabel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        backbone = EfficientNetB7(
            weights="imagenet", include_top=False, input_shape=image_dims
        )
        self.model = Sequential(
            [
                InputLayer(input_shape=image_dims),
                backbone,
                GlobalMaxPool2D(),
                Dense(num_classes, activation="sigmoid"),
            ]
        )

    def call(self, x, **kwargs):
        return self.model(x)




## === cell 3
if __name__ == "__main__":
    model = MultiLabel()
    model.build(input_shape=[None, *image_dims])

    images_path_list = sorted(os.listdir(test_dir))

    def preprocess_path(fname):
        img_path = os.path.join(test_dir, fname)
        img_raw = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])
        img = tf.cast(img, tf.float32)  # EfficientNet expects float images
        return img

    image_tensors = [preprocess_path(fname) for fname in images_path_list]
    batch_tensor = tf.stack(image_tensors, axis=0)  # shape: (N, H, W, C)

    probs_batch = model(batch_tensor).numpy()  # shape: (N, num_classes)

    threshold = 0.5
    results = []
    for idx, probs in enumerate(probs_batch):
        pred_indices = [i for i, p in enumerate(probs) if p > threshold]
        pred_labels = " ".join([dataset_labels[i] for i in pred_indices])
        results.append([images_path_list[idx], pred_labels])

    submission_df = pd.DataFrame(results, columns=["image", "labels"])
    submission_df.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/tmp/ipykernel_11/440256156.py in <cell line: 0>()
     18 
     19     # Load all images into a list of tensors, then stack
---> 20     image_tensors = [preprocess_path(fname) for fname in images_path_list]
     21     batch_tensor = tf.stack(image_tensors, axis=0)  # shape: (N, H, W, C)
     22 

/tmp/ipykernel_11/440256156.py in <listcomp>(.0)
     18 
     19     # Load all images into a list of tensors, then stack
---> 20     image_tensors = [preprocess_path(fname) for fname in images_path_list]
     21     batch_tensor = tf.stack(image_tensors, axis=0)  # shape: (N, H, W, C)
     22 

/tmp/ipykernel_11/440256156.py in preprocess_path(fname)
     10     def preprocess_path(fname):
     11         img_path = os.path.join(test_dir, fname)
---> 12         img_raw = tf.io.read_file(img_path)
     13         # All competition images are JPEG; decode_jpeg is byte‑efficient
     14         img = tf.io.decode_jpeg(img_raw, channels=3)

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
