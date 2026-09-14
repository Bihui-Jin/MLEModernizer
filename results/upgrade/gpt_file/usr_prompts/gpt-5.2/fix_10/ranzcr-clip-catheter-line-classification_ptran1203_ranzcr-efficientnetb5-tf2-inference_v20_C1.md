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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.9388390211512304

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.53831) has done: 'I fix the import/runtime failure caused by an incompatible protobuf backend by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` override, which triggers the `MessageFactory.GetPrototype` error in this environment. Then I fix the prediction loop crash by not calling `.numpy()` on the output of `predict_on_batch`, since it already returns a NumPy array. Finally, I make the submission robust to the provided sample_submission having only 10 columns by explicitly ensuring all 11 required target columns are present (filling missing ones with 0.0) and writing a valid `submission.csv`.'
- What this solution (achieved 0.54622) has done: 'I remove the forced protobuf backend override that’s currently breaking TensorFlow import in this environment, so `tf` is available and downstream cells don’t cascade into `NameError`. I also make dataset creation robust by falling back to reading JPEGs when the TFRecords directory doesn’t exist (it doesn’t in the provided input tree). Finally, I guarantee the submission matches the *required 11 target columns* even though the provided `sample_submission.csv` has only 10 columns, by using `train.csv` to define targets and then filling any missing target column(s) with 0.0.'
- What this solution (achieved 0.46133) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf package by forcing the pure-Python protobuf backend *before* importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error in this environment). I also correct the submission schema to always include all 11 required target columns (the provided `sample_submission.csv` here has only 10), adding any missing columns (e.g., `Swan Ganz Catheter Present`) filled with the model’s predictions when available or a safe default if not. Finally, I keep the existing model and inference loop intact, but ensure deterministic, correctly ordered `StudyInstanceUID` alignment and always write a valid `submission.csv`.'
- What this solution (achieved 0.50697) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf override that triggers `MessageFactory.GetPrototype` with the installed TF/protobuf versions, and keep imports in a safe order. I also make the submission schema always match the 11 required target columns by basing them on `train.csv` (since the provided sample submission here has only 10 columns). Finally, I keep your model/inference logic intact but make the JPEG dataset return `StudyInstanceUID` as a Tensor (so `.numpy()` works reliably) and ensure output row order aligns to the sample submission IDs.'
- What this solution (achieved 0.48698) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing a compatible pure-Python protobuf backend *before* importing TensorFlow (this is the root of the `MessageFactory.GetPrototype` failure). Then I keep your model/inference logic unchanged, but make the submission schema robust to the provided `sample_submission.csv` having only 10 columns by always using the 11 target columns from `train.csv` and inserting any missing columns with zeros. Finally, I ensure deterministic, correctly ordered `StudyInstanceUID` alignment to the sample submission and always write a valid `submission.csv`.'
- What this solution (achieved 0.49495) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf backend override, which is incompatible with the installed TF/protobuf versions and triggers the `MessageFactory.GetPrototype` error. Then I keep your model and inference loop intact, but ensure the input tensor shape matches EfficientNet expectations (use `(H, W, 3)` consistently) to avoid subtle mismatches. Finally, I make submission generation robust to the provided `sample_submission.csv` missing some required target columns by always producing all 11 target columns (adding any missing ones with 0.0) while preserving the sample submission row order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
import pandas as pd
import numpy as np

W = H = 338
N_CLASSES = 11
autotune = tf.data.experimental.AUTOTUNE

features = {
    "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}

train_csv_path = "../input/ranzcr-clip-catheter-line-classification/train.csv"
train_df_cols = pd.read_csv(train_csv_path, nrows=1).columns.tolist()
required_target_cols = [
    c for c in train_df_cols if c not in ("StudyInstanceUID", "PatientID")
]
target_cols = required_target_cols
assert (
    len(target_cols) == N_CLASSES
), f"Expected {N_CLASSES} targets, got {len(target_cols)}: {target_cols}"

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

test_tfrecords_dir = "../input/ranzcr-clip-catheter-line-classification/test_tfrecords"

weight_dir = "../input/cassava2020weights"
test_images_dir = "../input/ranzcr-clip-catheter-line-classification/test"

model_map = {
    "efficientb3": [
        tf.keras.applications.EfficientNetB3,
        weight_dir + "/ranzcr_efficientb3.h5",
    ],
    "efficientb5": [
        tf.keras.applications.EfficientNetB5,
        weight_dir + "/ranzcr_efficientb5.h5",
    ],
    "efficientb7": [
        tf.keras.applications.EfficientNetB7,
        weight_dir + "/ranzcr_efficientb5.h5",
    ],
}

print("TensorFlow:", tf.__version__)
print(
    "Protobuf backend env vars:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None),
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None),
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/3951565565.py in <cell line: 0>()
      6 os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
      7 
----> 8 import tensorflow as tf
      9 import pandas as pd
     10 import numpy as np

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
def triple_image(image):
    return tf.concat([image] * 3, axis=-1)


def parse_example(sample):
    sample = tf.io.parse_single_example(sample, features)
    image = tf.image.decode_png(sample["image"], channels=1)
    image = tf.image.resize(image, (H, W))
    image = triple_image(image)
    image_id = sample["StudyInstanceUID"]
    return image, image_id


def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = (images - mean) / std
    return images, labels


def build_test_dataset_from_tfrecords(batch_size=16):
    test_tfrecords = sorted(
        [
            f
            for f in os.listdir(test_tfrecords_dir)
            if f.endswith((".tfrec", ".tfrecord", ".tfrec.gz"))
        ]
    )
    if not test_tfrecords:
        test_tfrecords = sorted(os.listdir(test_tfrecords_dir))
    files = [os.path.join(test_tfrecords_dir, c) for c in test_tfrecords]
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=autotune)
    ds = ds.map(parse_example, num_parallel_calls=autotune)
    ds = ds.batch(batch_size)
    ds = ds.map(preprocess, num_parallel_calls=autotune)
    ds = ds.prefetch(1)
    return ds


def build_test_dataset_from_jpegs(batch_size=16):
    sample_sub = pd.read_csv(
        "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
    )
    uids = sample_sub["StudyInstanceUID"].astype(str).tolist()
    paths = [os.path.join(test_images_dir, f"{uid}.jpg") for uid in uids]

    def _load(uid, path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=1)
        img = tf.image.resize(img, (H, W))
        img = triple_image(img)
        uid = tf.cast(uid, tf.string)
        return img, uid

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(uids, dtype=tf.string), tf.constant(paths, dtype=tf.string))
    )
    ds = ds.map(_load, num_parallel_calls=autotune)
    ds = ds.batch(batch_size)
    ds = ds.map(preprocess, num_parallel_calls=autotune)
    ds = ds.prefetch(1)
    return ds


def get_model(
    base_model,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model(
        include_top=False, input_shape=(H, W, 3), pooling="avg", weights=baseline_weight
    )
    base_out = base_model.output
    out = tf.keras.layers.Dropout(0.3)(base_out)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(out)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )
    if init_weight:
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed, {e}")
    return model




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/406133621.py in <cell line: 0>()
     67     init_weight=None,
     68     lr=0.001,
---> 69     optimizer=tf.optimizers.Adam,
     70 ):
     71     base_model = base_model(

NameError: name 'tf' is not defined

## === cell 2
if os.path.isdir(test_tfrecords_dir) and len(os.listdir(test_tfrecords_dir)) > 0:
    test_data = build_test_dataset_from_tfrecords(batch_size=16)
else:
    test_data = build_test_dataset_from_jpegs(batch_size=16)

base_model_fn, weight_path = model_map["efficientb5"]
model = get_model(base_model_fn, init_weight=weight_path)

sample_sub_path = (
    "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
)
test_df = pd.read_csv(sample_sub_path)

test_uids_order = test_df["StudyInstanceUID"].astype(str).tolist()
uid_to_index = {u: i for i, u in enumerate(test_uids_order)}



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2505460335.py in <cell line: 0>()
----> 1 if os.path.isdir(test_tfrecords_dir) and len(os.listdir(test_tfrecords_dir)) > 0:
      2     test_data = build_test_dataset_from_tfrecords(batch_size=16)
      3 else:
      4     test_data = build_test_dataset_from_jpegs(batch_size=16)
      5 

NameError: name 'test_tfrecords_dir' is not defined

## === cell 3
preds = []
image_ids = []

for images, image_id in test_data:
    batch_ids = image_id.numpy().astype(str).tolist()
    image_ids.extend(batch_ids)

    pred = model.predict_on_batch(images)  # already numpy
    preds.append(pred)

preds = np.concatenate(preds, axis=0)

if len(image_ids) != preds.shape[0]:
    raise RuntimeError(
        f"Prediction count mismatch: {len(image_ids)} ids vs {preds.shape[0]} preds"
    )

pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df.insert(0, "StudyInstanceUID", image_ids)

for c in target_cols:
    if c not in pred_df.columns:
        pred_df[c] = 0.0

submission_cols = ["StudyInstanceUID"] + target_cols
submission = pred_df[submission_cols].copy()

submission["__order"] = submission["StudyInstanceUID"].map(uid_to_index)
missing_mask = submission["__order"].isna()
if missing_mask.any():
    submission = submission.loc[~missing_mask].copy()
submission = submission.sort_values("__order").drop(columns="__order")

submission = (
    submission.set_index("StudyInstanceUID").reindex(test_uids_order).reset_index()
)

for c in target_cols:
    if c not in submission.columns:
        submission[c] = 0.0
submission[target_cols] = submission[target_cols].fillna(0.0)

for c in target_cols:
    submission[c] = submission[c].astype(np.float32).clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print(submission.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3557033968.py in <cell line: 0>()
      2 image_ids = []
      3 
----> 4 for images, image_id in test_data:
      5     batch_ids = image_id.numpy().astype(str).tolist()
      6     image_ids.extend(batch_ids)

NameError: name 'test_data' is not defined
