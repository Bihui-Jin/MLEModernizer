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

0.9030920693473292

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.50753) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf setting that breaks TF 2.18 in this environment, and add a small compatibility guard so the notebook still runs if the variable is set externally. I also fix the inference error by treating `predict_on_batch` as already returning a NumPy array (so `.numpy()` is invalid) while keeping the model and pipeline unchanged. Finally, I ensure the submission matches the required `sample_submission.csv` columns even though the provided sample file is missing some targets, by backfilling the missing required columns from `train.csv` and defaulting any truly absent ones to 0.0, so a valid `submission.csv` is always produced end-to-end.'

# 9. Code solution

## === cell 0
import os

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() == "python":
    print(
        "Warning: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is set; overriding to 'cpp' "
        "to avoid TF/protobuf incompatibility."
    )
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

W = H = 448
N_CLASSES = 11
autotune = tf.data.AUTOTUNE

target_cols = [
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "Swan Ganz Catheter Present",
]

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

DATA_DIR = "../input/ranzcr-clip-catheter-line-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")

model_map = {
    "efficientb3": tf.keras.applications.EfficientNetB3,
    "efficientb5": tf.keras.applications.EfficientNetB5,
    "efficientb7": tf.keras.applications.EfficientNetB7,
}

print("TensorFlow:", tf.__version__)
print("Train image dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/740116637.py in <cell line: 0>()
     10 os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
     11 
---> 12 import tensorflow as tf
     13 import pandas as pd
     14 import numpy as np

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
    if image.shape.rank is not None and image.shape.rank >= 3 and image.shape[-1] == 3:
        return image
    return tf.concat([image] * 3, axis=-1)


def _decode_and_preprocess(img_bytes):
    image = tf.image.decode_jpeg(img_bytes, channels=1)  # x-rays are grayscale
    image = tf.image.resize(image, (H, W), method="bilinear")
    image = triple_image(image)
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - mean) / std
    return image


def parse_jpg(path):
    img_bytes = tf.io.read_file(path)
    image = _decode_and_preprocess(img_bytes)
    uid = tf.strings.split(path, os.sep)[-1]
    uid = tf.strings.regex_replace(uid, r"\.jpg$", "")
    return image, uid


def parse_train_jpg(path, label_vec):
    img_bytes = tf.io.read_file(path)
    image = _decode_and_preprocess(img_bytes)
    label_vec = tf.cast(label_vec, tf.float32)
    return image, label_vec


def get_model(
    base_model,
    baseline_weight="imagenet",
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model(
        include_top=False,
        input_shape=(W, H, 3),
        pooling="avg",
        weights=baseline_weight,
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
/tmp/ipykernel_55/3983484829.py in <cell line: 0>()
     34     init_weight=None,
     35     lr=0.001,
---> 36     optimizer=tf.optimizers.Adam,
     37 ):
     38     base_model = base_model(

NameError: name 'tf' is not defined

## === cell 2
id_col = "StudyInstanceUID"
train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)

req_target_cols = [c for c in target_cols if c in train_df.columns]
missing_in_train = [c for c in target_cols if c not in train_df.columns]
if missing_in_train:
    print(
        "Warning: these required targets are missing from train.csv and will be defaulted to 0:",
        missing_in_train,
    )

train_df["path"] = (
    train_df[id_col].astype(str).map(lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.jpg"))
)

exists_mask = train_df["path"].map(tf.io.gfile.exists)
if hasattr(exists_mask, "values"):
    exists_mask = exists_mask.values
train_df = train_df.loc[exists_mask].reset_index(drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(gss.split(train_df, groups=train_df["PatientID"]))
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train/val sizes:", len(tr_df), len(va_df))

y_tr = tr_df[req_target_cols].astype(np.float32).values
y_va = va_df[req_target_cols].astype(np.float32).values

if missing_in_train:
    y_tr = np.concatenate(
        [y_tr, np.zeros((len(y_tr), len(missing_in_train)), dtype=np.float32)], axis=1
    )
    y_va = np.concatenate(
        [y_va, np.zeros((len(y_va), len(missing_in_train)), dtype=np.float32)], axis=1
    )
    req_target_cols = req_target_cols + missing_in_train

assert (
    len(req_target_cols) == N_CLASSES
), f"Expected {N_CLASSES} targets, got {len(req_target_cols)}"

BATCH_SIZE = 16

train_ds = tf.data.Dataset.from_tensor_slices((tr_df["path"].values, y_tr))
train_ds = train_ds.shuffle(
    min(len(tr_df), 4096), seed=42, reshuffle_each_iteration=True
)
train_ds = (
    train_ds.map(parse_train_jpg, num_parallel_calls=autotune)
    .batch(BATCH_SIZE)
    .prefetch(autotune)
)

val_ds = tf.data.Dataset.from_tensor_slices((va_df["path"].values, y_va))
val_ds = (
    val_ds.map(parse_train_jpg, num_parallel_calls=autotune)
    .batch(BATCH_SIZE)
    .prefetch(autotune)
)

test_paths = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
test_paths = sorted(test_paths)
if len(test_paths) == 0:
    raise FileNotFoundError(f"No test .jpg files found in: {TEST_IMG_DIR}")

test_data = tf.data.Dataset.from_tensor_slices(test_paths)
test_data = test_data.map(parse_jpg, num_parallel_calls=autotune)
test_data = test_data.batch(BATCH_SIZE).prefetch(1)

print("Test images:", len(test_paths))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1915488701.py in <cell line: 0>()
      1 # --- Build train/val split by PatientID to reduce leakage (score-relevant, minimal change).
      2 id_col = "StudyInstanceUID"
----> 3 train_csv_path = os.path.join(DATA_DIR, "train.csv")
      4 train_df = pd.read_csv(train_csv_path)
      5 

NameError: name 'DATA_DIR' is not defined

## === cell 3
base_mode = model_map["efficientb3"]
model = get_model(base_mode, baseline_weight="imagenet", init_weight=None, lr=1e-3)

model.layers[1].trainable = (
    False  # EfficientNet base is typically the second layer; robustly handle below if needed
)
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
        layer.trainable = False

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC()],
)

model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)

for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
        layer.trainable = True

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC()],
)

model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2261043531.py in <cell line: 0>()
      1 # --- Train a task-specific head (and fine-tune backbone) to move score toward target.
      2 # Core architecture/loss remains unchanged; we only add the missing training step.
----> 3 base_mode = model_map["efficientb3"]
      4 model = get_model(base_mode, baseline_weight="imagenet", init_weight=None, lr=1e-3)
      5 

NameError: name 'model_map' is not defined

## === cell 4
preds = []
image_ids = []

for batch_images, batch_uids in test_data:
    batch_pred = model.predict_on_batch(batch_images)  # numpy array (or tensor)
    if isinstance(batch_pred, tf.Tensor):
        batch_pred = batch_pred.numpy()
    batch_pred = np.asarray(batch_pred, dtype=np.float32)

    preds.append(batch_pred)
    batch_uids = batch_uids.numpy()
    image_ids.extend([u.decode("utf-8") for u in batch_uids])

preds = np.concatenate(preds, axis=0)
assert (
    preds.shape[0] == len(image_ids) == len(test_paths)
), "Prediction/ID count mismatch"
assert (
    preds.shape[1] == N_CLASSES
), f"Expected {N_CLASSES} outputs, got {preds.shape[1]}"

sub_cols = [id_col] + target_cols  # official order we want
test_df = pd.DataFrame({id_col: image_ids})

for j, c in enumerate(req_target_cols):
    test_df[c] = preds[:, j].astype(np.float32)

for c in target_cols:
    if c not in test_df.columns:
        test_df[c] = 0.0

test_df = test_df[sub_cols]
test_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/312517672.py in <cell line: 0>()
      3 image_ids = []
      4 
----> 5 for batch_images, batch_uids in test_data:
      6     batch_pred = model.predict_on_batch(batch_images)  # numpy array (or tensor)
      7     if isinstance(batch_pred, tf.Tensor):

NameError: name 'test_data' is not defined
