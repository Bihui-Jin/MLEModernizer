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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.7263523723179208

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11173) has done: 'I remove the incompatible `tensorflow_hub` import that is triggering the protobuf `MessageFactory.GetPrototype` crash under Python 3.11, since it isn’t used by the solution. Because the referenced external weight file path doesn’t exist in this environment, I keep the same ResNet50-based core logic but instantiate a standard ImageNet-pretrained ResNet50 classifier (so `my_model` is defined and predictions can run). I also fix the test image path to the actual `/kaggle/input/...` location and add lightweight safety checks to ensure we always write a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.25448) has done: 'I remove the protobuf-implementation override that’s triggering the `MessageFactory.GetPrototype` crash, since it’s not needed for this script in Kaggle’s TF build. Then I fix the ResNet50 ImageNet fallback so it can actually load weights by using the required `(224, 224, 3)` input/target size when `include_top=True`. Finally, I keep the rest of the prediction/submission logic intact but ensure the generator target size matches the chosen model input size so `my_model` is defined and inference completes, producing a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl  # kept since originally imported (even if unused)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt  # kept since originally imported (even if unused)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

weight_path = "/kaggle/input/resnet50-trasnfer-learning-on-tpu/cassava_base.h5"

if os.path.exists(weight_path):
    my_model = tf.keras.models.load_model(weight_path, compile=False)
    model_mode = "cassava_5class"
    target_size = (256, 256)
else:
    target_size = (224, 224)
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(target_size[0], target_size[1], 3),
        pooling="avg",
    )
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(5, activation="softmax")(x)
    my_model = tf.keras.Model(inputs=base.input, outputs=out)
    model_mode = "trained_5class_from_imagenet_backbone"




## === cell 2
if not os.path.exists(TRAIN_CSV):
    TRAIN_CSV = "/kaggle/input/train.csv"
train_df = pd.read_csv(TRAIN_CSV)

train_df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)

if (not os.path.exists(TRAIN_IMG_DIR)) or (
    len(train_df) and (not os.path.exists(train_df["path"].iloc[0]))
):
    TRAIN_IMG_DIR = "../input/cassava-leaf-disease-classification/train_images"
    train_df["path"] = (
        TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
    )

preprocess = tf.keras.applications.resnet50.preprocess_input

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

df_tr = train_df.iloc[tr_idx].copy()
df_val = train_df.iloc[val_idx].copy()

BATCH = 32

train_idg = ImageDataGenerator(
    preprocessing_function=preprocess,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
)
val_idg = ImageDataGenerator(preprocessing_function=preprocess)

train_gen = train_idg.flow_from_dataframe(
    dataframe=df_tr,
    x_col="path",
    y_col="label",
    target_size=target_size,
    batch_size=BATCH,
    seed=SEED,
    shuffle=True,
    class_mode="raw",  # yields numeric labels directly
)

val_gen = val_idg.flow_from_dataframe(
    dataframe=df_val,
    x_col="path",
    y_col="label",
    target_size=target_size,
    batch_size=BATCH,
    seed=SEED,
    shuffle=False,
    class_mode="raw",
)




## === cell 3
if model_mode != "cassava_5class":
    for layer in my_model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = False
    for layer in my_model.layers:
        if "resnet50" in layer.name:
            layer.trainable = False

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    EPOCHS = 3

    my_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1,
        workers=min(8, (os.cpu_count() or 2)),
        use_multiprocessing=True,
        max_queue_size=32,
    )




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2024390963.py in <cell line: 0>()
     18     # Use Keras multiprocessing for CPU-side image loading/augmentation to keep the training step fed.
     19     # This does not change the model/training loop, just speeds up input pipeline.
---> 20     my_model.fit(
     21         train_gen,
     22         validation_data=val_gen,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 4
test_images = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
if len(test_images) == 0:
    test_images = tf.io.gfile.glob(
        "../input/cassava-leaf-disease-classification/test_images/*.jpg"
    )

df_test = pd.DataFrame({"path": test_images})
if df_test.empty:
    raise RuntimeError("No test images found. Check the test_images path.")


def make_test_gen(batch_size=32):
    my_test_idg = ImageDataGenerator(preprocessing_function=preprocess)
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=target_size,
    )
    return test_gen




## === cell 5
test_gen = make_test_gen(batch_size=32)

pred = my_model.predict(
    test_gen,
    verbose=1,
    workers=min(8, (os.cpu_count() or 2)),
    use_multiprocessing=True,
    max_queue_size=32,
)

pred_labels = np.argmax(pred, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].map(os.path.basename)
final_submission["label"] = pred_labels
final_csv = final_submission[["image_id", "label"]]

if os.path.exists(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
else:
    sample = pd.read_csv("/kaggle/input/sample_submission.csv")

final_csv = sample[["image_id"]].merge(final_csv, on="image_id", how="left")

if final_csv["label"].isna().any():
    fill_label = int(pd.Series(pred_labels).mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(fill_label)

final_csv["label"] = final_csv["label"].astype(int)
final_csv.to_csv("submission.csv", index=False)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2228308812.py in <cell line: 0>()
      3 # CHANGE (timeout fix, correctness preserved):
      4 # Parallelize CPU-side image decoding/resizing during predict; output is identical for same weights/inputs.
----> 5 pred = my_model.predict(
      6     test_gen,
      7     verbose=1,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 6
final_csv.head()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1842027079.py in <cell line: 0>()
----> 1 final_csv.head()

NameError: name 'final_csv' is not defined
