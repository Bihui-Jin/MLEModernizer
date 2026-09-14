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

3.9

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

0.7963130855243276

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
weight_path_candidates = [
    "/kaggle/input/model-ensembling-with-k-fold/fineTuned_v0.39.h5",
    "../input/model-ensembling-with-k-fold/fineTuned_v0.39.h5",
]
weight_path = next((p for p in weight_path_candidates if os.path.exists(p)), None)

my_model = None
if weight_path is not None:
    print("Loading model from:", weight_path)
    my_model = load_model(weight_path, compile=False)
else:
    print(
        "Pretrained .h5 not found; will train a model from scratch/fine-tune locally to produce submission."
    )




## === cell 2
base_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]


def find_file(relpath):
    for b in base_candidates:
        p = os.path.join(b, relpath)
        if os.path.exists(p):
            return p
    return None


train_csv_path = find_file("train.csv")
sample_sub_path = find_file("sample_submission.csv")

test_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/test_images",
    "../input/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
]
train_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/train_images",
    "../input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
]


def first_existing_dir(cands, desc):
    for d in cands:
        if os.path.isdir(d):
            print(f"Using {desc} dir:", d)
            return d
    raise FileNotFoundError(f"Could not find {desc} directory. Checked: {cands}")


test_dir = first_existing_dir(test_dir_candidates, "test_images")
train_dir = first_existing_dir(train_dir_candidates, "train_images")

if train_csv_path is None:
    raise FileNotFoundError("Could not locate train.csv under known base paths.")
if sample_sub_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under known base paths."
    )

test_images = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No .jpg files found in {test_dir}")

df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_gen(batch_size=128):
    my_test_idg = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=(512, 512),
    )
    return test_gen




## === cell 3
if my_model is None:
    df_train = pd.read_csv(train_csv_path)

    df_train["path"] = train_dir.rstrip("/") + "/" + df_train["image_id"].astype(str)

    sample_check = df_train["path"].iloc[:: max(1, len(df_train) // 2000)].tolist()
    if not all(os.path.exists(p) for p in sample_check):
        missing = [p for p in sample_check if not os.path.exists(p)][:5]
        raise FileNotFoundError(
            f"Some training image paths do not exist (sample check); examples: {missing}"
        )

    df_train["label"] = df_train["label"].astype(str)

    idx = np.arange(len(df_train))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
    df_va = df_train.iloc[va_idx].reset_index(drop=True)

    train_idg = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    valid_idg = ImageDataGenerator(rescale=1.0 / 255.0)

    batch_size = 32 if not DEBUG else 8

    train_gen = train_idg.flow_from_dataframe(
        df_tr,
        x_col="path",
        y_col="label",
        target_size=(512, 512),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True,
        seed=SEED,
    )
    valid_gen = valid_idg.flow_from_dataframe(
        df_va,
        x_col="path",
        y_col="label",
        target_size=(512, 512),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
        seed=SEED,
    )

    inputs = tf.keras.Input(shape=(512, 512, 3))

    x = tf.keras.layers.Rescaling(1.0)(inputs)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x
    )
    backbone.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(0.2, seed=SEED)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    my_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    epochs = 2 if not DEBUG else 1

    autotune = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    ).prefetch(autotune)
    valid_ds = tf.data.Dataset.from_generator(
        lambda: valid_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    ).prefetch(autotune)

    my_model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        verbose=1,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=16,
        steps_per_epoch=len(train_gen),
        validation_steps=len(valid_gen),
    )




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3264945670.py in <cell line: 0>()
    105     ).prefetch(autotune)
    106 
--> 107     my_model.fit(
    108         train_ds,
    109         validation_data=valid_ds,

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
test_gen = make_test_gen(batch_size=128)

test_ds = tf.data.Dataset.from_generator(
    lambda: test_gen,
    output_signature=tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
).prefetch(tf.data.AUTOTUNE)

pred_list = []
for i in range(1):
    pred_test = my_model.predict(
        test_ds,
        verbose=1,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=16,
        steps=len(test_gen),
    )
    pred_list.append(pred_test)

pred_test = np.mean(pred_list, axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].apply(os.path.basename)
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]

sample_sub = pd.read_csv(sample_sub_path)
final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
if final_csv["label"].isna().any():
    mode_label = int(pd.Series(pred_test_labels).mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(mode_label).astype(int)
else:
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print(f"Wrote submission.csv with shape: {final_csv.shape}")
print("submission.csv exists:", os.path.exists("submission.csv"))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3950590848.py in <cell line: 0>()
      9 pred_list = []
     10 for i in range(1):
---> 11     pred_test = my_model.predict(
     12         test_ds,
     13         verbose=1,

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

## === cell 5
final_csv.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1842027079.py in <cell line: 0>()
----> 1 final_csv.head()

NameError: name 'final_csv' is not defined
