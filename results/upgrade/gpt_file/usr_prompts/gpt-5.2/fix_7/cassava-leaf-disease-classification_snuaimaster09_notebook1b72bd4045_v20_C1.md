# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.10

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

SUBMISSION_MODE = 1

import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import InceptionV3, Xception
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

_CPU_COUNT = os.cpu_count() or 2
DATA_WORKERS = min(8, max(2, _CPU_COUNT // 2))
MAX_QUEUE_SIZE = 32

df_train = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
df_train["label"] = df_train["label"].astype(str)

batch_size = 32
image_size = 300
input_shape = (image_size, image_size, 3)
target_size = (image_size, image_size)




## === cell 1
def create_Inception():
    base_model = InceptionV3(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(5, activation="softmax", name="dense", dtype="float32")(x)

    inception = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.2, from_logits=False
    )

    inception.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return inception


def create_Xception():
    base_model = Xception(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(5, activation="softmax", name="dense", dtype="float32")(x)

    xception = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.2, from_logits=False
    )

    xception.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return xception




## === cell 2
def _make_dir_iter(
    datagen, directory, batch_size, target_size, seed, shuffle, subset=None
):
    return datagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        class_mode="categorical",
        subset=subset,
    )


if SUBMISSION_MODE == 0:
    fold_number = 0
    n_splits = 3
    epochs = 8

    tf.keras.backend.clear_session()
    KFoldSplit = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=0.1, random_state=SEED
    )

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        validation_split=0.0,
    )

    for train_index, val_index in KFoldSplit.split(
        df_train["image_id"], df_train["label"]
    ):
        train_set = df_train.loc[train_index]
        val_set = df_train.loc[val_index]

        train_datagen = datagen.flow_from_dataframe(
            dataframe=train_set,
            directory="../input/cassava-leaf-disease-classification/train_images",
            x_col="image_id",
            y_col="label",
            target_size=target_size,
            batch_size=batch_size,
            shuffle=True,
            class_mode="categorical",
            seed=SEED,
        )

        val_datagen = datagen.flow_from_dataframe(
            dataframe=val_set,
            directory="../input/cassava-leaf-disease-classification/train_images",
            x_col="image_id",
            y_col="label",
            target_size=target_size,
            batch_size=batch_size,
            shuffle=False,
            class_mode="categorical",
            seed=SEED,
        )

        model = create_Inception()
        print("Training fold no.: " + str(fold_number + 1))

        model_name = "inception "
        fold_name = "fold.h5"
        filepath = model_name + str(fold_number + 1) + fold_name
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", patience=1, verbose=1, factor=0.2),
            EarlyStopping(monitor="val_loss", patience=3),
            ModelCheckpoint(filepath=filepath, monitor="val_loss", save_best_only=True),
        ]

        history = model.fit(
            train_datagen,
            epochs=epochs,
            validation_data=val_datagen,
            callbacks=callbacks,
        )
        fold_number += 1
        if fold_number == n_splits:
            print("Training finished!")

if SUBMISSION_MODE == 1:
    sample_path = os.path.join(
        "../input/cassava-leaf-disease-classification", "sample_submission.csv"
    )
    SampleSubmit = pd.read_csv(sample_path)

    candidate_model_paths = [
        "/kaggle/input/inception2fold/inception 2fold.h5",
        "/kaggle/input/inception2fold/inception2fold.h5",
        "../input/inception2fold/inception 2fold.h5",
        "../input/inception2fold/inception2fold.h5",
        "/kaggle/working/inception_fallback.h5",
    ]
    model_path = next((p for p in candidate_model_paths if os.path.exists(p)), None)

    preprocess = tf.keras.applications.inception_v3.preprocess_input

    if model_path is not None:
        model = load_model(model_path)
        try:
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=0.01, momentum=0.9, nesterov=True
            )
            loss = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.2, from_logits=False
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        except Exception:
            pass
    else:
        epochs = 3

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        tr_idx, va_idx = next(splitter.split(df_train["image_id"], df_train["label"]))
        train_set = df_train.iloc[tr_idx].reset_index(drop=True)
        val_set = df_train.iloc[va_idx].reset_index(drop=True)

        datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess,
            horizontal_flip=True,
            rotation_range=20,
            zoom_range=0.2,
        )
        datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess
        )

        tf.keras.backend.clear_session()
        model = create_Inception()

        train_dir = "../input/cassava-leaf-disease-classification/train_images"

        train_datagen = datagen_train.flow_from_dataframe(
            dataframe=train_set,
            directory=train_dir,
            x_col="image_id",
            y_col="label",
            target_size=target_size,
            batch_size=batch_size,
            shuffle=True,
            class_mode="categorical",
            seed=SEED,
        )

        val_datagen = datagen_val.flow_from_dataframe(
            dataframe=val_set,
            directory=train_dir,
            x_col="image_id",
            y_col="label",
            target_size=target_size,
            batch_size=batch_size,
            shuffle=False,
            class_mode="categorical",
            seed=SEED,
        )

        local_model_path = "/kaggle/working/inception_fallback.h5"
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", patience=1, verbose=1, factor=0.2),
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            ModelCheckpoint(
                filepath=local_model_path, monitor="val_loss", save_best_only=True
            ),
        ]

        model.fit(
            train_datagen,
            epochs=epochs,
            validation_data=val_datagen,
            callbacks=callbacks,
            verbose=1,
        )

        model.save(local_model_path)
        model_path = local_model_path

    test_dir = os.path.join(
        "../input/cassava-leaf-disease-classification", "test_images"
    )
    test_df = SampleSubmit[["image_id"]].copy()
    test_df["label"] = "0"

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess
    )

    infer_batch_size = 64

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=test_dir,
        x_col="image_id",
        y_col="label",
        target_size=target_size,
        batch_size=infer_batch_size,
        shuffle=False,
        class_mode="categorical",
    )

    preds = model.predict(
        test_generator,
        verbose=1,
    )
    results = np.argmax(preds, axis=1).astype(int).tolist()

    SampleSubmit["label"] = results[: len(SampleSubmit)]

    SampleSubmit["label"] = SampleSubmit["label"].astype(int)

    out_path = os.path.join("/kaggle/working", "submission.csv")
    SampleSubmit.to_csv(out_path, index=False)

    print("Loaded model from:", model_path)
    print("Wrote:", out_path)
    print(SampleSubmit.head())
