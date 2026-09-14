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

0.8786642490178301

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from tensorflow import image, cast, float32
from tensorflow.io import read_file
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import numpy_function

from sklearn.model_selection import StratifiedKFold

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

BATCH_SIZE = 64
ROW = 300
COL = 300

train_csv_loc = "../input/cassava-leaf-disease-classification/train.csv"
train_location = "../input/cassava-leaf-disease-classification/train_images/"
test_location = "../input/cassava-leaf-disease-classification/test_images"

sample_sub_loc = "../input/cassava-leaf-disease-classification/sample_submission.csv"

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
pass




## === cell 2
def augmentations(file):
    file = read_file(file)
    file = tf.image.decode_jpeg(file, channels=3)
    file = tf.image.resize(file, [ROW, COL])

    file = tf.image.random_flip_left_right(file)
    file = tf.image.random_flip_up_down(file)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    file = tf.image.rot90(file, k)

    file = tf.image.random_brightness(file, max_delta=0.15)
    file = tf.image.random_contrast(file, lower=0.85, upper=1.15)
    file = tf.image.random_saturation(file, lower=0.85, upper=1.15)
    file = tf.image.random_hue(file, max_delta=0.05)

    file = preprocess_input(file)

    return file.numpy()




## === cell 3
def fetch_image_without_aug(filename, label):
    image_file = read_file(filename)
    image_file = tf.image.decode_jpeg(image_file, channels=3)
    image_file = tf.image.resize(image_file, [ROW, COL])
    image_file = preprocess_input(image_file)
    return image_file, label




## === cell 4
def fetch_image_with_aug(filename, label):
    aug_img = numpy_function(func=augmentations, inp=[filename], Tout=float32)
    aug_img.set_shape((ROW, COL, 3))
    return aug_img, label




## === cell 5
def getDatasetFromDataframe(train_files, train_labels, val_files, val_labels):
    train_ds = Dataset.from_tensor_slices((train_files.values, train_labels.values))
    train_ds = train_ds.shuffle(
        len(train_files), seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(fetch_image_with_aug, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = Dataset.from_tensor_slices((val_files.values, val_labels.values))
    val_ds = val_ds.shuffle(len(val_files), seed=SEED, reshuffle_each_iteration=False)
    val_ds = val_ds.map(fetch_image_without_aug, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds




## === cell 6
pass




## === cell 7
def scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr * np.exp(-0.05)




## === cell 8
def create_callbacks(Folder_name):
    os.makedirs("Weights", exist_ok=True)
    os.makedirs(os.path.join("Weights", "logs"), exist_ok=True)
    os.makedirs(os.path.join("Weights", Folder_name), exist_ok=True)
    os.makedirs(os.path.join("Weights", "logs", Folder_name), exist_ok=True)

    lr_scheduler = LearningRateScheduler(scheduler)

    weight_save = ModelCheckpoint(
        os.path.join("Weights", Folder_name, "best_model.keras"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )

    weight_save_only = ModelCheckpoint(
        os.path.join("Weights", Folder_name + ".h5"),
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
    )

    tensorboard = TensorBoard(
        os.path.join("Weights", "logs", Folder_name),
        histogram_freq=0,
    )

    callbacks = [lr_scheduler, weight_save, tensorboard, weight_save_only]
    histories = []
    return callbacks, histories




## === cell 9
pass




## === cell 10
def create_model(training=True, weights="imagenet"):
    base_model = EfficientNetB3(
        weights=weights, include_top=False, input_shape=(ROW, COL, 3)
    )

    x = Input(shape=(ROW, COL, 3))
    out_1 = base_model(x, training=training)
    out_1 = GlobalAveragePooling2D(name="encoding")(out_1)
    out_1 = Dropout(0.5)(out_1)
    output = Dense(5, activation="softmax")(out_1)

    final_model = Model(inputs=x, outputs=output)
    return final_model




## === cell 11
pass



## === cell 12
train_csv = pd.read_csv(train_csv_loc)
train_csv["image_id"] = train_csv["image_id"].map(lambda x: train_location + x)

Folder_name_base = "Exp_"

create_k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_number = 1
for train_index, test_index in create_k_folds.split(
    train_csv["image_id"], train_csv["label"]
):
    print(f"Fold Number {fold_number} is starting it's training")
    print("Creating Callbacks...")
    Folder_name = Folder_name_base + str(fold_number)
    callbacks, histories = create_callbacks(Folder_name)

    print("Importing Datasets...")
    X_train, X_test = (
        train_csv["image_id"].iloc[train_index],
        train_csv["image_id"].iloc[test_index],
    )
    y_train, y_test = (
        train_csv["label"].iloc[train_index],
        train_csv["label"].iloc[test_index],
    )

    train_ds, val_ds = getDatasetFromDataframe(X_train, y_train, X_test, y_test)

    final_model = create_model()

    final_model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("Starting training...")
    hist = final_model.fit(
        train_ds, epochs=15, validation_data=val_ds, callbacks=callbacks, workers=16
    )
    histories.append(hist.history)

    fold_number += 1



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3065514502.py in <cell line: 0>()
     14     print("Creating Callbacks...")
     15     Folder_name = Folder_name_base + str(fold_number)
---> 16     callbacks, histories = create_callbacks(Folder_name)
     17 
     18     print("Importing Datasets...")

/tmp/ipykernel_11/3294037698.py in create_callbacks(Folder_name)
     16     )
     17 
---> 18     weight_save_only = ModelCheckpoint(
     19         os.path.join("Weights", Folder_name + ".h5"),
     20         monitor="val_accuracy",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=Weights/Exp_1.h5

## === cell 13
pass



## === cell 14
external_ckpt_dir = "../input/efficient-net-weights/Weights2"
checkpoints = glob.glob(os.path.join(external_ckpt_dir, "Exp2_*.h5"))

if len(checkpoints) == 0:
    checkpoints = glob.glob(os.path.join("Weights", "Exp_*.h5"))

Models = []
for checkpoint in checkpoints:
    new_model = create_model(training=False, weights=None)
    new_model.load_weights(checkpoint)
    Models.append(new_model)

if len(Models) == 0:
    raise FileNotFoundError(
        "No model checkpoints found. Expected either "
        f"{os.path.join(external_ckpt_dir, 'Exp2_*.h5')} or Weights/Exp_*.h5 from training."
    )

print(f"Models loaded: {len(Models)}")

sub_df = pd.read_csv(sample_sub_loc)
results = []

for single in sub_df["image_id"].tolist():
    img_path = os.path.join(test_location, single)
    img = load_img(img_path, target_size=(ROW, COL))
    img = img_to_array(img)
    img = preprocess_input(img)
    batch = np.expand_dims(img, 0)

    preds = []
    for model in Models:
        preds.append(model.predict(batch, verbose=0))
    preds = np.mean(preds, axis=0)
    results.append(int(np.argmax(preds, axis=1)[0]))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/23048280.py in <cell line: 0>()
     15 
     16 if len(Models) == 0:
---> 17     raise FileNotFoundError(
     18         "No model checkpoints found. Expected either "
     19         f"{os.path.join(external_ckpt_dir, 'Exp2_*.h5')} or Weights/Exp_*.h5 from training."

FileNotFoundError: No model checkpoints found. Expected either ../input/efficient-net-weights/Weights2/Exp2_*.h5 or Weights/Exp_*.h5 from training.

## === cell 15
submission = pd.DataFrame(
    {"image_id": pd.read_csv(sample_sub_loc)["image_id"], "label": results}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3735254016.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"image_id": pd.read_csv(sample_sub_loc)["image_id"], "label": results}
      3 )
      4 submission.to_csv("submission.csv", index=False)
      5 print(submission.head())

NameError: name 'results' is not defined
