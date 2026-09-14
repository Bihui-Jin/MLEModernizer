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

0.4348745844666062

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.12108) has done: 'I remove the unavailable KaggleDatasets import, replace the missing pretrained model with a tiny randomly‑initialized Keras model that can still produce predictions, and simplify the augmentation pipeline (using a no‑op function when imgaug isn’t installed). These fixes ensure the script runs without errors, creates a valid `submission.csv`, and keeps the core logic unchanged apart from the minimal fall‑backs needed for execution.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import math, re, glob
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from functools import partial
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from PIL import Image

print("Tensorflow version " + tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMAGE_SIZE = 128  # smaller size for fast training
BATCH_SIZE = 64
EPOCHS = 3  # few epochs are enough for this tiny model



## === cell 2
train_dir = "../input/cassava-leaf-disease-classification/train_images"
if not os.path.isdir(train_dir):
    train_dir = "./train_images"

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
if not os.path.isfile(train_csv_path):
    train_csv_path = "./train.csv"

train_df = pd.read_csv(train_csv_path)

file_paths = train_df["image_id"].apply(lambda x: os.path.join(train_dir, x)).values
labels = train_df["label"].values.astype(np.int32)




## === cell 3
def _parse_function(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255.0
    return image, label


dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_size = int(0.1 * len(train_df))
train_ds = dataset.skip(val_size)
val_ds = dataset.take(val_size)




## === cell 4
def create_small_cnn():
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = keras.layers.Conv2D(16, 3, activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


model_path = "small_cnn.h5"
if os.path.exists(model_path):
    modeleffb4_0 = keras.models.load_model(model_path)
else:
    modeleffb4_0 = create_small_cnn()
    callbacks = [
        EarlyStopping(patience=2, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(patience=1, factor=0.5, monitor="val_accuracy"),
    ]
    modeleffb4_0.fit(
        train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=2
    )
    modeleffb4_0.save(model_path)

mod_lst = [modeleffb4_0]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1053624320.py in <cell line: 0>()
     26         ReduceLROnPlateau(patience=1, factor=0.5, monitor="val_accuracy"),
     27     ]
---> 28     modeleffb4_0.fit(
     29         train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=2
     30     )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py in update(self, current, values, finalize)
    186         elif self.verbose == 2:
    187             if finalize:
--> 188                 numdigits = int(math.log10(self.target)) + 1
    189                 count = ("%" + str(numdigits) + "d/%d") % (current, self.target)
    190                 info = f"{count} - {now - self._start:.0f}s"

ValueError: math domain error

## === cell 5
test_dir = "../input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_dir):
    test_dir = "./test_images"



## === cell 6
try:
    from imgaug import augmenters as iaa

    seq = iaa.Sequential(
        [
            iaa.Crop(px=(0, 128)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.GaussianBlur(sigma=(0, 2.0)),
            iaa.Dropout((0.01, 0.15), per_channel=0.5),
        ]
    )
except Exception:

    class DummySeq:
        def __call__(self, image):
            return image

    seq = DummySeq()




## === cell 7
def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    """
    Generate predictions for all JPG images in `image_dir` using the supplied
    list of Keras models. Supports optional Test‑Time Augmentation (TTA).
    """
    preds = []
    img_ids = []
    for img_path in glob.glob(os.path.join(image_dir, "*.jpg")):
        image = Image.open(img_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(image)

        if TTA:
            if normalize:
                aug_imgs = [seq(image=img_array) / 255.0 for _ in range(aug_num)]
            else:
                aug_imgs = [seq(image=img_array) for _ in range(aug_num)]
            all_preds = []
            for aug in aug_imgs:
                aug_exp = np.expand_dims(aug, axis=0)
                for mod in model_obj_list:
                    all_preds.append(mod.predict(aug_exp, verbose=0))
            avg_pred = np.mean(np.concatenate(all_preds, axis=0), axis=0)
        else:
            img = img_array / 255.0 if normalize else img_array
            img_exp = np.expand_dims(img, axis=0)
            all_preds = [mod.predict(img_exp, verbose=0) for mod in model_obj_list]
            avg_pred = np.mean(np.concatenate(all_preds, axis=0), axis=0)

        preds.append(int(np.argmax(avg_pred)))
        img_ids.append(os.path.basename(img_path))

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 8
predict_df = get_preds_model_list(
    image_dir=test_dir,
    model_obj_list=mod_lst,
    normalize=False,  # model expects raw pixels normalized inside pipeline
    aug_num=3,
    TTA=False,  # disable TTA for speed with the lightweight model
)

submission_path = "submission.csv"
predict_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2688445643.py in <cell line: 0>()
      1 predict_df = get_preds_model_list(
      2     image_dir=test_dir,
----> 3     model_obj_list=mod_lst,
      4     normalize=False,  # model expects raw pixels normalized inside pipeline
      5     aug_num=3,

NameError: name 'mod_lst' is not defined

## === cell 9
print(predict_df.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2160939148.py in <cell line: 0>()
----> 1 print(predict_df.head())

NameError: name 'predict_df' is not defined
