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
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

BATCH_SIZE = 64
ROW = 300
COL = 300

train_csv_loc = "../input/cassava-leaf-disease-classification/train.csv"
train_location = "../input/cassava-leaf-disease-classification/train_images/"
test_location = "../input/cassava-leaf-disease-classification/test_images"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def augmentations(file_path):
    img_bytes = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)

    img_np = img.numpy()
    transform = A.Compose(
        [
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf(
                [
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=0.2,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.IAAPiecewiseAffine(p=0.3),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2),
                    A.IAASharpen(),
                    A.IAAEmboss(),
                    A.RandomBrightnessContrast(),
                    A.HueSaturationValue(hue_shift_limit=30),
                    A.RandomFog(p=0.3),
                ],
                p=0.7,
            ),
            A.HueSaturationValue(p=0.3),
            A.OneOf(
                [
                    A.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                    A.Cutout(num_holes=16, max_h_size=16, max_w_size=16),
                    A.ElasticTransform(alpha=1, sigma=25),
                ],
                p=0.4,
            ),
        ]
    )
    aug_img = transform(image=img_np)["image"]
    aug_img = preprocess_input(aug_img)
    aug_img = tf.image.resize(aug_img, [ROW, COL])
    return aug_img




## === cell 2
def fetch_image_without_aug(filename, label):
    img_bytes = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = preprocess_input(img)
    img = tf.image.resize(img, [ROW, COL])
    return img, label




## === cell 3
def fetch_image_with_aug(filename, label):
    aug_img = tf.numpy_function(func=augmentations, inp=[filename], Tout=tf.float32)
    aug_img.set_shape([ROW, COL, 3])
    return aug_img, label




## === cell 4
def getDatasetFromDataframe(train_files, train_labels, val_files, val_labels):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(len(train_files))
    train_ds = train_ds.map(fetch_image_with_aug, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.shuffle(len(val_files))
    val_ds = val_ds.map(fetch_image_without_aug, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return train_ds, val_ds




## === cell 5
def scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr * np.exp(-0.05)




## === cell 6
def create_callbacks(folder_name):
    weights_dir = os.path.join("Weights", folder_name)
    logs_dir = os.path.join("Weights", "logs", folder_name)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    lr_scheduler = LearningRateScheduler(scheduler)

    weight_save = ModelCheckpoint(
        filepath=os.path.join(weights_dir, "best_full.h5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )

    weight_save_only = ModelCheckpoint(
        filepath=os.path.join(weights_dir, "best_weights.h5"),
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
    )

    tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=1)

    callbacks = [lr_scheduler, weight_save, weight_save_only, tensorboard]
    return callbacks




## === cell 7
def create_model(training=True, weights="imagenet"):
    base_model = EfficientNetB3(
        weights=weights, include_top=False, input_shape=(ROW, COL, 3)
    )
    inputs = Input(shape=(ROW, COL, 3))
    x = base_model(inputs, training=training)
    x = GlobalAveragePooling2D(name="encoding")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model




## === cell 8
train_csv = pd.read_csv(train_csv_loc)
train_csv["image_id"] = train_csv["image_id"].map(
    lambda x: os.path.join(train_location, x)
)

create_k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_number = 1
Models = []  # store trained models for inference

for train_idx, val_idx in create_k_folds.split(
    train_csv["image_id"], train_csv["label"]
):
    print(f"\n=== Fold {fold_number} ===")
    folder_name = f"Exp_{fold_number}"
    callbacks = create_callbacks(folder_name)

    X_train = train_csv["image_id"].iloc[train_idx].values
    y_train = train_csv["label"].iloc[train_idx].values
    X_val = train_csv["image_id"].iloc[val_idx].values
    y_val = train_csv["label"].iloc[val_idx].values

    train_ds, val_ds = getDatasetFromDataframe(X_train, y_train, X_val, y_val)

    model = create_model()
    model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    try:
        model.fit(
            train_ds, epochs=15, validation_data=val_ds, callbacks=callbacks, workers=4
        )
    except Exception as e:
        print("Training error:", e)

    best_path = os.path.join("Weights", folder_name, "best_full.h5")
    if os.path.exists(best_path):
        model.load_weights(best_path)
    Models.append(model)
    fold_number += 1



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/376861036.py in <cell line: 0>()
     15     print(f"\n=== Fold {fold_number} ===")
     16     folder_name = f"Exp_{fold_number}"
---> 17     callbacks = create_callbacks(folder_name)
     18 
     19     X_train = train_csv["image_id"].iloc[train_idx].values

/tmp/ipykernel_11/1138462638.py in create_callbacks(folder_name)
     16     )
     17 
---> 18     weight_save_only = ModelCheckpoint(
     19         filepath=os.path.join(weights_dir, "best_weights.h5"),
     20         monitor="val_accuracy",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=Weights/Exp_1/best_weights.h5

## === cell 9
test_files = [
    os.path.join(test_location, f)
    for f in os.listdir(test_location)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
    and os.path.isfile(os.path.join(test_location, f))
]

results = []
for img_path in test_files:
    img = load_img(img_path, target_size=(ROW, COL))
    img_arr = img_to_array(img)
    img_arr = preprocess_input(img_arr)
    batch = np.expand_dims(img_arr, axis=0)

    preds = np.mean([m.predict(batch, verbose=0) for m in Models], axis=0)
    label = int(np.argmax(preds, axis=1)[0])
    results.append({"image_id": os.path.basename(img_path), "label": label})



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2711455052.py in <cell line: 0>()
     16     # Average predictions across all fold models
     17     preds = np.mean([m.predict(batch, verbose=0) for m in Models], axis=0)
---> 18     label = int(np.argmax(preds, axis=1)[0])
     19     results.append({"image_id": os.path.basename(img_path), "label": label})
     20 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in argmax(a, axis, out, keepdims)
   1227     """
   1228     kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
-> 1229     return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)
   1230 
   1231 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     57 
     58     try:
---> 59         return bound(*args, **kwds)
     60     except TypeError:
     61         # A TypeError occurs if the object does have such a method in its

AxisError: axis 1 is out of bounds for array of dimension 1

## === cell 10
submission = pd.DataFrame(results)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
