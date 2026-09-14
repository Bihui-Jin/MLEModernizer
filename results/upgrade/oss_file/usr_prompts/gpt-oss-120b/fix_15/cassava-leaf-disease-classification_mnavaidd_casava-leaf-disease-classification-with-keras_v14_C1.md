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

0.7440314294348745

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'Optimized data generators to use multiple worker processes and enabled multiprocessing in model training, which speeds up image loading and augmentation without altering any model architecture, training logic, or results. Added the `workers` argument to both training and validation generators and passed `workers=NUM_WORKERS, use_multiprocessing=True` to `model.fit`. These changes keep the same data flow and training behavior while reducing overall runtime.'
- What this solution (achieved 0.05531) has done: 'The changes increase TensorFlow’s CPU thread usage, enable multiprocessing for data loading, compute the full number of steps per epoch (using `ceil`), and, when a GPU is available, turn on mixed‑precision to accelerate training – all without altering the model architecture, augmentations, loss, or training schedule.'
- What this solution (achieved 0.05531) has done: 'Optimized data loading and training by limiting CPU workers to a reasonable number (max 8) to reduce overhead, and enabled multiprocessing in `model.fit` with a larger queue size, which speeds up image augmentation without altering the model architecture or training logic. These adjustments keep the exact same augmentations, epochs, and model while making the pipeline efficiently parallelized, ensuring the script completes within the 600‑second limit without affecting result accuracy.'
- What this solution (achieved 0.05531) has done: 'We replace the Python‑generator based ImageDataGenerator pipelines with a cached tf.data pipeline that performs the same resizing, rescaling and augmentations in compiled TensorFlow ops, and we drop multiprocessing overhead. The model architecture, loss, optimizer and training callbacks stay unchanged; only the data loading is made much faster, keeping the same batch size, epochs and validation split, so the final accuracy is unaffected while execution finishes well under the 600 s limit.'

# 9. Code solution

## === cell 0
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

NUM_WORKERS = min(8, os.cpu_count() or 1)
tf.config.threading.set_intra_op_parallelism_threads(NUM_WORKERS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_WORKERS)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_path = data_path + "train.csv"
label_json_path = data_path + "label_num_to_disease_map.json"
train_images_dir = data_path + "train_images/"
test_images_dir = data_path + "test_images/"



## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(str)

label_map = pd.read_json(label_json_path, orient="index")
label_names = label_map.values.flatten().tolist()
print("Label names :", label_names)

label_to_index = {name: i for i, name in enumerate(label_names)}
train_df["label_idx"] = train_df["label"].map(label_to_index)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.15,
    stratify=train_df["label_idx"],
    random_state=42,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/765201467.py in <cell line: 0>()
     11 
     12 # split once; validation split = 15%
---> 13 train_split, val_split = train_test_split(
     14     train_df,
     15     test_size=0.15,

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2581         cv = CVClass(test_size=n_test, train_size=n_train, random_state=random_state)
   2582 
-> 2583         train, test = next(cv.split(X=arrays[0], y=stratify))
   2584 
   2585     return list(

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
   2158         to an integer.
   2159         """
-> 2160         y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
   2161         return super().split(X, y, groups)
   2162 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input y contains NaN.

## === cell 3
BATCH_SIZE = 32
IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE


def decode_and_resize(path):
    img = tf.io.read_file(tf.strings.join([train_images_dir, path]))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img


def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(
        img, max_delta=0.9 - 0.1
    )  # brightness_range [0.1,0.9]
    img = tf.image.random_contrast(img, lower=0.1, upper=1.0)
    angle = tf.random.uniform([], 0, 2 * np.pi)
    img = tfa.image.rotate(img, angle, fill_mode="nearest")
    scales = tf.random.uniform([], 0.7, 1.3)
    new_size = tf.cast(scales * IMG_SIZE, tf.int32)
    img = tf.image.resize(img, [new_size, new_size])
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)
    return img


def preprocess(path, label, training):
    img = decode_and_resize(path)
    if training:
        img = augment(img)
    img = img / 255.0
    label_onehot = tf.one_hot(label, depth=5)
    return img, label_onehot


def make_dataset(df, training):
    ds = tf.data.Dataset.from_tensor_slices(
        (df["image_id"].values, df["label_idx"].values)
    )
    ds = ds.map(lambda p, l: preprocess(p, l, training), num_parallel_calls=AUTOTUNE)
    if not training:
        ds = ds.cache()
    ds = ds.shuffle(1000) if training else ds
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_dataset = make_dataset(train_split, training=True)
val_dataset = make_dataset(val_split, training=False)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3538862216.py in <cell line: 0>()
     51 
     52 
---> 53 train_dataset = make_dataset(train_split, training=True)
     54 val_dataset = make_dataset(val_split, training=False)
     55 

NameError: name 'train_split' is not defined

## === cell 4
base_model = applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # freeze for initial training

model = tf.keras.Sequential(
    [base_model, GlobalAveragePooling2D(), Dense(5, activation="softmax")]
)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.summary()



## === cell 5
checkpoint_path = "best_model.weights.h5"
model_ckpt = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)
early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1)

EPOCHS = 15

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[model_ckpt, early_stop, reduce_lr],
    verbose=2,
)
model.load_weights(checkpoint_path)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3555575520.py in <cell line: 0>()
     15 
     16 model.fit(
---> 17     train_dataset,
     18     validation_data=val_dataset,
     19     epochs=EPOCHS,

NameError: name 'train_dataset' is not defined

## === cell 6
sample_sub_path = data_path + "sample_submission.csv"
sub_df = pd.read_csv(sample_sub_path)


def decode_test(path):
    img = tf.io.read_file(tf.strings.join([test_images_dir, path]))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img


test_ds = tf.data.Dataset.from_tensor_slices(sub_df["image_id"].values)
test_ds = test_ds.map(lambda p: decode_test(p), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

pred_probs = model.predict(test_ds, verbose=2)
pred_labels = np.argmax(pred_probs, axis=1)

submission = pd.DataFrame({"image_id": sub_df["image_id"], "label": pred_labels})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
