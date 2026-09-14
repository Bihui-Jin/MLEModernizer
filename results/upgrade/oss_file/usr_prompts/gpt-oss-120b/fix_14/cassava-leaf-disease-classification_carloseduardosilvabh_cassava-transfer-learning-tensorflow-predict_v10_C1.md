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

0.8318223028105167

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/cassava-leaf-disease-classification"



## === cell 2
train_images = os.listdir(os.path.join(path, "train_images"))
print("Total images for Train: ", len(train_images))



## === cell 3
with open(os.path.join(path, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)
print(json.dumps(classes, indent=4))



## === cell 4
df_train = pd.read_csv(os.path.join(path, "train.csv"))
df_train.head()



## === cell 5
df_train["class"] = df_train["label"].map({int(i): c for i, c in classes.items()})
df_train["label"] = df_train["label"].astype(str)  # needed for flow_from_dataframe
train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42)



## === cell 6
img_size = 224
size = (img_size, img_size)
batch_size = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE

class_names_sorted = sorted(df_train["class"].unique())
class_to_index = {name: idx for idx, name in enumerate(class_names_sorted)}
num_classes = len(class_names_sorted)


def decode_and_preprocess(path, label, augment=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size)
    image = image / 255.0  # rescale
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = (
            tf.image.random_zoom(image, (0.8, 1.2))
            if hasattr(tf.image, "random_zoom")
            else image
        )
    return image, tf.one_hot(label, depth=num_classes, dtype=tf.float32)


def make_dataset(df, augment=False, shuffle=False):
    paths = tf.convert_to_tensor(
        [os.path.join(path, "train_images", fname) for fname in df["image_id"]]
    )
    labels = tf.convert_to_tensor(
        df["class"].map(class_to_index).values, dtype=tf.int32
    )
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer=len(df), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: decode_and_preprocess(p, l, augment), num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


train_dataset = make_dataset(train_df, augment=True, shuffle=True)
val_dataset = make_dataset(val_df, augment=False, shuffle=False)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/639374870.py in <cell line: 0>()
     49 
     50 
---> 51 train_dataset = make_dataset(train_df, augment=True, shuffle=True)
     52 val_dataset = make_dataset(val_df, augment=False, shuffle=False)
     53 

/tmp/ipykernel_11/639374870.py in make_dataset(df, augment, shuffle)
     41     ds = tf.data.Dataset.from_tensor_slices((paths, labels))
     42     if shuffle:
---> 43         ds = ds.shuffle(buffer=len(df), reshuffle_each_iteration=True)
     44     ds = ds.map(
     45         lambda p, l: decode_and_preprocess(p, l, augment), num_parallel_calls=AUTOTUNE

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 7
def build_model():
    model = models.Sequential()
    model.add(
        EfficientNetB3(
            input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
        )
    )
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation="softmax", dtype="float32"))
    return model




## === cell 8
model = build_model()
model.summary()



## === cell 9
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)



## === cell 10
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, mode="min", restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "modelB3.keras", monitor="val_loss", verbose=1, mode="min", save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5, mode="min", verbose=1
)



## === cell 11
if not os.path.exists("modelB3.keras"):
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1,
    )
    best_model = tf.keras.models.load_model("modelB3.keras")
else:
    print("Pre‑trained model found; loading.")
    best_model = tf.keras.models.load_model("modelB3.keras")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3688774909.py in <cell line: 0>()
      1 if not os.path.exists("modelB3.keras"):
      2     history = model.fit(
----> 3         train_dataset,
      4         validation_data=val_dataset,
      5         epochs=30,

NameError: name 'train_dataset' is not defined

## === cell 12
class_names = [None] * num_classes
for name, idx in class_to_index.items():
    class_names[idx] = name
name_to_label = {v: int(k) for k, v in classes.items()}



## === cell 13
test_path = os.path.join(path, "test_images")
test_files = sorted(os.listdir(test_path))
batch_size_test = 32
pred_labels = []

for i in range(0, len(test_files), batch_size_test):
    batch_files = test_files[i : i + batch_size_test]
    batch_imgs = []
    for fname in batch_files:
        img = load_img(os.path.join(test_path, fname), target_size=size)
        img_arr = img_to_array(img) / 255.0  # same rescaling as training
        batch_imgs.append(img_arr)
    batch_array = np.stack(batch_imgs, axis=0)
    probs = best_model.predict(batch_array, verbose=0)
    pred_idxs = np.argmax(probs, axis=1)
    for pred_idx in pred_idxs:
        pred_name = class_names[pred_idx]
        pred_label = name_to_label[pred_name]
        pred_labels.append(pred_label)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3831588728.py in <cell line: 0>()
     12         batch_imgs.append(img_arr)
     13     batch_array = np.stack(batch_imgs, axis=0)
---> 14     probs = best_model.predict(batch_array, verbose=0)
     15     pred_idxs = np.argmax(probs, axis=1)
     16     for pred_idx in pred_idxs:

NameError: name 'best_model' is not defined

## === cell 14
submission = pd.DataFrame({"image_id": test_files, "label": pred_labels})
print(submission.head())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1211890525.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": test_files, "label": pred_labels})
      2 print(submission.head())
      3 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length

## === cell 15
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/781720812.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Submission file written to submission.csv")

NameError: name 'submission' is not defined
