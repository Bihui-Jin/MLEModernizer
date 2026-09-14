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

0.8637050468419462

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()



## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 4
train_csv.head()



## === cell 5
BATCH_SIZE = 24
IMG_SIZE = 320



## === cell 6
tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass



## === cell 7
train_gen = ImageDataGenerator(
    rotation_range=270,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.1, 0.9],
    shear_range=25,
    zoom_range=0.3,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
    validation_split=0.2,
)

valid_gen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)



## === cell 8
_classes = [str(i) for i in range(5)]



## === cell 9
train_generator = train_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset="training",
    classes=_classes,
)

valid_generator = valid_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    subset="validation",
    classes=_classes,
)



## === cell 10
print("Class indices:", train_generator.class_indices)



## === cell 11
pass



## === cell 12
tf.keras.backend.clear_session()



## === cell 13
base = None



## === cell 14
model = None




## === cell 15
def scheduler(epoch, lr):
    if epoch > 3 and epoch % 2 == 0:
        return lr / 1.25
    else:
        return lr


callback0 = tf.keras.callbacks.ModelCheckpoint(
    "./CasavaLeafDiseaseModel.h5", monitor="val_loss", save_best_only=True
)

callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)



## === cell 16
EPOCHS = 12



## === cell 17
pretrained_path = (
    "../input/casavaleafdiseasemodel-tf/CasavaLeafDiseaseModel_epoch_12_acc_85.h5"
)

loaded = False
try:
    model = tf.keras.models.load_model(pretrained_path, compile=False)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        metrics=["acc", tf.keras.metrics.TruePositives(name="tp")],
    )
    loaded = True
    print("Loaded saved model from input dataset:", pretrained_path)
except Exception as e:
    print(
        "ERROR: Pretrained model could not be loaded; training would likely timeout in this environment."
    )
    print("Load error:", repr(e))
    raise

if not loaded:
    base = applications.InceptionResNetV2(
        include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
    )
    model = tf.keras.Sequential()
    model.add(base)
    model.add(BatchNormalization(axis=-1))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(5, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        metrics=["acc", tf.keras.metrics.TruePositives(name="tp")],
    )

    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, valid_generator.samples // BATCH_SIZE)
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[callback0, callback1],
        verbose=1,
    )
    model = tf.keras.models.load_model("./CasavaLeafDiseaseModel.h5")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/724163829.py in <cell line: 0>()
      5 loaded = False
      6 try:
----> 7     model = tf.keras.models.load_model(pretrained_path, compile=False)
      8     model.compile(
      9         loss=tf.keras.losses.CategoricalCrossentropy(),

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/casavaleafdiseasemodel-tf/CasavaLeafDiseaseModel_epoch_12_acc_85.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 18
AUTOTUNE = tf.data.AUTOTUNE

n_total = len(train_csv)
val_start = int(np.floor(n_total * (1.0 - 0.2)))
val_df = train_csv.iloc[val_start:].copy()

val_paths = (images_dir_path + "/" + val_df["image_id"].astype(str)).to_numpy()
val_labels = val_df["label"].astype(np.int32).to_numpy()


def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _val_map_fn(path, label):
    img = _decode_resize_rescale(path)
    label_oh = tf.one_hot(label, depth=5, dtype=tf.float32)
    return img, label_oh


valid_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(_val_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))
val_metrics = model.evaluate(valid_ds, steps=val_steps, verbose=0)
print("Validation metrics:", dict(zip(model.metrics_names, val_metrics)))



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1429579596.py in <cell line: 0>()
     39 
     40 val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))
---> 41 val_metrics = model.evaluate(valid_ds, steps=val_steps, verbose=0)
     42 print("Validation metrics:", dict(zip(model.metrics_names, val_metrics)))
     43 

AttributeError: 'NoneType' object has no attribute 'evaluate'

## === cell 19
ss = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
test_img_path = os.path.join(
    "../input/cassava-leaf-disease-classification/test_images", ss.image_id.iloc[0]
)
print("Example test image path:", test_img_path)



## === cell 20
img = cv2.imread(test_img_path)
if img is None:
    raise FileNotFoundError(f"Could not read test image at: {test_img_path}")
resized_img = (
    cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255
)



## === cell 21
pred = model.predict(resized_img, verbose=0)
print("Pred probs:", pred[0], "Pred label:", int(np.argmax(pred[0])))



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2054155996.py in <cell line: 0>()
----> 1 pred = model.predict(resized_img, verbose=0)
      2 print("Pred probs:", pred[0], "Pred label:", int(np.argmax(pred[0])))
      3 

AttributeError: 'NoneType' object has no attribute 'predict'

## === cell 22
test_dir = "../input/cassava-leaf-disease-classification/test_images"
test_df = ss[["image_id"]].copy()
test_paths = (test_dir + "/" + test_df["image_id"].astype(str)).to_numpy()

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_decode_resize_rescale, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))
test_pred = model.predict(test_ds, steps=test_steps, verbose=0)
preds = np.argmax(test_pred, axis=1).astype(int).tolist()

my_submission = pd.DataFrame({"image_id": ss.image_id, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: submission.csv")

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1174121138.py in <cell line: 0>()
     14 
     15 test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))
---> 16 test_pred = model.predict(test_ds, steps=test_steps, verbose=0)
     17 preds = np.argmax(test_pred, axis=1).astype(int).tolist()
     18 

AttributeError: 'NoneType' object has no attribute 'predict'
