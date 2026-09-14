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

3.12

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

0.8401329706860079

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.models import load_model



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification/"




## === cell 2
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)


custom_objects = {"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy}



## === cell 3
model_path = "/kaggle/input/finalmodel/myfinal_model.h5"
try:
    model = load_model(model_path, custom_objects=custom_objects)
    print("Loaded existing model.")
except Exception as e:
    print(f"Model load failed ({e}); building a new model.")
    base_model = efficientnet_v2.EfficientNetV2B0(
        include_top=False, input_shape=(224, 224, 3), weights="imagenet"
    )
    base_model.trainable = False  # freeze base
    inputs = layers.Input(shape=(224, 224, 3))
    x = efficientnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(5, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    train_df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label"], random_state=42
    )

    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        preprocessing_function=efficientnet_v2.preprocess_input,
    )
    val_gen = ImageDataGenerator(
        preprocessing_function=efficientnet_v2.preprocess_input
    )

    train_flow = train_gen.flow_from_dataframe(
        dataframe=train_df,
        directory=os.path.join(WORK_DIR, "train_images"),
        x_col="image_id",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
    )
    val_flow = val_gen.flow_from_dataframe(
        dataframe=val_df,
        directory=os.path.join(WORK_DIR, "train_images"),
        x_col="image_id",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_flow,
        epochs=3,
        validation_data=val_flow,
        callbacks=[
            callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=1, factor=0.5),
        ],
        verbose=2,
    )
    print("Training completed.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/235413569.py in <cell line: 0>()
      3 try:
----> 4     model = load_model(model_path, custom_objects=custom_objects)
      5     print("Loaded existing model.")

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/finalmodel/myfinal_model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/235413569.py in <cell line: 0>()
     21     train_df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
     22     # Split into train/validation
---> 23     train_df, val_df = train_test_split(
     24         train_df, test_size=0.1, stratify=train_df["label"], random_state=42
     25     )

NameError: name 'train_test_split' is not defined

## === cell 4
test_dir = os.path.join(WORK_DIR, "test_images")
test_images = sorted(os.listdir(test_dir))

submission = pd.DataFrame(columns=["image_id", "label"])

batch_size = 32
for i in range(0, len(test_images), batch_size):
    batch_names = test_images[i : i + batch_size]
    batch_paths = [os.path.join(test_dir, name) for name in batch_names]
    batch_imgs = []
    for p in batch_paths:
        img = kimage.load_img(p, target_size=(224, 224))
        img_arr = kimage.img_to_array(img)
        batch_imgs.append(img_arr)
    batch_array = np.stack(batch_imgs, axis=0)
    batch_preprocessed = efficientnet_v2.preprocess_input(batch_array.astype("float32"))
    preds = model.predict(batch_preprocessed, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    for name, label in zip(batch_names, pred_labels):
        submission = submission.append(
            {"image_id": name, "label": int(label)}, ignore_index=True
        )

submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written with", len(submission), "records.")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/352112716.py in <cell line: 0>()
     19     pred_labels = np.argmax(preds, axis=1)
     20     for name, label in zip(batch_names, pred_labels):
---> 21         submission = submission.append(
     22             {"image_id": name, "label": int(label)}, ignore_index=True
     23         )

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'
