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

0.8735267452402539

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05531) has done: 'The main timeout comes from doing heavy Python-side augmentation via `tf.numpy_function` on every image, plus training two separate models sequentially for 32 epochs each. To keep the same architecture, losses, epochs, and data semantics while cutting overhead, I (1) remove the unnecessary `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` slowdown, (2) switch augmentation/standardization to fully-TensorFlow equivalents (same transforms set, still deterministic per-example seed), and (3) cache the decoded+resized images so JPEG decode/resize isn’t repeated every epoch. These changes keep the training loop, model, and evaluation identical in intent, but drastically reduce Python overhead and redundant work so it can fit within 600 seconds.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    gpus = tf.config.list_physical_devices("GPU")
    for _g in gpus:
        tf.config.experimental.set_memory_growth(_g, True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




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

test_images_dir = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"




## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()

print(train_csv.head())
print("Num classes in map:", len(label_class))




## === cell 3
IMG_SIZE = 288
BATCH_SIZE = 12
EPOCHS = 32
lr = 1e-5




## === cell 4
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


def _cv2_imread(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


train_gen.preprocessing_function = None
valid_gen.preprocessing_function = None




## === cell 5
from sklearn.model_selection import train_test_split

idx = np.arange(len(train_csv))
train_idx, valid_idx = train_test_split(
    idx,
    test_size=0.2,
    random_state=SEED,
    shuffle=True,
    stratify=train_csv["label"].values,
)
train_df = train_csv.iloc[train_idx].reset_index(drop=True)
valid_df = train_csv.iloc[valid_idx].reset_index(drop=True)

classes = sorted(train_csv["label"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(classes)}
NUM_CLASSES = len(classes)

train_paths = (images_dir_path + "/" + train_df["image_id"].astype(str)).values
valid_paths = (images_dir_path + "/" + valid_df["image_id"].astype(str)).values
train_labels = train_df["label"].map(class_to_idx).astype(np.int32).values
valid_labels = valid_df["label"].map(class_to_idx).astype(np.int32).values


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


def _one_hot(y):
    return tf.one_hot(y, depth=NUM_CLASSES, dtype=tf.float32)


def _augment_tf(img, seed_i32):
    seed2 = tf.stack([tf.cast(SEED, tf.int32), tf.cast(seed_i32, tf.int32)], axis=0)

    x = img  # float32 in [0,255] range

    k = tf.random.stateless_uniform([], seed=seed2, minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)

    x = tf.image.stateless_random_flip_left_right(
        x, seed=seed2 + tf.constant([1, 0], tf.int32)
    )
    x = tf.image.stateless_random_flip_up_down(
        x, seed=seed2 + tf.constant([2, 0], tf.int32)
    )

    b = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([3, 0], tf.int32), minval=0.1, maxval=0.9
    )
    x = x * b

    cshift = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([4, 0], tf.int32), minval=-0.1, maxval=0.1
    )
    x = x + (cshift * 255.0)

    shift_y = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([5, 0], tf.int32), minval=-0.2, maxval=0.2
    )
    shift_x = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([6, 0], tf.int32), minval=-0.2, maxval=0.2
    )
    scale = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([7, 0], tf.int32), minval=0.7, maxval=1.3
    )

    new_size = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * scale), tf.int32)
    new_size = tf.clip_by_value(new_size, IMG_SIZE // 2, IMG_SIZE * 2)
    x = tf.image.resize(x, [new_size, new_size], method="bilinear")

    pad = IMG_SIZE
    x = tf.pad(x, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    center = pad + new_size // 2
    dy = tf.cast(tf.round(shift_y * tf.cast(IMG_SIZE, tf.float32)), tf.int32)
    dx = tf.cast(tf.round(shift_x * tf.cast(IMG_SIZE, tf.float32)), tf.int32)
    start_y = center - IMG_SIZE // 2 + dy
    start_x = center - IMG_SIZE // 2 + dx
    x = tf.image.crop_to_bounding_box(x, start_y, start_x, IMG_SIZE, IMG_SIZE)

    x = tf.clip_by_value(x, 0.0, 255.0) * (1.0 / 255.0)
    return x


def _valid_standardize_tf(img):
    return tf.cast(img, tf.float32) * (1.0 / 255.0)


def _make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(labels), seed=SEED, reshuffle_each_iteration=True)

    def _decode_map(p, y):
        return _decode_resize(p), y

    ds = ds.map(_decode_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()

    ds = ds.enumerate()

    def _aug_map(i, xy):
        img, y = xy
        seed = tf.cast(i, tf.int32) + tf.constant(SEED, tf.int32)
        img = _augment_tf(img, seed)
        return img, _one_hot(y)

    ds = ds.map(_aug_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _make_valid_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(p, y):
        img = _decode_resize(p)
        img = _valid_standardize_tf(img)
        return img, _one_hot(y)

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = _make_train_ds(train_paths, train_labels)
valid_dataset = _make_valid_ds(valid_paths, valid_labels)

steps_per_epoch = int(np.ceil(len(train_labels) / BATCH_SIZE))
validation_steps = int(np.ceil(len(valid_labels) / BATCH_SIZE))

print("Train/valid sizes:", len(train_labels), len(valid_labels))
print("Steps:", steps_per_epoch, validation_steps)




## === cell 6
@tf.function
def F1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred_bin = tf.cast(y_pred >= 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred_bin)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred_bin)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred_bin))

    precision = tp / (tp + fp + 1e-23)
    recall = tp / (tp + fn + 1e-23)
    return 2.0 * (precision * recall) / (precision + recall + 1e-23)




## === cell 7
BASE0 = applications.MobileNet(
    include_top=False, input_shape=[IMG_SIZE, IMG_SIZE, 3], weights=None, pooling="max"
)


def build_model(input_size=[IMG_SIZE, IMG_SIZE, 3]):
    model = tf.keras.Sequential()
    model.add(BASE0)
    model.add(Dropout(0.5))
    model.add(Dense(5, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
        metrics=["accuracy", F1_score],
    )
    return model


model0 = build_model()
model1 = build_model()

model0.summary()




## === cell 8
callback0 = tf.keras.callbacks.ModelCheckpoint(
    "CasavaLeafDiseaseModel_0.h5", monitor="val_loss", save_best_only=True
)
callback1 = tf.keras.callbacks.ModelCheckpoint(
    "CasavaLeafDiseaseModel_1.h5", monitor="val_loss", save_best_only=True
)




## === cell 9
history0 = model0.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[callback0],
    verbose=2,
)

history1 = model1.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[callback1],
    verbose=2,
)

if os.path.exists("CasavaLeafDiseaseModel_0.h5"):
    model0 = tf.keras.models.load_model(
        "CasavaLeafDiseaseModel_0.h5", custom_objects={"F1_score": F1_score}
    )
if os.path.exists("CasavaLeafDiseaseModel_1.h5"):
    model1 = tf.keras.models.load_model(
        "CasavaLeafDiseaseModel_1.h5", custom_objects={"F1_score": F1_score}
    )




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/4124852092.py in <cell line: 0>()
----> 1 history0 = model0.fit(
      2     train_dataset,
      3     validation_data=valid_dataset,
      4     epochs=EPOCHS,
      5     steps_per_epoch=steps_per_epoch,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node MirrorPad defined at (most recent call last):
<stack traces unavailable>
paddings must be less than the dimension size: 288, 288 not less than 247
	 [[{{node MirrorPad}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_11833]

## === cell 10
ss = pd.read_csv(sample_sub_path)


def _load_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


test_image_ids = ss["image_id"].astype(str).tolist()
test_paths = [os.path.join(test_images_dir, img_id) for img_id in test_image_ids]

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)




## === cell 11
p0 = model0.predict(test_ds, verbose=0)
p1 = model1.predict(test_ds, verbose=0)
p = (p0 + p1) / 2.0
preds = np.argmax(p, axis=1)

my_submission = pd.DataFrame({"image_id": ss["image_id"], "label": preds.astype(int)})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)




## === cell 12
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nLabel value counts:\n", my_submission["label"].value_counts().sort_index())
