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

0.2311876699909338

# 6. Current score

0.12631

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.12631) has done: 'I fix the two blockers that prevent the notebook from running end-to-end: (1) the protobuf/TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation, and (2) the missing EfficientNet weight file path by falling back to `weights="imagenet"` when the provided local `.h5` does not exist. These changes keep the same model architecture and training loop, but make them robust to the Kaggle environment. I also make the train/test image paths resilient to the dataset being mounted in either `/kaggle/input/...` or `/kaggle/data/...`, and ensure `submission.csv` is always written with the required columns and aligned to `sample_submission.csv`. No score-tuning beyond restoring the intended pretrained backbone is introduced; this should naturally move accuracy above the target band from “not yielded” to a valid baseline.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import random

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow.keras.backend as K

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)

CANDIDATE_BASES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
BASE_PATH = None
for p in CANDIDATE_BASES:
    if tf.io.gfile.exists(p):
        BASE_PATH = p
        break
if BASE_PATH is None:
    BASE_PATH = CANDIDATE_BASES[0]

TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
LABEL_MAP_PATH = os.path.join(BASE_PATH, "label_num_to_disease_map.json")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

print("Using BASE_PATH:", BASE_PATH)
print("Train CSV exists:", tf.io.gfile.exists(TRAIN_CSV_PATH))
print("Train images dir exists:", tf.io.gfile.exists(TRAIN_IMG_DIR))
print("Test images dir exists:", tf.io.gfile.exists(TEST_IMG_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv = pd.read_csv(TRAIN_CSV_PATH)
print("Number of train images: {}".format(len(train_csv)))



## === cell 2
train_csv.head()



## === cell 3
with open(LABEL_MAP_PATH, "r") as fp:
    class_map = json.load(fp)
class_map



## === cell 4
ax = train_csv.pivot_table(columns="label", aggfunc="size").plot(kind="barh")
ax.set_yticklabels(list(class_map.values()))
ax.set_xlabel("count")



## === cell 5
pass



## === cell 6
pass




## === cell 7
@tf.function
def process_data(path, label):
    path = tf.strings.join([tf.constant(TRAIN_IMG_DIR + "/"), path])
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img, tf.one_hot(label, 5)




## === cell 8
train, val = train_test_split(
    train_csv, test_size=0.1, random_state=SEED, stratify=train_csv["label"]
)



## === cell 9
oversampled_df = []
target_count = int(train.pivot_table(columns="label", aggfunc="size").values[3] * 0.8)
for i in range(len(class_map)):
    class_i = train[train.label == i]
    oversampled_df.append(class_i.sample(target_count, replace=True))



## === cell 10
resampled_train = pd.concat(oversampled_df, axis=0)
resampled_train.pivot_table(columns="label", aggfunc="size")



## === cell 11
resampled_train = resampled_train.sample(frac=1, random_state=SEED).reset_index(
    drop=True
)
resampled_train.head()



## === cell 12
batch_size = 8



## === cell 13
print("Planned batch_size:", batch_size)
print(
    "Train rows:",
    len(train),
    "Resampled train rows:",
    len(resampled_train),
    "Val rows:",
    len(val),
)



## === cell 14
train_ds = (
    tf.data.Dataset.from_tensor_slices(
        (resampled_train.image_id.values, resampled_train.label.values)
    )
    .map(process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .shuffle(buffer_size=2000, seed=SEED, reshuffle_each_iteration=True)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val.image_id.values, val.label.values))
    .map(process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)



## === cell 15
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomCrop(height=512, width=512),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom((-0.2, 0.0)),
        tf.keras.layers.RandomContrast((0.0, 0.2)),
    ]
)



## === cell 16
pass



## === cell 17
for images, labels in train_ds.take(1):
    images_aug = data_augmentation(images, training=True)
    print(images_aug.shape, labels.shape)
    plt.figure(figsize=(10, 10))
    labels_np = np.argmax(labels.numpy(), -1)
    for i in range(min(8, images_aug.shape[0])):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images_aug.numpy()[i].astype(np.uint8))
        plt.title(class_map[str(labels_np[i])])
        plt.axis("off")
    plt.show()



## === cell 18
pass




## === cell 19
def plot_metrics(history, metrics=["loss", "accuracy"]):
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label="Train")
        plt.plot(
            history.epoch, history.history["val_" + metric], linestyle="--", label="Val"
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0, 1])
    plt.legend()




## === cell 20
def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()




## === cell 21
def sigmoid_focal_crossentropy(
    y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False
):
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)




## === cell 22
def build_efficient_model(
    input_layer, input_shape, model_inputs, num_classes, dropout_rate=0.2
):
    model = tf.keras.applications.EfficientNetB3(
        weights="/kaggle/input/efficientnetb3notop/efficientnetb3_notop.h5",
        include_top=False,
        input_shape=input_shape,
    )

    model.trainable = False
    model_output = model(model_inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model_output)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = tf.keras.Model(input_layer, outputs, name="EfficientNet")
    return model




## === cell 23
input_shape = (300, 300, 3)
dropout_rate = 0.2
num_classes = len(class_map)



## === cell 24
EFFICIENTNET_LOCAL_WEIGHTS = "/kaggle/input/efficientnetb3notop/efficientnetb3_notop.h5"
effnet_weights = (
    EFFICIENTNET_LOCAL_WEIGHTS
    if tf.io.gfile.exists(EFFICIENTNET_LOCAL_WEIGHTS)
    else "imagenet"
)
print("EfficientNetB3 weights:", effnet_weights)

input_layer = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
x = tf.keras.layers.Lambda(lambda t: tf.cast(t, tf.float32))(input_layer)
x = data_augmentation(x, training=False)
x = tf.keras.layers.Resizing(input_shape[0], input_shape[1])(x)

base_model = tf.keras.applications.EfficientNetB3(
    weights=effnet_weights,
    include_top=False,
    input_shape=input_shape,
)
base_model.trainable = False

model_output = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model_output)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
model = tf.keras.Model(input_layer, outputs, name="EfficientNet")



## === cell 25
model.summary()



## === cell 26
layers = [layer.name for layer in base_model.layers]
(layers.index("block7a_expand_conv"), len(layers))



## === cell 27
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]
optimizer = tf.keras.optimizers.Adam()



## === cell 28
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=METRICS)



## === cell 29
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, verbose=1
    ),
]

hist = model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_11/1632000195.py in <cell line: 0>()
     13 ]
     14 
---> 15 hist = model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks)
     16 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

OperatorNotAllowedInGraphError: Exception encountered when calling RandomCrop.call().

Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.

Arguments received by RandomCrop.call():
  • data=tf.Tensor(shape=(None, None, None, 3), dtype=float32)
  • training=True

## === cell 30
plot_metrics(hist, ["loss", "auc", "precision", "recall"])
plt.show()



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1257766677.py in <cell line: 0>()
----> 1 plot_metrics(hist, ["loss", "auc", "precision", "recall"])
      2 plt.show()
      3 

NameError: name 'hist' is not defined

## === cell 31
val_probs = model.predict(val_ds)
val_preds = tf.math.argmax(val_probs, -1)



## === cell 32
plot_cm(val.label.values, val_preds.numpy())



## === cell 33
np.mean(np.where(val.label.values == val_preds.numpy(), 1, 0))



## === cell 34
fine_tune_at = layers.index("block7a_expand_conv")
base_model.trainable = True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4, decay_steps=300 * 10
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule),
    loss="categorical_crossentropy",
    metrics=METRICS,
)



## === cell 35
fine_tune_epochs = 5
total_epochs = 5 + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=hist.epoch[-1] + 1,
    validation_data=val_ds,
    callbacks=callbacks,
)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3001517893.py in <cell line: 0>()
      5     train_ds,
      6     epochs=total_epochs,
----> 7     initial_epoch=hist.epoch[-1] + 1,
      8     validation_data=val_ds,
      9     callbacks=callbacks,

NameError: name 'hist' is not defined

## === cell 36
pass



## === cell 37
pass



## === cell 38
pass



## === cell 39
pass



## === cell 40
pass




## === cell 41
@tf.function
def process_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img




## === cell 42
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
print("Found test images:", len(TEST_FILENAMES))



## === cell 43
test_ds = (
    tf.data.Dataset.from_tensor_slices(TEST_FILENAMES)
    .map(process_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)



## === cell 44
probabilities = model.predict(test_ds)
predictions = np.argmax(probabilities, axis=-1)



## === cell 45
test_ids = [os.path.split(path)[1] for path in TEST_FILENAMES]
submission = pd.DataFrame({"image_id": test_ids, "label": predictions})

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
if submission["label"].isna().any():
    majority = int(train_csv["label"].mode().iloc[0])
    submission["label"] = submission["label"].fillna(majority).astype(int)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
submission.head()
