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

# 5. Code solution

## === cell 0
import math, re, os, multiprocessing, glob
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from functools import partial
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from PIL import Image

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception:
        pass
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    mixed_precision.set_policy("mixed_float16")

tf.random.set_seed(42)
np.random.seed(42)
tf.config.threading.set_intra_op_parallelism_threads(multiprocessing.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(multiprocessing.cpu_count())

print("Tensorflow version " + tf.__version__)



## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64



## === cell 2
base = tf.keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
output = tf.keras.layers.Dense(5, activation="softmax")(x)
model = tf.keras.Model(inputs=base.input, outputs=output)

base.trainable = False
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
mod_lst = [model]



## === cell 3
train_dir = os.path.abspath("../input/cassava-leaf-disease-classification/train_images")
train_csv_path = os.path.abspath(
    "../input/cassava-leaf-disease-classification/train.csv"
)

train_df = pd.read_csv(train_csv_path)
train_df["path"] = train_df["image_id"].apply(lambda x: os.path.join(train_dir, x))

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_df["path"].values,
    train_df["label"].values,
    test_size=0.2,
    stratify=train_df["label"].values,
    random_state=42,
)


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _process(path, label):
    return _decode_resize(path), label


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

early_stop = EarlyStopping(
    monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=2, mode="max"
)

print("Starting model fine‑tuning on training data …")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
)



## === cell 4
test_dir = "../input/cassava-leaf-disease-classification/test_images"




## === cell 5
def _fast_augment(img, aug_idx):
    """
    Deterministic augmentation matching the original implementation.
    img is a NumPy array of shape (H, W, 3), dtype uint8 or float.
    """
    if aug_idx == 0:
        return img
    elif aug_idx == 1:
        return np.fliplr(img)
    elif aug_idx == 2:
        return np.flipud(img)
    elif aug_idx == 3:
        return np.fliplr(np.flipud(img))
    else:
        crop_px = (aug_idx - 3) * 10
        h, w = img.shape[:2]
        crop_px = min(crop_px, h // 2 - 1, w // 2 - 1)
        cropped = img[crop_px : h - crop_px, crop_px : w - crop_px, :]
        pil_img = Image.fromarray(cropped).resize((IMAGE_SIZE, IMAGE_SIZE))
        return np.array(pil_img)




## === cell 6
def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    """
    Batch‑process test‑time augmentation using fully‑vectorized TensorFlow ops.
    The averaging logic over augmentations and models follows the original code.
    """
    preds = []
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(p) for p in image_paths]

    CHUNK = 512  # larger chunk reduces the number of Python‑level loops
    for start in range(0, len(image_paths), CHUNK):
        batch_paths = image_paths[start : start + CHUNK]

        raw_data = [tf.io.read_file(p) for p in batch_paths]
        decoded = [tf.image.decode_jpeg(b, channels=3) for b in raw_data]
        resized = [
            tf.image.resize(d, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
            for d in decoded
        ]
        orig_imgs = tf.stack(
            [tf.cast(r, tf.uint8) for r in resized], axis=0
        )  # (B, H, W, 3)

        B = tf.shape(orig_imgs)[0]
        aug_tensors = []

        for a_idx in range(aug_num):
            if a_idx == 0:
                aug = orig_imgs
            elif a_idx == 1:
                aug = tf.image.flip_left_right(orig_imgs)
            elif a_idx == 2:
                aug = tf.image.flip_up_down(orig_imgs)
            elif a_idx == 3:
                aug = tf.image.flip_up_down(tf.image.flip_left_right(orig_imgs))
            else:
                crop_px = (a_idx - 3) * 10
                h = tf.shape(orig_imgs)[1]
                w = tf.shape(orig_imgs)[2]
                crop_px = tf.minimum(crop_px, h // 2 - 1)
                crop_px = tf.minimum(crop_px, w // 2 - 1)
                cropped = tf.image.crop_to_bounding_box(
                    orig_imgs,
                    offset_height=crop_px,
                    offset_width=crop_px,
                    target_height=h - 2 * crop_px,
                    target_width=w - 2 * crop_px,
                )
                aug = tf.image.resize(
                    cropped, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear"
                )
                aug = tf.cast(aug, tf.uint8)
            aug_tensors.append(aug)

        aug_batch = tf.concat(aug_tensors, axis=0)

        if normalize:
            aug_batch = tf.cast(aug_batch, tf.float32) / 255.0

        pred_sum = 0.0
        for mod in model_obj_list:
            pred_sum += mod.predict(aug_batch, verbose=0)

        pred_reshaped = tf.reshape(pred_sum, (B, aug_num, -1))
        avg_pred = tf.reduce_mean(pred_reshaped, axis=1)  # (B, num_classes)

        batch_preds = tf.argmax(avg_pred, axis=1).numpy()
        preds.extend(batch_preds.tolist())

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 7
def get_preds(image_dir, model_obj, normalize=True):
    preds = []
    img_ids = []

    for i in glob.glob(image_dir + "/*.jpg"):
        image = Image.open(i).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(image)
        img_norm = img_array / 255.0 if normalize else img_array
        img_batch = np.expand_dims(img_norm, axis=0)
        preds.append(int(np.argmax(model_obj.predict(img_batch))))
        img_ids.append(i.replace(image_dir + "/", ""))

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 8
predict_df = get_preds_model_list(test_dir, mod_lst, normalize=True, aug_num=9)



## === cell 9
predict_df.to_csv("submission.csv", index=False)



## === cell 10
display(predict_df.head())
