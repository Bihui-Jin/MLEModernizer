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
import numpy as np



## === cell 1
np.concatenate([[1, 2, 3], [2, 3, 4]])



## === cell 2
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import math, re, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

print("Tensorflow version " + tf.__version__)



## === cell 3
IMAGE_SIZE = 512
BATCH_SIZE = 64
NUM_CLASSES = 5

tf.random.set_seed(42)
np.random.seed(42)




## === cell 4
def build_fallback_model(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def safe_load_model(path, fallback_name):
    if os.path.exists(path):
        try:
            return keras.models.load_model(path)
        except Exception as e:
            print(
                f"Warning: failed to load {path} ({e}); using fallback model {fallback_name}."
            )
            return build_fallback_model()
    print(
        f"Warning: model file not found: {path}; using fallback model {fallback_name}."
    )
    return build_fallback_model()


model101_4 = safe_load_model(
    "../input/tpus-resnet101-with-5-fold/resnet101_4.h5", "resnet101_4_fallback"
)
model50_4 = safe_load_model(
    "../input/resnet50-5fold-0/resnet50_4.h5", "resnet50_4_fallback"
)
modeleffb4_0 = safe_load_model(
    "../input/efficientb4-net-0117/efficientnet_0.h5", "efficientnetb4_0_fallback"
)

mod_lst = [modeleffb4_0, model50_4, model101_4]



## === cell 5
test_dir = "../input/cassava-leaf-disease-classification/test_images"
assert os.path.isdir(test_dir), f"Test directory not found: {test_dir}"




## === cell 6
@tf.function
def tta_augment_tf(img_uint8):
    img = tf.cast(img_uint8, tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.08 * 255.0)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


@tf.function
def tta_augment_tf_batch(imgs_uint8):
    imgs = tf.cast(imgs_uint8, tf.float32)
    imgs = tf.image.random_flip_left_right(imgs)
    imgs = tf.image.random_flip_up_down(imgs)
    imgs = tf.image.random_brightness(imgs, max_delta=0.08 * 255.0)
    imgs = tf.image.random_contrast(imgs, lower=0.9, upper=1.1)
    imgs = tf.clip_by_value(imgs, 0.0, 255.0)
    return imgs


def augment_np(image_np_uint8):
    out = tta_augment_tf(image_np_uint8)
    return out.numpy()




## === cell 7
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_img_dir = "../input/cassava-leaf-disease-classification/train_images"


def _is_fallback(model):
    return len(model.layers) <= 8


def build_train_dataset(df, image_dir, batch_size=BATCH_SIZE, shuffle=True):
    paths = [os.path.join(image_dir, fn) for fn in df["image_id"].values]
    labels = df["label"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        img = tf.cast(
            img, tf.float32
        )  # keep 0..255 to match inference normalization control
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(min(len(df), 8192), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


if os.path.exists(train_csv_path) and os.path.isdir(train_img_dir):
    train_df = pd.read_csv(train_csv_path)
    split = int(len(train_df) * 0.95)
    train_part = train_df.iloc[:split].reset_index(drop=True)
    val_part = train_df.iloc[split:].reset_index(drop=True)

    train_ds = build_train_dataset(
        train_part, train_img_dir, batch_size=BATCH_SIZE, shuffle=True
    )
    val_ds = build_train_dataset(
        val_part, train_img_dir, batch_size=BATCH_SIZE, shuffle=False
    )

    for i, m in enumerate(mod_lst):
        if _is_fallback(m):
            m.fit(train_ds, validation_data=val_ds, epochs=2, verbose=1)
else:
    print(
        "Warning: training data not found; proceeding without training fallback models."
    )




## === cell 8
def get_preds_model_list_norm_inds(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize_indices=None
):
    if normalize_indices is None:
        normalize_indices = [1] * len(model_obj_list)
    if len(normalize_indices) != len(model_obj_list):
        raise ValueError("normalize_indices must have same length as model_obj_list")

    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(fp) for fp in files]
    n = len(files)
    if n == 0:
        return pd.DataFrame({"image_id": [], "label": []})

    base_uint8 = np.empty((n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i, fp in enumerate(files):
        with Image.open(fp) as im:
            im = im.convert("RGB")
            im = im.resize((IMAGE_SIZE, IMAGE_SIZE))
            base_uint8[i] = np.asarray(im, dtype=np.uint8)

    if TTA:
        base_tf = tf.convert_to_tensor(base_uint8, dtype=tf.uint8)
        aug_batches = [tta_augment_tf_batch(base_tf) for _ in range(aug_num)]
    else:
        aug_batches = [tf.cast(base_uint8, tf.float32)]

    pred_sum = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    denom = float(len(model_obj_list) * len(aug_batches))

    for aug_tf in aug_batches:
        aug_np = aug_tf.numpy()  # (N,H,W,3) float32 0..255
        for mod, norm_flag in zip(model_obj_list, normalize_indices):
            if norm_flag == 1:
                x = aug_np / 255.0
            else:
                x = aug_np
            pred_sum += mod.predict(x, batch_size=BATCH_SIZE, verbose=0)

    avg_pred = pred_sum / denom
    labels = np.argmax(avg_pred, axis=1).astype(np.int64)

    return pd.DataFrame({"image_id": img_ids, "label": labels})


def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    normalize_indices = [1 if normalize else 0] * len(model_obj_list)
    return get_preds_model_list_norm_inds(
        image_dir,
        model_obj_list,
        TTA=TTA,
        aug_num=aug_num,
        normalize_indices=normalize_indices,
    )


def get_preds(image_dir, model_obj, normalize=True):
    return get_preds_model_list_norm_inds(
        image_dir,
        [model_obj],
        TTA=False,
        aug_num=1,
        normalize_indices=[1 if normalize else 0],
    )




## === cell 9
try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() or 1))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 10
predict_df = get_preds_model_list_norm_inds(
    test_dir, mod_lst, TTA=True, aug_num=5, normalize_indices=[0, 1, 1]
)

predict_df = predict_df[["image_id", "label"]]
predict_df.to_csv("submission.csv", index=False)

print(predict_df.head())
print("Wrote submission.csv with rows:", len(predict_df))



## === cell 11
print(predict_df.describe(include="all"))
