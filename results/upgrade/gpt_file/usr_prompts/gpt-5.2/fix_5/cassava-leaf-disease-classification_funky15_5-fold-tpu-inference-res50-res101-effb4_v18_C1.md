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
import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

print("Tensorflow version " + tf.__version__)
tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.optimizer.set_jit(
        False
    )  # keep numerics stable; avoid compilation latency
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() or 1))
tf.config.threading.set_inter_op_parallelism_threads(max(1, (os.cpu_count() or 1) // 2))



## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64



## === cell 2
model_path = "../input/efficient-net-0115/efficientnet_0.h5"

if os.path.exists(model_path):
    modeleffb4_0 = keras.models.load_model(model_path, compile=False)
else:
    base = keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
    )
    x = keras.layers.Dropout(0.2)(base.output)
    out = keras.layers.Dense(5, activation="softmax")(x)
    modeleffb4_0 = keras.Model(inputs=base.input, outputs=out)

mod_lst = [modeleffb4_0]



## === cell 3
test_dir = "../input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_dir):
    test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_dir):
    test_dir = "/kaggle/data/input/cassava-leaf-disease-classification/test_images"

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "/kaggle/data/input/cassava-leaf-disease-classification/sample_submission.csv"
    )

assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"
assert os.path.exists(
    sample_sub_path
), f"sample_submission.csv not found: {sample_sub_path}"




## === cell 4
def _random_resized_crop(img, target_size=IMAGE_SIZE, scale=(0.75, 1.0)):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    s = tf.random.uniform([], scale[0], scale[1])
    new_h = tf.cast(tf.cast(h, tf.float32) * s, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * s, tf.int32)
    new_h = tf.maximum(new_h, 1)
    new_w = tf.maximum(new_w, 1)
    offset_h = tf.random.uniform([], 0, tf.maximum(h - new_h + 1, 1), dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, tf.maximum(w - new_w + 1, 1), dtype=tf.int32)
    cropped = tf.image.crop_to_bounding_box(img, offset_h, offset_w, new_h, new_w)
    cropped = tf.image.resize(cropped, (target_size, target_size), method="bilinear")
    return cropped


def tta_augment(img_np):
    x = _random_resized_crop(img_np, target_size=IMAGE_SIZE, scale=(0.75, 1.0))
    if tf.random.uniform([]) < 0.5:
        x = tf.image.flip_left_right(x)
    if tf.random.uniform([]) < 0.5:
        x = tf.image.flip_up_down(x)
    return x.numpy()




## === cell 5
@tf.function
def _decode_jpeg_to_float32(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    return tf.cast(img, tf.float32)


@tf.function
def _make_tta_batch(img, aug_num, normalize):
    ta = tf.TensorArray(tf.float32, size=aug_num)
    i = tf.constant(0, dtype=tf.int32)

    def cond(i, ta):
        return i < aug_num

    def body(i, ta):
        x = _random_resized_crop(img, target_size=IMAGE_SIZE, scale=(0.75, 1.0))
        if tf.random.uniform([]) < 0.5:
            x = tf.image.flip_left_right(x)
        if tf.random.uniform([]) < 0.5:
            x = tf.image.flip_up_down(x)
        if normalize:
            x = x / 255.0
        ta = ta.write(i, x)
        return i + 1, ta

    _, ta = tf.while_loop(cond, body, [i, ta], parallel_iterations=1)
    return ta.stack()  # [aug_num, IMAGE_SIZE, IMAGE_SIZE, 3]


def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    assert len(model_obj_list) >= 1

    model = model_obj_list[0]
    other_models = model_obj_list[1:]

    paths_ds = tf.data.Dataset.from_tensor_slices(img_paths)

    def _load_one(p):
        img = _decode_jpeg_to_float32(p)
        return p, img

    ds = paths_ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)

    if TTA:
        aug_num_t = tf.constant(int(aug_num), tf.int32)
        norm_t = tf.constant(bool(normalize))

        def _tta_map(p, img):
            tta = _make_tta_batch(img, aug_num_t, norm_t)
            return p, tta

        ds = ds.map(_tta_map, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

        preds = []
        img_ids = []

        for p_batch, tta_batch in ds:
            b = tf.shape(tta_batch)[0]
            a = tf.shape(tta_batch)[1]
            flat = tf.reshape(tta_batch, [b * a, IMAGE_SIZE, IMAGE_SIZE, 3])

            pred = model(flat, training=False)
            if other_models:
                for m in other_models:
                    pred += m(flat, training=False)
                pred /= float(1 + len(other_models))

            pred = tf.reshape(pred, [b, a, -1])  # [B, aug, 5]
            avg_pred = tf.reduce_mean(pred, axis=1)  # [B, 5]
            labels = tf.argmax(avg_pred, axis=-1)  # [B]

            preds.extend(labels.numpy().astype(np.int32).tolist())
            img_ids.extend(
                [os.path.basename(x.decode("utf-8")) for x in p_batch.numpy()]
            )

        return pd.DataFrame({"image_id": img_ids, "label": preds})

    else:
        def _norm_map(p, img):
            img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
            if normalize:
                img = img / 255.0
            return p, img

        ds = ds.map(_norm_map, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

        n = len(img_paths)
        out_labels = np.empty((n,), dtype=np.int32)
        out_ids = [None] * n
        k = 0

        for p_batch, x_batch in ds:
            pred = model(x_batch, training=False)
            if other_models:
                for m in other_models:
                    pred += m(x_batch, training=False)
                pred /= float(1 + len(other_models))
            labels = tf.argmax(pred, axis=-1).numpy().astype(np.int32)

            bsz = labels.shape[0]
            out_labels[k : k + bsz] = labels

            p_np = p_batch.numpy()
            out_ids[k : k + bsz] = [os.path.basename(p).decode("utf-8") for p in p_np]
            k += bsz

        return pd.DataFrame({"image_id": out_ids, "label": out_labels})




## === cell 6
def get_preds(image_dir, model_obj, normalize=True):
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    paths_ds = tf.data.Dataset.from_tensor_slices(img_paths)

    def _load_and_norm(p):
        img = _decode_jpeg_to_float32(p)
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
        if normalize:
            img = img / 255.0
        return p, img

    ds = paths_ds.map(_load_and_norm, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    preds = []
    img_ids = []
    for p_batch, x_batch in ds:
        pred = model_obj(x_batch, training=False)
        labels = tf.argmax(pred, axis=-1)
        preds.extend(labels.numpy().astype(np.int32).tolist())
        img_ids.extend([os.path.basename(x.decode("utf-8")) for x in p_batch.numpy()])

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 7
predict_df = get_preds_model_list(test_dir, mod_lst, TTA=False, normalize=False)

sub = pd.read_csv(sample_sub_path)
predict_df = sub[["image_id"]].merge(predict_df, on="image_id", how="left")

if predict_df["label"].isna().any():
    predict_df["label"] = predict_df["label"].fillna(0).astype(int)
else:
    predict_df["label"] = predict_df["label"].astype(int)

predict_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predict_df.shape)
print(predict_df.head())



## === cell 8
print(predict_df.describe(include="all"))
print("Unique labels:", sorted(predict_df["label"].unique().tolist()))
print("submission.csv exists:", os.path.exists("submission.csv"))
