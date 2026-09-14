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
import os, glob, math, re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from PIL import Image

print("Tensorflow version " + tf.__version__)
tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() or 1))
tf.config.threading.set_inter_op_parallelism_threads(max(1, (os.cpu_count() or 1) // 2))




## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64




## === cell 2
model_path = "../input/efficientb4-net-0117/efficientnet_0.h5"


def build_fallback_efficientnetb4(num_classes=5, image_size=512):
    inp = keras.Input(shape=(image_size, image_size, 3))
    base = keras.applications.EfficientNetB4(
        include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
    )
    x = base.output
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inp, outputs=out)
    return model


if os.path.exists(model_path):
    modeleffb4_0 = keras.models.load_model(model_path, compile=False)
else:
    modeleffb4_0 = build_fallback_efficientnetb4(num_classes=5, image_size=IMAGE_SIZE)

mod_lst = [modeleffb4_0]
print("Models loaded:", len(mod_lst))




## === cell 3
test_dir = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"

assert os.path.isdir(test_dir), f"Missing test_dir: {test_dir}"
assert os.path.exists(sample_sub_path), f"Missing sample submission: {sample_sub_path}"

sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()
print("Test images in sample_submission:", len(test_image_ids))




## === cell 4
def tta_augment_tf(x):
    """
    x: [H,W,3] tensor float32 in [0..255] (same range as original tta_augment_np input/output).
    Returns: [IMAGE_SIZE, IMAGE_SIZE, 3] float32 in [0..255].
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    crop_frac = tf.random.uniform([], 0.75, 1.0)
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]
    ch = tf.cast(tf.cast(h, tf.float32) * crop_frac, tf.int32)
    cw = tf.cast(tf.cast(w, tf.float32) * crop_frac, tf.int32)
    x = tf.image.random_crop(x, size=[ch, cw, 3])
    x = tf.image.resize(x, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    return x




## === cell 5
def _preprocess_for_model_np(img_batch, model_obj, normalize=True):
    """
    Original semantics preserved:
    - If model name contains 'efficientnet' => keras.applications.efficientnet.preprocess_input
    - Else if normalize=True => /255
    """
    img_batch = img_batch.astype(np.float32)
    name = (getattr(model_obj, "name", "") or "").lower()
    if "efficientnet" in name:
        return keras.applications.efficientnet.preprocess_input(img_batch)
    return (img_batch / 255.0) if normalize else img_batch


def _make_preprocess_for_model_tf(model_obj, normalize=True):
    name = (getattr(model_obj, "name", "") or "").lower()
    is_effnet = "efficientnet" in name

    def _fn(x):
        x = tf.cast(x, tf.float32)
        if is_effnet:
            return keras.applications.efficientnet.preprocess_input(x)
        return (x / 255.0) if normalize else x

    return _fn


def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    image_ids = test_image_ids  # preserve the exact ordering from sample_submission
    paths = [os.path.join(image_dir, img_id) for img_id in image_ids]

    autotune = tf.data.AUTOTUNE

    def _decode_resize(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        return tf.cast(img, tf.float32)

    ds_base = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True  # preserve ordering / determinism
    try:
        options.threading.private_threadpool_size = max(1, (os.cpu_count() or 1) // 2)
    except Exception:
        pass
    ds_base = ds_base.with_options(options)

    ds_base = ds_base.map(_decode_resize, num_parallel_calls=autotune)
    ds_base = (
        ds_base.cache()
    )  # ~2.7k * 512*512*3 float32 fits; avoids repeated disk+decode

    if TTA:
        def _make_tta_batch(img):
            img_rep = tf.broadcast_to(img, [aug_num, IMAGE_SIZE, IMAGE_SIZE, 3])
            img_rep = tf.map_fn(
                tta_augment_tf,
                img_rep,
                fn_output_signature=tf.float32,
                parallel_iterations=aug_num,
            )
            return img_rep

        ds = ds_base.map(_make_tta_batch, num_parallel_calls=autotune)
        ds = (
            ds.unbatch()
        )  # now yields aug_num elements per original image (same ordering)
        num_aug = aug_num
    else:
        ds = ds_base
        num_aug = 1

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(autotune)

    all_model_probs = []
    preprocess_fns = [
        _make_preprocess_for_model_tf(m, normalize=normalize) for m in model_obj_list
    ]

    num_imgs = len(image_ids)
    total = num_imgs * num_aug

    for mod, pre_fn in zip(model_obj_list, preprocess_fns):
        ds_p = ds.map(pre_fn, num_parallel_calls=autotune).prefetch(autotune)
        probs = mod.predict(ds_p, verbose=0)
        if probs.shape[0] != total:
            probs = probs[:total]
        probs = probs.reshape((num_imgs, num_aug, -1)).mean(axis=1)  # [N, C]
        all_model_probs.append(probs)

    avg_probs = np.mean(np.stack(all_model_probs, axis=0), axis=0)  # [N, C]
    preds = avg_probs.argmax(axis=1).astype(int)

    return pd.DataFrame({"image_id": image_ids, "label": preds})




## === cell 6
def get_preds(image_dir, model_obj, normalize=True):
    preds = []
    img_ids = []
    image_ids = test_image_ids

    for image_id in image_ids:
        img_path = os.path.join(image_dir, image_id)
        image = Image.open(img_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(image)
        batch = np.expand_dims(img, axis=0)
        batch = _preprocess_for_model_np(batch, model_obj, normalize=normalize)
        preds.append(int(np.argmax(model_obj.predict(batch, verbose=0))))
        img_ids.append(image_id)

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 7
predict_df = get_preds_model_list(
    test_dir, mod_lst, TTA=True, aug_num=5, normalize=False
)

predict_df = sample_sub[["image_id"]].merge(predict_df, on="image_id", how="left")
assert predict_df["label"].notna().all(), "Some predictions are missing."
predict_df["label"] = predict_df["label"].astype(int)

predict_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predict_df.shape)
print(predict_df.head())




## === cell 8
print(predict_df.describe(include="all"))
print("Label distribution:\n", predict_df["label"].value_counts().sort_index())
