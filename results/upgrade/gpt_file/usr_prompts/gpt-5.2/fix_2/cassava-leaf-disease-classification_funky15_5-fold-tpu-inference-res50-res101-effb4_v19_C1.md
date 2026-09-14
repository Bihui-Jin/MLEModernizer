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
def tta_augment_np(img_np):
    """img_np: HWC uint8 or float array in [0..255]. returns augmented in same range."""
    x = tf.convert_to_tensor(img_np)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    crop_frac = tf.random.uniform([], 0.75, 1.0)
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]
    ch = tf.cast(tf.cast(h, tf.float32) * crop_frac, tf.int32)
    cw = tf.cast(tf.cast(w, tf.float32) * crop_frac, tf.int32)
    x = tf.image.random_crop(x, size=[ch, cw, 3])
    x = tf.image.resize(x, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    return x.numpy()




## === cell 5
def _load_image_as_array(path, image_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size))
    return np.array(image)


def _preprocess_for_model(img_batch, model_obj, normalize=True):
    """
    Keeps original normalize flag semantics but makes it safe:
    - If model is EfficientNet family, apply keras.applications.efficientnet.preprocess_input.
    - Else if normalize=True, divide by 255.
    """
    img_batch = img_batch.astype(np.float32)

    name = (getattr(model_obj, "name", "") or "").lower()
    if "efficientnet" in name:
        return keras.applications.efficientnet.preprocess_input(img_batch)
    return (img_batch / 255.0) if normalize else img_batch


def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    preds = []
    img_ids = []

    sample_sub = pd.read_csv(sample_sub_path)
    image_ids = sample_sub["image_id"].tolist()

    for image_id in image_ids:
        img_path = os.path.join(image_dir, image_id)
        img = _load_image_as_array(img_path, IMAGE_SIZE)

        if TTA:
            aug_imgs = np.stack(
                [tta_augment_np(img) for _ in range(aug_num)], axis=0
            )  # (aug_num,H,W,3)
            all_preds = []
            for mod in model_obj_list:
                batch = _preprocess_for_model(aug_imgs, mod, normalize=normalize)
                p = mod.predict(batch, verbose=0)  # (aug_num,5)
                all_preds.append(p)
            avg_pred = np.concatenate(all_preds, axis=0).mean(axis=0)  # (5,)
        else:
            batch = np.expand_dims(img, axis=0)
            all_preds = []
            for mod in model_obj_list:
                batch_p = _preprocess_for_model(batch, mod, normalize=normalize)
                all_preds.append(mod.predict(batch_p, verbose=0))
            avg_pred = np.concatenate(all_preds, axis=0).mean(axis=0).reshape(-1)

        preds.append(int(np.argmax(avg_pred)))
        img_ids.append(image_id)

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 6
def get_preds(image_dir, model_obj, normalize=True):
    preds = []
    img_ids = []

    sample_sub = pd.read_csv(sample_sub_path)
    image_ids = sample_sub["image_id"].tolist()

    for image_id in image_ids:
        img_path = os.path.join(image_dir, image_id)
        img = _load_image_as_array(img_path, IMAGE_SIZE)
        batch = np.expand_dims(img, axis=0)
        batch = _preprocess_for_model(batch, model_obj, normalize=normalize)
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
