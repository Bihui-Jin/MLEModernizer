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
from PIL import Image

print("Tensorflow version " + tf.__version__)
tf.random.set_seed(42)
np.random.seed(42)



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
def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    preds = []
    img_ids = []

    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    for p in img_paths:
        image = Image.open(p).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_np = np.array(image)

        if TTA:
            if normalize:
                aug_imgs = [tta_augment(img_np) / 255.0 for _ in range(aug_num)]
            else:
                aug_imgs = [tta_augment(img_np) for _ in range(aug_num)]

            all_preds = []
            for aug in aug_imgs:
                aug_b = np.expand_dims(aug, axis=0)
                for mod in model_obj_list:
                    all_preds.append(mod.predict(aug_b, verbose=0))
            avg_pred = np.concatenate(all_preds, axis=0).mean(
                axis=0, keepdims=True
            )  # (1,5)
        else:
            x = img_np.astype(np.float32)
            if normalize:
                x = x / 255.0
            x = np.expand_dims(x, axis=0)
            all_preds = [mod.predict(x, verbose=0) for mod in model_obj_list]
            avg_pred = np.concatenate(all_preds, axis=0).mean(axis=0, keepdims=True)

        preds.append(int(np.argmax(avg_pred, axis=1)[0]))
        img_ids.append(os.path.basename(p))

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 6
def get_preds(image_dir, model_obj, normalize=True):
    preds = []
    img_ids = []

    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    for p in img_paths:
        image = Image.open(p).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        x = np.array(image).astype(np.float32)
        if normalize:
            x = x / 255.0
        x = np.expand_dims(x, axis=0)

        preds.append(int(np.argmax(model_obj.predict(x, verbose=0), axis=1)[0]))
        img_ids.append(os.path.basename(p))

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 7
predict_df = get_preds_model_list(test_dir, mod_lst, normalize=False)

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
