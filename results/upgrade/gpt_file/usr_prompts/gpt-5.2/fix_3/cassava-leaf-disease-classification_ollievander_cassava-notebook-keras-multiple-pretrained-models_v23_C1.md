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
import os

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")



## === cell 1
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

import albumentations as A
import cv2

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

SEED = 100
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass



## === cell 2
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
train.head()



## === cell 3
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)
classes



## === cell 4
train["class"] = train["label"].apply(lambda x: classes[str(x)])
train[["image_id", "label", "class"]].head()



## === cell 5
if False:
    plt.figure(figsize=(15, 7))
    sns.countplot(x=train["class"], order=train["class"].value_counts().index)
    plt.tight_layout()
    plt.show()



## === cell 6
train["path"] = train["image_id"].apply(lambda x: os.path.join(TRAIN_PATH, str(x)))
train_df = train.copy()
train_df["label"] = train_df["label"].astype(str)

train_df, val_df = train_test_split(
    train_df, test_size=0.05, random_state=SEED, stratify=train_df["label"].values
)
train_df.head()



## === cell 7
batch_size = 4

_train_aug = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=40, p=0.5),
        A.Transpose(p=0.5),
    ]
)


def transform(image):
    return _train_aug(image=image)["image"]


datagen = ImageDataGenerator(
    preprocessing_function=transform, rescale=1.0 / 255.0
).flow_from_dataframe(
    batch_size=batch_size,
    dataframe=train_df,
    directory=TRAIN_PATH,
    shuffle=True,
    x_col="image_id",
    y_col="label",
    target_size=(512, 512),
    class_mode="categorical",
    seed=SEED,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_dataframe(
    batch_size=batch_size,
    dataframe=val_df,
    directory=TRAIN_PATH,
    shuffle=False,
    x_col="image_id",
    y_col="label",
    target_size=(512, 512),
    class_mode="categorical",
    seed=SEED,
)



## === cell 8
NUM_CLASSES = 5

model = models.Sequential(
    [
        layers.Input(shape=(512, 512, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

steps_per_epoch = max(1, len(train_df) // batch_size)
val_steps = max(1, len(val_df) // batch_size)

history = model.fit(
    datagen,
    validation_data=val_datagen,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 9
def agg_preds(predictions, y):
    y_classes = np.argmax(y, axis=1)
    acc_hist = []
    for i in range(predictions.shape[0]):
        pred_agg = np.mean(predictions[: i + 1], axis=0)
        preds = np.argmax(pred_agg, axis=1)
        acc = preds == y_classes
        acc = np.mean(acc)
        acc_hist.append(acc)
    return acc_hist


def agg_acc(predictions, y):
    pred_agg = np.mean(predictions, axis=0)
    preds = np.argmax(pred_agg, axis=1)
    acc = np.mean(preds == y)
    return acc




## === cell 10
def _ensure_hwc(img):
    if img.ndim == 4:
        return img[0]
    return img


def _back_to_batch(img_hwc, like):
    if like.ndim == 4:
        return np.expand_dims(img_hwc, axis=0)
    return img_hwc


_tta_flip_v = A.Compose([A.VerticalFlip(p=1)])
_tta_rotate = A.Compose(
    [A.Rotate(limit=40, border_mode=cv2.BORDER_CONSTANT, value=0, p=1)]
)
_tta_flip_h = A.Compose([A.HorizontalFlip(p=1)])
_tta_dropout = A.Compose(
    [
        A.GridDropout(
            ratio=0.5,
            unit_size_min=None,
            unit_size_max=None,
            holes_number_x=None,
            holes_number_y=None,
            shift_x=0,
            shift_y=0,
            random_offset=False,
            fill_value=0,
            mask_fill_value=None,
            p=1,
        )
    ]
)
_tta_perspec = A.Compose([A.Perspective(scale=(0.02, 0.1), p=1)])


def flip_lr(image):
    img = _ensure_hwc(image)
    out = _tta_flip_v(image=img)["image"]
    return _back_to_batch(out, image)


def rotate(image):
    img = _ensure_hwc(image)
    out = _tta_rotate(image=img)["image"]
    return _back_to_batch(out, image)


def flip_hor(image):
    img = _ensure_hwc(image)
    out = _tta_flip_h(image=img)["image"]
    return _back_to_batch(out, image)


def dropout(image):
    img = _ensure_hwc(image)
    out = _tta_dropout(image=img)["image"]
    return _back_to_batch(out, image)


def perspec(image):
    img = _ensure_hwc(image)
    out = _tta_perspec(image=img)["image"]
    return _back_to_batch(out, image)




## === cell 11
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
test_images = sample_sub["image_id"].tolist()

TEST_DIR = TEST_PATH if TEST_PATH.endswith("/") else (TEST_PATH + "/")
pred_labels = []




## === cell 12
def _load_img_512_rgb_float01(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    return img_rgb.astype(np.float32) / 255.0


INFER_BATCH = 32

n = len(test_images)
i = 0
while i < n:
    batch_ids = test_images[i : i + INFER_BATCH]

    base_imgs = []
    valid_mask = []
    for image_id in batch_ids:
        arr = _load_img_512_rgb_float01(os.path.join(TEST_PATH, image_id))
        if arr is None:
            valid_mask.append(False)
        else:
            valid_mask.append(True)
            base_imgs.append(arr)

    if len(base_imgs) == 0:
        pred_labels.extend([0] * len(batch_ids))
        i += INFER_BATCH
        continue

    base = np.stack(base_imgs, axis=0)  # (B,512,512,3) float32 in [0,1]

    base_u8 = np.clip(base * 255.0, 0, 255).astype(np.uint8)

    vflip = np.empty_like(base_u8)
    hflip = np.empty_like(base_u8)
    gdrop = np.empty_like(base_u8)
    for j in range(base_u8.shape[0]):
        vflip[j] = _tta_flip_v(image=base_u8[j])["image"]
        hflip[j] = _tta_flip_h(image=base_u8[j])["image"]
        gdrop[j] = _tta_dropout(image=base_u8[j])["image"]

    vflip = vflip.astype(np.float32) / 255.0
    hflip = hflip.astype(np.float32) / 255.0
    gdrop = gdrop.astype(np.float32) / 255.0

    views = np.concatenate([base, hflip, vflip, gdrop], axis=0)
    preds_all = model.predict(views, verbose=0)  # (4B,5)
    B = base.shape[0]
    pred0 = preds_all[0:B]
    pred_h = preds_all[B : 2 * B]
    pred_v = preds_all[2 * B : 3 * B]
    pred_d = preds_all[3 * B : 4 * B]

    predi = (pred0 + pred_h + pred_v + pred_d) / 4.0
    batch_pred_classes = np.argmax(predi, axis=1).astype(int).tolist()

    out_idx = 0
    for is_valid in valid_mask:
        if not is_valid:
            pred_labels.append(0)
        else:
            pred_labels.append(batch_pred_classes[out_idx])
            out_idx += 1

    i += INFER_BATCH



## === cell 13
submission = pd.DataFrame({"image_id": test_images, "label": pred_labels})
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
submission.head()
