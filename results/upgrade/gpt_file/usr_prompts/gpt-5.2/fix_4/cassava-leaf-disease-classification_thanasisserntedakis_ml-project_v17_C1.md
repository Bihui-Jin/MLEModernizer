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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
try:
    from google.protobuf.internal import api_implementation

    try:
        api_implementation._SetType("python")
    except Exception:
        pass
except Exception:
    pass

import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

tf.random.set_seed(42)
np.random.seed(42)

print("TF:", tf.__version__)



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
ALT_DATA_DIR = "../input/cassava-leaf-disease-classification"

if not os.path.exists(DATA_DIR) and os.path.exists(ALT_DATA_DIR):
    DATA_DIR = ALT_DATA_DIR

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_GLOB = os.path.join(DATA_DIR, "test_images", "*.jpg")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

test_images = glob.glob(TEST_GLOB)
if len(test_images) == 0:
    raise FileNotFoundError(
        f"Could not find test images under expected path: {TEST_GLOB}"
    )

df_test = pd.DataFrame(test_images, columns=["path"])
df_test["image_id"] = df_test["path"].str.split("/").str[-1]

df_train = pd.read_csv(TRAIN_CSV)
df_train["path"] = df_train["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

missing = (~df_train["path"].apply(os.path.exists)).sum()
if missing:
    raise FileNotFoundError(
        f"{missing} training images referenced in train.csv were not found under {TRAIN_IMG_DIR}"
    )

IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32

print("Train rows:", len(df_train), "Test rows:", len(df_test))



## === cell 2


def build_base_model(app_model_fn, preprocess_fn, name):
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name=f"{name}_input")
    x = tf.keras.layers.Lambda(lambda z: preprocess_fn(z), name=f"{name}_preprocess")(
        inp
    )
    backbone = app_model_fn(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    backbone.trainable = False  # inference only
    out = Dense(5, activation="softmax", name=f"{name}_cls")(backbone.output)
    model = Model(inp, out, name=name)
    return model


resnet = build_base_model(
    tf.keras.applications.ResNet50,
    tf.keras.applications.resnet.preprocess_input,
    "resnet50",
)
mobilenet = build_base_model(
    tf.keras.applications.MobileNetV2,
    tf.keras.applications.mobilenet_v2.preprocess_input,
    "mobilenetv2",
)
densenet = build_base_model(
    tf.keras.applications.DenseNet121,
    tf.keras.applications.densenet.preprocess_input,
    "densenet121",
)
efficientnet = build_base_model(
    tf.keras.applications.EfficientNetB0,
    tf.keras.applications.efficientnet.preprocess_input,
    "efficientnetb0",
)

models = [resnet, mobilenet, densenet, efficientnet]

final_model = Sequential(
    [
        Input(shape=(20,)),
        Dense(64, activation="relu"),
        Dense(5, activation="softmax"),
    ],
    name="final_stacker",
)

final_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)



## === cell 3
AUTOTUNE = tf.data.AUTOTUNE


def decode_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    if label is None:
        return img
    return img, label


def make_ds_from_df(df, labeled=True, shuffle=False, batch_size=BATCH_SIZE):
    paths = df["path"].values
    if labeled:
        labels = df["label"].values.astype(np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(len(df), 8192), seed=42, reshuffle_each_iteration=True
            )
        ds = ds.map(lambda p, y: decode_image(p, y), num_parallel_calls=AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: decode_image(p, None), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds




## === cell 4
from sklearn.model_selection import StratifiedKFold

try:
    _ = StratifiedKFold
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


def get_model_preds_for_df(model, df, labeled=False):
    ds = make_ds_from_df(df, labeled=labeled, shuffle=False)
    return model.predict(ds, verbose=1)


def build_stacked_features_from_preds(pred_list):
    return np.concatenate(pred_list, axis=1).astype(np.float32)


NUM_CLASSES = 5
y = df_train["label"].values.astype(np.int32)

if HAVE_SKLEARN:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred_per_model = [
        np.zeros((len(df_train), NUM_CLASSES), dtype=np.float32) for _ in models
    ]

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(df_train)), y), start=1
    ):
        df_va = df_train.iloc[va_idx].reset_index(drop=True)
        print(
            f"\nFold {fold}: generating base predictions for {len(df_va)} validation samples"
        )

        for m_i, m in enumerate(models):
            preds_va = get_model_preds_for_df(m, df_va, labeled=False)
            oof_pred_per_model[m_i][va_idx] = preds_va.astype(np.float32)

    X_stack = build_stacked_features_from_preds(oof_pred_per_model)
    Y_onehot = tf.keras.utils.to_categorical(y, NUM_CLASSES).astype(np.float32)

    print("\nTraining stacker on OOF stacked features:", X_stack.shape, Y_onehot.shape)
    final_model.fit(X_stack, Y_onehot, epochs=20, batch_size=128, verbose=1)
else:
    idx = np.arange(len(df_train))
    np.random.RandomState(42).shuffle(idx)
    cut = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:cut], idx[cut:]

    df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
    df_va = df_train.iloc[va_idx].reset_index(drop=True)

    pred_tr_per_model = []
    pred_va_per_model = []
    for m in models:
        pred_tr_per_model.append(get_model_preds_for_df(m, df_tr, labeled=False))
        pred_va_per_model.append(get_model_preds_for_df(m, df_va, labeled=False))

    X_tr = build_stacked_features_from_preds(pred_tr_per_model)
    X_va = build_stacked_features_from_preds(pred_va_per_model)
    y_tr = tf.keras.utils.to_categorical(df_tr["label"].values, NUM_CLASSES)
    y_va = tf.keras.utils.to_categorical(df_va["label"].values, NUM_CLASSES)

    final_model.fit(
        X_tr, y_tr, validation_data=(X_va, y_va), epochs=20, batch_size=128, verbose=1
    )



## === cell 5
test_pred_per_model = []
for m in models:
    preds_test = get_model_preds_for_df(m, df_test, labeled=False)
    test_pred_per_model.append(preds_test.astype(np.float32))

x_flat = build_stacked_features_from_preds(test_pred_per_model)
print("Test stacked feature shape:", x_flat.shape)

pred_test = final_model.predict(x_flat, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_test_labels
final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")
