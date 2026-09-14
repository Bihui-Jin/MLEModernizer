# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ.setdefault("KERAS_BACKEND", "numpy")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
import numpy as np

import keras
from keras.models import load_model

from PIL import Image



## === cell 1
import glob

candidate_paths = [
    "../input/keras-available-models-part-1-training/model.h5",
    "/kaggle/input/keras-available-models-part-1-training/model.h5",
]

search_roots = [
    "../input",
    "../data",
    "/kaggle/input",
    "/kaggle/data",
]

found_paths = []
for p in candidate_paths:
    if os.path.exists(p):
        found_paths.append(p)

for root in search_roots:
    if os.path.isdir(root):
        found_paths.extend(
            glob.glob(os.path.join(root, "**", "model.h5"), recursive=True)
        )

found_paths = sorted(set(found_paths))
model_path = found_paths[0] if found_paths else None

if model_path is None:
    keras.utils.set_random_seed(42)
    np.random.seed(42)

    img_size = 224
    train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(224, 224, 3)),
            keras.layers.Rescaling(1.0 / 255.0),
            keras.layers.Conv2D(16, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(5, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    df_train = pd.read_csv(train_csv_path)
    idx = np.arange(len(df_train))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
    df_va = df_train.iloc[va_idx].reset_index(drop=True)

    def _load_x_y_from_df(df, max_items=None):
        n = len(df) if max_items is None else min(len(df), max_items)
        x = np.zeros((n, img_size, img_size, 3), dtype=np.float32)
        y = df["label"].to_numpy(dtype=np.int64)[:n]
        for i in range(n):
            fname = df.loc[i, "image_id"]
            path = os.path.join(train_dir, fname)
            with Image.open(path) as im:
                im = im.convert("RGB")
                im = im.resize((img_size, img_size))
                arr = np.asarray(im, dtype=np.float32)
            x[i] = arr
        return x, y

    max_train_items = 6000
    max_val_items = 1000

    x_tr, y_tr = _load_x_y_from_df(df_tr, max_items=max_train_items)
    x_va, y_va = _load_x_y_from_df(df_va, max_items=max_val_items)

    model.fit(
        x_tr,
        y_tr,
        validation_data=(x_va, y_va),
        epochs=3,
        batch_size=32,
        verbose=2,
        shuffle=True,
    )
else:
    model = load_model(model_path)



## --- ERROR in cell 1, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNotImplementedError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2015407217.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     92[0m     [0mx_va[0m[0;34m,[0m [0my_va[0m [0;34m=[0m [0m_load_x_y_from_df[0m[0;34m([0m[0mdf_va[0m[0;34m,[0m [0mmax_items[0m[0;34m=[0m[0mmax_val_items[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     93[0m [0;34m[0m[0m
[0;32m---> 94[0;31m     model.fit(
[0m[1;32m     95[0m         [0mx_tr[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     96[0m         [0my_tr[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/numpy/trainer.py[0m in [0;36mfit[0;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq)[0m
[1;32m    167[0m         [0mvalidation_freq[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    168[0m     ):
[0;32m--> 169[0;31m         [0;32mraise[0m [0mNotImplementedError[0m[0;34m([0m[0;34m"fit not implemented for NumPy backend."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    170[0m [0;34m[0m[0m
[1;32m    171[0m     [0;34m@[0m[0mtraceback_utils[0m[0;34m.[0m[0mfilter_traceback[0m[0;34m[0m[0;34m[0m[0m

[0;31mNotImplementedError[0m: fit not implemented for NumPy backend.

## === cell 2
img_size = 224
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)
test_images = sample_sub["image_id"].tolist()

missing = [f for f in test_images if not os.path.isfile(os.path.join(test_dir, f))]
if len(missing) > 0:
    test_images = [
        f
        for f in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, f))
        and f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
        )
    ]

y_preds = []

for fname in test_images:
    with Image.open(os.path.join(test_dir, fname)) as im:
        im = im.convert("RGB")
        im = im.resize((img_size, img_size))
        x = np.asarray(im, dtype=np.float32)
        x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)
    y_preds.append(int(np.argmax(probs, axis=1)[0]))
