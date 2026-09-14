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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import sys

if "tensorflow" not in sys.modules:
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import glob
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow import keras

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
import os, subprocess, sys

try:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "../input/kerasapplication/Keras_Applications-1.0.8-py3-none-any.whl",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "../input/efficientnet/efficientnet-1.1.1-py3-none-any.whl",
        ]
    )
except Exception:
    pass



## === cell 2
try:
    from efficientnet.tfkeras import EfficientNetB0  # type: ignore
    import efficientnet.tfkeras as efn  # type: ignore

    _HAS_EFN = True
except ModuleNotFoundError:
    from tensorflow.keras.applications import EfficientNetB0

    efn = None
    _HAS_EFN = False

import os
import glob

expected_fname = "model_v0.38.h5"
candidate_paths = []

base_dirs = [
    "../input",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data/kaggle/data/input",
    "/kaggle/data/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]

for base in base_dirs:
    if os.path.isdir(base):
        candidate_paths.extend(
            glob.glob(os.path.join(base, "**", expected_fname), recursive=True)
        )

if not candidate_paths:
    all_h5 = []
    for base in base_dirs:
        if os.path.isdir(base):
            all_h5.extend(glob.glob(os.path.join(base, "**", "*.h5"), recursive=True))
    v038_like = [p for p in all_h5 if "v0.38" in os.path.basename(p).lower()]
    if v038_like:
        candidate_paths = sorted(v038_like)
    else:
        b3_like = [p for p in all_h5 if "b3" in os.path.basename(p).lower()]
        candidate_paths = sorted(b3_like) if b3_like else sorted(all_h5)

if candidate_paths:
    weight_path = candidate_paths[0]

    custom_objects = {}
    if _HAS_EFN:
        custom_objects.update(getattr(efn, "__dict__", {}))
    custom_objects["EfficientNetB0"] = EfficientNetB0

    my_model = load_model(weight_path, custom_objects=custom_objects)
else:
    weight_path = None
    my_model = EfficientNetB0(weights=None, include_top=True, classes=5)

try:
    _h, _w = my_model.input_shape[1], my_model.input_shape[2]
    _target_size = (
        (int(_h), int(_w)) if _h is not None and _w is not None else (224, 224)
    )
except Exception:
    _target_size = (224, 224)



## === cell 3
import os

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
test_dir = "../input/cassava-leaf-disease-classification/test_images"
sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"

sample_sub = pd.read_csv(sample_path)
df_test = sample_sub[["image_id"]].copy()
df_test["path"] = df_test["image_id"].apply(lambda x: os.path.join(test_dir, x))

missing = df_test.loc[~df_test["path"].apply(os.path.exists), "image_id"].tolist()
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images (first 5): {missing[:5]}"
    )

df_train = pd.read_csv(train_csv_path)
train_dir = "../input/cassava-leaf-disease-classification/train_images"
df_train = df_train[["image_id", "label"]].copy()
df_train["path"] = df_train["image_id"].apply(lambda x: os.path.join(train_dir, x))

df_tr, df_va = train_test_split(
    df_train,
    test_size=0.15,
    random_state=SEED,
    stratify=df_train["label"],
)

max_val = 1024
if len(df_va) > max_val:
    df_va = (
        df_va.groupby("label", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=max(1, int(max_val * len(x) / len(df_va))), random_state=SEED
            )
        )
        .reset_index(drop=True)
    )


def _get_effnet_preprocess():
    if _HAS_EFN and efn is not None and hasattr(efn, "preprocess_input"):
        return efn.preprocess_input
    try:
        from tensorflow.keras.applications.efficientnet import (
            preprocess_input as tf_eff_pre,
        )

        return tf_eff_pre
    except Exception:
        return None


_eff_pre = _get_effnet_preprocess()


def make_gen(
    dataframe,
    batch_size=64,
    target_size=_target_size,
    preprocess_mode="rescale_1_255",
    shuffle=False,
):
    if preprocess_mode == "effnet" and _eff_pre is not None:
        idg = ImageDataGenerator(preprocessing_function=_eff_pre)
    else:
        idg = ImageDataGenerator(rescale=1.0 / 255.0)

    has_label = "label" in dataframe.columns
    gen = idg.flow_from_dataframe(
        dataframe=dataframe,
        x_col="path",
        y_col=("label" if has_label else None),
        batch_size=batch_size,
        seed=SEED,
        shuffle=shuffle,
        class_mode=("sparse" if has_label else None),
        target_size=target_size,
    )
    return gen


def eval_preprocess(preprocess_mode):
    va_gen = make_gen(
        df_va, batch_size=64, preprocess_mode=preprocess_mode, shuffle=False
    )
    pred = my_model.predict(va_gen, verbose=0, steps=len(va_gen))
    pred_labels = np.argmax(pred, axis=-1).astype(int)

    y_true = va_gen.classes.astype(int)
    acc = float(np.mean(pred_labels[: len(y_true)] == y_true))
    return acc


acc_rescale = eval_preprocess("rescale_1_255")
acc_effnet = eval_preprocess("effnet") if _eff_pre is not None else -1.0

best_mode = "rescale_1_255"
best_acc = acc_rescale
if acc_effnet > best_acc:
    best_mode = "effnet"
    best_acc = acc_effnet

print(f"Validation acc (rescale_1/255): {acc_rescale:.4f}")
if _eff_pre is not None:
    print(f"Validation acc (effnet preprocess): {acc_effnet:.4f}")
print(f"Chosen preprocess_mode: {best_mode} (val acc {best_acc:.4f})")

test_gen = make_gen(df_test, batch_size=128, preprocess_mode=best_mode, shuffle=False)
pred_test = my_model.predict(test_gen, verbose=1, steps=len(test_gen))
pred_test_labels = np.argmax(pred_test, axis=-1)

final_submission = df_test[["image_id"]].copy()
final_submission["label"] = pred_test_labels.astype(int)

final_csv = final_submission[["image_id", "label"]]
final_csv.to_csv("submission.csv", index=False)
final_csv.head()



## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1911070913.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     99[0m [0;34m[0m[0m
[1;32m    100[0m [0;34m[0m[0m
[0;32m--> 101[0;31m [0macc_rescale[0m [0;34m=[0m [0meval_preprocess[0m[0;34m([0m[0;34m"rescale_1_255"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    102[0m [0macc_effnet[0m [0;34m=[0m [0meval_preprocess[0m[0;34m([0m[0;34m"effnet"[0m[0;34m)[0m [0;32mif[0m [0m_eff_pre[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0;34m-[0m[0;36m1.0[0m[0;34m[0m[0;34m[0m[0m
[1;32m    103[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1911070913.py[0m in [0;36meval_preprocess[0;34m(preprocess_mode)[0m
[1;32m     86[0m [0;34m[0m[0m
[1;32m     87[0m [0;32mdef[0m [0meval_preprocess[0m[0;34m([0m[0mpreprocess_mode[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 88[0;31m     va_gen = make_gen(
[0m[1;32m     89[0m         [0mdf_va[0m[0;34m,[0m [0mbatch_size[0m[0;34m=[0m[0;36m64[0m[0;34m,[0m [0mpreprocess_mode[0m[0;34m=[0m[0mpreprocess_mode[0m[0;34m,[0m [0mshuffle[0m[0;34m=[0m[0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m     90[0m     )

[0;32m/tmp/ipykernel_11/1911070913.py[0m in [0;36mmake_gen[0;34m(dataframe, batch_size, target_size, preprocess_mode, shuffle)[0m
[1;32m     72[0m [0;34m[0m[0m
[1;32m     73[0m     [0mhas_label[0m [0;34m=[0m [0;34m"label"[0m [0;32min[0m [0mdataframe[0m[0;34m.[0m[0mcolumns[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 74[0;31m     gen = idg.flow_from_dataframe(
[0m[1;32m     75[0m         [0mdataframe[0m[0;34m=[0m[0mdataframe[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     76[0m         [0mx_col[0m[0;34m=[0m[0;34m"path"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py[0m in [0;36mflow_from_dataframe[0;34m(self, dataframe, directory, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset, interpolation, validate_filenames, **kwargs)[0m
[1;32m   1206[0m             )
[1;32m   1207[0m [0;34m[0m[0m
[0;32m-> 1208[0;31m         return DataFrameIterator(
[0m[1;32m   1209[0m             [0mdataframe[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1210[0m             [0mdirectory[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py[0m in [0;36m__init__[0;34m(self, dataframe, directory, image_data_generator, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, subset, interpolation, keep_aspect_ratio, dtype, validate_filenames)[0m
[1;32m    749[0m         [0mself[0m[0;34m.[0m[0mdtype[0m [0;34m=[0m [0mdtype[0m[0;34m[0m[0;34m[0m[0m
[1;32m    750[0m         [0;31m# check that inputs match the required class_mode[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 751[0;31m         [0mself[0m[0;34m.[0m[0m_check_params[0m[0;34m([0m[0mdf[0m[0;34m,[0m [0mx_col[0m[0;34m,[0m [0my_col[0m[0;34m,[0m [0mweight_col[0m[0;34m,[0m [0mclasses[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    752[0m         if (
[1;32m    753[0m             [0mvalidate_filenames[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py[0m in [0;36m_check_params[0;34m(self, df, x_col, y_col, weight_col, classes)[0m
[1;32m    817[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mclass_mode[0m [0;32min[0m [0;34m{[0m[0;34m"binary"[0m[0;34m,[0m [0;34m"sparse"[0m[0;34m}[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    818[0m             [0;32mif[0m [0;32mnot[0m [0mall[0m[0;34m([0m[0mdf[0m[0;34m[[0m[0my_col[0m[0;34m][0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0misinstance[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mstr[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 819[0;31m                 raise TypeError(
[0m[1;32m    820[0m                     [0;34m'If class_mode="{}", y_col="{}" column '[0m[0;34m[0m[0;34m[0m[0m
[1;32m    821[0m                     [0;34m"values must be strings."[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mclass_mode[0m[0;34m,[0m [0my_col[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: If class_mode="sparse", y_col="label" column values must be strings.

## === cell 4
final_csv.head()
