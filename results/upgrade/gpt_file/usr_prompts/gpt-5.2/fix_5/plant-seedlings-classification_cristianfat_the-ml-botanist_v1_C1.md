# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.62468

# 6. Current score

0.83183

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.83183) has done: 'The timeout is dominated by per-image feature extraction calling `model.predict()` ~5k times (each with its own graph/session overhead) and by storing large feature arrays inside pandas objects. I keep the exact same InceptionV3(avg_pool) feature representation and the exact same XGBoost training logic, but batch the CNN forward passes using a `tf.data` pipeline (same preprocessing) to amortize overhead and fully utilize vectorization. I also avoid plotting cells (they cost time but don’t affect outputs) and build NumPy feature matrices directly (no per-row object arrays), while keeping determinism and all model hyperparameters identical.'
- What this solution (achieved 0.81532) has done: 'I fix the TensorFlow import crash by switching from `tf_keras` to `tensorflow.keras`, which avoids the protobuf `MessageFactory.GetPrototype` incompatibility in this environment while keeping the exact same InceptionV3(avg_pool) feature extraction. I keep the batching `tf.data` pipeline and the XGBoost training/prediction logic unchanged so the score should remain essentially the same (and still within the target ±10% band). I also remove the no-op “cell 1 pass” and renumber cells to start at 1, but won’t change any model hyperparameters or preprocessing. The script run end-to-end and write a valid `submission.csv` with the required `file,species` columns.'
- What this solution (achieved 0.83183) has done: 'The crash happens at TensorFlow import time due to a protobuf incompatibility in this Kaggle image; switching to the installed `tf_keras` package avoids that while keeping the exact same InceptionV3(avg_pool) feature extractor, preprocessing, batching, and XGBoost logic (so score behavior should remain essentially unchanged). I also add a safe CPU-only fallback if GPU init fails, and make the `decode_image` output shape explicit to prevent rare shape-related runtime errors during batching. Finally, I keep the submission formatting checks and ensure `submission.csv` is always written with the required `file,species` columns.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
random.seed(42)
np.random.seed(42)

p = "/kaggle/input/plant-seedlings-classification"


def df_of_images(folder_name, path="/kaggle/input/plant-seedlings-classification"):
    itms = []
    folder_path = os.path.join(path, folder_name)
    for cls in sorted(os.listdir(folder_path)):
        cls_path = os.path.join(folder_path, cls)
        if not os.path.isdir(cls_path):
            continue
        for img in sorted(os.listdir(cls_path)):
            img_path = os.path.join(cls_path, img)
            if os.path.isfile(img_path) and img.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                itms.append(
                    {
                        "label": cls.lower()
                        .strip()
                        .replace(" ", "_")
                        .replace("-", "_"),
                        "image_path": img_path,
                    }
                )
    return pd.DataFrame(itms)


train = df_of_images("train")

test_dir = os.path.join(p, "test")
test_files = sorted(
    [
        fn
        for fn in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, fn))
        and fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)
test = pd.DataFrame({"image_path": [os.path.join(test_dir, fn) for fn in test_files]})

print("train:", train.shape, "test:", test.shape)
print("train labels:", train["label"].nunique())



## === cell 1
import tensorflow as tf
import tf_keras as keras
from tf_keras.applications.inception_v3 import InceptionV3, preprocess_input
from tf_keras.models import Model

tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
base_model = InceptionV3(weights="imagenet", include_top=True)
feat_model = Model(
    inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output
)




## === cell 3
def _decode_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [299, 299], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # identical preprocessing intent
    return img


def extract_features_batched(image_paths, model, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(image_paths, dtype=tf.string)
    )
    ds = ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    feats = model.predict(ds, verbose=0)
    return feats  # shape: (N, 2048)




## === cell 4
train_paths = train["image_path"].tolist()
test_paths = test["image_path"].tolist()

X_train_cnn = extract_features_batched(train_paths, feat_model, batch_size=64)
X_test_cnn = extract_features_batched(test_paths, feat_model, batch_size=64)

print("Feature vector length:", X_train_cnn.shape[1])



## === cell 5
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder



## === cell 6
train_idx, valid_idx = train_test_split(
    np.arange(len(train)),
    test_size=0.33,
    random_state=42,
    stratify=train["label"].values,
)

train_ = train.iloc[train_idx].reset_index(drop=True)
test_ = train.iloc[valid_idx].reset_index(drop=True)

print(
    "train label distribution head:\n",
    (train_["label"].value_counts() / len(train_)).head(),
    "\nvalid label distribution head:\n",
    (test_["label"].value_counts() / len(test_)).head(),
)



## === cell 7
le = LabelEncoder()
le.fit(train["label"])

X_train = X_train_cnn[train_idx]
y_train = le.transform(train.iloc[train_idx]["label"].values)

X_valid = X_train_cnn[valid_idx]
y_valid = le.transform(train.iloc[valid_idx]["label"].values)

xgc = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=train.label.nunique(),
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
)
xgc.fit(X_train, y_train)



## === cell 8
results = test_.copy()
valid_pred = xgc.predict(X_valid)
results["y_pred"] = le.inverse_transform(valid_pred)

print(metrics.classification_report(results.label, results.y_pred))



## === cell 9
X_full = X_train_cnn
y_full = le.transform(train["label"].values)

xgc_full = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=train.label.nunique(),
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
)
xgc_full.fit(X_full, y_full)

X_test = X_test_cnn
test_pred = xgc_full.predict(X_test)
test["species_norm"] = le.inverse_transform(test_pred)

label_map = {
    x.lower().strip().replace(" ", "_").replace("-", "_"): x
    for x in os.listdir(os.path.join(p, "train"))
}

results_sub = pd.DataFrame()
results_sub["file"] = test["image_path"].apply(lambda x: os.path.basename(x))
results_sub["species"] = test["species_norm"].replace(label_map)

sample_path = os.path.join(p, "sample_submission.csv")
sample = pd.read_csv(sample_path)
assert list(sample.columns) == [
    "file",
    "species",
], "Unexpected sample_submission columns"
assert len(results_sub) == len(
    sample
), f"Row mismatch vs sample_submission: {len(results_sub)} vs {len(sample)}"

results_sub.to_csv("submission.csv", index=False)
print(results_sub.head())
print("Wrote submission.csv with shape:", results_sub.shape)
