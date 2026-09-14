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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
imutils==0.5.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9996491666666668

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("Listing ../input:")
print(os.listdir("../input"))



## === cell 1
DATA_ROOT = "../input/aerial-cactus-identification"
print("Listing dataset root:", DATA_ROOT)
print(os.listdir(DATA_ROOT))

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
train_dir = os.path.join(DATA_ROOT, "train", "train")
test_dir = os.path.join(DATA_ROOT, "test", "test")

print("train_csv_path exists:", os.path.exists(train_csv_path))
print("sample_sub_path exists:", os.path.exists(sample_sub_path))
print("train_dir exists:", os.path.exists(train_dir))
print("test_dir exists:", os.path.exists(test_dir))



## === cell 2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(SEED)
print("TF version:", tf.__version__)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def focal_loss_fn(gamma=2.0, alpha=0.25):
    EPSILON = 1e-6

    def ce(y_true, y_pred, weights=None):
        mask = y_pred < EPSILON
        true_vals = tf.fill(tf.shape(y_pred), EPSILON)
        y_pred = tf.where(mask, true_vals, y_pred)
        ce_val = y_true * (-tf.math.log(y_pred)) + (1 - y_true) * (
            -tf.math.log(1 - y_pred)
        )

        if weights is not None:
            ce_val = ce_val * weights

        ce_loss = tf.reduce_mean(ce_val + EPSILON)
        return ce_loss

    def focal_loss_fixed(y_true, y_pred):
        t = y_true
        p = y_pred

        pt = p * t + (1 - p) * (1 - t)
        w = alpha * t + (1 - alpha) * (1 - t)
        w = tf.pow((1 - pt), gamma)

        fl = ce(y_true, y_pred, w)
        return fl

    return focal_loss_fixed




## === cell 4
train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

assert set(train_df.columns) == {"id", "has_cactus"}
assert set(sample_sub.columns) == {"id", "has_cactus"}

print("Train rows:", len(train_df), "Sample submission rows:", len(sample_sub))
print(train_df["has_cactus"].value_counts().to_dict())



## === cell 5
from imutils import paths
import cv2



## === cell 6
test_imgs_dir = test_dir
train_imgs_dir = train_dir

test_img_paths = list(paths.list_images(test_imgs_dir))
train_img_paths = list(paths.list_images(train_imgs_dir))

print("Num train images found:", len(train_img_paths))
print("Num test images found:", len(test_img_paths))
assert len(train_img_paths) == len(
    train_df
), "Train images count should match train.csv rows."
assert len(test_img_paths) == len(
    sample_sub
), "Test images count should match sample_submission rows."



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/1359654213.py in <cell line: 0>()
      8 print("Num train images found:", len(train_img_paths))
      9 print("Num test images found:", len(test_img_paths))
---> 10 assert len(train_img_paths) == len(
     11     train_df
     12 ), "Train images count should match train.csv rows."

AssertionError: Train images count should match train.csv rows.

## === cell 7
train_path_map = {os.path.basename(p): p for p in train_img_paths}
test_path_map = {os.path.basename(p): p for p in test_img_paths}

missing_train = set(train_df["id"]) - set(train_path_map.keys())
missing_test = set(sample_sub["id"]) - set(test_path_map.keys())
print("Missing train ids:", len(missing_train))
print("Missing test ids:", len(missing_test))
assert len(missing_train) == 0
assert len(missing_test) == 0



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/80136741.py in <cell line: 0>()
      8 print("Missing train ids:", len(missing_train))
      9 print("Missing test ids:", len(missing_test))
---> 10 assert len(missing_train) == 0
     11 assert len(missing_test) == 0
     12 

AssertionError: 

## === cell 8
img_size = 96


def load_images_from_ids(ids, path_map, img_size):
    X = np.empty((len(ids), img_size, img_size, 3), dtype=np.float32)
    for i, img_id in enumerate(ids):
        img_path = path_map[img_id]
        img = cv2.imread(img_path)  # BGR uint8
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        X[i] = img.astype(np.float32) / 255.0
    return X


X = load_images_from_ids(train_df["id"].values, train_path_map, img_size)
y = train_df["has_cactus"].values.astype(np.float32)

print(
    "X shape:",
    X.shape,
    "y shape:",
    y.shape,
    "X min/max:",
    float(X.min()),
    float(X.max()),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/481840576.py in <cell line: 0>()
     16 
     17 
---> 18 X = load_images_from_ids(train_df["id"].values, train_path_map, img_size)
     19 y = train_df["has_cactus"].values.astype(np.float32)
     20 

/tmp/ipykernel_55/481840576.py in load_images_from_ids(ids, path_map, img_size)
      6     X = np.empty((len(ids), img_size, img_size, 3), dtype=np.float32)
      7     for i, img_id in enumerate(ids):
----> 8         img_path = path_map[img_id]
      9         img = cv2.imread(img_path)  # BGR uint8
     10         if img is None:

KeyError: '2de8f189f1dce439766637e75df0ee27.jpg'

## === cell 9
model = keras.Sequential(
    [
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)
model.summary()



## === cell 10
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)

history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=8, batch_size=64, verbose=2
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/548457502.py in <cell line: 0>()
      3 
      4 X_train, X_val, y_train, y_val = train_test_split(
----> 5     X, y, test_size=0.15, random_state=SEED, stratify=y
      6 )
      7 

NameError: name 'X' is not defined

## === cell 11
X_test = load_images_from_ids(sample_sub["id"].values, test_path_map, img_size)
predictions = model.predict(X_test, batch_size=128, verbose=1).reshape(-1)

print(
    "Pred shape:",
    predictions.shape,
    "min/max:",
    float(predictions.min()),
    float(predictions.max()),
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1478936808.py in <cell line: 0>()
      1 # Predict on test in the exact order of sample_submission to ensure proper row alignment.
----> 2 X_test = load_images_from_ids(sample_sub["id"].values, test_path_map, img_size)
      3 predictions = model.predict(X_test, batch_size=128, verbose=1).reshape(-1)
      4 
      5 print(

/tmp/ipykernel_55/481840576.py in load_images_from_ids(ids, path_map, img_size)
      6     X = np.empty((len(ids), img_size, img_size, 3), dtype=np.float32)
      7     for i, img_id in enumerate(ids):
----> 8         img_path = path_map[img_id]
      9         img = cv2.imread(img_path)  # BGR uint8
     10         if img is None:

KeyError: '09034a34de0e2015a8a28dfe18f423f6.jpg'

## === cell 12
submit_csv = pd.DataFrame(
    {"id": sample_sub["id"].values, "has_cactus": predictions.astype(np.float64)}
)
submit_csv.to_csv("submission_resnet50.csv", index=False)

print(submit_csv.head())
print("Wrote submission_resnet50.csv with rows:", len(submit_csv))

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/666084351.py in <cell line: 0>()
      1 # Build submission and write to CSV with required columns and .csv suffix.
      2 submit_csv = pd.DataFrame(
----> 3     {"id": sample_sub["id"].values, "has_cactus": predictions.astype(np.float64)}
      4 )
      5 submit_csv.to_csv("submission_resnet50.csv", index=False)

NameError: name 'predictions' is not defined
