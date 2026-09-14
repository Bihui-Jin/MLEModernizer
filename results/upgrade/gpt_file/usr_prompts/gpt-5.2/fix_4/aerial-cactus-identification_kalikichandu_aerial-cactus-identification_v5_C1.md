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
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

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

0.5055

# 6. Current score

0.99852

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.69578) has done: 'I fix the environment/import crash by switching to `tf_keras` (the Kaggle image has keras 3 + tf_keras 2.18, and your current `keras.preprocessing` import triggers a protobuf/MessageFactory error). I also fix pathing to use the actual extracted dataset folders under `../input/aerial-cactus-identification/`, ensure class_weight/tqdm imports are available, and update callback monitors to valid Keras metric names so training/checkpointing runs. Finally, I generate predictions in the exact `sample_submission.csv` order and output floating probabilities (not `int`), which fixes the “same number of rows” submission error and is metric-correct for AUC.'
- What this solution (achieved 0.99984) has done: 'I fix the import crash by avoiding `tf_keras.preprocessing` (which triggers the protobuf `MessageFactory` error in this environment) and instead use `tf_keras.utils.load_img/img_to_array` plus a small custom augmentation generator that preserves the same augmentation logic. I also keep the model architecture, loss, optimizer, and training loop semantics intact, but make callbacks monitor `val_auc` (AUC-aligned) instead of `val_accuracy` to avoid score drift in the wrong direction. Finally, I ensure the submission is written as a valid `.csv` file with the exact `id,has_cactus` columns in `sample_submission.csv` order. Since your current score (0.69578) is already far above the target (0.5055), these fixes focus on stability and correctness rather than further improvement.'
- What this solution (achieved 0.99852) has done: 'I fix the environment crash in the first cell by importing TensorFlow before `tf_keras` (this prevents the protobuf `MessageFactory.GetPrototype` failure in this Kaggle image). Then I fix the `InvalidArgumentError` during `model.fit()` by making sure the generators yield `float32` arrays with consistent shapes (and by removing the unused/unsafe augmentation helper that can trigger graph slicing issues). Since your current score (0.99984) is far above the target (0.5055), I won’t make any modeling improvements; these changes are score-neutral and focus only on stability and producing a valid `submission.csv` with the correct columns/order.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tensorflow as tf

import tf_keras as keras
from tf_keras import layers
from tf_keras.utils import load_img, img_to_array
from tf_keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight

print("TF version:", tf.__version__)
print("Listing ../input:", os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "../input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

output_dir = os.path.join(BASE_DIR, "model_output", "CNN")
os.makedirs(output_dir, exist_ok=True)

seed = 7
np.random.seed(seed)
tf.random.set_seed(seed)

print("BASE_DIR exists:", os.path.exists(BASE_DIR))
print(
    "TRAIN_DIR exists:",
    os.path.exists(TRAIN_DIR),
    "num_files:",
    len(os.listdir(TRAIN_DIR)) if os.path.exists(TRAIN_DIR) else 0,
)
print(
    "TEST_DIR exists:",
    os.path.exists(TEST_DIR),
    "num_files:",
    len(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else 0,
)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df.head()



## === cell 3
cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"].values,
)
class_weights = {0: float(cw[0]), 1: float(cw[1])}
print("class_weights:", class_weights)



## === cell 4
train_image = []
for i in tqdm(range(len(train_df)), desc="Loading train images"):
    img = load_img(
        os.path.join(TRAIN_DIR, train_df["id"].iloc[i]), target_size=(32, 32)
    )
    img = img_to_array(img).astype("float32") / 255.0
    train_image.append(img)

X = np.array(train_image, dtype="float32")
print("X:", X.shape, X.dtype)



## === cell 5
X.shape



## === cell 6
plt.imshow(X[1])
plt.axis("off")
plt.show()



## === cell 7
y = train_df["has_cactus"].values.astype("float32").reshape(-1, 1)
y.shape



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)



## === cell 9
X_train.shape, X_val.shape, y_train.shape, y_val.shape




## === cell 10
def train_generator_np(X_arr, y_arr, batch_size=32, shuffle=True):
    n = len(X_arr)
    idx = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            bx = X_arr[batch_idx].astype("float32", copy=True)
            by = y_arr[batch_idx].astype("float32", copy=False)

            for j in range(len(bx)):
                if np.random.rand() < 0.5:
                    bx[j] = bx[j, :, ::-1, :]
                if np.random.rand() < 0.5:
                    bx[j] = bx[j, ::-1, :, :]
                if np.random.rand() < 0.5:
                    z = 1.0 + np.random.uniform(-0.1, 0.1)
                    new_size = max(28, min(36, int(round(32 * z))))
                    bx_j = tf.image.resize(
                        bx[j], (new_size, new_size), method="bilinear"
                    ).numpy()
                    bx_j = tf.image.resize_with_crop_or_pad(bx_j, 32, 32).numpy()
                    bx[j] = bx_j.astype("float32", copy=False)
                b = np.random.uniform(0.5, 1.0)
                bx[j] = np.clip(bx[j] * b, 0.0, 1.0)

            yield bx, by


def val_generator_np(X_arr, y_arr, batch_size=32):
    n = len(X_arr)
    idx = np.arange(n)
    while True:
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            yield (
                X_arr[batch_idx].astype("float32", copy=False),
                y_arr[batch_idx].astype("float32", copy=False),
            )


train_generator = train_generator_np(X_train, y_train, batch_size=32, shuffle=True)
validation_generator = val_generator_np(X_val, y_val, batch_size=32)

steps_per_epoch = int(np.ceil(len(X_train) / 32))
val_steps = int(np.ceil(len(X_val) / 32))



## === cell 11
model = keras.Sequential()
model.add(
    layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()



## === cell 12
best_path = os.path.join(output_dir, "weights.best.hdf5")
callbacks = [
    ModelCheckpoint(
        filepath=best_path,
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        mode="auto",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", mode="auto", patience=3, min_lr=0.0001, verbose=1
    ),
]



## === cell 13
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=80,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2,
)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2258794390.py in <cell line: 0>()
      5 )
      6 
----> 7 history = model.fit(
      8     train_generator,
      9     steps_per_epoch=steps_per_epoch,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__StridedSlice_device_/job:localhost/replica:0/task:0/device:GPU:0}} Expected begin and size arguments to be 1-D tensors of size 4, but got shapes [1] and [4] instead. [Op:StridedSlice] name: strided_slice/

## === cell 14
def predict_one(img_path):
    img = load_img(img_path, target_size=(32, 32))
    arr = img_to_array(img).astype("float32") / 255.0
    proba = model.predict(arr.reshape(1, 32, 32, 3), verbose=0)[0, 0]
    return float(proba)




## === cell 15
if os.path.exists(best_path):
    model.load_weights(best_path)
    print("Loaded best weights:", best_path)
else:
    print("Best weights not found; using final epoch weights.")



## === cell 16
y_hat = model.predict(
    X_val.astype("float32", copy=False), batch_size=128, verbose=0
).reshape(-1)
auc = roc_auc_score(y_val.reshape(-1), y_hat)
print("Validation ROC AUC:", auc)



## === cell 17
sub = pd.read_csv(SAMPLE_SUB)
test_ids = sub["id"].tolist()

preds = []
for fname in tqdm(test_ids, desc="Predicting test"):
    preds.append(predict_one(os.path.join(TEST_DIR, fname)))



## === cell 18
pred_df = pd.DataFrame({"id": test_ids, "has_cactus": preds})
pred_df.shape, pred_df.head()



## === cell 19
out_path = "submission.csv"
pred_df.to_csv(out_path, index=False)
print(
    "Wrote submission:", out_path, "rows:", len(pred_df), "cols:", list(pred_df.columns)
)
print(pred_df.head())
