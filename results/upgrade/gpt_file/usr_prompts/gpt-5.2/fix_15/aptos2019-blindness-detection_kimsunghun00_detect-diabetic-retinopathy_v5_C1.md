# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import time
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

pd.set_option("display.max_rows", 10)

np.random.seed(123)
tf.random.set_seed(123)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    _cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _cpu))
    tf.config.threading.set_inter_op_parallelism_threads(min(8, _cpu))
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
base_data_folder = "/kaggle/input/aptos2019-blindness-detection"
train_data_folder = os.path.join(base_data_folder, "train_images")

print("Base folder exists:", os.path.exists(base_data_folder))
print("Train images folder exists:", os.path.exists(train_data_folder))
print("Base folder listing (first 20):", sorted(os.listdir(base_data_folder))[:20])




## === cell 2
train_files_names = sorted(os.listdir(train_data_folder))
train_files_names[:5]




## === cell 3
train_df = pd.read_csv(os.path.join(base_data_folder, "train.csv"))
train_df.head()




## === cell 4
train_ids = train_df["id_code"].values
train_files_set = set(train_files_names)
exists_mask = np.fromiter(
    (f"{id_code}.png" in train_files_set for id_code in train_ids),
    dtype=bool,
    count=len(train_ids),
)
missing = int((~exists_mask).sum())
if missing == 0:
    labels = train_df.copy()
else:
    labels = train_df.loc[exists_mask].reset_index(drop=True)

y_data = labels["diagnosis"].astype(int)
print("Missing train images:", missing)
labels.head()




## === cell 5
labels["diagnosis"].hist()
print(labels["diagnosis"].value_counts())




## === cell 6
def crop_image_from_gray(img, tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img > tol
    if not mask.any():
        return img
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    return img[np.ix_(rows, cols)]




## === cell 7
def circle_crop(img):
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(
        img, dsize=(largest_side, largest_side), interpolation=cv2.INTER_CUBIC
    )

    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    background = np.zeros(shape=(height, width), dtype=np.uint8)
    cv2.circle(background, (x, y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=background)
    return img




## === cell 8
pic_num = 43
img = None
img_rgb = None
if False and len(labels) > pic_num:
    fp = os.path.join(train_data_folder, f"{labels['id_code'].iloc[pic_num]}.png")
    img = cv2.imread(fp, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.resize(img, dsize=(0, 0), fx=0.12, fy=0.12)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




## === cell 9
img1 = None
if img_rgb is not None:
    img1 = crop_image_from_gray(img_rgb)




## === cell 10
img2 = None
if img_rgb is not None:
    img2 = circle_crop(img_rgb)




## === cell 11
def _preprocess_one_image_from_decoded_rgb(img_rgb_uint8):
    image_small = cv2.resize(img_rgb_uint8, dsize=(0, 0), fx=0.12, fy=0.12)
    circle_img = circle_crop(image_small)
    image_224 = cv2.resize(circle_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return image_224.astype(np.uint8, copy=False)


def _tf_preprocess(path, label_onehot=None):
    def _py_fn(path_tensor):
        p = path_tensor.numpy().decode("utf-8")
        data = np.frombuffer(tf.io.read_file(p).numpy(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _preprocess_one_image_from_decoded_rgb(rgb)

    img = tf.py_function(func=_py_fn, inp=[path], Tout=tf.uint8)
    img.set_shape((224, 224, 3))
    if label_onehot is None:
        return img
    return img, label_onehot


train_ids_kept = labels["id_code"].values
train_paths = np.array(
    [os.path.join(train_data_folder, f"{id_code}.png") for id_code in train_ids_kept],
    dtype=object,
)
print("Num train paths:", len(train_paths))




## === cell 12
_CIRCLE_MASK_224 = np.zeros((224, 224), dtype=np.uint8)
_CIRCLE_MASK_224 = cv2.circle(
    _CIRCLE_MASK_224, (112, 112), 110, 1, thickness=-1
).astype(bool)


def values_in_mask(X):
    dim1 = X[:, :, 0]
    dim2 = X[:, :, 1]
    dim3 = X[:, :, 2]

    circle_locations = _CIRCLE_MASK_224
    R = dim1[circle_locations]
    G = dim2[circle_locations]
    B = dim3[circle_locations]
    return R, G, B




## === cell 13
def min_max_scaler_rgb(X):
    R, G, B = values_in_mask(X)

    dim1 = X[:, :, 0].astype("float32")
    dim2 = X[:, :, 1].astype("float32")
    dim3 = X[:, :, 2].astype("float32")

    min_R, min_G, min_B = np.min(R), np.min(G), np.min(B)
    max_R, max_G, max_B = np.max(R), np.max(G), np.max(B)

    denom_R = (max_R - min_R) if (max_R - min_R) != 0 else 1.0
    denom_G = (max_G - min_G) if (max_G - min_G) != 0 else 1.0
    denom_B = (max_B - min_B) if (max_B - min_B) != 0 else 1.0

    img_R = (dim1 - min_R) / denom_R
    img_G = (dim2 - min_G) / denom_G
    img_B = (dim3 - min_B) / denom_B

    img_R = np.clip(img_R, 0.0, 1.0)
    img_G = np.clip(img_G, 0.0, 1.0)
    img_B = np.clip(img_B, 0.0, 1.0)

    img = np.stack([img_R, img_G, img_B], axis=-1)
    return img




## === cell 14
def min_max_scaler_gray(X):
    denom = np.max(X) - np.min(X)
    denom = denom if denom != 0 else 1.0
    img = (X - np.min(X)) / denom
    return img




## === cell 15
pass




## === cell 16
pass




## === cell 17
pass




## === cell 18
from sklearn.model_selection import train_test_split

train_paths_train, train_paths_valid, y_train, y_valid = train_test_split(
    train_paths, y_data, test_size=0.2, stratify=y_data, random_state=123
)

print(train_paths_train.shape, y_train.shape)
print(train_paths_valid.shape, y_valid.shape)




## === cell 19
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_valid_onehot = tf.keras.utils.to_categorical(y_valid, num_classes=5)

print(y_train_onehot.shape)
print(y_valid_onehot.shape)




## === cell 20
pass




## === cell 21
X_resampled = train_paths_train
y_resampled = y_train
y_resampled_onehot = y_train_onehot

print("Train paths:", X_resampled.shape, "y_onehot:", y_resampled_onehot.shape)




## === cell 22
pass




## === cell 23
pass




## === cell 24
model = models.Sequential()




## === cell 25
model.add(
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        input_shape=(224, 224, 3),
        name="Conv1-1",
    )
)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool1"))




## === cell 26
model.add(
    layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv2-1",
    )
)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool2"))




## === cell 27
model.add(
    layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv3-1",
    )
)
model.add(
    layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv3-2",
    )
)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool3"))




## === cell 28
model.add(
    layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv4-1",
    )
)
model.add(
    layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv4-2",
    )
)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool4"))




## === cell 29
model.add(
    layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv5-1",
    )
)
model.add(
    layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv5-2",
    )
)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool5"))




## === cell 30
model.add(layers.Flatten())




## === cell 31
model.add(layers.Dense(256, activation="relu", name="Dense1"))
model.add(layers.Dropout(0.3))




## === cell 32
model.add(layers.Dense(256, activation="relu", name="Dense2"))
model.add(layers.Dropout(0.3))




## === cell 33
model.add(layers.Dense(5, activation="softmax", name="Final"))




## === cell 34
model.summary()




## === cell 35
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])




## === cell 36
checkpoint_path = "cnn_checkpoint.weights.h5"

callback_list = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=0,
    )
]




## === cell 37
batch_size = 200

_ds_opts = tf.data.Options()
_ds_opts.experimental_deterministic = True
try:
    _ds_opts.experimental_optimization.map_parallelization = True
    _ds_opts.experimental_optimization.parallel_batch = True
except Exception:
    pass


from concurrent.futures import ThreadPoolExecutor


def _paths_fingerprint(paths: np.ndarray) -> str:
    h = hashlib.md5()
    for p in paths.tolist():
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _cache_path(prefix: str, paths: np.ndarray) -> str:
    return os.path.join("/kaggle/working", f"{prefix}_{_paths_fingerprint(paths)}.npy")


def _read_and_preprocess_one(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _preprocess_one_image_from_decoded_rgb(rgb)


def _preprocess_paths_to_array(paths, desc, max_workers=None):
    cache_fp = _cache_path(
        desc.replace(" ", "_").lower(), np.asarray(paths, dtype=object)
    )
    if os.path.exists(cache_fp):
        arr = np.load(cache_fp, mmap_mode=None)
        return arr

    n = len(paths)
    X = np.empty((n, 224, 224, 3), dtype=np.uint8)

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 4)

    def _task(i_p):
        i, p = i_p
        return i, _read_and_preprocess_one(p)

    t_iter = enumerate(paths)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, img in tqdm(ex.map(_task, t_iter), total=n, desc=desc):
            X[i] = img

    np.save(cache_fp, X)
    return X


t0 = time.time()
X_train_arr = _preprocess_paths_to_array(train_paths_train, desc="Preprocess train")
X_valid_arr = _preprocess_paths_to_array(train_paths_valid, desc="Preprocess valid")
print("Preprocessing time (s):", round(time.time() - t0, 2))
print("X_train_arr:", X_train_arr.shape, X_train_arr.dtype)
print("X_valid_arr:", X_valid_arr.shape, X_valid_arr.dtype)

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train_arr, y_resampled_onehot))
    .with_options(_ds_opts)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((X_valid_arr, y_valid_onehot))
    .with_options(_ds_opts)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

history = model.fit(
    train_ds,
    epochs=100,
    validation_data=valid_ds,
    callbacks=callback_list,
    verbose=2,
)




## === cell 38
pass




## === cell 39
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
else:
    print(f"Warning: {checkpoint_path} not found; using last-epoch in-memory weights")

restored_model = model




## === cell 40
restored_model.evaluate(valid_ds, verbose=0)




## === cell 41
y_pred = np.argmax(restored_model.predict(valid_ds, verbose=0), axis=1)

print("Predict:", y_pred[:10])
print("Validation:", np.array(y_valid[:10]))




## === cell 42
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=y_valid, y_pred=y_pred)

print("Confusion Matrix")
print(cm)
print()
print("Shape :", cm.shape)
print("Accurcy: {0:.2f}%".format(np.trace(cm) / np.sum(cm) * 100))




## === cell 43
from sklearn.metrics import classification_report

print(
    classification_report(
        y_valid,
        y_pred,
        digits=4,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
    )
)




## === cell 44
test_data_folder = os.path.join(base_data_folder, "test_images")
print("Test images folder exists:", os.path.exists(test_data_folder))




## === cell 45
test_df = pd.read_csv(os.path.join(base_data_folder, "test.csv"))
test_df.head()




## === cell 46
test_ids = test_df["id_code"].values
test_paths = np.array(
    [os.path.join(test_data_folder, f"{id_code}.png") for id_code in test_ids],
    dtype=object,
)




## === cell 47
t0 = time.time()
X_test_arr = _preprocess_paths_to_array(test_paths, desc="Preprocess test")
print("Test preprocessing time (s):", round(time.time() - t0, 2))
test_ds = (
    tf.data.Dataset.from_tensor_slices(X_test_arr)
    .with_options(_ds_opts)
    .batch(256, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
preds = np.argmax(restored_model.predict(test_ds, verbose=0), axis=1)
print("Predicted (first 10):", preds[:10])




## === cell 48
submission = pd.DataFrame({"id_code": test_ids, "diagnosis": preds.astype(int)})
submission.head()




## === cell 49
submission = submission.sort_values("id_code").reset_index(drop=True)
submission.head()




## === cell 50
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())




## === cell 51
sample_sub = pd.read_csv(os.path.join(base_data_folder, "sample_submission.csv"))
print("sample_submission shape:", sample_sub.shape)
print(
    "id_code match (as sets):", set(sample_sub["id_code"]) == set(submission["id_code"])
)
print(
    "id_code match (sorted equality):",
    sample_sub.sort_values("id_code")["id_code"].values.tolist()
    == submission["id_code"].values.tolist(),
)
