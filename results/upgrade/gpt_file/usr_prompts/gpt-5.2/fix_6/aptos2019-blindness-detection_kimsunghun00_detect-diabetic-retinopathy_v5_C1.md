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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import time
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
exists_mask = np.fromiter(
    (
        os.path.exists(os.path.join(train_data_folder, f"{id_code}.png"))
        for id_code in train_ids
    ),
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

    img1 = img[:, :, 0][np.ix_(mask.any(axis=1), mask.any(axis=0))]
    img2 = img[:, :, 1][np.ix_(mask.any(axis=1), mask.any(axis=0))]
    img3 = img[:, :, 2][np.ix_(mask.any(axis=1), mask.any(axis=0))]
    img = np.stack([img1, img2, img3], axis=-1)
    return img




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
if len(labels) > pic_num:
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
def _preprocess_one_image_cv2(path_bytes):
    if hasattr(path_bytes, "numpy"):
        path = path_bytes.numpy().decode("utf-8")
    else:
        path = path_bytes.decode("utf-8")

    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        out = np.zeros((224, 224, 3), dtype=np.uint8)
        return out

    image_small = cv2.resize(image_bgr, dsize=(0, 0), fx=0.12, fy=0.12)
    img_rgb_local = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
    circle_img = circle_crop(img_rgb_local)
    image_224 = cv2.resize(circle_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return image_224.astype(np.uint8)


def _tf_preprocess(path, label_onehot=None):
    img = tf.py_function(func=_preprocess_one_image_cv2, inp=[path], Tout=tf.uint8)
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
        verbose=1,
    )
]



## === cell 37
from concurrent.futures import ThreadPoolExecutor, as_completed

cache_root = "/kaggle/working/aptos_preprocessed_224"
train_cache_dir = os.path.join(cache_root, "train")
test_cache_dir = os.path.join(cache_root, "test")
os.makedirs(train_cache_dir, exist_ok=True)
os.makedirs(test_cache_dir, exist_ok=True)


def _cached_path(src_path, split_dir):
    base = os.path.splitext(os.path.basename(src_path))[0]
    return os.path.join(split_dir, f"{base}.png")


def _precompute_one(src_path, out_path):
    if os.path.exists(out_path):
        return True
    img224 = _preprocess_one_image_cv2(src_path.encode("utf-8"))
    img_bgr = cv2.cvtColor(img224, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(out_path, img_bgr)
    return bool(ok)


def _precompute_many(src_paths, out_dir, max_workers):
    t0 = time.time()
    out_paths = np.array([_cached_path(p, out_dir) for p in src_paths], dtype=object)
    missing = [i for i, op in enumerate(out_paths) if not os.path.exists(op)]
    if len(missing) == 0:
        print(f"Precompute: 0 files needed in {out_dir} (already cached).")
        return out_paths

    print(
        f"Precompute: generating {len(missing)}/{len(out_paths)} files in {out_dir} ..."
    )
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_precompute_one, src_paths[i], out_paths[i]) for i in missing
        ]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
    print(f"Precompute done in {time.time()-t0:.1f}s for dir {out_dir}")
    return out_paths


max_workers = min(8, os.cpu_count() or 4)
train_cached_paths_all = _precompute_many(
    train_paths, train_cache_dir, max_workers=max_workers
)

train_cached_map = {
    os.path.splitext(os.path.basename(p))[0]: _cached_path(p, train_cache_dir)
    for p in train_paths
}
train_paths_train_cached = np.array(
    [
        train_cached_map[os.path.splitext(os.path.basename(p))[0]]
        for p in train_paths_train
    ],
    dtype=object,
)
train_paths_valid_cached = np.array(
    [
        train_cached_map[os.path.splitext(os.path.basename(p))[0]]
        for p in train_paths_valid
    ],
    dtype=object,
)

print(
    "Cached train split paths:",
    train_paths_train_cached.shape,
    train_paths_valid_cached.shape,
)


def _tf_read_cached_png(path, label_onehot=None):
    data = tf.io.read_file(path)
    img = tf.image.decode_png(data, channels=3)  # uint8 RGB
    img.set_shape((224, 224, 3))
    if label_onehot is None:
        return img
    return img, label_onehot




## === cell 38
batch_size = 200

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths_train_cached, y_resampled_onehot))
    .map(_tf_read_cached_png, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths_valid_cached, y_valid_onehot))
    .map(_tf_read_cached_png, num_parallel_calls=AUTOTUNE, deterministic=True)
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



## === cell 39
pass



## === cell 40
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
else:
    print(f"Warning: {checkpoint_path} not found; using last-epoch in-memory weights")

restored_model = model



## === cell 41
restored_model.evaluate(valid_ds, verbose=0)



## === cell 42
y_pred = np.argmax(restored_model.predict(valid_ds, verbose=0), axis=1)

print("Predict:", y_pred[:10])
print("Validation:", np.array(y_valid[:10]))



## === cell 43
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=y_valid, y_pred=y_pred)

print("Confusion Matrix")
print(cm)
print()
print("Shape :", cm.shape)
print("Accurcy: {0:.2f}%".format(np.trace(cm) / np.sum(cm) * 100))



## === cell 44
from sklearn.metrics import classification_report

print(
    classification_report(
        y_valid,
        y_pred,
        digits=4,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
    )
)



## === cell 45
test_data_folder = os.path.join(base_data_folder, "test_images")
print("Test images folder exists:", os.path.exists(test_data_folder))



## === cell 46
test_df = pd.read_csv(os.path.join(base_data_folder, "test.csv"))
test_df.head()



## === cell 47
test_ids = test_df["id_code"].values
test_paths = np.array(
    [os.path.join(test_data_folder, f"{id_code}.png") for id_code in test_ids],
    dtype=object,
)



## === cell 48
test_cached_paths = _precompute_many(
    test_paths, test_cache_dir, max_workers=max_workers
)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_cached_paths)
    .map(_tf_read_cached_png, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(256, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
preds = np.argmax(restored_model.predict(test_ds, verbose=0), axis=1)
print("Predicted (first 10):", preds[:10])



## === cell 49
submission = pd.DataFrame({"id_code": test_ids, "diagnosis": preds.astype(int)})
submission.head()



## === cell 50
submission = submission.sort_values("id_code").reset_index(drop=True)
submission.head()



## === cell 51
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())



## === cell 52
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
