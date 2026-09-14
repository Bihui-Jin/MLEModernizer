# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Code solution

## === cell 0
import os, zipfile

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

np.random.seed(19)
tf.random.set_seed(19)

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"


def _maybe_extract(zip_path, out_path, must_exist_relpath):
    target = os.path.join(out_path, must_exist_relpath)
    if os.path.exists(target) and (os.path.isdir(target) or os.path.isfile(target)):
        return
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_path)


_maybe_extract(path_zip + "train.zip", path, "train")
_maybe_extract(path_zip + "test.zip", path, "test")
_maybe_extract(path_zip + "train_cleaned.zip", path, "train_cleaned")
_maybe_extract(path_zip + "sampleSubmission.csv.zip", path, "sampleSubmission.csv")

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

train_img = sorted([f for f in os.listdir(train_dir) if f.lower().endswith(".png")])
train_cleaned_img = sorted(
    [f for f in os.listdir(train_cleaned_dir) if f.lower().endswith(".png")]
)
test_img = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".png")])

common = sorted(set(train_img).intersection(set(train_cleaned_img)))
if len(common) == 0:
    raise RuntimeError(
        "No overlapping filenames between train and train_cleaned. Check extracted paths."
    )
train_img = common
train_cleaned_img = common

print(
    "Train images:",
    len(train_img),
    "Train cleaned images:",
    len(train_cleaned_img),
    "Test images:",
    len(test_img),
)




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [cv2.imread(os.path.join(train_dir, f)) for f in train_img[:10]]
print("Sample Dimensions (H,W):", [img.shape[:2] for img in imgs if img is not None])
del imgs




## === cell 3
def process_image(path_):
    img = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path_}")
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    img = img.astype("float32") / 255.0
    img = img.reshape((*config.IMG_SIZE, 1))
    return img




## === cell 4
n_train = len(train_img)
n_test = len(test_img)
H, W = config.IMG_SIZE

train = np.empty((n_train, H, W, 1), dtype=np.float32)
train_cleaned = np.empty((n_train, H, W, 1), dtype=np.float32)
test = np.empty((n_test, H, W, 1), dtype=np.float32)

for i, f in enumerate(tqdm(train_img, desc="Loading train")):
    train[i] = process_image(os.path.join(train_dir, f))

for i, f in enumerate(tqdm(train_cleaned_img, desc="Loading train_cleaned")):
    train_cleaned[i] = process_image(os.path.join(train_cleaned_dir, f))

for i, f in enumerate(tqdm(test_img, desc="Loading test")):
    test[i] = process_image(os.path.join(test_dir, f))



## === cell 5
train.shape, train_cleaned.shape, test.shape




## === cell 6
def _resize_back_batch(batch_hw1):
    out = np.empty(
        (batch_hw1.shape[0], config.IMG_SIZE[0], config.IMG_SIZE[1], 1),
        dtype=batch_hw1.dtype,
    )
    for i in range(batch_hw1.shape[0]):
        out[i, ..., 0] = cv2.resize(
            batch_hw1[i, ..., 0],
            (config.IMG_SIZE[1], config.IMG_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return out


def augment_pipeline_fast(images, seed=19):
    np.random.seed(seed)
    x0 = images
    n = x0.shape[0]

    out = np.empty((n * 32, *x0.shape[1:]), dtype=x0.dtype)
    out[:n] = x0
    cur_len = n

    y = np.rot90(out[:cur_len], k=1, axes=(1, 2)).copy()
    y = _resize_back_batch(y)
    out[cur_len : cur_len * 2] = y
    cur_len *= 2

    y = np.rot90(out[:cur_len], k=2, axes=(1, 2)).copy()
    out[cur_len : cur_len * 2] = y
    cur_len *= 2

    y = np.rot90(out[:cur_len], k=3, axes=(1, 2)).copy()
    y = _resize_back_batch(y)
    out[cur_len : cur_len * 2] = y
    cur_len *= 2

    y = np.flip(out[:cur_len], axis=2).copy()
    out[cur_len : cur_len * 2] = y
    cur_len *= 2

    y = np.flip(out[:cur_len], axis=1).copy()
    out[cur_len : cur_len * 2] = y
    cur_len *= 2

    return out




## === cell 7
def _apply_aug_by_index(x, k):
    k = tf.cast(k, tf.int32)

    flip_lr = tf.greater_equal(k, 16)
    k2 = tf.where(flip_lr, k - 16, k)

    flip_ud = tf.greater_equal(k2, 8)
    k3 = tf.where(flip_ud, k2 - 8, k2)

    base = k3  # 0..7

    lr_inner = tf.greater_equal(base, 4)
    base4 = tf.where(lr_inner, base - 4, base)  # 0..3

    def _t0(z):  # identity
        return z

    def _t1(z):  # rot90 then resize back
        return tf.image.resize(
            tf.image.rot90(z, k=1),
            size=config.IMG_SIZE,
            method="bilinear",
            antialias=False,
        )

    def _t2(z):  # rot180
        return tf.image.rot90(z, k=2)

    def _t3(z):  # rot270 then resize back
        return tf.image.resize(
            tf.image.rot90(z, k=3),
            size=config.IMG_SIZE,
            method="bilinear",
            antialias=False,
        )

    z = tf.switch_case(
        base4,
        branch_fns={
            0: lambda: _t0(x),
            1: lambda: _t1(x),
            2: lambda: _t2(x),
            3: lambda: _t3(x),
        },
    )

    z = tf.cond(lr_inner, lambda: tf.image.flip_left_right(z), lambda: z)
    z = tf.cond(flip_ud, lambda: tf.image.flip_up_down(z), lambda: z)
    z = tf.cond(flip_lr, lambda: tf.image.flip_left_right(z), lambda: z)
    return z


def make_augmented_dataset(x, y, batch_size):
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)

    base_ds = tf.data.Dataset.from_tensor_slices((x_tf, y_tf))
    aug_idx = tf.data.Dataset.range(32)
    ds = aug_idx.flat_map(
        lambda k: base_ds.map(
            lambda a, b: (_apply_aug_by_index(a, k), _apply_aug_by_index(b, k)),
            num_parallel_calls=AUTOTUNE,
        )
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    opt = tf.data.Options()
    opt.experimental_deterministic = True
    ds = ds.with_options(opt)
    return ds


BATCH_SIZE = 12
train_ds = make_augmented_dataset(train, train_cleaned, batch_size=BATCH_SIZE)

steps_per_epoch = int(np.ceil((n_train * 32) / BATCH_SIZE))
print("Augmented samples:", n_train * 32, "steps_per_epoch:", steps_per_epoch)




## === cell 8
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = DenoisingAutoencoder()

autoencoder.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error"],
    jit_compile=True,
)



## === cell 9
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)
rlp = callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.8, patience=5, min_lr=1e-6, mode="min", verbose=1
)

history = autoencoder.fit(
    train_ds,
    shuffle=True,
    callbacks=[es, rlp],
    epochs=500,
    steps_per_epoch=steps_per_epoch,
    batch_size=None,
)



## === cell 10
print("Final training loss:", float(history.history["loss"][-1]))



## === cell 11
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 12
decoded_imgs = autoencoder(train[:4], training=False).numpy()
decoded_imgs = np.clip(decoded_imgs, 0.0, 1.0)
print(
    "Sanity decoded batch:", decoded_imgs.shape, decoded_imgs.min(), decoded_imgs.max()
)
del decoded_imgs



## === cell 13
sample_path = os.path.join(path, "sampleSubmission.csv")
if not os.path.exists(sample_path):
    sample_path = "../input/denoising-dirty-documents/sampleSubmission.csv"

sample_sub = pd.read_csv(sample_path, usecols=["id"], dtype={"id": "string"})
ids = sample_sub["id"].to_numpy()

id_bytes = ids.astype("S")
us = np.frombuffer(b"_", dtype="S1")[0]
p1 = np.char.find(id_bytes, us)
p2 = np.char.find(id_bytes, us, start=p1 + 1)

img_ids = np.char.decode(np.char.substr(id_bytes, 0, p1)).astype(np.int32)
rows = (
    np.char.decode(np.char.substr(id_bytes, p1 + 1, p2 - (p1 + 1))).astype(np.int32) - 1
)
cols = (
    np.char.decode(
        np.char.substr(id_bytes, p2 + 1, np.char.str_len(id_bytes) - (p2 + 1))
    ).astype(np.int32)
    - 1
)

del sample_sub, id_bytes, us, p1, p2

decoded_test = autoencoder(test, training=False).numpy()
decoded_test = np.clip(decoded_test, 0.0, 1.0)

test_imgids = np.fromiter(
    (int(f[:-4]) for f in test_img), dtype=np.int32, count=len(test_img)
)
imgid_to_index = {imgid: i for i, imgid in enumerate(test_imgids)}

h0, w0 = cv2.imread(os.path.join(test_dir, test_img[0]), cv2.IMREAD_GRAYSCALE).shape

if (h0, w0) != config.IMG_SIZE:
    decoded_test_rs = (
        tf.image.resize(decoded_test, size=(h0, w0), method="bilinear", antialias=False)
        .numpy()
        .astype(np.float32)
    )
else:
    decoded_test_rs = decoded_test.astype(np.float32)

del decoded_test

idx = np.fromiter(
    (imgid_to_index[i] for i in img_ids), dtype=np.int32, count=img_ids.shape[0]
)
values = decoded_test_rs[idx, rows, cols, 0].astype(np.float32, copy=False)

sub = pd.DataFrame({"id": ids, "value": values})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", sub.shape)
print(sub.head())
