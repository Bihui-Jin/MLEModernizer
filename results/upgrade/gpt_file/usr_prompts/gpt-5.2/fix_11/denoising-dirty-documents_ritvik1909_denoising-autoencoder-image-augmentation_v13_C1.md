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

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide / use env
    tf.config.threading.set_inter_op_parallelism_threads(0)
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
def _apply_aug_np(x_hw1: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    flip_lr = k >= 16
    k2 = k - 16 if flip_lr else k

    flip_ud = k2 >= 8
    k3 = k2 - 8 if flip_ud else k2

    base = k3  # 0..7
    lr_inner = base >= 4
    base4 = base - 4 if lr_inner else base  # 0..3

    img = x_hw1[..., 0]  # (H, W)

    if base4 == 0:
        z = img
    elif base4 == 1:
        z = np.rot90(img, k=1)
    elif base4 == 2:
        z = np.rot90(img, k=2)
    elif base4 == 3:
        z = np.rot90(img, k=3)
    else:
        raise ValueError("base4 out of range")

    if lr_inner:
        z = np.fliplr(z)
    if flip_ud:
        z = np.flipud(z)
    if flip_lr:
        z = np.fliplr(z)

    if z.shape != (H, W):
        z = cv2.resize(z, (W, H), interpolation=cv2.INTER_LINEAR)

    return z[..., None].astype(np.float32, copy=False)


def make_augmented_arrays(x: np.ndarray, y: np.ndarray, n_aug: int = 32):
    n = x.shape[0]
    xa = np.empty((n * n_aug, H, W, 1), dtype=np.float32)
    ya = np.empty((n * n_aug, H, W, 1), dtype=np.float32)
    out = 0
    for k in range(n_aug):
        for i in range(n):
            xa[out] = _apply_aug_np(x[i], k)
            ya[out] = _apply_aug_np(y[i], k)
            out += 1
    return xa, ya


BATCH_SIZE = 12

train_aug, train_cleaned_aug = make_augmented_arrays(train, train_cleaned, n_aug=32)

train_ds = tf.data.Dataset.from_tensor_slices((train_aug, train_cleaned_aug))
train_ds = train_ds.cache().batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
opt = tf.data.Options()
opt.experimental_deterministic = True
train_ds = train_ds.with_options(opt)

steps_per_epoch = int(np.ceil(train_aug.shape[0] / BATCH_SIZE))
print("Augmented samples:", train_aug.shape[0], "steps_per_epoch:", steps_per_epoch)




## === cell 7
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



## === cell 8
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)
rlp = callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.8, patience=5, min_lr=1e-6, mode="min", verbose=1
)

history = autoencoder.fit(
    train_ds,
    callbacks=[es, rlp],
    epochs=500,
    steps_per_epoch=steps_per_epoch,
    batch_size=None,
)



## === cell 9
print("Final training loss:", float(history.history["loss"][-1]))



## === cell 10
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 11
decoded_imgs = autoencoder(train[:4], training=False).numpy()
decoded_imgs = np.clip(decoded_imgs, 0.0, 1.0)
print(
    "Sanity decoded batch:", decoded_imgs.shape, decoded_imgs.min(), decoded_imgs.max()
)
del decoded_imgs



## === cell 12
sample_path = os.path.join(path, "sampleSubmission.csv")
if not os.path.exists(sample_path):
    sample_path = "../input/denoising-dirty-documents/sampleSubmission.csv"

sample_sub = pd.read_csv(sample_path, usecols=["id"], dtype={"id": "string"})
ids = sample_sub["id"].to_numpy()
parts = sample_sub["id"].str.split("_", n=2, expand=True)
img_ids = parts[0].astype("int32").to_numpy()
rows = parts[1].astype("int32").to_numpy() - 1
cols = parts[2].astype("int32").to_numpy() - 1
del sample_sub, parts

decoded_test = autoencoder.predict(test, batch_size=BATCH_SIZE, verbose=0)
decoded_test = np.clip(decoded_test, 0.0, 1.0)

test_imgids = np.fromiter(
    (int(f[:-4]) for f in test_img), dtype=np.int32, count=len(test_img)
)

h0, w0 = cv2.imread(os.path.join(test_dir, test_img[0]), cv2.IMREAD_GRAYSCALE).shape
if (h0, w0) != config.IMG_SIZE:
    decoded_test_rs = (
        tf.image.resize(decoded_test, size=(h0, w0), method="bilinear", antialias=False)
        .numpy()
        .astype(np.float32)
    )
else:
    decoded_test_rs = decoded_test.astype(np.float32, copy=False)
del decoded_test

order = np.argsort(test_imgids)
test_sorted = test_imgids[order]
pos = np.searchsorted(test_sorted, img_ids)
idx = order[pos].astype(np.int32, copy=False)

values = decoded_test_rs[idx, rows, cols, 0].astype(np.float32, copy=False)
del decoded_test_rs, idx, order, test_sorted, pos, img_ids, rows, cols, test_imgids

sub = pd.DataFrame({"id": ids, "value": values})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", sub.shape)
print(sub.head())
