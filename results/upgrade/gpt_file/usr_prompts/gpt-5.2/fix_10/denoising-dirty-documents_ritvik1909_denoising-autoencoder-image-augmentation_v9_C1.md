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
import os, zipfile, sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

sns.set_style("darkgrid")

np.random.seed(19)
tf.random.set_seed(19)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
path_zip_candidates = [
    "/kaggle/input/denoising-dirty-documents",
    "/kaggle/input/denoising-dirty-documents/denoising-dirty-documents",
]
path_zip = None
for cand in path_zip_candidates:
    if os.path.exists(os.path.join(cand, "train.zip")):
        path_zip = cand
        break
if path_zip is None:
    path_zip = "/kaggle/input/denoising-dirty-documents/"

path = "/kaggle/working/"

for zname, out_dir in [
    ("train.zip", "train"),
    ("test.zip", "test"),
    ("train_cleaned.zip", "train_cleaned"),
]:
    target_dir = os.path.join(path, out_dir)
    if (not os.path.exists(target_dir)) or (len(os.listdir(target_dir)) == 0):
        with zipfile.ZipFile(os.path.join(path_zip, zname), "r") as zip_ref:
            zip_ref.extractall(path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

train_img = sorted(os.listdir(train_dir))
train_cleaned_img = sorted(os.listdir(train_cleaned_dir))
test_img = sorted(os.listdir(test_dir))

print(
    "Train images:",
    len(train_img),
    "Train_cleaned images:",
    len(train_cleaned_img),
    "Test images:",
    len(test_img),
)




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [
    cv2.imread(os.path.join(train_dir, f)) for f in sorted(os.listdir(train_dir))[:10]
]
print("Example Dimensions (first 10):", [(img.shape[0], img.shape[1]) for img in imgs])
print(
    "Median Dimensions (first 10):",
    np.median([img.shape[0] for img in imgs]),
    np.median([img.shape[1] for img in imgs]),
)
del imgs




## === cell 3
def process_image(img_path):
    img = cv2.imread(img_path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
def _stem(fn: str) -> str:
    return os.path.splitext(os.path.basename(fn))[0]


train_files = sorted(os.listdir(train_dir))
clean_files = sorted(os.listdir(train_cleaned_dir))

clean_map = {_stem(f): f for f in clean_files}

paired_train_files = []
paired_clean_files = []
missing = []
for f in train_files:
    s = _stem(f)
    if s in clean_map:
        paired_train_files.append(f)
        paired_clean_files.append(clean_map[s])
    else:
        missing.append(f)

if missing:
    print(
        "Warning: missing cleaned counterparts for",
        len(missing),
        "train files. Example:",
        missing[:5],
    )

print("Paired train/clean count:", len(paired_train_files))

n_train = len(paired_train_files)
n_test = len(test_img)
h, w = config.IMG_SIZE
train = np.empty((n_train, h, w, 1), dtype=np.float32)
train_cleaned = np.empty((n_train, h, w, 1), dtype=np.float32)
test = np.empty((n_test, h, w, 1), dtype=np.float32)

for i, f in enumerate(paired_train_files):
    train[i] = process_image(os.path.join(train_dir, f))

for i, f in enumerate(paired_clean_files):
    train_cleaned[i] = process_image(os.path.join(train_cleaned_dir, f))

for i, f in enumerate(sorted(os.listdir(test_dir))):
    test[i] = process_image(os.path.join(test_dir, f))

train_img = paired_train_files



## === cell 5
print(train.shape, train_cleaned.shape, test.shape)



## === cell 6
fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train[i]), cmap="gray")
    ax[i][0].set_title("Noise image: {}".format(train_img[i]))

    ax[i][1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][1].set_title("Denoised image: {}".format(train_img[i]))

    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)




## === cell 7
def _resize_to_img_size(x):
    return tf.image.resize(x, config.IMG_SIZE, method="bilinear", antialias=True)


@tf.function(reduce_retracing=True)
def augment_pipeline_tf(images):
    images = tf.cast(images, tf.float32)

    r1 = _resize_to_img_size(tf.image.rot90(images, k=1))
    r2 = _resize_to_img_size(tf.image.rot90(images, k=2))
    r3 = _resize_to_img_size(tf.image.rot90(images, k=3))

    out = tf.concat(
        [
            images,
            r1,
            r2,
            r3,
            tf.reverse(images, axis=[2]),  # horizontal flip (W axis)
            tf.reverse(images, axis=[1]),  # vertical flip (H axis)
        ],
        axis=0,
    )
    return out


processed_train = augment_pipeline_tf(train).numpy().astype(np.float32, copy=False)
processed_train_cleaned = (
    augment_pipeline_tf(train_cleaned).numpy().astype(np.float32, copy=False)
)

print(processed_train.shape, processed_train_cleaned.shape)




## === cell 8
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(48, (5, 5), activation="relu", padding="same"),
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
                layers.Conv2D(48, (5, 5), activation="relu", padding="same"),
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
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)



## === cell 9
batch_size = 12

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True

train_ds = (
    tf.data.Dataset.from_tensor_slices((processed_train, processed_train_cleaned))
    .with_options(options)
    .shuffle(buffer_size=len(processed_train), seed=19, reshuffle_each_iteration=True)
    .batch(batch_size, drop_remainder=False)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

history = autoencoder.fit(
    train_ds,
    epochs=500,
)



## === cell 10
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
plt.show()
del history



## === cell 11
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 12
decoded_imgs = autoencoder.predict_on_batch(train[:4])

fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][0].set_title("Denoised image: {}".format(train_img[i]))

    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
    ax[i][1].set_title("Predicted image: {}".format(train_img[i]))

    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)

plt.show()
del decoded_imgs



## === cell 13
sample_path_candidates = [
    os.path.join(path, "sampleSubmission.csv"),
    os.path.join(path_zip, "sampleSubmission.csv"),
    os.path.join("/kaggle/input", "sampleSubmission.csv"),
]
sample_path = None
for p in sample_path_candidates:
    if os.path.exists(p):
        sample_path = p
        break
if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sampleSubmission.csv. Tried: {sample_path_candidates}"
    )

with open(sample_path, "rb") as f:
    n_rows = sum(1 for _ in f) - 1  # subtract header
print("Using sample submission at:", sample_path)
print("Sample submission rows:", n_rows)

sub_path = os.path.join(path, "submission.csv")

with open(sub_path, "w", newline="") as f_out:
    f_out.write("id,value\n")
    total_written = 0

    id_cache = {}  # (imgid, H, W) -> id_grid (object array)

    for i, fname in tqdm(list(enumerate(test_img)), total=len(test_img)):
        file = os.path.join(test_dir, fname)
        imgid = os.path.splitext(fname)[0]

        img0 = cv2.imread(file, 0)
        H, W = img0.shape

        decoded_img = np.squeeze(
            autoencoder.predict_on_batch(test[i : i + 1])
        )  # (420,540)

        preds_reshaped = cv2.resize(decoded_img, (W, H), interpolation=cv2.INTER_LINEAR)
        preds_reshaped = np.clip(preds_reshaped, 0.0, 1.0).astype(
            np.float32, copy=False
        )
        val_flat = preds_reshaped.ravel()

        cache_key = (imgid, H, W)
        id_grid = id_cache.get(cache_key)
        if id_grid is None:
            rows = np.arange(1, H + 1, dtype=np.int32).astype(str)
            cols = np.arange(1, W + 1, dtype=np.int32).astype(str)

            base = np.char.add(np.char.add(imgid, "_"), rows)
            base = np.char.add(base, "_")
            id_grid = np.char.add(base[:, None], cols[None, :]).ravel()
            id_cache[cache_key] = id_grid

        val_str = np.char.mod("%.8f", val_flat.astype(np.float64, copy=False))
        lines = np.char.add(np.char.add(id_grid, ","), val_str)
        f_out.write("\n".join(lines.tolist()))
        f_out.write("\n")

        total_written += val_flat.size

print("Generated rows:", total_written)
assert (
    total_written == n_rows
), f"Row count mismatch vs sampleSubmission: generated {total_written}, expected {n_rows}"

print(f"Results saved to {sub_path}!")
print(pd.read_csv(sub_path, nrows=5))
