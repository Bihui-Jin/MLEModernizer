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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

sns.set_style("darkgrid")

np.random.seed(19)
tf.random.set_seed(19)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
path_zip = "/kaggle/input/denoising-dirty-documents/"
path = "/kaggle/working/"

os.makedirs(path, exist_ok=True)

need_extract = False
for folder in ["train", "test", "train_cleaned"]:
    if not os.path.isdir(os.path.join(path, folder)) and not os.path.isdir(
        os.path.join(path, "denoising-dirty-documents", folder)
    ):
        need_extract = True
        break

if need_extract:
    for zname in [
        "train.zip",
        "test.zip",
        "train_cleaned.zip",
        "sampleSubmission.csv.zip",
    ]:
        zpath = os.path.join(path_zip, zname)
        with zipfile.ZipFile(zpath, "r") as zip_ref:
            zip_ref.extractall(path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

if not (
    os.path.isdir(train_dir)
    and os.path.isdir(train_cleaned_dir)
    and os.path.isdir(test_dir)
):
    nested_base = os.path.join(path, "denoising-dirty-documents")
    train_dir = os.path.join(nested_base, "train")
    train_cleaned_dir = os.path.join(nested_base, "train_cleaned")
    test_dir = os.path.join(nested_base, "test")


def _list_pngs(d):
    files = []
    for fn in os.listdir(d):
        full = os.path.join(d, fn)
        if os.path.isfile(full) and fn.lower().endswith(".png"):
            files.append(fn)
    return sorted(files)


train_img = _list_pngs(train_dir)
train_cleaned_img = _list_pngs(train_cleaned_dir)
test_img = _list_pngs(test_dir)

print(
    "Found:",
    len(train_img),
    "train,",
    len(train_cleaned_img),
    "train_cleaned,",
    len(test_img),
    "test images",
)

if len(test_img) == 0:
    raise RuntimeError(f"No test .png images found in: {test_dir}")




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [cv2.imread(os.path.join(train_dir, f)) for f in _list_pngs(train_dir)[:10]]
imgs = [im for im in imgs if im is not None]
print(
    "Median Dimensions:",
    np.median([img.shape[0] for img in imgs]),
    np.median([img.shape[1] for img in imgs]),
)
del imgs



## === cell 3
from concurrent.futures import ThreadPoolExecutor


def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cv2.imread failed for path: {img_path}")
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img.reshape((*config.IMG_SIZE, 1))
    return img




## === cell 4
train_files = set(_list_pngs(train_dir))
clean_files = set(_list_pngs(train_cleaned_dir))
paired_files = sorted(train_files.intersection(clean_files))

if len(paired_files) == 0:
    raise RuntimeError(
        "No overlapping filenames between train and train_cleaned folders."
    )

missing_in_clean = sorted(train_files - clean_files)
missing_in_train = sorted(clean_files - train_files)
if missing_in_clean:
    print(
        f"Warning: {len(missing_in_clean)} files exist in train but not in train_cleaned. They will be ignored."
    )
if missing_in_train:
    print(
        f"Warning: {len(missing_in_train)} files exist in train_cleaned but not in train. They will be ignored."
    )

train_paths = [os.path.join(train_dir, f) for f in paired_files]
clean_paths = [os.path.join(train_cleaned_dir, f) for f in paired_files]
test_img = _list_pngs(test_dir)
test_paths = [os.path.join(test_dir, f) for f in test_img]

max_workers = min(8, (os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    train = list(ex.map(process_image, train_paths))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    train_cleaned = list(ex.map(process_image, clean_paths))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    test = list(ex.map(process_image, test_paths))

train = np.asarray(train, dtype=np.float32)
train_cleaned = np.asarray(train_cleaned, dtype=np.float32)
test = np.asarray(test, dtype=np.float32)

train_img = paired_files
train_cleaned_img = paired_files

print("Loaded paired train set:", train.shape, train_cleaned.shape)
print("Loaded test set:", test.shape)



## === cell 5
train.shape, train_cleaned.shape, test.shape



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
plt.show()




## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    """
    Score/logic fix preserved:
    - Noisy and clean must receive identical deterministic transforms.
    - Generate each augmented variant from the same original tensor x0.
    - Augmentations: [original, rot90, rot180, rot270, hflip, vflip].
    - Keep shape stable for rot90/rot270 by resizing back to (H, W).
    """
    tf.random.set_seed(seed)
    x0 = tf.convert_to_tensor(images, dtype=tf.float32)
    target_hw = config.IMG_SIZE  # (H, W)

    out = [x0]

    for step in pipeline:
        x = x0
        if step == "rot90":
            x = tf.image.rot90(x, k=1)
            x = tf.image.resize(x, target_hw, method="bilinear")
        elif step == "rot180":
            x = tf.image.rot90(x, k=2)
        elif step == "rot270":
            x = tf.image.rot90(x, k=3)
            x = tf.image.resize(x, target_hw, method="bilinear")
        elif step == "hflip":
            x = tf.image.flip_left_right(x)
        elif step == "vflip":
            x = tf.image.flip_up_down(x)
        else:
            raise ValueError(f"Unknown augmentation step: {step}")

        x = tf.cast(x, tf.float32)
        out.append(x)

    return tf.concat(out, axis=0)




## === cell 8
pipeline = ["rot90", "rot180", "rot270", "hflip", "vflip"]




## === cell 9
def _apply_one_aug(x, y, aug_name):
    target_hw = config.IMG_SIZE
    if aug_name == "orig":
        return x, y
    if aug_name == "rot90":
        x2 = tf.image.rot90(x, k=1)
        y2 = tf.image.rot90(y, k=1)
        x2 = tf.image.resize(x2, target_hw, method="bilinear")
        y2 = tf.image.resize(y2, target_hw, method="bilinear")
        return tf.cast(x2, tf.float32), tf.cast(y2, tf.float32)
    if aug_name == "rot180":
        x2 = tf.image.rot90(x, k=2)
        y2 = tf.image.rot90(y, k=2)
        return tf.cast(x2, tf.float32), tf.cast(y2, tf.float32)
    if aug_name == "rot270":
        x2 = tf.image.rot90(x, k=3)
        y2 = tf.image.rot90(y, k=3)
        x2 = tf.image.resize(x2, target_hw, method="bilinear")
        y2 = tf.image.resize(y2, target_hw, method="bilinear")
        return tf.cast(x2, tf.float32), tf.cast(y2, tf.float32)
    if aug_name == "hflip":
        return tf.image.flip_left_right(x), tf.image.flip_left_right(y)
    if aug_name == "vflip":
        return tf.image.flip_up_down(x), tf.image.flip_up_down(y)
    raise ValueError(f"Unknown augmentation step: {aug_name}")


AUG_NAMES = ["orig"] + pipeline


def make_augmented_dataset(x_np, y_np, batch_size, seed=19):
    base = tf.data.Dataset.from_tensor_slices((x_np, y_np)).cache()
    ds_list = []
    for name in AUG_NAMES:
        ds_list.append(
            base.map(
                lambda a, b, n=name: _apply_one_aug(a, b, n),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )
    ds = ds_list[0]
    for d in ds_list[1:]:
        ds = ds.concatenate(d)

    ds = ds.shuffle(
        buffer_size=int(x_np.shape[0]) * len(AUG_NAMES),
        seed=seed,
        reshuffle_each_iteration=True,
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


processed_train = None
processed_train_cleaned = None




## === cell 10
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
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
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



## === cell 11
y_hat_shape = autoencoder(train[:1], training=False).shape
if tuple(y_hat_shape[1:]) != tuple(train_cleaned[:1].shape[1:]):
    raise RuntimeError(
        f"Model output shape {y_hat_shape} does not match target shape {train_cleaned[:1].shape}."
    )
print("Output shape matches target:", y_hat_shape)



## === cell 12
BATCH_SIZE = 12

train_ds = make_augmented_dataset(train, train_cleaned, batch_size=BATCH_SIZE, seed=19)

es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    train_ds,
    callbacks=[es],
    epochs=500,
    verbose=2,
)



## === cell 13
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
plt.show()
del history



## === cell 14
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 15
decoded_imgs = autoencoder(train[:4], training=False).numpy()

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



## === cell 16
import csv

pred_test = autoencoder(test, training=False).numpy().astype(np.float32)

out_path = "submission.csv"

sample_path = "/kaggle/input/denoising-dirty-documents/sampleSubmission.csv"
if os.path.exists(sample_path):
    sample_n = sum(1 for _ in open(sample_path)) - 1  # minus header
    print("Sample submission rows:", sample_n)

shape_cache = {}  # (h,w) -> (rr_str, cc_str) flattened arrays of strings

with open(out_path, "w", newline="") as f:
    f.write("id,value\n")
    for i, fname in tqdm(list(enumerate(test_img)), total=len(test_img)):
        imgid = int(fname[:-4])
        img0 = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img0 is None:
            raise FileNotFoundError(f"cv2.imread failed for test image: {fname}")
        h, w = img0.shape

        decoded_img = np.squeeze(pred_test[i]).astype(np.float32)
        preds = cv2.resize(decoded_img, (w, h), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )
        preds = np.clip(preds, 0.0, 1.0)

        key = (h, w)
        if key not in shape_cache:
            r = np.arange(1, h + 1, dtype=np.int32)
            c = np.arange(1, w + 1, dtype=np.int32)
            rr = np.repeat(r, w)
            cc = np.tile(c, h)
            shape_cache[key] = (rr.astype(np.str_), cc.astype(np.str_))
        rr_s, cc_s = shape_cache[key]

        prefix = f"{imgid}_"
        ids = np.char.add(np.char.add(prefix, rr_s), "_")
        ids = np.char.add(ids, cc_s)

        vals = preds.reshape(-1).astype(np.float32)

        block = np.empty((vals.size, 2), dtype=object)
        block[:, 0] = ids
        block[:, 1] = vals.astype(np.str_)

        np.savetxt(f, block, fmt="%s,%s")

print(f"Results saved to {out_path}!")
print(pd.read_csv(out_path, nrows=5))
if os.path.exists(sample_path):
    sub_rows = sum(1 for _ in open(out_path)) - 1
    print("Submission rows:", sub_rows, "Expected:", sample_n)
