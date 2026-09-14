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

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
CANDIDATE_INPUT_DIRS = [
    "/kaggle/input/denoising-dirty-documents/",
    "/kaggle/input/",
    "../input/denoising-dirty-documents/",
    "../input/",
]
path_zip = None
for d in CANDIDATE_INPUT_DIRS:
    if os.path.isdir(d) and (
        os.path.exists(os.path.join(d, "train.zip"))
        or os.path.exists(os.path.join(d, "denoising-dirty-documents", "train.zip"))
    ):
        if os.path.exists(os.path.join(d, "denoising-dirty-documents", "train.zip")):
            path_zip = os.path.join(d, "denoising-dirty-documents") + "/"
        else:
            path_zip = d if d.endswith("/") else d + "/"
        break

if path_zip is None:
    raise FileNotFoundError(
        "Could not find denoising-dirty-documents zips under expected /kaggle/input paths."
    )

path = "/kaggle/working/"


def maybe_extract(zip_name, target_dir):
    zip_path = os.path.join(path_zip, zip_name)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing {zip_path}")
    if zip_name.endswith(".csv.zip"):
        out_csv = os.path.join(path, "sampleSubmission.csv")
        if os.path.exists(out_csv):
            return
    else:
        folder = zip_name.replace(".zip", "")
        out_dir = os.path.join(path, folder)
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(path)


maybe_extract("train.zip", path)
maybe_extract("test.zip", path)
maybe_extract("train_cleaned.zip", path)
maybe_extract("sampleSubmission.csv.zip", path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")
sample_sub_path = os.path.join(path, "sampleSubmission.csv")

train_img = sorted(os.listdir(train_dir))
train_cleaned_img = sorted(os.listdir(train_cleaned_dir))
test_img = sorted(os.listdir(test_dir))

print(
    "Num train:",
    len(train_img),
    "Num train_cleaned:",
    len(train_cleaned_img),
    "Num test:",
    len(test_img),
)
print("Sample submission exists:", os.path.exists(sample_sub_path))




## === cell 2
_probe = cv2.imread(os.path.join(test_dir, test_img[0]), cv2.IMREAD_UNCHANGED)
if _probe is None:
    raise FileNotFoundError(f"Could not read test image: {test_img[0]}")
if _probe.ndim == 3:
    _probe = cv2.cvtColor(_probe, cv2.COLOR_BGR2GRAY)
max_row, max_col = int(_probe.shape[0]), int(_probe.shape[1])
del _probe


class config:
    IMG_SIZE = (max_row, max_col)  # (H, W)


print("Using IMG_SIZE from image shape:", config.IMG_SIZE)

imgs = []
for f in sorted(os.listdir(train_dir))[:10]:
    im = cv2.imread(os.path.join(train_dir, f), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Could not read train image: {f}")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgs.append(im)
print(
    "Median raw Dimensions:",
    int(np.median([img.shape[0] for img in imgs])),
    int(np.median([img.shape[1] for img in imgs])),
    "Median dtype:",
    str(np.median([img.dtype == np.uint16 for img in imgs])),
)
del imgs




## === cell 3
def process_image(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_AREA)
    img = img.astype("float32")
    denom = 65535.0 if img.max() > 255.0 else 255.0
    img = img / denom
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img


AUTOTUNE = tf.data.AUTOTUNE


def _tf_read_and_preprocess(path_tensor):
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.io.decode_png(img_bytes, channels=1)  # uint8 or uint16 [H,W,1]
    img = tf.image.convert_image_dtype(img, tf.float32)  # now in [0,1]
    img = tf.image.resize(img, config.IMG_SIZE, method=tf.image.ResizeMethod.AREA)
    img = tf.ensure_shape(img, (*config.IMG_SIZE, 1))
    return img


def _make_path_list(dir_path, file_list):
    return [os.path.join(dir_path, fn) for fn in file_list]


train_paths = _make_path_list(train_dir, train_img)
train_cleaned_paths = _make_path_list(train_cleaned_dir, train_cleaned_img)
test_paths = _make_path_list(test_dir, test_img)

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True

train_ds = tf.data.Dataset.from_tensor_slices(train_paths).with_options(data_opts)
train_cleaned_ds = tf.data.Dataset.from_tensor_slices(train_cleaned_paths).with_options(
    data_opts
)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(data_opts)

cache_dir = os.path.join(path, "tf_cache")
os.makedirs(cache_dir, exist_ok=True)

train_ds = train_ds.map(_tf_read_and_preprocess, num_parallel_calls=AUTOTUNE).cache(
    os.path.join(cache_dir, "train.cache")
)
train_cleaned_ds = train_cleaned_ds.map(
    _tf_read_and_preprocess, num_parallel_calls=AUTOTUNE
).cache(os.path.join(cache_dir, "train_cleaned.cache"))
test_ds = test_ds.map(_tf_read_and_preprocess, num_parallel_calls=AUTOTUNE).cache(
    os.path.join(cache_dir, "test.cache")
)

train_pair_ds = tf.data.Dataset.zip((train_ds, train_cleaned_ds)).with_options(
    data_opts
)




## === cell 4
num_train = len(train_paths)
num_test = len(test_paths)

print("Prepared tf.data datasets. num_train:", num_train, "num_test:", num_test)




## === cell 5
print(
    (num_train, config.IMG_SIZE[0], config.IMG_SIZE[1], 1),
    (num_train, config.IMG_SIZE[0], config.IMG_SIZE[1], 1),
    (num_test, config.IMG_SIZE[0], config.IMG_SIZE[1], 1),
)




## === cell 6
if False:
    fig, ax = plt.subplots(4, 2, figsize=(15, 25))
    pass




## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    return images




## === cell 8
pipeline = []  # not used




## === cell 9
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        th, tw = int(config.IMG_SIZE[0]), int(config.IMG_SIZE[1])
        decoded = tf.image.resize_with_crop_or_pad(decoded, th, tw)
        decoded = tf.ensure_shape(decoded, (None, th, tw, 1))
        return decoded


autoencoder = DenoisingAutoencoder()
autoencoder.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)




## === cell 10
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

BATCH_SIZE = 24
train_fit_ds = (
    train_pair_ds.shuffle(buffer_size=num_train, seed=19, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

history = autoencoder.fit(
    train_fit_ds,
    shuffle=False,  # shuffle handled in dataset (equivalent to shuffle=True on arrays)
    callbacks=[es],
    epochs=500,
    verbose=1,
)




## === cell 11
if False:
    fig, ax = plt.subplots(figsize=(20, 6))
    pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
del history




## === cell 12
autoencoder.encoder.summary()
autoencoder.decoder.summary()




## === cell 13
if False:
    pass




## === cell 14
def _pad_or_crop_to(imgs_nhwc, target_hw):
    th, tw = int(target_hw[0]), int(target_hw[1])
    h = int(imgs_nhwc.shape[1])
    w = int(imgs_nhwc.shape[2])

    imgs = imgs_nhwc[:, : min(h, th), : min(w, tw), :]

    pad_h = th - imgs.shape[1]
    pad_w = tw - imgs.shape[2]
    if pad_h > 0 or pad_w > 0:
        imgs = np.pad(
            imgs,
            pad_width=((0, 0), (0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    return imgs[:, :th, :tw, :]


test_pred_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
pred_test = autoencoder.predict(test_pred_ds, verbose=1)
pred_test = _pad_or_crop_to(pred_test, config.IMG_SIZE)  # safety no-op
pred_test = np.clip(pred_test, 0.0, 1.0).astype(np.float32)
pred_test = np.squeeze(pred_test, axis=-1)  # (N,H,W)

H, W = config.IMG_SIZE
test_ids = np.array(
    [int(fn[:-4]) for fn in test_img], dtype=np.int32
)  # in the same order as test_paths/pred_test

rows = np.arange(1, H + 1, dtype=np.int32)
cols = np.arange(1, W + 1, dtype=np.int32)
grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")  # (H,W)
grid_r = grid_r.reshape(-1)
grid_c = grid_c.reshape(-1)

vals = pred_test.reshape(pred_test.shape[0], -1)  # (N, H*W)

out_path = "/kaggle/working/submission.csv"
with open(out_path, "w", newline="") as f:
    f.write("id,value\n")
    for i, img_id in enumerate(test_ids.tolist()):
        ids_img = (
            pd.Series(grid_r)
            .astype(str)
            .radd(f"{img_id}_")
            .str.cat(pd.Series(grid_c).astype(str), sep="_")
        )
        df_img = pd.DataFrame(
            {"id": ids_img.values, "value": vals[i].astype(np.float32)}
        )
        df_img.to_csv(f, index=False, header=False)

print("Saved submission.csv")
print("Path:", out_path)
print(pd.read_csv(out_path, nrows=5))
