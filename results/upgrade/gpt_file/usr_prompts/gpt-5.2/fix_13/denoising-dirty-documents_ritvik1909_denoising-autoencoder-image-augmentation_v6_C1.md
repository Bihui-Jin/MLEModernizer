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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

np.random.seed(19)
tf.random.set_seed(19)

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
path_zip_candidates = [
    "/kaggle/input/denoising-dirty-documents/",
    "../input/denoising-dirty-documents/",
    "/kaggle/input/denoising-dirty-documents/denoising-dirty-documents/",
]
path_zip = None
for p in path_zip_candidates:
    if os.path.exists(p) and (
        os.path.exists(os.path.join(p, "train.zip"))
        or os.path.exists(os.path.join(p, "train/"))
    ):
        path_zip = p
        break
if path_zip is None:
    raise FileNotFoundError(
        "Could not find denoising-dirty-documents input folder in expected locations."
    )

path = "/kaggle/working/"


def _safe_extract(zip_path, out_dir):
    if not os.path.exists(zip_path):
        return
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)


if os.path.exists(os.path.join(path_zip, "train.zip")):
    _safe_extract(os.path.join(path_zip, "train.zip"), path)
    _safe_extract(os.path.join(path_zip, "test.zip"), path)
    _safe_extract(os.path.join(path_zip, "train_cleaned.zip"), path)
    _safe_extract(os.path.join(path_zip, "sampleSubmission.csv.zip"), path)

if (
    os.path.exists(os.path.join(path, "train"))
    and len(os.listdir(os.path.join(path, "train"))) > 0
):
    base_img_path = path
else:
    base_img_path = path_zip if path_zip.endswith("/") else path_zip + "/"


def _img_id(fname: str) -> int:
    return int(os.path.splitext(fname)[0])


train_files = [
    f
    for f in os.listdir(os.path.join(base_img_path, "train"))
    if f.lower().endswith(".png")
]
train_cleaned_files = [
    f
    for f in os.listdir(os.path.join(base_img_path, "train_cleaned"))
    if f.lower().endswith(".png")
]
test_files = [
    f
    for f in os.listdir(os.path.join(base_img_path, "test"))
    if f.lower().endswith(".png")
]

train_map = {_img_id(f): f for f in train_files}
clean_map = {_img_id(f): f for f in train_cleaned_files}

common_ids = sorted(set(train_map.keys()) & set(clean_map.keys()))
if len(common_ids) == 0:
    raise RuntimeError("No overlapping image ids between train and train_cleaned.")

missing_in_clean = sorted(set(train_map.keys()) - set(clean_map.keys()))
missing_in_train = sorted(set(clean_map.keys()) - set(train_map.keys()))
if missing_in_clean or missing_in_train:
    raise RuntimeError(
        f"Train/train_cleaned mismatch. Missing in cleaned: {missing_in_clean[:10]} "
        f"(+{max(0, len(missing_in_clean)-10)} more). Missing in train: {missing_in_train[:10]} "
        f"(+{max(0, len(missing_in_train)-10)} more)."
    )

train_img = [train_map[i] for i in common_ids]
train_cleaned_img = [clean_map[i] for i in common_ids]
test_img = sorted(test_files, key=_img_id)

print("Base image path:", base_img_path)
print(
    "Num train:",
    len(train_img),
    "Num train_cleaned:",
    len(train_cleaned_img),
    "Num test:",
    len(test_img),
)




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [cv2.imread(os.path.join(base_img_path, "train", f)) for f in train_img[:10]]
print(
    "Median Dimensions (sample of 10):",
    int(np.median([img.shape[0] for img in imgs])),
    int(np.median([img.shape[1] for img in imgs])),
)
del imgs




## === cell 3
def process_image(p):
    img = cv2.imread(p)
    if img is None:
        raise RuntimeError(f"Failed to read image: {p}")
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1)).astype("float32", copy=False)
    return img




## === cell 4
n_train = len(train_img)
n_test = len(test_img)

train = np.empty((n_train, config.IMG_SIZE[0], config.IMG_SIZE[1], 1), dtype=np.float32)
train_cleaned = np.empty_like(train)
test = np.empty((n_test, config.IMG_SIZE[0], config.IMG_SIZE[1], 1), dtype=np.float32)

for idx, f in enumerate(train_img):
    train[idx] = process_image(os.path.join(base_img_path, "train", f))

for idx, f in enumerate(train_cleaned_img):
    train_cleaned[idx] = process_image(os.path.join(base_img_path, "train_cleaned", f))

for idx, f in enumerate(test_img):
    test[idx] = process_image(os.path.join(base_img_path, "test", f))

print("Shapes:", train.shape, train_cleaned.shape, test.shape)




## === cell 5
def augment_pipeline(pipeline, images, seed=19):
    rng = np.random.default_rng(seed)
    processed_images = images.copy()
    for step in pipeline:
        temp = step(processed_images, rng)
        processed_images = np.append(processed_images, temp, axis=0)
    return processed_images


def _resize_back(imgs):
    if imgs.shape[1:3] == config.IMG_SIZE:
        return imgs
    out = np.empty(
        (imgs.shape[0], config.IMG_SIZE[0], config.IMG_SIZE[1], imgs.shape[3]),
        dtype=imgs.dtype,
    )
    for i in range(imgs.shape[0]):
        out[i, ..., 0] = cv2.resize(
            imgs[i, ..., 0], config.IMG_SIZE[::-1], interpolation=cv2.INTER_LINEAR
        )
    return out


def aug_rot90_k(k):
    def _fn(images, rng):
        rotated = np.rot90(images, k=k, axes=(1, 2)).copy()
        return _resize_back(rotated)

    return _fn


def aug_hflip(images, rng):
    return images[:, :, ::-1, :].copy()


def aug_vflip(images, rng):
    return images[:, ::-1, :, :].copy()


def augment_pair_pipeline(pipeline, x_images, y_images, seed=19):
    rng = np.random.default_rng(seed)

    processed_x = x_images.copy()
    processed_y = y_images.copy()
    for step in pipeline:
        temp_x = step(processed_x, rng)
        temp_y = step(processed_y, rng)
        processed_x = np.append(processed_x, temp_x, axis=0)
        processed_y = np.append(processed_y, temp_y, axis=0)

    return processed_x, processed_y




## === cell 6
pipeline = [aug_rot90_k(1), aug_rot90_k(2), aug_rot90_k(3), aug_hflip, aug_vflip]

processed_train, processed_train_cleaned = augment_pair_pipeline(
    pipeline, train, train_cleaned, seed=19
)
processed_train = processed_train.astype("float32", copy=False)
processed_train_cleaned = processed_train_cleaned.astype("float32", copy=False)

print("Augmented shapes:", processed_train.shape, processed_train_cleaned.shape)




## === cell 7
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
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
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
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



## === cell 8
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    processed_train,
    processed_train_cleaned,
    shuffle=True,
    callbacks=[es],
    epochs=500,
    batch_size=12,
)

print("Final training loss:", float(history.history["loss"][-1]))
print("Final training MAE:", float(history.history["mean_absolute_error"][-1]))



## === cell 9
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 10
decoded_imgs = autoencoder(train[:2]).numpy()
print("Decoded batch shape:", decoded_imgs.shape)
del decoded_imgs



## === cell 11
pred_batch_size = 8
decoded_test = autoencoder.predict(test, batch_size=pred_batch_size, verbose=0).astype(
    np.float32, copy=False
)
print("Decoded test shape:", decoded_test.shape)

sub_path = "submission.csv"

h, w = config.IMG_SIZE
rows = np.repeat(np.arange(1, h + 1, dtype=np.int32), w)
cols = np.tile(np.arange(1, w + 1, dtype=np.int32), h)
rows_s = rows.astype(str)
cols_s = cols.astype(str)

with open(sub_path, "w", buffering=8 * 1024 * 1024) as f:
    f.write("id,value\n")
    for i, fname in enumerate(tqdm(test_img, total=len(test_img))):
        imgid = int(os.path.splitext(fname)[0])

        decoded_img = np.squeeze(decoded_test[i]).astype("float32", copy=False)
        if decoded_img.shape != (h, w):
            preds_img = cv2.resize(decoded_img, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            preds_img = decoded_img
        pv = np.clip(preds_img, 0.0, 1.0).astype("float32", copy=False).reshape(-1)

        id_str = np.char.add(
            np.char.add(str(imgid) + "_", rows_s),
            np.char.add("_", cols_s),
        )
        val_str = np.char.mod("%.8f", pv)
        lines = np.char.add(np.char.add(id_str, ","), val_str)
        f.write("\n".join(lines.tolist()) + "\n")

print(f"Results saved to {sub_path}!")

sub = pd.read_csv(sub_path)
assert list(sub.columns) == ["id", "value"]
print("Submission rows:", len(sub))
print(sub.head())
