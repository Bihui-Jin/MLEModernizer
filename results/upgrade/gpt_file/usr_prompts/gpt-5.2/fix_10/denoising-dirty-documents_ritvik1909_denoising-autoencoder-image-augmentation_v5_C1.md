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
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

sns.set_style("darkgrid")

np.random.seed(19)
tf.random.set_seed(19)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass




## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"

os.makedirs(path, exist_ok=True)

with zipfile.ZipFile(os.path.join(path_zip, "train.zip"), "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(os.path.join(path_zip, "test.zip"), "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(os.path.join(path_zip, "train_cleaned.zip"), "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(
    os.path.join(path_zip, "sampleSubmission.csv.zip"), "r"
) as zip_ref:
    zip_ref.extractall(path)

train_img = sorted(os.listdir(os.path.join(path, "train")))
train_cleaned_img = sorted(os.listdir(os.path.join(path, "train_cleaned")))
test_img = sorted(os.listdir(os.path.join(path, "test")))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


_subset_files = sorted(os.listdir(os.path.join(path, "train")))[:10]
imgs = [cv2.imread(os.path.join(path, "train", f)) for f in _subset_files]
print(
    "Median Dimensions (subset of 10):",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs, _subset_files




## === cell 3
def process_image(path_):
    img = cv2.imread(path_)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
train_files = sorted(os.listdir(os.path.join(path, "train")))
train_cleaned_files = sorted(os.listdir(os.path.join(path, "train_cleaned")))
test_files = sorted(os.listdir(os.path.join(path, "test")))

train = np.empty((len(train_files), *config.IMG_SIZE, 1), dtype=np.float32)
train_cleaned = np.empty(
    (len(train_cleaned_files), *config.IMG_SIZE, 1), dtype=np.float32
)
test = np.empty((len(test_files), *config.IMG_SIZE, 1), dtype=np.float32)

for i, f in enumerate(train_files):
    train[i] = process_image(os.path.join(path, "train", f))

for i, f in enumerate(train_cleaned_files):
    train_cleaned[i] = process_image(os.path.join(path, "train_cleaned", f))

for i, f in enumerate(test_files):
    test[i] = process_image(os.path.join(path, "test", f))




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




## === cell 7
def _resize_back(images, target_hw):
    out = np.empty(
        (images.shape[0], target_hw[0], target_hw[1], images.shape[-1]),
        dtype=images.dtype,
    )
    for i in range(images.shape[0]):
        out[i, ..., 0] = cv2.resize(
            images[i, ..., 0],
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return out


def _rot90(images, k):
    rotated = np.rot90(images, k=k, axes=(1, 2)).copy()
    if rotated.shape[1] != config.IMG_SIZE[0] or rotated.shape[2] != config.IMG_SIZE[1]:
        rotated = _resize_back(rotated, config.IMG_SIZE)
    return rotated


def _hflip(images):
    return np.flip(images, axis=2).copy()  # flip width


def _vflip(images):
    return np.flip(images, axis=1).copy()  # flip height


def augment_pipeline(pipeline, images, seed=19):
    processed_images = images.copy()
    for step in pipeline:
        temp = np.asarray(step(images), dtype=processed_images.dtype)
        if temp.shape[1:] != processed_images.shape[1:]:
            raise ValueError(
                f"Augmentation produced shape {temp.shape} but expected {processed_images.shape}"
            )
        processed_images = np.concatenate([processed_images, temp], axis=0)
    return processed_images




## === cell 8
rotate90 = lambda x: _rot90(x, 1)
rotate180 = lambda x: _rot90(x, 2)
rotate270 = lambda x: _rot90(x, 3)
hflip = lambda x: _hflip(x)
vflip = lambda x: _vflip(x)




## === cell 9
pipeline = [rotate90, rotate180, rotate270, hflip, vflip]




## === cell 10
processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned)

processed_train.shape, processed_train_cleaned.shape




## === cell 11
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(192, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(192, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
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




## === cell 12
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

batch_size = 12
train_ds = (
    tf.data.Dataset.from_tensor_slices((processed_train, processed_train_cleaned))
    .shuffle(buffer_size=len(processed_train), seed=19, reshuffle_each_iteration=True)
    .batch(batch_size, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

history = autoencoder.fit(
    train_ds,
    callbacks=[es],
    epochs=500,
)




## === cell 13
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
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

del decoded_imgs




## === cell 16

test_img = sorted(test_img, key=lambda x: int(os.path.splitext(x)[0]))

test_files_loaded = sorted(os.listdir(os.path.join(path, "test")))
idx_map = {fn: i for i, fn in enumerate(test_files_loaded)}
test = test[[idx_map[fn] for fn in test_img]]


def _get_hw_fast(filepath):
    im = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return im.shape  # (H, W)


_suffix_cache = {}  # (H,W) -> np.ndarray of strings like "_r_c"


def _get_suffixes(H, W):
    key = (H, W)
    if key in _suffix_cache:
        return _suffix_cache[key]
    r = np.repeat(np.arange(1, H + 1, dtype=np.int32), W).astype(str)
    c = np.tile(np.arange(1, W + 1, dtype=np.int32), H).astype(str)
    suffix = np.char.add(np.char.add("_", r), np.char.add("_", c))  # "_r_c"
    _suffix_cache[key] = suffix
    return suffix


out_path = "submission.csv"
with open(out_path, "w") as f:
    f.write("id,value\n")

batch = 8  # does not change predictions, only groups inference calls
n = len(test_img)

for start in tqdm(range(0, n, batch)):
    end = min(start + batch, n)
    batch_imgs = test[start:end]
    decoded_batch = autoencoder(batch_imgs, training=False).numpy()  # (B,420,540,1)

    for j in range(end - start):
        fname = test_img[start + j]
        file = os.path.join(path, "test", fname)
        imgid = int(os.path.splitext(fname)[0])

        H, W = _get_hw_fast(file)

        decoded_img = decoded_batch[j, ..., 0]  # (420,540)
        preds_reshaped = cv2.resize(decoded_img, (W, H), interpolation=cv2.INTER_LINEAR)
        preds_reshaped = (
            np.clip(preds_reshaped, 0.0, 1.0).astype(np.float32).reshape(-1)
        )

        suffix = _get_suffixes(H, W)
        ids = np.char.add(str(imgid), suffix)

        df = pd.DataFrame({"id": ids, "value": preds_reshaped})
        df.to_csv(out_path, mode="a", header=False, index=False)

print(f"Results saved to {out_path}!")
