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
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

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
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass




## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"

for zname in ["train.zip", "test.zip", "train_cleaned.zip", "sampleSubmission.csv.zip"]:
    zpath = os.path.join(path_zip, zname)
    expected = os.path.join(path, zname.replace(".zip", ""))
    expected_csv = os.path.join(path, "sampleSubmission.csv")
    if zname == "sampleSubmission.csv.zip":
        if os.path.exists(expected_csv):
            continue
    else:
        if os.path.isdir(expected):
            continue
    with zipfile.ZipFile(zpath, "r") as zip_ref:
        zip_ref.extractall(path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

train_img = sorted(os.listdir(train_dir))
train_cleaned_img = sorted(os.listdir(train_cleaned_dir))
test_img = sorted(os.listdir(test_dir))

print("Counts:", len(train_img), len(train_cleaned_img), len(test_img))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [
    cv2.imread(os.path.join(train_dir, f)) for f in sorted(os.listdir(train_dir))[:10]
]
print(
    "Median Dimensions (sample):",
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
from concurrent.futures import ThreadPoolExecutor


def _load_images_to_array(file_list, folder, n_workers=8):
    n = len(file_list)
    arr = np.empty((n, config.IMG_SIZE[0], config.IMG_SIZE[1], 1), dtype=np.float32)

    def _one(i_f):
        i, f = i_f
        arr[i] = process_image(os.path.join(folder, f))

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(
            tqdm(
                ex.map(_one, enumerate(file_list)),
                total=n,
                desc=f"Loading {os.path.basename(folder)}",
            )
        )
    return arr


train = _load_images_to_array(
    train_img, train_dir, n_workers=min(8, (os.cpu_count() or 2))
)
train_cleaned = _load_images_to_array(
    train_cleaned_img, train_cleaned_dir, n_workers=min(8, (os.cpu_count() or 2))
)
test = _load_images_to_array(
    test_img, test_dir, n_workers=min(8, (os.cpu_count() or 2))
)

train.shape, train_cleaned.shape, test.shape




## === cell 5
def augment_pipeline(pipeline, images, seed=19):
    processed_images = images.copy()
    for aug_fn in pipeline:
        temp = aug_fn(images)
        processed_images = np.concatenate([processed_images, temp], axis=0)
    return processed_images


def aug_rot90(images):
    return np.rot90(images, k=1, axes=(1, 2)).transpose(0, 2, 1, 3).copy()


def aug_rot180(images):
    return np.rot90(images, k=2, axes=(1, 2)).copy()


def aug_rot270(images):
    return np.rot90(images, k=3, axes=(1, 2)).transpose(0, 2, 1, 3).copy()


def aug_hflip(images):
    return images[:, :, ::-1, :].copy()


def aug_vflip(images):
    return images[:, ::-1, :, :].copy()




## === cell 6
pipeline = [aug_rot90, aug_rot180, aug_rot270, aug_hflip, aug_vflip]

processed_train = augment_pipeline(pipeline, train, seed=19)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned, seed=19)

print(processed_train.shape, processed_train_cleaned.shape)




## === cell 7
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.DepthwiseConv2D(
                    (3, 3), depth_multiplier=32, activation="relu", padding="same"
                ),
                layers.DepthwiseConv2D(
                    (3, 3), depth_multiplier=2, activation="relu", padding="same"
                ),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.DepthwiseConv2D(
                    (3, 3), depth_multiplier=1, activation="relu", padding="same"
                ),
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

rlp = callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.8, patience=5, min_lr=1e-15, mode="min", verbose=1
)

ds = tf.data.Dataset.from_tensor_slices((processed_train, processed_train_cleaned))
ds = ds.shuffle(
    buffer_size=len(processed_train), seed=19, reshuffle_each_iteration=True
)
ds = ds.cache().batch(12, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

history = autoencoder.fit(
    ds,
    callbacks=[es, rlp],
    epochs=500,
    verbose=2,
)




## === cell 9
del history




## === cell 10
pass




## === cell 11
pass




## === cell 12
batch_size = 4
decoded_test = autoencoder.predict(test, batch_size=batch_size, verbose=0).astype(
    np.float32
)  # (N,420,540,1)

sample_path = os.path.join(path, "sampleSubmission.csv")
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/denoising-dirty-documents/sampleSubmission.csv"
sample = pd.read_csv(sample_path)

test_ids = np.array([int(f[:-4]) for f in test_img], dtype=np.int32)
n_test = len(test_ids)
H0, W0 = config.IMG_SIZE
pixels_per_img = H0 * W0

vals = np.empty(len(sample), dtype=np.float32)

for i, fname in enumerate(test_img):
    img_id = test_ids[i]

    orig = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_GRAYSCALE)
    H, W = orig.shape
    del orig

    decoded_img = decoded_test[i, ..., 0]  # (420,540)
    pr = cv2.resize(decoded_img, (W, H), interpolation=cv2.INTER_LINEAR)
    pr = np.clip(pr, 0.0, 1.0).astype(np.float32)

    start = i * pixels_per_img
    end = start + pixels_per_img
    if (H, W) != (H0, W0):
        pr = cv2.resize(pr, (W0, H0), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    vals[start:end] = pr.reshape(-1)

sub_path = os.path.join(path, "submission.csv")
pd.DataFrame({"id": sample["id"].values, "value": vals}).to_csv(sub_path, index=False)
print(f"Results saved to {sub_path}!")
