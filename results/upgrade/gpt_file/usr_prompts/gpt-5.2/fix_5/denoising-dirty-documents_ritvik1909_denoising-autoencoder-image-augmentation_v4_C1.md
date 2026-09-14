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
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

np.random.seed(19)
tf.random.set_seed(19)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"


def _extract_if_missing(zip_path, target_dir, expected_subdir):
    if (
        not os.path.exists(os.path.join(target_dir, expected_subdir))
        or len(os.listdir(os.path.join(target_dir, expected_subdir))) == 0
    ):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)


_extract_if_missing(path_zip + "train.zip", path, "train")
_extract_if_missing(path_zip + "test.zip", path, "test")
_extract_if_missing(path_zip + "train_cleaned.zip", path, "train_cleaned")

if not os.path.exists(os.path.join(path, "sampleSubmission.csv")):
    with zipfile.ZipFile(path_zip + "sampleSubmission.csv.zip", "r") as zip_ref:
        zip_ref.extractall(path)

train_img = sorted(os.listdir(path + "/train"))
train_cleaned_img = sorted(os.listdir(path + "/train_cleaned"))
test_img = sorted(os.listdir(path + "/test"))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


train_files = sorted(os.listdir(path + "train/"))
heights = []
widths = []
for f in train_files:
    img = cv2.imread(path + "train/" + f)
    heights.append(img.shape[0])
    widths.append(img.shape[1])
del img
print("Median Dimensions:", np.median(heights), np.median(widths))
del heights, widths, train_files




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
from concurrent.futures import ThreadPoolExecutor


def _load_folder(folder_path):
    files = sorted(os.listdir(folder_path))
    full_paths = [os.path.join(folder_path, f) for f in files]
    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        imgs = list(ex.map(process_image, full_paths))
    return np.asarray(imgs, dtype=np.float32), files


train, _train_files = _load_folder(path + "train/")
train_cleaned, _trainc_files = _load_folder(path + "train_cleaned/")
test, _test_files = _load_folder(path + "test/")

train_img = sorted(_train_files)
train_cleaned_img = sorted(_trainc_files)
test_img = sorted(_test_files)

del _train_files, _trainc_files, _test_files



## === cell 5
train.shape, train_cleaned.shape, test.shape




## === cell 6
def augment_pipeline(images, seed=19):
    """
    Returns augmented set = original + [hflip, vflip, brightness/contrast jitter copies].
    All augmentations preserve (H,W) so concatenation is valid.
    """
    images_tf = tf.convert_to_tensor(images, dtype=tf.float32)

    hflip = tf.image.flip_left_right(images_tf)
    vflip = tf.image.flip_up_down(images_tf)

    jitter1 = tf.image.stateless_random_brightness(
        images_tf, max_delta=0.08, seed=[seed, 1]
    )
    jitter1 = tf.image.stateless_random_contrast(
        jitter1, lower=0.85, upper=1.15, seed=[seed, 2]
    )

    jitter2 = tf.image.stateless_random_brightness(
        images_tf, max_delta=0.05, seed=[seed, 3]
    )
    jitter2 = tf.image.stateless_random_contrast(
        jitter2, lower=0.90, upper=1.10, seed=[seed, 4]
    )

    augmented = tf.concat([images_tf, hflip, vflip, jitter1, jitter2], axis=0)
    augmented = tf.clip_by_value(augmented, 0.0, 1.0)
    return augmented.numpy()




## === cell 7
processed_train = augment_pipeline(train, seed=19)
processed_train_cleaned = augment_pipeline(train_cleaned, seed=19)

processed_train.shape, processed_train_cleaned.shape




## === cell 8
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
        return decoded


autoencoder = DenoisingAutoencoder()
autoencoder.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)



## === cell 9
BATCH_SIZE = 24
train_ds = tf.data.Dataset.from_tensor_slices(
    (processed_train, processed_train_cleaned)
)
train_ds = train_ds.shuffle(
    buffer_size=len(processed_train), seed=19, reshuffle_each_iteration=True
)
train_ds = (
    train_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)
)

es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    train_ds,
    callbacks=[es],
    epochs=500,
)



## === cell 10
_ = history.history  # no-op to ensure history exists
del history



## === cell 11
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 12
decoded_imgs = autoencoder(train[:4]).numpy()
print("Sanity decoded batch shape:", decoded_imgs.shape)
del decoded_imgs



## === cell 13
test_ids = [int(f[:-4]) for f in test_img]

test_hw = []
for fname in test_img:
    fp = os.path.join(path, "test", fname)
    im0 = cv2.imread(fp, 0)
    test_hw.append(im0.shape)  # (h,w)
del im0


@tf.function(reduce_retracing=True)
def _predict(x):
    return autoencoder(x, training=False)


batch_size = 8  # unchanged
n_test = test.shape[0]
out_path = "submission.csv"

_rc_cache = {}  # (h,w) -> (rr, cc) as strings


def _get_rc_strings(h, w):
    key = (h, w)
    cached = _rc_cache.get(key)
    if cached is not None:
        return cached
    r = np.arange(1, h + 1, dtype=np.int32)
    c = np.arange(1, w + 1, dtype=np.int32)
    rr = np.repeat(r, w).astype(str)
    cc = np.tile(c, h).astype(str)
    _rc_cache[key] = (rr, cc)
    return rr, cc


with open(out_path, "w", buffering=1024 * 1024) as f:
    f.write("id,value\n")

    for start in tqdm(range(0, n_test, batch_size)):
        end = min(n_test, start + batch_size)

        preds_batch = _predict(
            tf.convert_to_tensor(test[start:end], dtype=tf.float32)
        ).numpy()
        preds_batch = np.squeeze(preds_batch, axis=-1)  # (B,420,540)

        lines = []
        for bi in range(end - start):
            imgid = test_ids[start + bi]
            h, w = test_hw[start + bi]

            decoded_img = preds_batch[bi]
            preds_reshaped = cv2.resize(decoded_img, (w, h))
            preds_reshaped = (
                np.clip(preds_reshaped, 0.0, 1.0).reshape(-1).astype(np.float32)
            )

            rr_s, cc_s = _get_rc_strings(h, w)
            prefix = str(imgid) + "_"

            ids_arr = np.char.add(np.char.add(prefix, rr_s), "_")
            ids_arr = np.char.add(ids_arr, cc_s)

            vals_str = np.char.mod("%.6f", preds_reshaped)
            img_lines = np.char.add(np.char.add(ids_arr, ","), vals_str)
            lines.append("\n".join(img_lines.tolist()))

        f.write("\n".join(lines) + "\n")

print("Results saved to submission.csv!")
