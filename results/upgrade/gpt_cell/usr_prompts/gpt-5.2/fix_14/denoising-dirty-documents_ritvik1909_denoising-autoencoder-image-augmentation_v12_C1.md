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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers

DO_PLOTS = False

ia = None


class _ImgAugUnavailable:
    def __getattr__(self, name):
        raise ImportError(
            "imgaug could not be imported due to a protobuf incompatibility in this environment. "
            "Augmentation is unavailable unless the environment is changed."
        )


iaa = _ImgAugUnavailable()

SEED = 19
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass




## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"


def _ensure_extracted(zip_path, out_dir, sentinel_dir):
    if os.path.isdir(sentinel_dir) and len(os.listdir(sentinel_dir)) > 0:
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


_ensure_extracted(path_zip + "train.zip", path, os.path.join(path, "train"))
_ensure_extracted(path_zip + "test.zip", path, os.path.join(path, "test"))
_ensure_extracted(
    path_zip + "train_cleaned.zip", path, os.path.join(path, "train_cleaned")
)
_ensure_extracted(
    path_zip + "sampleSubmission.csv.zip",
    path,
    os.path.join(path, "sampleSubmission.csv"),
)

train_img = sorted(os.listdir(path + "/train"))
train_cleaned_img = sorted(os.listdir(path + "/train_cleaned"))
test_img = sorted(os.listdir(path + "/test"))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


print("Median Dimensions: (skipped to save time; not used in pipeline)")




## === cell 3
def process_image(path_):
    img = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)  # (H,W), uint8
    img = cv2.resize(img, config.IMG_SIZE[::-1])  # (w,h)
    img = img.astype(np.float32, copy=False) / 255.0
    img = img.reshape((*config.IMG_SIZE, 1))
    return np.ascontiguousarray(img)




## === cell 4
from concurrent.futures import ThreadPoolExecutor

train_files = sorted(os.listdir(path + "train/"))
train_cleaned_files = sorted(os.listdir(path + "train_cleaned/"))
test_files = sorted(os.listdir(path + "test/"))

n_train = len(train_files)
n_train_cleaned = len(train_cleaned_files)
n_test = len(test_files)

train_paths = [path + "train/" + f for f in train_files]
train_cleaned_paths = [path + "train_cleaned/" + f for f in train_cleaned_files]
test_paths = [path + "test/" + f for f in test_files]

train = np.empty((n_train, *config.IMG_SIZE, 1), dtype=np.float32)
train_cleaned = np.empty((n_train_cleaned, *config.IMG_SIZE, 1), dtype=np.float32)
test = np.empty((n_test, *config.IMG_SIZE, 1), dtype=np.float32)

max_workers = min(8, (os.cpu_count() or 8))

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, img_arr in enumerate(ex.map(process_image, train_paths, chunksize=16)):
        train[i] = img_arr

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, img_arr in enumerate(
        ex.map(process_image, train_cleaned_paths, chunksize=16)
    ):
        train_cleaned[i] = img_arr

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, img_arr in enumerate(ex.map(process_image, test_paths, chunksize=16)):
        test[i] = img_arr




## === cell 5
train.shape, train_cleaned.shape, test.shape




## === cell 6
if DO_PLOTS:
    import matplotlib.pyplot as plt

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
try:
    _ = iaa.Rot90  # will raise ImportError via _ImgAugUnavailable.__getattr__
    _IMG_AUG_AVAILABLE = True
except Exception:
    _IMG_AUG_AVAILABLE = False

if not _IMG_AUG_AVAILABLE:

    class _NP_Rot90:
        def __init__(self, k):
            self.k = int(k)

        def augment_images(self, images):
            return np.rot90(images, k=self.k, axes=(1, 2)).copy()

    class _NP_Fliplr:
        def __init__(self, p=1.0):
            self.p = float(p)

        def augment_images(self, images):
            if self.p >= 1.0:
                return images[:, :, ::-1, :].copy()
            rng = np.random.RandomState(SEED)
            out = images.copy()
            m = rng.rand(images.shape[0]) < self.p
            out[m] = out[m, :, ::-1, :]
            return out

    class _NP_Flipud:
        def __init__(self, p=1.0):
            self.p = float(p)

        def augment_images(self, images):
            if self.p >= 1.0:
                return images[:, ::-1, :, :].copy()
            rng = np.random.RandomState(SEED)
            out = images.copy()
            m = rng.rand(images.shape[0]) < self.p
            out[m] = out[m, ::-1, :, :]
            return out

    rotate90 = _NP_Rot90(1)
    rotate180 = _NP_Rot90(2)
    rotate270 = _NP_Rot90(3)

    hflip = _NP_Fliplr(1.0)
    vflip = _NP_Flipud(1.0)

else:
    rotate90 = iaa.Rot90(1)
    rotate180 = iaa.Rot90(2)
    rotate270 = iaa.Rot90(3)
    hflip = iaa.Fliplr(1)
    vflip = iaa.Flipud(1)




## === cell 8
pipeline = [rotate90, rotate180, rotate270, hflip, vflip]




## === cell 9
def augment_pipeline(pipeline, images, seed=19):
    if ia is not None and hasattr(ia, "seed"):
        ia.seed(seed)
    n = images.shape[0]
    k = len(pipeline)
    out = np.empty((n * (k + 1),) + images.shape[1:], dtype=images.dtype)
    out[:n] = images
    write = n
    for step in pipeline:
        temp = np.asarray(step.augment_images(images), dtype=images.dtype)

        if temp.shape[1:] != images.shape[1:]:
            if (
                temp.ndim == images.ndim
                and temp.shape[0] == n
                and temp.shape[1] == images.shape[2]
                and temp.shape[2] == images.shape[1]
                and temp.shape[3:] == images.shape[3:]
            ):
                temp = np.transpose(temp, (0, 2, 1) + tuple(range(3, temp.ndim)))
            else:
                raise ValueError(
                    f"Augmentation produced unexpected shape {temp.shape}; expected {images.shape} or swapped H/W."
                )

        out[write : write + n] = temp
        write += n
    return out


processed_train = augment_pipeline(pipeline, train, seed=SEED)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned, seed=SEED)

processed_train.shape, processed_train_cleaned.shape




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




## === cell 11
BATCH_SIZE = 12

_shuffle_buf = min(256, len(processed_train))

opts = tf.data.Options()
opts.deterministic = True

processed_train = np.ascontiguousarray(processed_train)
processed_train_cleaned = np.ascontiguousarray(processed_train_cleaned)

train_ds = tf.data.Dataset.from_tensor_slices(
    (processed_train, processed_train_cleaned)
).with_options(opts)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(
    buffer_size=_shuffle_buf, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

history = autoencoder.fit(train_ds, callbacks=[], epochs=500, verbose=1)




## === cell 12
if DO_PLOTS:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 6))
    pd.DataFrame(history.history).plot(ax=ax)
del history




## === cell 13
if DO_PLOTS:
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()




## === cell 14
if DO_PLOTS:
    import matplotlib.pyplot as plt

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




## === cell 15
import csv

test_shapes = []
test_ids = []
for f in test_img:
    imgid = int(f[:-4])
    img = cv2.imread(path + "test/" + f, cv2.IMREAD_GRAYSCALE)
    test_shapes.append(img.shape)  # (h,w)
    test_ids.append(imgid)

TEST_BATCH = 4  # unchanged
pred_batches = []
for i in range(0, len(test), TEST_BATCH):
    pred = autoencoder(test[i : i + TEST_BATCH], training=False).numpy()
    pred_batches.append(pred)
preds = np.concatenate(pred_batches, axis=0)
del pred_batches

_suffix_cache = {}  # (h,w) -> np.ndarray(dtype='<U..') like ["1_1","1_2",...]
CHUNK = 1 << 17  # 131072 rows/chunk: fewer writer calls, identical output.

with open("submission.csv", "w", newline="", buffering=8 * 1024 * 1024) as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])

    for i, imgid in tqdm(list(enumerate(test_ids)), total=len(test_ids)):
        h, w = test_shapes[i]
        decoded_img = np.squeeze(preds[i])  # (420,540)
        preds_reshaped = cv2.resize(decoded_img, (w, h))  # (h,w), float32

        key = (h, w)
        suffix = _suffix_cache.get(key)
        if suffix is None:
            r = np.repeat(np.arange(1, h + 1, dtype=np.int32), w).astype(str)
            c = np.tile(np.arange(1, w + 1, dtype=np.int32), h).astype(str)
            suffix = np.char.add(np.char.add(r, "_"), c)
            _suffix_cache[key] = suffix

        prefix = f"{imgid}_"
        ids = np.char.add(prefix, suffix)

        vals = preds_reshaped.reshape(-1).astype(np.float32, copy=False)

        total = vals.size
        for j in range(0, total, CHUNK):
            jj = min(j + CHUNK, total)
            writer.writerows(zip(ids[j:jj], vals[j:jj]))

print("Results saved to submission.csv!")
