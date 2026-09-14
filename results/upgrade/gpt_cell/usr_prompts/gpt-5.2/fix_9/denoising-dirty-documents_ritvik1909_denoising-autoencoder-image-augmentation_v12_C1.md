# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.02688

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.46129) has done: 'The crash happens immediately in cell 0 while importing `imgaug`, due to an incompatibility between `imgaug` and the installed `protobuf==6.33.0` (it triggers a removed protobuf API: `MessageFactory.GetPrototype`). The minimal fix is to avoid importing `imgaug` so the notebook can proceed; this is safe as long as augmentation is not strictly required at import-time. To preserve compatibility with later cells that may reference `ia`/`iaa`, this patch provides small placeholders that raise a clear error only if augmentation is actually invoked. No other logic (TensorFlow/model/training/data loading) is changed.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens at import time in cell 0 because `imgaug` is incompatible with the installed `protobuf==6.33.0` and raises `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during import. The current `try/except Exception:` does not reliably catch this failure before the exception aborts execution in this environment. We should avoid importing `imgaug` entirely (since it is optional here) and keep a safe stub so downstream references won’t crash unless actually used.

Patch summary: In cell 0, remove the eager `imgaug` import attempt and replace it with a deterministic “augmentation unavailable” stub (`ia=None`, `iaa` stub) unconditionally. This prevents the protobuf-related import crash while preserving the existing behavior that augmentation is unavailable in this environment.

Updated cells: Only cell 0 is changed.

Compatibility notes for cell k+1: Cell 1 does not depend on `imgaug`; variables and imports used there (`np`, `pd`, `zipfile`, `os`, `cv2`, etc.) remain unchanged, so execution continues normally.

Assumptions: Any later cell that tries to use `iaa.*` should already be prepared to run without augmentation (as suggested by the existing stub); if it does attempt to use augmentation, it raise a clear `ImportError` as before.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens during imports in cell 0 due to a protobuf incompatibility: packages that rely on older protobuf APIs (commonly `imgaug`/`tensorboard`/other deps) can trigger `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` with protobuf 6.x. This prevents the notebook from even reaching the existing fallback augmentation stub. The most minimal, deterministic fix is to force protobuf to use the pure-Python implementation (which retains the older behavior) before importing TensorFlow and anything that indirectly touches protobuf-generated descriptors.

Patch summary: Modify cell 0 only to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and its version flag) at the very top, before importing TensorFlow. Keep the rest of the logic unchanged, including the augmentation-unavailable stub, so downstream cells and variables remain identical.

Updated cells: Cell 0 only.

Compatibility notes for cell k+1: No variable names or interfaces are changed; TensorFlow import succeed so cell 1 can run as-is.

Assumptions: This environment allows switching protobuf to the Python implementation via environment variables, and performance impact is acceptable since the goal is to unblock execution.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens during imports in cell 0: `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known incompatibility between older code expecting `MessageFactory.GetPrototype` and the installed `protobuf==6.x`, and it can be triggered indirectly by importing TensorFlow/Keras in this environment. Since later cells rely on `tf` being importable (and augmentation is already disabled), the minimal fix is to pin protobuf to the pure-Python implementation and proactively add a small compatibility shim that defines `GetPrototype` in terms of `GetMessageClass` when missing.

Patch summary: In cell 0 only, add a safe monkey-patch for `google.protobuf.message_factory.MessageFactory.GetPrototype` before importing TensorFlow. This avoids the AttributeError without changing any model/training logic and keeps the existing “imgaug unavailable” behavior intact.

Updated cells: cell 0 only.

Compatibility notes for cell k+1: Cell 1 and later still receive the same variables (`np`, `pd`, `tf`, `layers`, etc.) with unchanged semantics; only the protobuf import path is patched to prevent the crash.

Assumptions: The error is raised during import-time protobuf usage (triggered by TensorFlow/Keras or another imported dependency), and adding the missing method is sufficient to make imports succeed under `protobuf==6.33.0`.'
- What this solution (achieved 0.28616) has done: 'Cell 8 is crashing because `iaa` is intentionally set to an `_ImgAugUnavailable` stub (due to an `imgaug` import failure earlier), and any attribute access like `iaa.Rot90(...)` raises `ImportError`. To unblock execution without changing the training/model logic, we keep the same variable names (`rotate90`, `seq_rp`, etc.) but provide no-op augmentation objects when `imgaug` is unavailable. These no-op objects implement `.augment_images(...)` so cell 7’s `augment_pipeline()` continues to work and simply returns unchanged images. This patch is localized to cell 8 and preserves interfaces used by cell 9 and later cells.'
- What this solution (achieved 0.28616) has done: 'The crash happens because `ia` was explicitly set to `None` in an earlier cell (because imgaug can’t be imported), but `augment_pipeline()` still unconditionally calls `ia.seed(...)`, causing an AttributeError. Since your augmentation objects are already swapped to no-op implementations when imgaug is unavailable, the minimal safe fix is to make `augment_pipeline()` seed only when `ia` exists and exposes `seed`. This preserves identical behavior when imgaug is available, and allows the notebook to run deterministically (no augmentation changes) when it is not. No other logic, shapes, or downstream interfaces are changed.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by two places: (1) augmentation uses repeated `np.append` (quadratic time + memory reallocations), and (2) submission building loops over every pixel in Python (millions of iterations). I make augmentation provably equivalent by preallocating the final array and filling it, and I generate the submission using vectorized NumPy (same ids/values ordering) while batching model inference to reduce per-image overhead. I also switch training/inference inputs to efficient `tf.data` pipelines (same data, same shuffling semantics, same epochs/batch size) and ensure deterministic seeds so results remain stable.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) a very expensive `shuffle(buffer_size=len(processed_train))` that forces huge in-memory shuffles each epoch and (2) extremely slow submission building using per-pixel pandas `Series.astype(str)` inside a Python loop (millions of string conversions). I keep the exact same model, loss, training loop semantics, and augmentation outputs, but make the data pipeline cheaper by using a fixed shuffle buffer (still randomized with the same seed) and enable TF `prefetch_to_device` when GPU is available. For submission, I generate ids using fast NumPy string ops and stream-write the CSV in chunks to avoid building 5.8M ids/values in RAM and to eliminate pandas string overhead; the produced CSV is identical in content/format. I also ensure OpenCV reads grayscale directly during preprocessing to cut I/O and conversion cost without changing pixel values.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

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
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

ia = None


class _ImgAugUnavailable:
    def __getattr__(self, name):
        raise ImportError(
            "imgaug could not be imported due to a protobuf incompatibility in this environment. "
            "Augmentation is unavailable unless the environment is changed."
        )


iaa = _ImgAugUnavailable()

sns.set_style("darkgrid")

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


imgs = [cv2.imread(path + "train/" + f) for f in sorted(os.listdir(path + "train/"))]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs




## === cell 3
def process_image(path_):
    img = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)  # (H,W), uint8
    img = cv2.resize(
        img, config.IMG_SIZE[::-1]
    )  # (540,420) -> (420,540) after resize target
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
train_files = sorted(os.listdir(path + "train/"))
train_cleaned_files = sorted(os.listdir(path + "train_cleaned/"))
test_files = sorted(os.listdir(path + "test/"))

n_train = len(train_files)
n_train_cleaned = len(train_cleaned_files)
n_test = len(test_files)

train = np.empty((n_train, *config.IMG_SIZE, 1), dtype=np.float32)
train_cleaned = np.empty((n_train_cleaned, *config.IMG_SIZE, 1), dtype=np.float32)
test = np.empty((n_test, *config.IMG_SIZE, 1), dtype=np.float32)

for i, f in enumerate(train_files):
    train[i] = process_image(path + "train/" + f)

for i, f in enumerate(train_cleaned_files):
    train_cleaned[i] = process_image(path + "train_cleaned/" + f)

for i, f in enumerate(test_files):
    test[i] = process_image(path + "test/" + f)



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
def augment_pipeline(pipeline, images, seed=19):
    ia.seed(seed)
    processed_images = images.copy()
    for step in pipeline:
        temp = np.array(step.augment_images(images))
        processed_images = np.append(processed_images, temp, axis=0)
    return processed_images




## === cell 8
try:
    _ = iaa.Rot90  # will raise ImportError via _ImgAugUnavailable.__getattr__
    _IMG_AUG_AVAILABLE = True
except Exception:
    _IMG_AUG_AVAILABLE = False

if not _IMG_AUG_AVAILABLE:

    class _NoOpAugmenter:
        def __init__(self, *args, **kwargs):
            pass

        def augment_images(self, images):
            return np.array(images, copy=True)

    class _NoOpSequential(_NoOpAugmenter):
        pass

    rotate90 = _NoOpAugmenter()  # rotate image 90 degrees
    rotate180 = _NoOpAugmenter()  # rotate image 180 degrees
    rotate270 = _NoOpAugmenter()  # rotate image 270 degrees
    random_rotate = _NoOpAugmenter()  # randomly rotate image from 90,180,270 degrees
    perc_transform = _NoOpAugmenter()  # perspective transform
    rotate10 = _NoOpAugmenter()  # rotate image 10 degrees
    rotate10r = _NoOpAugmenter()  # rotate image -10 degrees
    crop = _NoOpAugmenter()  # crop
    hflip = _NoOpAugmenter()  # horizontal flip
    vflip = _NoOpAugmenter()  # vertical flip
    gblur = _NoOpAugmenter()  # gaussian blur
    motionblur = _NoOpAugmenter()  # motion blur

    seq_rp = _NoOpSequential()
    seq_cfg = _NoOpSequential()
    seq_fm = _NoOpSequential()
else:
    rotate90 = iaa.Rot90(1)  # rotate image 90 degrees
    rotate180 = iaa.Rot90(2)  # rotate image 180 degrees
    rotate270 = iaa.Rot90(3)  # rotate image 270 degrees
    random_rotate = iaa.Rot90((1, 3))  # randomly rotate image from 90,180,270 degrees
    perc_transform = iaa.PerspectiveTransform(
        scale=(0.02, 0.1)
    )  # Skews and transform images without black bg
    rotate10 = iaa.Affine(rotate=(10))  # rotate image 10 degrees
    rotate10r = iaa.Affine(rotate=(-10))  # rotate image 30 degrees in reverse
    crop = iaa.Crop(px=(5, 32))  # Crop between 5 to 32 pixels
    hflip = iaa.Fliplr(1)  # horizontal flips for 100% of images
    vflip = iaa.Flipud(1)  # vertical flips for 100% of images
    gblur = iaa.GaussianBlur(
        sigma=(1, 1.5)
    )  # gaussian blur images with a sigma of 1.0 to 1.5
    motionblur = iaa.MotionBlur(8)  # motion blur images with a kernel size 8

    seq_rp = iaa.Sequential(
        [
            iaa.Rot90((1, 3)),  # randomly rotate image from 90,180,270 degrees
            iaa.PerspectiveTransform(
                scale=(0.02, 0.1)
            ),  # Skews and transform images without black bg
        ]
    )

    seq_cfg = iaa.Sequential(
        [
            iaa.Crop(
                px=(5, 32)
            ),  # crop images from each side by 5 to 32px (randomly chosen)
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.GaussianBlur(sigma=(0, 1.5)),  # blur images with a sigma of 0 to 1.5
        ]
    )

    seq_fm = iaa.Sequential(
        [
            iaa.Flipud(1),  # vertical flips all the images
            iaa.MotionBlur(k=6),  # motion blur images with a kernel size 6
        ]
    )



## === cell 9
pipeline = [rotate90, rotate180, rotate270, hflip, vflip]




## === cell 10
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
        out[write : write + n] = temp
        write += n
    return out


processed_train = augment_pipeline(pipeline, train, seed=SEED)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned, seed=SEED)

processed_train.shape, processed_train_cleaned.shape




## === cell 11
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
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)



## === cell 12
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

BATCH_SIZE = 12

_shuffle_buf = min(256, len(processed_train))

train_ds = tf.data.Dataset.from_tensor_slices(
    (processed_train, processed_train_cleaned)
).cache()
train_ds = train_ds.shuffle(
    buffer_size=_shuffle_buf, seed=SEED, reshuffle_each_iteration=True
).batch(BATCH_SIZE, drop_remainder=False)

try:
    gpus = tf.config.list_logical_devices("GPU")
    if gpus:
        train_ds = train_ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
except Exception:
    pass
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

history = autoencoder.fit(train_ds, callbacks=[es], epochs=500, verbose=1)



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
import csv

test_shapes = []
test_ids = []
for f in test_img:
    imgid = int(f[:-4])
    img = cv2.imread(path + "test/" + f, cv2.IMREAD_GRAYSCALE)
    test_shapes.append(img.shape)  # (h,w)
    test_ids.append(imgid)

TEST_BATCH = 4  # keep as in original attempt to avoid memory spikes; outputs preserved.
pred_batches = []
for i in range(0, len(test), TEST_BATCH):
    pred = autoencoder(
        test[i : i + TEST_BATCH], training=False
    ).numpy()  # (b,420,540,1)
    pred_batches.append(pred)
preds = np.concatenate(pred_batches, axis=0)
del pred_batches

with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])

    for i, imgid in tqdm(list(enumerate(test_ids)), total=len(test_ids)):
        h, w = test_shapes[i]
        decoded_img = np.squeeze(preds[i])  # (420,540)
        preds_reshaped = cv2.resize(decoded_img, (w, h))  # (h,w), float32

        r = np.repeat(np.arange(1, h + 1, dtype=np.int32), w)
        c = np.tile(np.arange(1, w + 1, dtype=np.int32), h)

        ids = np.char.add(
            np.char.add(str(imgid) + "_", r.astype(str)),
            np.char.add("_", c.astype(str)),
        )

        vals = preds_reshaped.reshape(-1)

        writer.writerows(zip(ids.tolist(), vals.tolist()))

print("Results saved to submission.csv!")
