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

0.0429

# 6. Current score

0.17478

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'The crash happens during the `import imgaug` step in cell 0 because the installed `protobuf==6.33.0` is incompatible with the version of `imgaug` available in this environment, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The minimal fix is to pin protobuf to the legacy Python implementation at runtime before importing TensorFlow and imgaug, which avoids the failing compiled-protobuf code path. This keeps all subsequent code unchanged while making the imports deterministic and compatible. No other cells need modification, and all variables used in cell 1 remain unaffected.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 0 due to an incompatibility between `protobuf==6.33.0` and TensorFlow 2.18 in this environment, manifesting as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The existing line setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is not sufficient to avoid the C++/protobuf API mismatch. Forcing TensorFlow to use the pure-Python protobuf implementation from process start also requires setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing TensorFlow.

Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` to `"2"` alongside the existing environment variable before importing TensorFlow, keeping all other logic unchanged.

Updated cells: Cell 0 only.

Compatibility notes for cell k+1: No variables, imports, or interfaces used by cell 1 are changed; only the protobuf runtime selection is adjusted so TensorFlow can import successfully.

Assumptions: The environment honors these protobuf environment variables at import time (standard behavior) and TensorFlow 2.18 works correctly with the Python protobuf backend when configured this way.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens during imports in cell 0: `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known incompatibility between `protobuf>=6` (installed: 6.33.0) and libraries that still expect the older protobuf Python API (often via `imgaug`/`imgaug` dependencies). The current environment variables set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, but with protobuf 6 that does not restore the missing `GetPrototype` method; the safest minimal fix is to monkey-patch `google.protobuf.message_factory.MessageFactory.GetPrototype` to call `GetMessageClass` when missing, before importing the affected libraries.

Patch summary: In cell 0 only, insert a small protobuf compatibility shim immediately after setting the protobuf environment variables and before importing other packages. This adds `MessageFactory.GetPrototype` if it doesn’t exist, preventing the import-time crash while preserving the notebook’s core logic unchanged.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: No variables or interfaces used by later cells are changed; all imports remain available and the rest of cell 0 executes as before.

Assumptions: The error is triggered by a protobuf API mismatch during import (likely via `imgaug` or related deps), and adding the `GetPrototype` shim is sufficient to unblock execution without changing model/training semantics.'
- What this solution (achieved 0.47217) has done: 'Diagnosis: The notebook crashes immediately in cell 0 because it imports `imgaug`, but `imgaug` is not installed in the provided environment, raising `ModuleNotFoundError`. This prevents all subsequent cells from running. Since augmentation is only a dependency and not part of TensorFlow/Keras core, the safest minimal fix is to make the `imgaug` import optional and provide a tiny no-op fallback so later code can still reference `ia`/`iaa` without crashing.

Patch summary: Update cell 0 to wrap the `imgaug` imports in a `try/except ModuleNotFoundError`. If unavailable, define minimal stub objects for `ia` and `iaa` that act as identity augmenters, preserving the expected names and preventing downstream NameErrors.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: Cell 1 does not depend on `imgaug`, and this change preserves the variables `ia` and `iaa` in the global namespace so any later cells that reference them still run.

Assumptions: Any later augmentation usage can safely degrade to a no-op when `imgaug` is unavailable, which is preferable to crashing and does not change I/O paths or model/training code structure.'
- What this solution (achieved 0.28616) has done: 'The crash happens because `imgaug` is not installed, so in cell 0 the fallback sets `ia = None`, but `augment_pipeline()` still calls `ia.seed(...)`, producing `'NoneType' object has no attribute 'seed'`. We should make cell 10 robust to the no-`imgaug` case without changing the augmentation logic: if `ia` is unavailable, simply return an unaugmented copy (which matches the fallback augmenters’ no-op behavior). This keeps `processed_train` and `processed_train_cleaned` defined with compatible shapes and types for the downstream model code in cell 11.'
- What this solution (achieved 0.28616) has done: 'You’re far worse than the target (0.28616 vs 0.0429, lower is better), so we should improve score with the smallest changes that don’t alter the model/training core. The biggest issue is that the code trains and predicts on resized images (420×540) but then resizes predictions back to the original size; this mismatch typically hurts RMSE a lot. I minimally change the pipeline to keep images at their native size (matching submission pixels), update the single `IMG_SIZE` config accordingly, and make prediction use `autoencoder(test[i:i+1])` directly (no manual encoder/decoder numpy hops). This preserves the same architecture/training loop/loss, but aligns preprocessing with the metric and reduces interpolation artifacts, which should move the score much closer to the target.'
- What this solution (achieved 0.17478) has done: 'The crash happens because `process_image()` forcibly reshapes every image to `config.IMG_SIZE`, but at least one file has different dimensions, so the reshape fails. The minimal fix is to make `process_image()` ensure a consistent size by resizing any non-matching image to `config.IMG_SIZE` before reshaping. This preserves the downstream expectation that `train`, `train_cleaned`, and `test` are regular NumPy arrays with identical shapes. The rest of the pipeline (normalization, grayscale conversion, and final `(H,W,1)` shape) is kept the same.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory  # noqa: F401

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

import zipfile, os, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except ModuleNotFoundError:

    class _NoOpAugmenter:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, images=None, **kwargs):
            return images

        def augment_image(self, image, **kwargs):
            return image

        def augment_images(self, images, **kwargs):
            return images

    class _IAAFallback:
        Sequential = _NoOpAugmenter

        def __getattr__(self, name):
            return _NoOpAugmenter

    ia = None
    iaa = _IAAFallback()

sns.set_style("darkgrid")




## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"

with zipfile.ZipFile(path_zip + "train.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "test.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "train_cleaned.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "sampleSubmission.csv.zip", "r") as zip_ref:
    zip_ref.extractall(path)

train_img = sorted(os.listdir(path + "/train"))
train_cleaned_img = sorted(os.listdir(path + "/train_cleaned"))
test_img = sorted(os.listdir(path + "/test"))




## === cell 2
class config:
    IMG_SIZE = None  # will be set after inspecting a sample image


imgs = [cv2.imread(path + "train/" + f) for f in sorted(os.listdir(path + "train/"))]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs

_sample = cv2.imread(path + "train/" + train_img[0], 0)
config.IMG_SIZE = _sample.shape  # (H, W)
print("Using IMG_SIZE (native):", config.IMG_SIZE)
del _sample




## === cell 3
def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))

    return img




## === cell 4
def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.shape != tuple(config.IMG_SIZE):
        img = cv2.resize(
            img, (config.IMG_SIZE[1], config.IMG_SIZE[0]), interpolation=cv2.INTER_AREA
        )

    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img


train = []
train_cleaned = []
test = []

for f in sorted(os.listdir(path + "train/")):
    train.append(process_image(path + "train/" + f))

for f in sorted(os.listdir(path + "train_cleaned/")):
    train_cleaned.append(process_image(path + "train_cleaned/" + f))

for f in sorted(os.listdir(path + "test/")):
    test.append(process_image(path + "test/" + f))

train = np.asarray(train)
train_cleaned = np.asarray(train_cleaned)
test = np.asarray(test)


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
if ia is None:
    processed_train = train.copy()
    processed_train_cleaned = train_cleaned.copy()
else:
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




## === cell 12
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

rlp = callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.8, patience=5, min_lr=1e-15, mode="min", verbose=1
)


history = autoencoder.fit(
    processed_train,
    processed_train_cleaned,
    shuffle=True,
    callbacks=[es, rlp],
    epochs=500,
    batch_size=12,
)




## === cell 13
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
del history




## === cell 14
autoencoder.encoder.summary()
autoencoder.decoder.summary()




## === cell 15
decoded_imgs = autoencoder(train[:4]).numpy()

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
ids = []
vals = []
for i, f in tqdm(enumerate(test_img), total=len(test_img)):
    file = path + "test/" + f
    imgid = int(f[:-4])
    img = cv2.imread(file, 0)
    img_shape = img.shape  # should match config.IMG_SIZE

    decoded_img = np.squeeze(autoencoder(test[i : i + 1]).numpy())

    for r in range(img_shape[0]):
        for c in range(img_shape[1]):
            ids.append(str(imgid) + "_" + str(r + 1) + "_" + str(c + 1))
            vals.append(float(decoded_img[r, c]))

print("Length of IDs: {}".format(len(ids)))
pd.DataFrame({"id": ids, "value": vals}).to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")
