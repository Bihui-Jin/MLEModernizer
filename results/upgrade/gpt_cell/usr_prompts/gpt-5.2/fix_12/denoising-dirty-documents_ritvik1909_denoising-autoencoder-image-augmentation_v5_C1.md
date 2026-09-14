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
import subprocess, sys

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _major = int(_pb_ver.split(".", 1)[0])
    if _major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
except Exception:
    pass

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils


class _NoOpAugmenter:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image=None, images=None, **kwargs):
        if images is not None:
            return images
        return image

    def augment_image(self, image):
        return image

    def augment_images(self, images):
        return images


class _IAAStub:
    class Sequential(_NoOpAugmenter):
        pass


class _IAStub:
    pass


ia = _IAStub()
iaa = _IAAStub()

sns.set_style("darkgrid")

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
np.random.seed(19)
tf.random.set_seed(19)
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"


def _extract_if_needed(zip_path, out_dir, sentinel_relpath):
    sentinel = os.path.join(out_dir, sentinel_relpath)
    if os.path.exists(sentinel):
        return
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)


_extract_if_needed(os.path.join(path_zip, "train.zip"), path, "train")
_extract_if_needed(os.path.join(path_zip, "test.zip"), path, "test")
_extract_if_needed(os.path.join(path_zip, "train_cleaned.zip"), path, "train_cleaned")

if not os.path.exists(os.path.join(path, "sampleSubmission.csv")):
    with zipfile.ZipFile(path_zip + "sampleSubmission.csv.zip", "r") as zip_ref:
        zip_ref.extractall(path)

train_img = sorted(os.listdir(path + "/train"))
train_cleaned_img = sorted(os.listdir(path + "/train_cleaned"))
test_img = sorted(os.listdir(path + "/test"))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


shapes = []
for f in train_img:
    im = cv2.imread(os.path.join(path, "train", f), cv2.IMREAD_UNCHANGED)
    if im is not None:
        shapes.append(im.shape[:2])
if shapes:
    hs = np.array([s[0] for s in shapes], dtype=np.int32)
    ws = np.array([s[1] for s in shapes], dtype=np.int32)
    print("Median Dimensions:", float(np.median(hs)), float(np.median(ws)))
else:
    print("Median Dimensions: (no images found)")




## === cell 3
def process_image(path_):
    img = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = img.astype("float32", copy=False) / 255.0
    img = img.reshape((*config.IMG_SIZE, 1))
    return np.ascontiguousarray(img)




## === cell 4
def _load_stack(folder, files):
    n = len(files)
    arr = np.empty((n, *config.IMG_SIZE, 1), dtype=np.float32)
    for i, f in enumerate(files):
        arr[i] = process_image(os.path.join(folder, f))
    return arr


train = _load_stack(os.path.join(path, "train"), train_img)
train_cleaned = _load_stack(os.path.join(path, "train_cleaned"), train_cleaned_img)
test = _load_stack(os.path.join(path, "test"), test_img)




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
if not hasattr(iaa, "Rot90"):

    class _NoOpAugmenterStep:
        def __init__(self, *args, **kwargs):
            pass

        def augment_images(self, images):
            return images

        def augment_image(self, image):
            return image

        def __call__(self, image=None, images=None, **kwargs):
            if images is not None:
                return images
            return image

    rotate90 = _NoOpAugmenterStep()  # rotate image 90 degrees (no-op fallback)
    rotate180 = _NoOpAugmenterStep()  # rotate image 180 degrees (no-op fallback)
    rotate270 = _NoOpAugmenterStep()  # rotate image 270 degrees (no-op fallback)
    random_rotate = _NoOpAugmenterStep()  # randomly rotate image (no-op fallback)
    perc_transform = _NoOpAugmenterStep()  # perspective transform (no-op fallback)
    rotate10 = _NoOpAugmenterStep()  # affine rotate (no-op fallback)
    rotate10r = _NoOpAugmenterStep()  # affine rotate reverse (no-op fallback)
    crop = _NoOpAugmenterStep()  # crop (no-op fallback)
    hflip = _NoOpAugmenterStep()  # horizontal flip (no-op fallback)
    vflip = _NoOpAugmenterStep()  # vertical flip (no-op fallback)
    gblur = _NoOpAugmenterStep()  # gaussian blur (no-op fallback)
    motionblur = _NoOpAugmenterStep()  # motion blur (no-op fallback)

    seq_rp = iaa.Sequential([]) if hasattr(iaa, "Sequential") else _NoOpAugmenterStep()
    seq_cfg = iaa.Sequential([]) if hasattr(iaa, "Sequential") else _NoOpAugmenterStep()
    seq_fm = iaa.Sequential([]) if hasattr(iaa, "Sequential") else _NoOpAugmenterStep()
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
def _build_augmented_dataset(images_x, images_y, pipeline, batch_size, seed=19):
    n = images_x.shape[0]
    n_steps = len(pipeline)

    def gen():
        if hasattr(ia, "seed") and callable(getattr(ia, "seed")):
            ia.seed(seed)
        for i in range(n):
            yield images_x[i], images_y[i]
        for step in pipeline:
            x_aug = np.asarray(step.augment_images(images_x))
            y_aug = np.asarray(step.augment_images(images_y))
            for i in range(n):
                yield x_aug[i], y_aug[i]

    output_signature = (
        tf.TensorSpec(shape=images_x.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=images_y.shape[1:], dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    ds = ds.shuffle(
        buffer_size=n * (n_steps + 1), seed=seed, reshuffle_each_iteration=True
    )

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


processed_train = None
processed_train_cleaned = None
processed_train_shape = (train.shape[0] * (len(pipeline) + 1),) + train.shape[1:]
processed_train_cleaned_shape = (
    train_cleaned.shape[0] * (len(pipeline) + 1),
) + train_cleaned.shape[1:]
print(processed_train_shape, processed_train_cleaned_shape)




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

train_ds = _build_augmented_dataset(
    train, train_cleaned, pipeline=pipeline, batch_size=12, seed=19
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
decoded_test = autoencoder(test, training=False).numpy()  # (N, 420, 540, 1)

_hw_cache = {}  # (H,W) -> (rowcol_ids_str_array)
out_path = "submission.csv"

with open(out_path, "w", newline="") as f:
    f.write("id,value\n")
    for i, fname in tqdm(list(enumerate(test_img)), total=len(test_img)):
        file = path + "test/" + fname
        imgid = int(fname[:-4])

        img0 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        H, W = img0.shape

        decoded_img = np.squeeze(decoded_test[i])  # (420, 540)
        preds_reshaped = cv2.resize(decoded_img, (W, H))  # (H, W)
        vals = preds_reshaped.reshape(-1)

        key = (H, W)
        if key not in _hw_cache:
            r = np.repeat(np.arange(1, H + 1, dtype=np.int32), W)
            c = np.tile(np.arange(1, W + 1, dtype=np.int32), H)
            suffix = np.char.add(np.char.add(r.astype(str), "_"), c.astype(str))
            _hw_cache[key] = suffix

        suffix = _hw_cache[key]
        prefix = str(imgid) + "_"
        ids = np.char.add(prefix, suffix)

        lines = "\n".join(
            ids.tolist()[j] + "," + repr(float(vals[j])) for j in range(vals.size)
        )
        f.write(lines)
        f.write("\n")

print(f"Results saved to {out_path}!")
