# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
    IMG_SIZE = None  # will be set to the submission grid size (H, W)


sample_path = os.path.join(path, "sampleSubmission.csv")
sample_df_head = pd.read_csv(sample_path, nrows=200000)  # enough to infer max row/col
parts = sample_df_head["id"].str.split("_", expand=True).astype(np.int32)
max_row = int(parts[1].max())
max_col = int(parts[2].max())
config.IMG_SIZE = (max_row, max_col)
print("Using IMG_SIZE (from sampleSubmission):", config.IMG_SIZE)




## === cell 3
def process_image(path_):
    img = cv2.imread(path_)
    img = np.asarray(img, dtype="float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0

    th, tw = config.IMG_SIZE
    H, W = img.shape[:2]

    if H > th:
        top = (H - th) // 2
        img = img[top : top + th, :]
        H = th
    if W > tw:
        left = (W - tw) // 2
        img = img[:, left : left + tw]
        W = tw

    if H < th or W < tw:
        out = np.ones((th, tw), dtype=np.float32)
        top = (th - H) // 2
        left = (tw - W) // 2
        out[top : top + H, left : left + W] = img
        img = out

    img = np.reshape(img, (th, tw, 1))
    return img


train = []
train_cleaned = []
test = []

for f in sorted(os.listdir(path + "train/")):
    train.append(process_image(path + "train/" + f))

for f in sorted(os.listdir(path + "train_cleaned/")):
    train_cleaned.append(process_image(path + "train_cleaned/" + f))

for f in test_img:
    test.append(process_image(path + "test/" + f))

train = np.asarray(train)
train_cleaned = np.asarray(train_cleaned)
test = np.asarray(test)




## === cell 4
train.shape, train_cleaned.shape, test.shape




## === cell 5
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




## === cell 6
def augment_pipeline(pipeline, images, seed=19):
    if ia is None:
        return images.copy()

    ia.seed(seed)
    processed_images = images.copy()
    for step in pipeline:
        temp = np.array(step.augment_images(images))
        processed_images = np.append(processed_images, temp, axis=0)
    return processed_images




## === cell 7
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




## === cell 8
pipeline = [rotate90, rotate180, rotate270, hflip, vflip]




## === cell 9
processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned)

processed_train.shape, processed_train_cleaned.shape




## === cell 10
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




## === cell 11
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




## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1765015730.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      7[0m )
[1;32m      8[0m [0;34m[0m[0m
[0;32m----> 9[0;31m history = autoencoder.fit(
[0m[1;32m     10[0m     [0mprocessed_train[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m     [0mprocessed_train_cleaned[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/losses/losses.py[0m in [0;36mmean_squared_error[0;34m(y_true, y_pred)[0m
[1;32m   1677[0m     [0my_true[0m [0;34m=[0m [0mops[0m[0;34m.[0m[0mconvert_to_tensor[0m[0;34m([0m[0my_true[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0my_pred[0m[0;34m.[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1678[0m     [0my_true[0m[0;34m,[0m [0my_pred[0m [0;34m=[0m [0msqueeze_or_expand_to_same_rank[0m[0;34m([0m[0my_true[0m[0;34m,[0m [0my_pred[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1679[0;31m     [0;32mreturn[0m [0mops[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mops[0m[0;34m.[0m[0msquare[0m[0;34m([0m[0my_true[0m [0;34m-[0m [0my_pred[0m[0;34m)[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;34m-[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1680[0m [0;34m[0m[0m
[1;32m   1681[0m [0;34m[0m[0m

[0;31mValueError[0m: Dimensions must be equal, but are 371 and 372 for '{{node compile_loss/mean_squared_error/sub}} = Sub[T=DT_FLOAT](data_1, denoising_autoencoder_1/sequential_1_2/conv2d_1_2/Sigmoid)' with input shapes: [?,371,540,1], [?,372,540,1].

## === cell 12
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
del history
