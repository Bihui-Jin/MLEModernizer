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

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils, optimizers

from sklearn.model_selection import train_test_split  # added for validation split

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
    IMG_SIZE = (420, 540)






## === cell 3
def process_image(pth):
    img = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)  # read in gray mode
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
train_paths = sorted(os.listdir(path + "train/"))
train_cleaned_paths = sorted(os.listdir(path + "train_cleaned/"))
test_paths = sorted(os.listdir(path + "test/"))

N_train = len(train_paths)
N_test = len(test_paths)

train = np.empty((N_train, *config.IMG_SIZE, 1), dtype=np.float32)
train_cleaned = np.empty((N_train, *config.IMG_SIZE, 1), dtype=np.float32)
test = np.empty((N_test, *config.IMG_SIZE, 1), dtype=np.float32)
test_orig_sizes = []  # store (h, w) for each test image

for i, f in enumerate(train_paths):
    train[i] = process_image(os.path.join(path, "train", f))

for i, f in enumerate(train_cleaned_paths):
    train_cleaned[i] = process_image(os.path.join(path, "train_cleaned", f))

for i, f in enumerate(test_paths):
    orig_path = os.path.join(path, "test", f)
    orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    test_orig_sizes.append(orig_img.shape[:2])  # (h, w)
    test[i] = process_image(orig_path)




## === cell 5
train.shape, train_cleaned.shape, test.shape




## === cell 6
fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train[i]), cmap="gray")
    ax[i][0].set_title("Noise image: {}".format(train_img[i]))

    ax[i][1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][1].set_title("Denoised image: {}".format(train_img[i]))

    ax[i][0].axis("off")
    ax[i][1].axis("off")




## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    return images.copy()




## === cell 8
pipeline = []




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
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.2),
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
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="mean_squared_error",
    metrics=["mean_absolute_error"],
)




## === cell 11
BATCH_SIZE = 64  # larger batch reduces number of steps per epoch
SHUFFLE_BUFFER = 1024  # smaller buffer avoids unnecessary memory copies

train_X, val_X, train_y, val_y = train_test_split(
    processed_train, processed_train_cleaned, test_size=0.1, random_state=19
)

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_X, train_y))
    .cache()
    .shuffle(buffer_size=SHUFFLE_BUFFER, seed=19, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_X, val_y))
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

es = callbacks.EarlyStopping(
    monitor="val_loss", patience=50, verbose=1, restore_best_weights=True
)
lr_sched = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
)

history = autoencoder.fit(
    train_ds,
    validation_data=val_ds,
    epochs=500,  # unchanged; early stopping will halt earlier if possible
    callbacks=[es, lr_sched],
    shuffle=False,  # shuffling already handled by dataset
)




## === cell 12
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
plt.show()
del history




## === cell 13
autoencoder.encoder.summary()
autoencoder.decoder.summary()




## === cell 14
decoded_imgs = autoencoder(train[:4]).numpy()

fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][0].set_title("Denoised image: {}".format(train_img[i]))

    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
    ax[i][1].set_title("Predicted image: {}".format(train_img[i]))

    ax[i][0].axis("off")
    ax[i][1].axis("off")
del decoded_imgs




## === cell 15
import csv

preds = autoencoder.predict(test, batch_size=64, verbose=0)  # (N, H, W, 1)

with open("submission.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "value"])  # header

    for i, fname in enumerate(tqdm(test_img, desc="Building submission")):
        imgid = int(fname[:-4])  # numeric id from filename
        orig_h, orig_w = test_orig_sizes[i]

        pred_img = preds[i, :, :, 0]
        pred_resized = cv2.resize(
            pred_img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        pred_resized = np.clip(pred_resized, 0.0, 1.0)

        rows = np.arange(1, orig_h + 1, dtype=np.int32)
        cols = np.arange(1, orig_w + 1, dtype=np.int32)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        ids_img = np.char.mod(
            "%d_%d_%d",
            np.stack([np.full(rr.size, imgid), rr.ravel(), cc.ravel()], axis=1),
        )

        writer.writerows(zip(ids_img, pred_resized.ravel()))

print("Results saved to submission.csv!")
