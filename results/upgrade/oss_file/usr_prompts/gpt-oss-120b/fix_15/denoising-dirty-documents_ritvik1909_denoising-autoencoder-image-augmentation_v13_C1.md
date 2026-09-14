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

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

from tensorflow.keras import mixed_precision

tf.config.optimizer.set_jit(True)

mixed_precision.set_global_policy("mixed_float16")

sns.set_style("darkgrid")

np.random.seed(42)
tf.random.set_seed(42)




## === cell 1
path_zip = "/kaggle/input/denoising-dirty-documents/"
path = "/kaggle/working/"

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
    IMG_SIZE = (420, 540)  # (height, width)


imgs = [cv2.imread(os.path.join(path, "train", f)) for f in train_img]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
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
train = []
train_cleaned = []
test = []

for f in train_img:
    train.append(process_image(os.path.join(path, "train", f)))
for f in train_cleaned_img:
    train_cleaned.append(process_image(os.path.join(path, "train_cleaned", f)))
for f in test_img:
    test.append(process_image(os.path.join(path, "test", f)))

train = np.asarray(train)
train_cleaned = np.asarray(train_cleaned)
test = np.asarray(test)

flipped_train_h = np.flip(train, axis=2)  # width dimension
flipped_train_cleaned_h = np.flip(train_cleaned, axis=2)

flipped_train_v = np.flip(train, axis=1)  # height dimension
flipped_train_cleaned_v = np.flip(train_cleaned, axis=1)

train = np.concatenate([train, flipped_train_h, flipped_train_v], axis=0)
train_cleaned = np.concatenate(
    [train_cleaned, flipped_train_cleaned_h, flipped_train_cleaned_v], axis=0
)

print("After augmentation:", train.shape, train_cleaned.shape, test.shape)




## === cell 5
def augment_pipeline(pipeline, images, seed=19):
    """
    Minimal augmentation: add a small amount of Gaussian noise.
    Only the noisy input images are perturbed; clean targets stay unchanged.
    """
    np.random.seed(seed)
    noise = np.random.normal(loc=0.0, scale=0.05, size=images.shape).astype(np.float32)
    noisy_images = np.clip(images + noise, 0.0, 1.0)
    return noisy_images




## === cell 6
pipeline = []




## === cell 7
processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = train_cleaned  # do NOT augment clean targets
print(processed_train.shape, processed_train_cleaned.shape)




## === cell 8
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
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)




## === cell 9
batch_size = 32  # larger batch fits in memory and speeds up each epoch
val_fraction = 0.1

dataset = tf.data.Dataset.from_tensor_slices((processed_train, processed_train_cleaned))
dataset = dataset.cache()  # cache the whole dataset (fits in RAM)
dataset = dataset.shuffle(
    buffer_size=len(processed_train), seed=42, reshuffle_each_iteration=True
)

val_size = int(val_fraction * len(processed_train))
train_ds = dataset.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = dataset.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

es = callbacks.EarlyStopping(
    monitor="val_loss", patience=300, verbose=1, restore_best_weights=True
)
rlp = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, mode="min", verbose=1
)

history = autoencoder.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[es, rlp],
    epochs=500,
    verbose=2,
)




## === cell 10
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
plt.show()




## === cell 11
autoencoder.encoder.summary()
autoencoder.decoder.summary()




## === cell 12
decoded_imgs = autoencoder(train[:4], training=False).numpy()
fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][0].set_title(f"Clean image: {train_img[i]}")
    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
    ax[i][1].set_title(f"Predicted image: {train_img[i]}")
    for a in ax[i]:
        a.axis("off")
plt.show()




## === cell 13
decoded_all = autoencoder(test, training=False).numpy()

ids = []
vals = []

for i, f in enumerate(test_img):
    imgid = int(f[:-4])
    orig_path = os.path.join(path, "test", f)
    orig_img = cv2.imread(orig_path, 0)
    img_shape = orig_img.shape  # (height, width)

    decoded_img = decoded_all[i]
    decoded_img = np.squeeze(decoded_img)

    blurred_uint8 = (decoded_img * 255).astype(np.uint8)
    blurred_uint8 = cv2.medianBlur(blurred_uint8, ksize=3)
    decoded_img = blurred_uint8.astype(np.float32) / 255.0

    decoded_img = decoded_img.astype(np.float32)

    preds_reshaped = cv2.resize(decoded_img, (img_shape[1], img_shape[0]))
    preds_reshaped = np.clip(preds_reshaped, 0.0, 1.0)

    rows = np.arange(1, img_shape[0] + 1, dtype=np.int32)
    cols = np.arange(1, img_shape[1] + 1, dtype=np.int32)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")

    ids_img = np.core.defchararray.add(
        np.core.defchararray.add(
            np.core.defchararray.add(
                np.core.defchararray.add(str(imgid) + "_", grid_r.astype(str)), "_"
            ),
            grid_c.astype(str),
        ),
        "",
    ).ravel()

    ids.extend(ids_img.tolist())
    vals.extend(preds_reshaped.ravel().tolist())

print("Length of IDs:", len(ids))
submission = pd.DataFrame({"id": ids, "value": vals})
submission.to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")
