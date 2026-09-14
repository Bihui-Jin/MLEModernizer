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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile, os, cv2
from tqdm.auto import tqdm

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

try:
    import imgaug as ia
    from imgaug import augmenters as iaa

    IMG_AUG_AVAILABLE = True
except Exception:
    IMG_AUG_AVAILABLE = False

sns.set_style("darkgrid")




## === cell 1
path_zip = "/kaggle/input/denoising-dirty-documents/"
path = "/kaggle/working/"

for fname in ["train.zip", "test.zip", "train_cleaned.zip", "sampleSubmission.csv.zip"]:
    with zipfile.ZipFile(os.path.join(path_zip, fname), "r") as zip_ref:
        zip_ref.extractall(path)

train_img = sorted(os.listdir(os.path.join(path, "train")))
train_cleaned_img = sorted(os.listdir(os.path.join(path, "train_cleaned")))
test_img = sorted(os.listdir(os.path.join(path, "test")))




## === cell 2
class config:
    IMG_SIZE = (420, 540)  # height, width




## === cell 3
def load_dataset(file_list, folder):
    """Efficiently load images as float32 tensors of shape (N, H, W, 1)."""
    N = len(file_list)
    H, W = config.IMG_SIZE
    data = np.empty((N, H, W, 1), dtype=np.float32)
    for i, f in enumerate(file_list):
        img = cv2.imread(os.path.join(path, folder, f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, config.IMG_SIZE[::-1])  # width, height order
        img = img.astype(np.float32) / 255.0
        data[i, :, :, 0] = img
    return data




## === cell 4
train = load_dataset(train_img, "train")
train_cleaned = load_dataset(train_cleaned_img, "train_cleaned")
test = load_dataset(test_img, "test")

print(
    "Shapes -> train:",
    train.shape,
    "cleaned:",
    train_cleaned.shape,
    "test:",
    test.shape,
)




## === cell 5
fig, ax = plt.subplots(4, 2, figsize=(12, 16))
for i in range(4):
    ax[i, 0].imshow(tf.squeeze(train[i]), cmap="gray")
    ax[i, 0].set_title(f"Noisy: {train_img[i]}")
    ax[i, 1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i, 1].set_title(f"Clean: {train_cleaned_img[i]}")
    for a in ax[i]:
        a.axis("off")
plt.close(fig)  # close to avoid display in non‑interactive env




## === cell 6
def simple_augment(images):
    """Return original images plus horizontally flipped versions."""
    flipped = np.flip(images, axis=2)  # flip width dimension
    return np.concatenate([images, flipped], axis=0)


if IMG_AUG_AVAILABLE:
    seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Affine(rotate=(-10, 10))])

    def augment(images):
        return np.concatenate([images, seq.augment_images(images)], axis=0)

else:
    augment = simple_augment

train_aug = augment(train)
train_cleaned_aug = augment(train_cleaned)
print("After augmentation ->", train_aug.shape)




## === cell 7
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(
                    256, (3, 3), activation="relu", padding="same"
                ),  # extra layer
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(
                    256, (3, 3), activation="relu", padding="same"
                ),  # match extra encoder layer
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        return self.decoder(self.encoder(x))


autoencoder = DenoisingAutoencoder()
autoencoder.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)




## === cell 8
total_samples = train_aug.shape[0]
val_fraction = 0.1
val_count = int(total_samples * val_fraction)

full_ds = tf.data.Dataset.from_tensor_slices((train_aug, train_cleaned_aug))
full_ds = full_ds.shuffle(
    buffer_size=total_samples, seed=42, reshuffle_each_iteration=False
)

val_ds = full_ds.take(val_count).batch(24).prefetch(tf.data.AUTOTUNE)
train_ds = full_ds.skip(val_count).batch(24).prefetch(tf.data.AUTOTUNE)

es = callbacks.EarlyStopping(
    monitor="val_loss", patience=100, verbose=1, restore_best_weights=True
)

lr_sched = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    verbose=1,
    min_lr=1e-6,
)

history = autoencoder.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[es, lr_sched],
    epochs=1000,
    verbose=2,
)




## === cell 9
fig, ax = plt.subplots(figsize=(8, 4))
pd.DataFrame(history.history)[["loss", "val_loss"]].plot(ax=ax)
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.close(fig)




## === cell 10
decoded_batch = autoencoder.predict(test, batch_size=24, verbose=0)  # (N, H, W, 1)

ids = []
vals = []

for i, f in enumerate(test_img):
    file_path = os.path.join(path, "test", f)
    imgid = int(f[:-4])  # numeric id from filename

    original = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    h_orig, w_orig = original.shape

    decoded_img = np.squeeze(decoded_batch[i])  # (config.IMG_SIZE)
    pred_resized = cv2.resize(decoded_img, (w_orig, h_orig))

    rows = np.repeat(np.arange(1, h_orig + 1), w_orig)
    cols = np.tile(np.arange(1, w_orig + 1), h_orig)
    ids.extend([f"{imgid}_{r}_{c}" for r, c in zip(rows, cols)])

    vals.extend(pred_resized.ravel().tolist())

print("Submission length:", len(ids))
submission = pd.DataFrame({"id": ids, "value": vals})
submission.to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")
