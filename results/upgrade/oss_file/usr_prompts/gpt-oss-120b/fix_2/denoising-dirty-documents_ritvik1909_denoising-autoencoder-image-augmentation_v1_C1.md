# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.02953

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.28616) has done: 'I remove the failing imgaug imports and related augmentation code, add a small validation split and a modest extra convolutional layer (keeping the auto‑encoder architecture essentially the same) to improve generalisation, and adjust early‑stopping to monitor validation loss. This fixes the runtime errors, ensures a valid submission.csv is written, and nudges the RMSE toward the target while preserving the core model logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile, os, cv2
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

try:
    import imgaug as ia
    from imgaug import augmenters as iaa

    IMG_AUG_AVAILABLE = True
except Exception:
    IMG_AUG_AVAILABLE = False

sns.set_style("darkgrid")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_zip = "../input/denoising-dirty-documents/"
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
def process_image(filepath):
    """Read, resize, grayscale, normalise and add channel dimension."""
    img = cv2.imread(filepath)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])  # cv2 uses (width, height)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
train = np.array([process_image(os.path.join(path, "train", f)) for f in train_img])
train_cleaned = np.array(
    [process_image(os.path.join(path, "train_cleaned", f)) for f in train_cleaned_img]
)
test = np.array([process_image(os.path.join(path, "test", f)) for f in test_img])

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
                ),  # extra layer (minor change)
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
es = callbacks.EarlyStopping(
    monitor="val_loss", patience=30, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    train_aug,
    train_cleaned_aug,
    validation_split=0.1,
    shuffle=True,
    callbacks=[es],
    epochs=500,
    batch_size=24,
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
ids = []
vals = []

for i, f in enumerate(test_img):
    file_path = os.path.join(path, "test", f)
    imgid = int(f[:-4])  # numeric id from filename
    original = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    h_orig, w_orig = original.shape

    decoded = autoencoder.decoder(autoencoder.encoder(test[i : i + 1])).numpy()
    decoded_img = np.squeeze(decoded)  # shape = config.IMG_SIZE
    pred_resized = cv2.resize(decoded_img, (w_orig, h_orig))

    for r in range(h_orig):
        for c in range(w_orig):
            ids.append(f"{imgid}_{r+1}_{c+1}")
            vals.append(float(pred_resized[r, c]))

print("Submission length:", len(ids))
submission = pd.DataFrame({"id": ids, "value": vals})
submission.to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")
