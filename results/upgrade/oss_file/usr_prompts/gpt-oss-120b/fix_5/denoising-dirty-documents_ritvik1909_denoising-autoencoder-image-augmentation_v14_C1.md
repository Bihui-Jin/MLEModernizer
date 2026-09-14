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

0.0429

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.05714) has done: 'I wrap the imgaug import in a safe try/except and provide dummy augmenters when the library cannot be loaded, so the pipeline can still run without errors. I also make the augmentation pipeline optional (empty if augmenters are unavailable) and ensure the `augment_pipeline` function simply returns the original images in that case. Finally, I add a small validation split to the model fitting to help early‑stopping without altering the core architecture. These fixes remove the runtime errors, allow the script to produce a valid `submission.csv`, and modestly improve training stability, moving the score toward the target.'
- What this solution (achieved 0.4726) has done: 'I fix the import error handling by providing simple NumPy‑based augmentations when `imgaug` cannot be loaded, and I make the early‑stopping and learning‑rate callbacks monitor validation loss instead of training loss. These minimal changes keep the original architecture intact while improving model generalisation and ensuring the script always runs, producing a proper `submission.csv`.'
- What this solution (achieved 0.47193) has done: 'I fixed the import error by skipping imgaug entirely, rewrote the augmentation logic so that noisy and clean images are augmented together and resized back to the original size (preventing shape mismatches), and updated the cells to use this new paired augmentation. These changes eliminate runtime failures, keep the image‑pair alignment during training (which should improve the denoising performance and move the RMSE closer to the target), and still produce a valid `submission.csv` file.'
- What this solution (achieved 0.28616) has done: 'Implemented fixes to resolve runtime errors and improve model performance:

- Wrapped seaborn import in a safe try/except to avoid protobuf‑related crashes.
- Simplified the augmentation pipeline: removed unnecessary resizing that caused dimension mismatches and directly concatenated augmented images, keeping all tensors 4‑D.
- Enhanced the autoencoder slightly by adding extra convolutional layers to boost capacity while preserving the original architecture style.
- Updated comments for clarity.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zipfile, os, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

try:
    import seaborn as sns

    sns.set_style("darkgrid")
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    IMG_SIZE = (420, 540)  # (height, width)




## === cell 3
imgs = [cv2.imread(path + "train/" + f) for f in sorted(os.listdir(path + "train/"))]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs




## === cell 4
def process_image(p):
    img = cv2.imread(p)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])  # width, height
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 5
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




## === cell 6
def np_rot90(images, k=1):
    return np.array([np.rot90(img, k) for img in images])


def np_hflip(images):
    return np.array([np.fliplr(img) for img in images])


def np_vflip(images):
    return np.array([np.flipud(img) for img in images])


pipeline = [
    lambda imgs: np_rot90(imgs, k=1),
    lambda imgs: np_rot90(imgs, k=2),
    lambda imgs: np_rot90(imgs, k=3),
    np_hflip,
    np_vflip,
]


def augment_pair(pipeline, noisy_imgs, clean_imgs, seed=19):
    """Apply each augmentation step to both noisy and clean batches and
    concatenate them, preserving the (H, W, 1) shape."""
    np.random.seed(seed)
    aug_noisy = noisy_imgs.copy()
    aug_clean = clean_imgs.copy()
    for step in pipeline:
        noisy_aug = step(noisy_imgs)
        clean_aug = step(clean_imgs)
        aug_noisy = np.concatenate([aug_noisy, noisy_aug], axis=0)
        aug_clean = np.concatenate([aug_clean, clean_aug], axis=0)
    return aug_noisy, aug_clean


processed_train, processed_train_cleaned = augment_pair(pipeline, train, train_cleaned)

print(
    "Shapes after augmentation:", processed_train.shape, processed_train_cleaned.shape
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3622096068.py in <cell line: 0>()
     34 
     35 
---> 36 processed_train, processed_train_cleaned = augment_pair(pipeline, train, train_cleaned)
     37 
     38 print(

/tmp/ipykernel_55/3622096068.py in augment_pair(pipeline, noisy_imgs, clean_imgs, seed)
     29         noisy_aug = step(noisy_imgs)
     30         clean_aug = step(clean_imgs)
---> 31         aug_noisy = np.concatenate([aug_noisy, noisy_aug], axis=0)
     32         aug_clean = np.concatenate([aug_clean, clean_aug], axis=0)
     33     return aug_noisy, aug_clean

ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 420 and the array at index 1 has size 540

## === cell 7
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
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
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



## === cell 8
es = callbacks.EarlyStopping(
    monitor="val_loss", patience=30, verbose=1, restore_best_weights=True
)

rlp = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.8, patience=5, min_lr=1e-15, mode="min", verbose=1
)

history = autoencoder.fit(
    processed_train,
    processed_train_cleaned,
    shuffle=True,
    callbacks=[es, rlp],
    epochs=500,
    batch_size=12,
    validation_split=0.1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3202257727.py in <cell line: 0>()
      8 
      9 history = autoencoder.fit(
---> 10     processed_train,
     11     processed_train_cleaned,
     12     shuffle=True,

NameError: name 'processed_train' is not defined

## === cell 9
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
plt.show()
del history



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1158270450.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(20, 6))
----> 2 pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
      3 plt.show()
      4 del history
      5 

NameError: name 'history' is not defined

## === cell 10
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 11
decoded_imgs = autoencoder(train[:4]).numpy()

fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][0].set_title(f"Clean image: {train_img[i]}")
    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
    ax[i][1].set_title(f"Predicted image: {train_img[i]}")
    ax[i][0].axis("off")
    ax[i][1].axis("off")
del decoded_imgs



## === cell 12
ids = []
vals = []
for i, f in tqdm(enumerate(test_img)):
    file = path + "test/" + f
    imgid = int(f[:-4])
    img = cv2.imread(file, 0)
    img_shape = img.shape
    decoded_img = np.squeeze(
        autoencoder.decoder(autoencoder.encoder(test[i : i + 1]).numpy()).numpy()
    )
    preds_reshaped = cv2.resize(decoded_img, (img_shape[1], img_shape[0]))
    for r in range(img_shape[0]):
        for c in range(img_shape[1]):
            ids.append(f"{imgid}_{r+1}_{c+1}")
            vals.append(preds_reshaped[r, c])

print("Length of IDs:", len(ids))
pd.DataFrame({"id": ids, "value": vals}).to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")
