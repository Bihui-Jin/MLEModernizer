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

0.02843

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I wrap the imgaug import in a try/except, provide a fallback augment_pipeline that simply returns the original images when augmentations are unavailable, and conditionally build the augmentation pipeline only if imgaug was successfully imported. This removes the import‑related crashes and the subsequent NameErrors, allowing the autoencoder to be trained and the script to produce a proper submission.csv file. The core model and training logic remain unchanged.'
- What this solution (achieved 0.28616) has done: 'I added a protobuf compatibility fix by forcing the pure‑Python implementation before TensorFlow is imported, and I relaxed the early‑stopping patience and increased the maximum epochs so the model can train longer and achieve a lower RMSE. No core architecture or data handling was changed.'
- What this solution (achieved 0.28616) has done: 'The fix adds a small protobuf compatibility monkey‑patch before importing imgaug to stop the `AttributeError` that halts execution.  
The early‑stopping patience is increased and the maximum epochs are raised so the auto‑encoder can train longer, which should lower the RMSE toward the target while keeping the original model unchanged.'
- What this solution (achieved 0.28616) has done: 'I increase the training capacity and data diversity while keeping the model architecture unchanged. First, I raise the EarlyStopping patience from 200 to 500 so the auto‑encoder can train longer before stopping. Then, when imgaug is unavailable I add a lightweight NumPy/OpenCV augmentation step (horizontal flip, vertical flip, and 90° rotation) and concatenate those images to the training batches. These small changes give the model more examples and training iterations, which should lower the RMSE toward the target without altering the core network logic.'
- What this solution (achieved 0.28616) has done: 'I speed up the augmentation routine by concatenating once instead of repeatedly appending, and replace the nested Python loops that build the submission with fully‑vectorized NumPy‑based construction of IDs and values. These changes keep the exact same data transformations and model training logic, but eliminate costly per‑iteration Python overhead, allowing the whole notebook to finish well under the 600‑second limit.'
- What this solution (achieved 0.28616) has done: 'The changes add mixed‑precision training and feed the data through a cached tf.data pipeline, which lets TensorFlow batch and pre‑fetch images efficiently on the GPU. This keeps the exact model architecture, loss, and training loop (still 3000 epochs with the same early‑stopping) while reducing per‑epoch overhead enough to stay under the 600‑second limit. No logic, augmentation, or output handling is altered.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _get_prototype
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

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

try:
    import imgaug as ia
    from imgaug import augmenters as iaa

    IA_AVAILABLE = True
except Exception as e:
    print(
        "imgaug not available or incompatible (",
        e,
        "); proceeding without augmentation.",
    )
    IA_AVAILABLE = False

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
    img = cv2.resize(img, config.IMG_SIZE[::-1])
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

train = np.asarray(train, dtype="float32")
train_cleaned = np.asarray(train_cleaned, dtype="float32")
test = np.asarray(test, dtype="float32")




## === cell 6
print(
    "train shape:",
    train.shape,
    "train_cleaned shape:",
    train_cleaned.shape,
    "test shape:",
    test.shape,
)




## === cell 7
fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train[i]), cmap="gray")
    ax[i][0].set_title(f"Noise image: {train_img[i]}")

    ax[i][1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][1].set_title(f"Denoised image: {train_img[i]}")

    ax[i][0].axis("off")
    ax[i][1].axis("off")




## === cell 8
def augment_pipeline(pipeline, images, seed=19):
    """Return augmented images if a pipeline is supplied, otherwise the original batch."""
    if not pipeline:
        return images.copy()
    ia.seed(seed)
    batches = [images]
    for step in pipeline:
        batches.append(step.augment_images(images))
    return np.concatenate(batches, axis=0)




## === cell 9
if IA_AVAILABLE:
    rotate90 = iaa.Rot90(1)
    rotate180 = iaa.Rot90(2)
    rotate270 = iaa.Rot90(3)
    hflip = iaa.Fliplr(1)
    vflip = iaa.Flipud(1)

    pipeline = [rotate90, rotate180, rotate270, hflip, vflip]
else:
    pipeline = []  # no augmentation

processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned)

if not IA_AVAILABLE:
    hflip = np.flip(train, axis=2)
    vflip = np.flip(train, axis=1)
    rot90 = np.rot90(train, k=1, axes=(0, 1))
    processed_train = np.concatenate([processed_train, hflip, vflip, rot90], axis=0)

    hflip_c = np.flip(train_cleaned, axis=2)
    vflip_c = np.flip(train_cleaned, axis=1)
    rot90_c = np.rot90(train_cleaned, k=1, axes=(0, 1))
    processed_train_cleaned = np.concatenate(
        [processed_train_cleaned, hflip_c, vflip_c, rot90_c], axis=0
    )

print(
    "Processed train shape:",
    processed_train.shape,
    "Processed train_cleaned shape:",
    processed_train_cleaned.shape,
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/450114321.py in <cell line: 0>()
     17     vflip = np.flip(train, axis=1)
     18     rot90 = np.rot90(train, k=1, axes=(0, 1))
---> 19     processed_train = np.concatenate([processed_train, hflip, vflip, rot90], axis=0)
     20 
     21     hflip_c = np.flip(train_cleaned, axis=2)

ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 420 and the array at index 3 has size 115

## === cell 10
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
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
train_ds = tf.data.Dataset.from_tensor_slices(
    (processed_train.astype("float32"), processed_train_cleaned.astype("float32"))
)
train_ds = (
    train_ds.shuffle(
        buffer_size=processed_train.shape[0], reshuffle_each_iteration=True
    )
    .batch(12)
    .prefetch(tf.data.AUTOTUNE)
)

es = callbacks.EarlyStopping(
    monitor="loss", patience=500, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    train_ds,
    epochs=3000,
    callbacks=[es],
    verbose=2,
)




## === cell 12
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
plt.title("Training History")
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
    ax[i][0].set_title(f"Denoised image: {train_img[i]}")

    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
    ax[i][1].set_title(f"Predicted image: {train_img[i]}")

    ax[i][0].axis("off")
    ax[i][1].axis("off")

del decoded_imgs




## === cell 15
ids_list = []
vals_list = []

for i, f in tqdm(enumerate(test_img), total=len(test_img)):
    file = path + "test/" + f
    imgid = int(f[:-4])
    img_raw = cv2.imread(file, 0)
    orig_h, orig_w = img_raw.shape

    decoded_img = np.squeeze(
        autoencoder.decoder(autoencoder.encoder(test[i : i + 1])).numpy()
    )
    preds_resized = cv2.resize(decoded_img, (orig_w, orig_h))

    rows = np.arange(1, orig_h + 1).reshape(-1, 1)
    cols = np.arange(1, orig_w + 1).reshape(1, -1)
    ids_grid = np.char.add(
        np.char.add(
            np.char.add(np.full((orig_h, orig_w), f"{imgid}_"), rows.astype(str)), "_"
        ),
        cols.astype(str),
    )
    ids_list.append(ids_grid.ravel())
    vals_list.append(preds_resized.ravel())

ids = np.concatenate(ids_list)
vals = np.concatenate(vals_list)

print("Length of IDs:", len(ids))
submission = pd.DataFrame({"id": ids, "value": vals})
submission.to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/2904342895.py in <cell line: 0>()
     11         autoencoder.decoder(autoencoder.encoder(test[i : i + 1])).numpy()
     12     )
---> 13     preds_resized = cv2.resize(decoded_img, (orig_w, orig_h))
     14 
     15     rows = np.arange(1, orig_h + 1).reshape(-1, 1)

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4086: error: (-215:Assertion failed) func != 0 in function 'resize'
