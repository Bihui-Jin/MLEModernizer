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

0.02688

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I remove the failing imgaug import and replace the augmentation pipeline with a no‑op version, so the script runs without errors and still trains the auto‑encoder on the original images. This fixes the import error, eliminates undefined augmenters, and ensures `processed_train` and `processed_train_cleaned` are valid arrays, allowing the model to train and a proper `submission.csv` to be written.'
- What this solution (achieved 0.28616) has done: 'The changes add a protobuf compatibility fix by setting an environment variable before importing TensorFlow, and adjust the training call to reserve a small validation split (10 %) so early stopping can use validation loss, which typically yields a better‑generalizing model and nudges the RMSE closer to the target score. No core architecture or processing logic is altered.'
- What this solution (achieved 0.28616) has done: 'I patched the protobuf incompatibility by adding a missing `GetPrototype` method before importing TensorFlow, reduced the dropout rate in the auto‑encoder to let the network learn richer features, increased the maximum epochs so early‑stopping can find a better‑fitting model, and switched the resizing interpolation to nearest‑neighbor when rebuilding the original‑size predictions for submission. These fixes eliminate the import error, improve model capacity modestly, and ensure the submission values more accurately reflect the model’s output, moving the RMSE toward the target score.'
- What this solution (achieved 0.28616) has done: 'I keep the existing auto‑encoder architecture but improve training and prediction so the model can learn better and produce more accurate pixel values. The changes are limited to the training call (using a lower learning rate, monitoring training loss, adding a ReduceLROnPlateau scheduler and a longer patience) and to the post‑processing step (using linear interpolation for resizing and clipping predictions to the valid [0, 1] range). These adjustments are expected to reduce the RMSE and move the score closer to the target while preserving the core logic of the solution.'
- What this solution (achieved 0.28616) has done: 'We replace the per‑pixel Python loops in the inference step with a fully vectorized construction of the submission IDs and values, and we run the model on the whole test set in one (or a few) batch predictions. This eliminates the inner double loop that caused the timeout while keeping the exact same predictions and submission format.'
- What this solution (achieved 0.28616) has done: 'The changes focus on eliminating the Python‑level training loop overhead.  
* In the training cell the NumPy arrays are wrapped in a `tf.data.Dataset`, shuffled, batched (size 32) and prefetched, then passed directly to `model.fit`. This keeps the exact model, loss and early‑stopping logic while dramatically reducing per‑epoch overhead, especially on CPU‑only runs.  
* In the prediction cell the batch size is increased to 32 to speed up inference without altering the results.  
All other code, model architecture, loss, callbacks and I/O remain unchanged.'

# 9. Code solution

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

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils, optimizers

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


imgs = [cv2.imread(path + "train/" + f) for f in sorted(os.listdir(path + "train/"))]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs




## === cell 3
def process_image(pth):
    img = cv2.imread(pth)
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
es = callbacks.EarlyStopping(
    monitor="loss", patience=50, verbose=1, restore_best_weights=True
)
lr_sched = callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
)

train_ds = (
    tf.data.Dataset.from_tensor_slices((processed_train, processed_train_cleaned))
    .shuffle(buffer=len(processed_train), seed=19)
    .batch(32)  # larger batch reduces per‑epoch overhead
    .prefetch(tf.data.AUTOTUNE)
)

history = autoencoder.fit(
    train_ds,
    epochs=2000,
    callbacks=[es, lr_sched],
    shuffle=False,  # shuffling already handled by the dataset
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4203932986.py in <cell line: 0>()
      9 train_ds = (
     10     tf.data.Dataset.from_tensor_slices((processed_train, processed_train_cleaned))
---> 11     .shuffle(buffer=len(processed_train), seed=19)
     12     .batch(32)  # larger batch reduces per‑epoch overhead
     13     .prefetch(tf.data.AUTOTUNE)

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 12
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
plt.show()
del history




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1304268417.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(20, 6))
----> 2 pd.DataFrame(history.history).plot(ax=ax)
      3 plt.show()
      4 del history
      5 

NameError: name 'history' is not defined

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
preds = autoencoder.predict(test, batch_size=32, verbose=0)  # shape (N, H, W, 1)

ids = []
vals = []

for i, f in enumerate(tqdm(test_img, desc="Building submission")):
    imgid = int(f[:-4])  # numeric id from filename
    orig_path = path + "test/" + f
    orig_img = cv2.imread(orig_path, 0)  # read original size (grayscale)
    orig_h, orig_w = orig_img.shape

    pred_img = preds[i, :, :, 0]

    pred_resized = cv2.resize(
        pred_img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )
    pred_resized = np.clip(pred_resized, 0.0, 1.0)

    rows = np.arange(orig_h) + 1  # 1‑based row indices
    cols = np.arange(orig_w) + 1  # 1‑based column indices
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    imgid_str = np.char.add(
        np.char.add(str(imgid) + "_", rr.astype(str)), "_" + cc.astype(str)
    )
    ids.extend(imgid_str.ravel().tolist())
    vals.extend(pred_resized.ravel().tolist())

print("Length of IDs: {}".format(len(ids)))
pd.DataFrame({"id": ids, "value": vals}).to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_55/299392100.py in <cell line: 0>()
     22     rr, cc = np.meshgrid(rows, cols, indexing="ij")
     23     imgid_str = np.char.add(
---> 24         np.char.add(str(imgid) + "_", rr.astype(str)), "_" + cc.astype(str)
     25     )
     26     ids.extend(imgid_str.ravel().tolist())

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U1'), dtype('<U21')) -> None
