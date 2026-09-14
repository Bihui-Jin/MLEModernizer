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

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
)

from tensorflow.keras import mixed_precision
import csv  # used for fast CSV writing

mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU memory growth setup failed:", e)

plt.rcParams["figure.figsize"] = (10.0, 5.0)  # default plot size




## === cell 1
import zipfile

zipfiles = ["train", "test", "train_cleaned"]
if not (
    os.path.isdir("./train")
    and os.path.isdir("./test")
    and os.path.isdir("./train_cleaned")
):
    for each_zip in zipfiles:
        zip_path = f"/kaggle/input/denoising-dirty-documents/{each_zip}.zip"
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("./")
else:
    print("Data already extracted; skipping zip extraction.")




## === cell 2
def load_image_from_dir(img_path):
    """Load all PNG images from a directory, resize to (258,540) and normalize."""
    file_list = sorted(glob.glob(os.path.join(img_path, "*.png")))
    n = len(file_list)
    img_array = np.empty((n, 258, 540, 1), dtype=np.float32)
    for i, fig in enumerate(file_list):
        raw = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(raw, (540, 258))  # width, height
        img_array[i, :, :, 0] = resized.astype(np.float32) / 255.0
    return img_array


def train_val_split(data, random_seed=55, split=0.75):
    """Simple shuffled split."""
    rng = np.random.RandomState(seed=random_seed)
    dsize = len(data)
    indices = rng.permutation(dsize)
    cut = int(split * dsize)
    return data[indices[:cut]], data[indices[cut:]]




## === cell 3
full_train = load_image_from_dir("./train")
full_target = load_image_from_dir("./train_cleaned")
test = load_image_from_dir("./test")

train, val = train_val_split(full_train, random_seed=9, split=0.8)
target_train, target_val = train_val_split(full_target, random_seed=9, split=0.8)




## === cell 4
optimizer = Adam(learning_rate=1e-4)




## === cell 5
input_layer = Input(shape=train[0].shape)  # (258,540,1)

h = Conv2D(128, (3, 3), activation="relu", padding="same")(input_layer)
h = BatchNormalization()(h)
h = Conv2D(256, (3, 3), activation="relu", padding="same")(h)
h = BatchNormalization()(h)
h = MaxPooling2D((2, 2), padding="same")(h)

h = Conv2D(256, (3, 3), activation="relu", padding="same")(h)
h = BatchNormalization()(h)

h = Conv2D(64, (3, 3), activation="relu", padding="same")(h)
h = BatchNormalization()(h)

h = Conv2D(128, (3, 3), activation="relu", padding="same")(h)
h = BatchNormalization()(h)
h = UpSampling2D((2, 2))(h)

output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(h)

ae_model = Model(inputs=input_layer, outputs=output_layer)
ae_model.compile(loss="mse", optimizer=optimizer)
ae_model.summary()




## === cell 6
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=80,  # longer patience to allow more training
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)




## === cell 7
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train, target_train))
    .shuffle(buffer_size=len(train), reshuffle_each_iteration=True)
    .cache()
    .batch(20, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((val, target_val))
    .cache()
    .batch(20, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

history = ae_model.fit(
    train_dataset,
    epochs=250,
    validation_data=val_dataset,
    callbacks=[early_stopping],
    verbose=2,
)




## === cell 8
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
plt.show()




## === cell 9
preds = ae_model.predict(test, batch_size=20)




## === cell 10
if test.shape[0] > 0:
    plt.imshow(test[0].reshape(258, 540), cmap="gray")
    plt.title("Noisy Input (first test image)")
    plt.show()
if preds.shape[0] > 0:
    plt.imshow(preds[0].reshape(258, 540), cmap="gray")
    plt.title("Denoised Output (first test image)")
    plt.show()




## === cell 11
file_list = sorted(glob.glob("./test/*.png"))  # same order as loading

print("Writing to csv file")
with open("submission.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "value"])  # header

    for i, f in enumerate(file_list):
        img_id = os.path.splitext(os.path.basename(f))[0]  # strip .png
        raw_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = raw_img.shape

        pred_img = (
            preds[i] if i < len(preds) else np.zeros((258, 540, 1), dtype=np.float32)
        )
        pred_resized = cv2.resize(pred_img.squeeze(), (orig_w, orig_h))

        rows = np.repeat(np.arange(1, orig_h + 1), orig_w)
        cols = np.tile(np.arange(1, orig_w + 1), orig_h)
        rows_str = rows.astype(str)
        cols_str = cols.astype(str)
        ids_arr = np.core.defchararray.add(
            np.core.defchararray.add(
                np.core.defchararray.add(img_id + "_", rows_str + "_"),
                cols_str,
            ),
            "",
        )

        values_arr = pred_resized.ravel().astype(str)

        writer.writerows(zip(ids_arr.tolist(), values_arr.tolist()))
