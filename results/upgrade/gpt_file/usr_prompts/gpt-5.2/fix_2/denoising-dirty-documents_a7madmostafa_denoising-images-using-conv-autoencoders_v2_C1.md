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

3.10

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.03636

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.28616) has done: 'I fix the crash in the first import cell caused by an incompatibility between TensorFlow 2.18 and protobuf 6 by forcing the pure‑Python protobuf implementation before importing TensorFlow/Keras. I also make the training deterministic (fixed seeds) and correct the EarlyStopping monitor to `val_loss` so it stops based on generalization instead of just training loss; this is a minimal change that typically improves RMSE without changing the model architecture or loss. Finally, I fix a subtle shape issue in submission generation (OpenCV resize expects 2D) and ensure the output values are valid floats clipped to [0, 1] and the submission is written as `submission.csv` in the working directory.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import cv2

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_zip = "/kaggle/input/denoising-dirty-documents/"
path = "/kaggle/working/"

with zipfile.ZipFile(path_zip + "train.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "test.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "train_cleaned.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "sampleSubmission.csv.zip", "r") as zip_ref:
    zip_ref.extractall(path)



## === cell 2
train_img = sorted(os.listdir(path + "/train"))
train_cleaned_img = sorted(os.listdir(path + "/train_cleaned"))
test_img = sorted(os.listdir(path + "/test"))




## === cell 3
def process_image(path_):
    img = cv2.imread(path_)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (540, 420))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (420, 540, 1))
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



## === cell 5
plt.figure(figsize=(15, 25))
for i in range(0, 8, 2):
    plt.subplot(4, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train[i][:, :, 0], cmap="gray")
    plt.title("Noise image: {}".format(train_img[i]))

    plt.subplot(4, 2, i + 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_cleaned[i][:, :, 0], cmap="gray")
    plt.title("Denoised image: {}".format(train_img[i]))

plt.show()



## === cell 6
X_train = np.asarray(train, dtype=np.float32)
y_train = np.asarray(train_cleaned, dtype=np.float32)
X_test = np.asarray(test, dtype=np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=SEED
)



## === cell 7
conv_autoencoder = Sequential()
conv_autoencoder.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        input_shape=(420, 540, 1),
        activation="relu",
        padding="same",
    )
)
conv_autoencoder.add(
    Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")
)
conv_autoencoder.add(
    Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")
)
conv_autoencoder.add(
    Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")
)
conv_autoencoder.add(
    Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")
)
conv_autoencoder.add(
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
)
conv_autoencoder.add(
    Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same")
)

conv_autoencoder.summary()



## === cell 8
conv_autoencoder.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

conv_autoencoder.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1,
)



## === cell 9
y_pred = conv_autoencoder.predict(X_test, batch_size=16, verbose=1)



## === cell 10
plt.figure(figsize=(15, 25))
for i in range(0, 8, 2):
    plt.subplot(4, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i][:, :, 0], cmap="gray")
    plt.title("Noisy image: {}".format(test_img[i]))

    plt.subplot(4, 2, i + 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_pred[i][:, :, 0], cmap="gray")
    plt.title("Denoised by autoencoder: {}".format(test_img[i]))

plt.show()



## === cell 11
ids = []
vals = []

for i, f in enumerate(test_img):
    file = path + "test/" + f
    imgid = int(f[:-4])
    img = cv2.imread(file, 0)
    img_shape = img.shape

    pred2d = y_pred[i, :, :, 0]
    preds_resized = cv2.resize(
        pred2d, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR
    )

    preds_resized = np.clip(preds_resized, 0.0, 1.0).astype(np.float32)

    for r in range(img_shape[0]):
        base = f"{imgid}_{r+1}_"
        for c in range(img_shape[1]):
            ids.append(base + str(c + 1))
            vals.append(float(preds_resized[r, c]))

submission = pd.DataFrame({"id": ids, "value": vals})
submission.to_csv("submission.csv", index=False)

print("Results saved to submission.csv!")
print("Length of IDs: {}".format(len(ids)))



## === cell 12
my_submission = pd.read_csv("submission.csv")
my_submission.head(5)
