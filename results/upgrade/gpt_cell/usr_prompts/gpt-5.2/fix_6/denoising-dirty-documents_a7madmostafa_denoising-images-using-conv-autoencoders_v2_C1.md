# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.04464

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens immediately on importing TensorFlow/Keras in cell 0, before any model code runs. With TensorFlow 2.18.0 and protobuf 6.33.0, this specific `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` is a known protobuf API incompatibility that occurs when TensorFlow ends up using the C++ protobuf implementation. Forcing protobuf to use the pure-Python implementation (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) avoids the missing method and lets TensorFlow import successfully.

Patch summary: In cell 0 only, set the protobuf implementation environment variables *before* importing TensorFlow/Keras. Keep all other imports and logic unchanged.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: No variables, paths, or interfaces used by cell 1 are changed; this only ensures TensorFlow/Keras imports without crashing.

Assumptions: It’s acceptable to use the Python protobuf runtime for compatibility; performance impact is negligible for this notebook’s import phase and does not change model semantics.'
- What this solution (achieved 0.28616) has done: 'The crash happens during TensorFlow import in cell 0 due to an incompatibility between TensorFlow 2.18 and the installed `protobuf==6.33.0`, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The most deterministic minimal fix is to force the pure-Python protobuf implementation and use the v3 API behavior, which avoids the missing `GetPrototype` path. This requires changing only the environment variables set before importing TensorFlow/Keras. No model/training logic is touched, and cell 1 remains compatible since it depends only on standard libs already imported.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens during the TensorFlow/Keras import in cell 0, and the traceback (`'MessageFactory' object has no attribute 'GetPrototype'`) indicates an incompatibility between TensorFlow 2.18.0 and the installed `protobuf==6.33.0`. The environment-variable workaround in the cell does not resolve this API mismatch because the missing `GetPrototype` method is part of protobuf’s Python API that TensorFlow expects. The minimal deterministic fix is to pin protobuf to a TensorFlow-compatible version (protobuf 4.x) before importing TensorFlow.

Patch summary: In cell 0 only, install a compatible protobuf version (`protobuf<5`, which resolve to a 4.x release) via pip at runtime, then proceed with the existing imports unchanged. This directly fixes the TensorFlow import crash without changing any modeling/training logic or downstream variables.

Updated cells: cell 0 only (buggy cell).

Compatibility notes for cell k+1: No variable names, types, or interfaces used by later cells are changed; the patch only ensures TensorFlow imports successfully so cell 1 and later cells can run as written.

Assumptions: The runtime allows `pip` installs (typical in notebook environments), and downgrading protobuf does not conflict with other already-imported packages (TensorFlow is imported after the downgrade in this cell).'
- What this solution (achieved 0.28616) has done: 'Your current score is far from the target (0.28616 vs 0.03636; lower is better), so we need a real quality improvement but still keep the same autoencoder core. The biggest issue hurting RMSE here is a mismatch between training preprocessing (resized to 540×420) and submission generation (predictions are resized back to original size using OpenCV, but the resize call is fed a 3D array and uses default interpolation), plus the submission values may end up as arrays instead of scalars. I make minimal, metric-aligned fixes: ensure deterministic splits, use correct 2D resizing with `INTER_AREA`, clip predictions to [0,1], and build the submission in the exact `sampleSubmission.csv` order to guarantee perfect id/value alignment. These changes preserve the model, loss, and training loop, but should substantially reduce RMSE toward your target.'
- What this solution (achieved 0.04464) has done: 'Your current RMSE (0.28616) is far from the target (0.03636; lower is better), so the most impactful *minimal* improvement is to fix a likely train/label misalignment caused by independently sorting filenames from different folders (string sorting can mismatch pairs like `1.png`, `10.png`, `2.png`). I enforce pairing by image id (numeric sort and intersection) so each noisy train image is matched to its correct cleaned target, without changing the model, loss, or training loop. I also keep the submission aligned strictly to `sampleSubmission.csv` ids (as you already do), while making the image read in `process_image` explicitly grayscale to avoid any subtle cv2 channel/read differences. These changes preserve core logic but should materially reduce RMSE toward the target by ensuring the autoencoder learns the correct mapping.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_C_DESCRIPTORS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_C_MESSAGE"] = "0"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_API_VERSION", "3")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import cv2

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Dropout,
    BatchNormalization,
    Input,
)
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)



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
def _list_png_ids(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    id_to_file = {}
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem.isdigit():
            id_to_file[int(stem)] = f
    return id_to_file


train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

train_id2f = _list_png_ids(train_dir)
clean_id2f = _list_png_ids(train_cleaned_dir)
test_id2f = _list_png_ids(test_dir)

common_train_ids = sorted(set(train_id2f).intersection(clean_id2f))
test_ids = sorted(test_id2f)

train_img = [train_id2f[i] for i in common_train_ids]
train_cleaned_img = [clean_id2f[i] for i in common_train_ids]
test_img = [test_id2f[i] for i in test_ids]

print("Train pairs:", len(train_img), "Test images:", len(test_img))




## === cell 3
def process_image(path_):
    img = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (540, 420))
    img = img / 255.0
    img = np.reshape(img, (420, 540, 1))
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
X_train = np.asarray(train)
y_train = np.asarray(train_cleaned)
X_test = np.asarray(test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, shuffle=True
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

early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
conv_autoencoder.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
)



## === cell 9
y_pred = conv_autoencoder.predict(X_test, batch_size=16)
y_pred = np.clip(y_pred, 0.0, 1.0)



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
sample_path = os.path.join(path, "sampleSubmission.csv")
sample = pd.read_csv(sample_path)

pred_map = {}
for i, f in enumerate(test_img):
    imgid = int(f[:-4])
    file = os.path.join(path, "test", f)
    img = cv2.imread(file, 0)
    h, w = img.shape

    pred_2d = y_pred[i, :, :, 0]  # (420, 540)
    pred_resized = cv2.resize(pred_2d, (w, h), interpolation=cv2.INTER_AREA)
    pred_resized = np.clip(pred_resized, 0.0, 1.0).astype(np.float32)
    pred_map[imgid] = pred_resized


def value_from_id(id_str: str) -> float:
    img_s, r_s, c_s = id_str.split("_")
    imgid = int(img_s)
    r = int(r_s) - 1
    c = int(c_s) - 1
    return float(pred_map[imgid][r, c])


sample["value"] = sample["id"].map(value_from_id).astype(np.float32)
sample.to_csv("submission.csv", index=False)

print("Results saved to submission.csv!")
print("Submission rows:", len(sample))



## === cell 12
my_submission = pd.read_csv("submission.csv")
my_submission.head(5)
