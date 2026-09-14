# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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
tf_keras==2.18.0
tqdm==4.67.1

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation,
    BatchNormalization,
    Flatten,
    Conv2D,
)
from tensorflow.keras.layers import MaxPooling2D, Dropout, UpSampling2D



## === cell 1
train_zip_path = "/kaggle/input/denoising-dirty-documents/train.zip"
test_zip_path = "/kaggle/input/denoising-dirty-documents/test.zip"
sample_zip_path = "/kaggle/input/denoising-dirty-documents/sampleSubmission.csv.zip"
trainclean_zip_path = "/kaggle/input/denoising-dirty-documents/train_cleaned.zip"
extracting_path = "/kaggle/working"



## === cell 2
import zipfile

with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
    zip_ref.extractall(extracting_path)

with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
    zip_ref.extractall(extracting_path)

with zipfile.ZipFile(sample_zip_path, "r") as zip_ref:
    zip_ref.extractall(extracting_path)

with zipfile.ZipFile(trainclean_zip_path, "r") as zip_ref:
    zip_ref.extractall(extracting_path)



## === cell 3
img_arr = mpimg.imread(extracting_path + "/train/107.png")
h, w = img_arr.shape
print("Height: ", h, "- Width: ", w)
print(img_arr.dtype)



## === cell 4
image_names = os.listdir(extracting_path + "/train")
data_size = len(image_names)
X = np.zeros([data_size, 2], dtype=np.uint16)
for i in tqdm(range(data_size)):
    image_name = image_names[i]
    img_dir = os.path.join(extracting_path + "/train", image_name)
    img_pixels = mpimg.imread(img_dir)
    X[i] = img_pixels.shape

print("Number of training images:", data_size)
print("Differnet image hights: {}".format(set(X[:, 0])))
print("Differnet image widths: {}".format(set(X[:, 1])))




## === cell 5
def images_to_array(data_dir, label_dir=None, img_size=(h, w)):
    """
    1- Read image samples from certain directory.
    2- Stack them into one big numpy array.
    -- And if there are labels images ..
    3- Read sample's label form the labels directory.
    4- Stack them into one big numpy array.
    5- Shuffle Data and label arrays.
    """
    image_names = sorted(os.listdir(data_dir))
    data_size = len(image_names)
    X = np.zeros([data_size, img_size[0], img_size[1]], dtype=np.uint8)
    for i in tqdm(range(data_size)):
        image_name = image_names[i]
        img_dir = os.path.join(data_dir, image_name)
        img_pixels = load_img(img_dir, color_mode="grayscale", target_size=(h, w))
        X[i] = img_pixels
    X = X.reshape(data_size, h, w, 1)

    if label_dir:
        label_names = sorted(os.listdir(label_dir))
        data_size = len(label_names)
        y = np.zeros([data_size, img_size[0], img_size[1]], dtype=np.uint8)
        for i in tqdm(range(data_size)):
            image_name = label_names[i]
            img_dir = os.path.join(label_dir, image_name)
            img_pixels = load_img(img_dir, color_mode="grayscale", target_size=(h, w))
            y[i] = img_pixels
        y = y.reshape(data_size, h, w, 1)
        ind = np.random.permutation(data_size)
        X = X[ind]
        y = y[ind]
        print("Ouptut Data Size: ", X.shape)
        print("Ouptut Label Size: ", y.shape)
        return X / 255.0, y / 255.0

    print("Ouptut Data Size: ", X.shape)
    return X / 255.0




## === cell 6
X, y = images_to_array(extracting_path + "/train", extracting_path + "/train_cleaned")



## === cell 7
val_split = int(0.3 * data_size)
X_val, y_val = X[:val_split], y[:val_split]
X_train, y_train = X[val_split:], y[val_split:]
print("Train data shape: ", X_train.shape)
print("Test data shape: ", X_val.shape)



## === cell 8
samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)

f, ax = plt.subplots(2, 3, figsize=(20, 10))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 9
import keras

input_layer = Input(shape=(None, None, 1))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
model = keras.models.Model(inputs=[input_layer], outputs=[output_layer])

sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
rms = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
ada = keras.optimizers.Adagrad(learning_rate=0.01)

model.compile(optimizer="adam", loss="mean_squared_error")



## === cell 10
LR_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=4, verbose=10, factor=0.4, min_lr=0.00001
)



## === cell 11
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[LR_callback],
    verbose=1,
)



## === cell 12
model.evaluate(X_val, y_val)



## === cell 13
test_samples, test_labels = X_val[:3], y_val[:3]
test_pred = model.predict(X_val[:3])

samples = np.concatenate((test_samples, test_labels, test_pred), axis=0)

f, ax = plt.subplots(3, 3, figsize=(25, 15))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 14
image_names = sorted(os.listdir(extracting_path + "/test"))
data_size = len(image_names)
X_test = []
for i in tqdm(range(data_size)):
    image_name = image_names[i]
    img_dir = os.path.join(extracting_path + "/test", image_name)
    img_pixels = load_img(img_dir, color_mode="grayscale", target_size=(h, w))
    X_test.append(np.array(img_pixels).reshape(1, h, w, 1) / 255.0)

print("Test sample shape: ", X_test[0].shape)
print("Test sample dtype: ", X_test[0].dtype)



## === cell 15
yh_test = []
for img in X_test:
    pred = model.predict(img, verbose=0)[0, :, :, 0]
    yh_test.append(np.clip(pred, 0.0, 1.0))



## === cell 16
f, ax = plt.subplots(3, 2, figsize=(20, 10))
for i, (img, lbl) in enumerate(zip(X_test[:3], yh_test[:3])):
    ax[i, 0].imshow(img[0, :, :, 0], cmap="gray")
    ax[i, 0].axis("off")

    ax[i, 1].imshow(lbl, cmap="gray")
    ax[i, 1].axis("off")
plt.show()



## === cell 17
submit_vector = []
for img in yh_test:
    hi, wi = img.shape
    for i in range(wi):
        for j in range(hi):
            submit_vector.append(float(img[j, i]))
print(len(submit_vector))



## === cell 18
sample_csv = pd.read_csv(extracting_path + "/sampleSubmission.csv")
sample_csv.head(10)



## === cell 19
submit_vector = []
for pred_img, x_img in zip(yh_test, X_test):
    hi, wi = x_img.shape[1], x_img.shape[2]  # original test image size
    if pred_img.shape != (hi, wi):
        pred_img_resized = tf.image.resize(
            pred_img[..., None], size=(hi, wi), method="bilinear"
        ).numpy()[:, :, 0]
        pred_img_resized = np.clip(pred_img_resized, 0.0, 1.0)
    else:
        pred_img_resized = pred_img

    for i in range(wi):
        for j in range(hi):
            submit_vector.append(float(pred_img_resized[j, i]))

if len(submit_vector) != len(sample_csv):
    raise ValueError(
        f"Submission length mismatch: predicted {len(submit_vector)} pixels, "
        f"but sampleSubmission has {len(sample_csv)} rows."
    )


## --- ERROR in cell 19, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1060619412.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     18[0m [0;34m[0m[0m
[1;32m     19[0m [0;32mif[0m [0mlen[0m[0;34m([0m[0msubmit_vector[0m[0;34m)[0m [0;34m!=[0m [0mlen[0m[0;34m([0m[0msample_csv[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 20[0;31m     raise ValueError(
[0m[1;32m     21[0m         [0;34mf"Submission length mismatch: predicted {len(submit_vector)} pixels, "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m         [0;34mf"but sampleSubmission has {len(sample_csv)} rows."[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Submission length mismatch: predicted 6577200 pixels, but sampleSubmission has 5789880 rows.

## === cell 20
c = 0
for img in yh_test:
    hi, wi = img.shape
    c += hi * wi
