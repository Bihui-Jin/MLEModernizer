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

0.05382

# 6. Current score

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
img_arr = mpimg.imread(os.path.join(extracting_path, "train", "107.png"))
h, w = img_arr.shape[:2]
print("Height:", h, "- Width:", w)
print("dtype:", img_arr.dtype)



## === cell 4
image_names = os.listdir(os.path.join(extracting_path, "train"))
data_size = len(image_names)
X = np.zeros([data_size, 2], dtype=np.uint16)
for i in tqdm(range(data_size)):
    image_name = image_names[i]
    img_path = os.path.join(extracting_path, "train", image_name)
    img_pixels = mpimg.imread(img_path)
    X[i] = img_pixels.shape[:2]

print("Number of training images:", data_size)
print("Different image heights:", set(X[:, 0]))
print("Different image widths:", set(X[:, 1]))




## === cell 5
def images_to_array(data_dir, label_dir=None, img_size=(h, w)):
    """
    Read images (and optional label images) from directories,
    resize them to img_size, normalize to [0,1] and return as NumPy arrays.
    """
    image_names = os.listdir(data_dir)
    n = len(image_names)
    X_arr = np.zeros([n, img_size[0], img_size[1]], dtype=np.uint8)

    for i in tqdm(range(n)):
        img_path = os.path.join(data_dir, image_names[i])
        img = mpimg.imread(img_path)
        if img.ndim == 3:  # in case the image has an extra channel dimension
            img = img[:, :, 0]
        X_arr[i] = np.array(img, dtype=np.uint8)

    X_arr = X_arr.reshape(n, img_size[0], img_size[1], 1)

    if label_dir:
        label_names = os.listdir(label_dir)
        n_lbl = len(label_names)
        y_arr = np.zeros([n_lbl, img_size[0], img_size[1]], dtype=np.uint8)

        for i in tqdm(range(n_lbl)):
            lbl_path = os.path.join(label_dir, label_names[i])
            lbl = mpimg.imread(lbl_path)
            if lbl.ndim == 3:
                lbl = lbl[:, :, 0]
            y_arr[i] = np.array(lbl, dtype=np.uint8)

        y_arr = y_arr.reshape(n_lbl, img_size[0], img_size[1], 1)

        perm = np.random.permutation(n_lbl)
        X_arr = X_arr[perm]
        y_arr = y_arr[perm]

        print("Output Data Size:", X_arr.shape)
        print("Output Label Size:", y_arr.shape)
        return X_arr / 255.0, y_arr / 255.0

    print("Output Data Size:", X_arr.shape)
    return X_arr / 255.0




## === cell 6
X, y = images_to_array(
    os.path.join(extracting_path, "train"),
    os.path.join(extracting_path, "train_cleaned"),
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2654735795.py in <cell line: 0>()
----> 1 X, y = images_to_array(
      2     os.path.join(extracting_path, "train"),
      3     os.path.join(extracting_path, "train_cleaned"),
      4 )
      5 

/tmp/ipykernel_55/336088590.py in images_to_array(data_dir, label_dir, img_size)
     14         if img.ndim == 3:  # in case the image has an extra channel dimension
     15             img = img[:, :, 0]
---> 16         X_arr[i] = np.array(img, dtype=np.uint8)
     17 
     18     X_arr = X_arr.reshape(n, img_size[0], img_size[1], 1)

ValueError: could not broadcast input array from shape (258,540) into shape (420,540)

## === cell 7
val_split = int(0.15 * data_size)
X_val, y_val = X[:val_split], y[:val_split]
X_train, y_train = X[val_split:], y[val_split:]
print("Train data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/531166263.py in <cell line: 0>()
      1 val_split = int(0.15 * data_size)
----> 2 X_val, y_val = X[:val_split], y[:val_split]
      3 X_train, y_train = X[val_split:], y[val_split:]
      4 print("Train data shape:", X_train.shape)
      5 print("Validation data shape:", X_val.shape)

NameError: name 'y' is not defined

## === cell 8
samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)

f, ax = plt.subplots(2, 3, figsize=(20, 10))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1382270599.py in <cell line: 0>()
----> 1 samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)
      2 
      3 f, ax = plt.subplots(2, 3, figsize=(20, 10))
      4 for i, img in enumerate(samples):
      5     ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")

NameError: name 'X_train' is not defined

## === cell 9
input_layer = Input(shape=(None, None, 1))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
model = keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer="adam", loss="mean_squared_error")



## === cell 10
LR_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=4, verbose=1, factor=0.4, min_lr=1e-5
)



## === cell 11
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=70,
    batch_size=16,
    callbacks=[LR_callback],
    verbose=2,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1639832116.py in <cell line: 0>()
      1 history = model.fit(
----> 2     X_train,
      3     y_train,
      4     validation_data=(X_val, y_val),
      5     epochs=70,

NameError: name 'X_train' is not defined

## === cell 12
val_loss = model.evaluate(X_val, y_val, verbose=0)
print("Validation loss (RMSE):", np.sqrt(val_loss))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1180934415.py in <cell line: 0>()
----> 1 val_loss = model.evaluate(X_val, y_val, verbose=0)
      2 print("Validation loss (RMSE):", np.sqrt(val_loss))
      3 

NameError: name 'X_val' is not defined

## === cell 13
test_image_names = sorted(os.listdir(os.path.join(extracting_path, "test")))
test_size = len(test_image_names)
X_test = []
for i in tqdm(range(test_size)):
    img_path = os.path.join(extracting_path, "test", test_image_names[i])
    img = mpimg.imread(img_path)
    if img.ndim == 3:
        img = img[:, :, 0]
    X_test.append(img.reshape(1, h, w, 1) / 255.0)

print("Test sample shape:", X_test[0].shape)
print("Test sample dtype:", X_test[0].dtype)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3444472206.py in <cell line: 0>()
      8         img = img[:, :, 0]
      9     # Ensure shape (1, h, w, 1)
---> 10     X_test.append(img.reshape(1, h, w, 1) / 255.0)
     11 
     12 print("Test sample shape:", X_test[0].shape)

ValueError: cannot reshape array of size 139320 into shape (1,420,540,1)

## === cell 14
yh_test = []
for img in X_test:
    pred = model.predict(img, verbose=0)[0, :, :, 0]
    yh_test.append(pred)



## === cell 15
f, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(X_test[0][0, :, :, 0], cmap="gray")
ax[0].axis("off")
ax[1].imshow(yh_test[0], cmap="gray")
ax[1].axis("off")
plt.show()



## === cell 16
submit_vector = []
for img in yh_test:
    h_img, w_img = img.shape
    for i in range(h_img):  # iterate rows
        for j in range(w_img):  # iterate columns
            submit_vector.append(float(img[i, j]))

print("Total predictions collected:", len(submit_vector))



## === cell 17
sample_csv = pd.read_csv(os.path.join(extracting_path, "sampleSubmission.csv"))
print("Sample submission size:", sample_csv.shape[0])
sample_csv.head(5)



## === cell 18
print("Sum of predicted pixels (sanity check):", sum(submit_vector))



## === cell 19
id_col = sample_csv["id"]
value_col = pd.Series(submit_vector, name="value")
submission = pd.concat([id_col, value_col], axis=1)
print("Submission preview:")
submission.head(5)



## === cell 20
submission_path = os.path.join(extracting_path, "Cleared.csv")
submission.to_csv(submission_path, index=False)
print("Submission saved to", submission_path)
