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

0.03254

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.28616) has done: 'I first fix the crash caused by an incompatibility between `keras` (v3) and the protobuf runtime by removing the standalone `keras` import and consistently using `tf.keras` throughout. Then I fix two data/logic issues that are hurting score: ensure images and labels are paired by filename (not separately listed/shuffled), and do a proper train/validation split before fitting (so validation is meaningful and the LR scheduler can work). Finally, I make submission generation deterministic and correctly aligned to the sample submission order by parsing each `id` and indexing into the correct predicted image/row/col, and I write the output to a `.csv` file as required.'

# 9. Code solution

## === cell 0
import os
import zipfile
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

np.random.seed(42)
tf.random.set_seed(42)



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
for zpath in [train_zip_path, test_zip_path, sample_zip_path, trainclean_zip_path]:
    with zipfile.ZipFile(zpath, "r") as zip_ref:
        zip_ref.extractall(extracting_path)

print("Extracted to:", extracting_path)
print("Exists train:", os.path.exists(os.path.join(extracting_path, "train")))
print("Exists test:", os.path.exists(os.path.join(extracting_path, "test")))
print(
    "Exists train_cleaned:",
    os.path.exists(os.path.join(extracting_path, "train_cleaned")),
)
print(
    "Exists sampleSubmission.csv:",
    os.path.exists(os.path.join(extracting_path, "sampleSubmission.csv")),
)



## === cell 3
img_arr = mpimg.imread(os.path.join(extracting_path, "train", "107.png"))
h, w = img_arr.shape
print("Height: ", h, "- Width: ", w)
print("dtype:", img_arr.dtype)



## === cell 4
image_names = os.listdir(os.path.join(extracting_path, "train"))
data_size = len(image_names)
X_shapes = np.zeros([data_size, 2], dtype=np.uint16)
for i in tqdm(range(data_size)):
    image_name = image_names[i]
    img_dir = os.path.join(extracting_path, "train", image_name)
    img_pixels = mpimg.imread(img_dir)
    X_shapes[i] = img_pixels.shape

print("Number of training images:", data_size)
print("Differnet image hights: {}".format(set(X_shapes[:, 0])))
print("Differnet image widths: {}".format(set(X_shapes[:, 1])))




## === cell 5
def images_to_array(data_dir, label_dir=None, img_size=(h, w)):
    """
    Fixes a key logic bug: ensure input images are paired with their correct cleaned labels
    by sorting filenames and using the same names in both directories (no independent listing).
    """
    image_names = sorted(
        [f for f in os.listdir(data_dir) if f.lower().endswith(".png")]
    )
    data_size_local = len(image_names)

    X = np.zeros([data_size_local, img_size[0], img_size[1]], dtype=np.uint8)
    for i, image_name in enumerate(tqdm(image_names, desc="Loading X")):
        img_dir = os.path.join(data_dir, image_name)
        img_pixels = load_img(img_dir, color_mode="grayscale", target_size=img_size)
        X[i] = np.array(img_pixels, dtype=np.uint8)

    X = X.reshape(data_size_local, img_size[0], img_size[1], 1)

    if label_dir is not None:
        y = np.zeros([data_size_local, img_size[0], img_size[1]], dtype=np.uint8)
        missing = []
        for i, image_name in enumerate(tqdm(image_names, desc="Loading y")):
            lbl_path = os.path.join(label_dir, image_name)
            if not os.path.exists(lbl_path):
                missing.append(image_name)
                continue
            img_pixels = load_img(
                lbl_path, color_mode="grayscale", target_size=img_size
            )
            y[i] = np.array(img_pixels, dtype=np.uint8)

        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} label files, e.g. {missing[:5]}"
            )

        y = y.reshape(data_size_local, img_size[0], img_size[1], 1)

        print("Output Data Size:", X.shape)
        print("Output Label Size:", y.shape)
        return X / 255.0, y / 255.0, image_names

    print("Output Data Size:", X.shape)
    return X / 255.0, image_names




## === cell 6
X, y, train_names = images_to_array(
    os.path.join(extracting_path, "train"),
    os.path.join(extracting_path, "train_cleaned"),
    img_size=(h, w),
)



## === cell 7
n = X.shape[0]
val_split = int(0.3 * n)

X_val, y_val = X[:val_split], y[:val_split]
X_train, y_train = X[val_split:], y[val_split:]

print("Train data shape:", X_train.shape)
print("Val data shape:", X_val.shape)



## === cell 8
samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)
f, ax = plt.subplots(2, 3, figsize=(20, 10))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 9
input_layer = Input(shape=(None, None, 1))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()



## === cell 10
LR_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=4, verbose=1, factor=0.4, min_lr=1e-5
)



## === cell 11
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=[LR_callback],
    verbose=2,
)



## === cell 12
val_loss = model.evaluate(X_val, y_val, verbose=0)
print("Validation loss (MSE):", val_loss)



## === cell 13
test_samples, test_labels = X_val[:3], y_val[:3]
test_pred = model.predict(X_val[:3], verbose=0)

samples = np.concatenate((test_samples, test_labels, test_pred), axis=0)
f, ax = plt.subplots(3, 3, figsize=(25, 15))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 14
test_dir = os.path.join(extracting_path, "test")
test_image_names = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith(".png")],
    key=lambda x: int(os.path.splitext(x)[0]),
)
test_ids = [int(os.path.splitext(f)[0]) for f in test_image_names]

X_test = []
for image_name in tqdm(test_image_names, desc="Loading test"):
    img_dir = os.path.join(test_dir, image_name)
    img_pixels = load_img(img_dir, color_mode="grayscale")
    w0, h0 = img_pixels.size  # PIL gives (width, height)
    X_test.append(np.array(img_pixels, dtype=np.uint8).reshape(1, h0, w0, 1) / 255.0)

print("Test sample shape:", X_test[0].shape)
print("Test sample dtype:", X_test[0].dtype)
print("Num test images:", len(X_test))



## === cell 15
yh_test = []
for img in tqdm(X_test, desc="Predicting test"):
    yh_test.append(model.predict(img, verbose=0)[0, :, :, 0])

pred_by_imgid = {img_id: pred for img_id, pred in zip(test_ids, yh_test)}



## === cell 16
f, ax = plt.subplots(3, 2, figsize=(20, 10))
for i, (img, pred) in enumerate(zip(X_test[:3], yh_test[:3])):
    ax[i, 0].imshow(img[0, :, :, 0], cmap="gray")
    ax[i, 0].axis("off")

    ax[i, 1].imshow(pred, cmap="gray")
    ax[i, 1].axis("off")
plt.show()



## === cell 17
sample_csv = pd.read_csv(os.path.join(extracting_path, "sampleSubmission.csv"))
ids = sample_csv["id"].astype(str).values

parts = np.char.split(ids, sep="_")
img_ids = np.fromiter((int(p[0]) for p in parts), dtype=np.int32, count=len(parts))
rows = np.fromiter((int(p[1]) for p in parts), dtype=np.int32, count=len(parts)) - 1
cols = np.fromiter((int(p[2]) for p in parts), dtype=np.int32, count=len(parts)) - 1

values = np.empty(len(ids), dtype=np.float32)
for img_id in np.unique(img_ids):
    mask = img_ids == img_id
    pred_img = pred_by_imgid[img_id]
    values[mask] = pred_img[rows[mask], cols[mask]]

submission = pd.DataFrame({"id": ids, "value": values})
submission.to_csv("Cleared.csv", index=False)
print(submission.head())
print("Wrote submission:", os.path.abspath("Cleared.csv"), "rows:", len(submission))



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/966726700.py in <cell line: 0>()
      4 
      5 # Parse ids like "image_row_col" (1-indexed)
----> 6 parts = np.char.split(ids, sep="_")
      7 img_ids = np.fromiter((int(p[0]) for p in parts), dtype=np.int32, count=len(parts))
      8 rows = np.fromiter((int(p[1]) for p in parts), dtype=np.int32, count=len(parts)) - 1

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in split(a, sep, maxsplit)
   1537     # This will return an array of lists of different sizes, so we
   1538     # leave it as an object array
-> 1539     return _vec_string(
   1540         a, object_, 'split', [sep] + _clean_args(maxsplit))
   1541 

TypeError: string operation on non-string array

## === cell 18
for folder in ["train", "test", "train_cleaned"]:
    p = os.path.join(extracting_path, folder)
    if os.path.exists(p):
        shutil.rmtree(p)
print("Cleanup done.")
