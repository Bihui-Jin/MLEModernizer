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
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
        input/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
            test/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
            train/
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

# 5. Target score

0.05382

# 6. Current score

Not yielded

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
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D




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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4205682735.py in <cell line: 0>()
      1 import zipfile
      2 
----> 3 with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
      4     zip_ref.extractall(extracting_path)
      5 with zipfile.ZipFile(test_zip_path, "r") as zip_ref:

/usr/lib/python3.11/zipfile.py in __init__(self, file, mode, compression, allowZip64, compresslevel, strict_timestamps, metadata_encoding)
   1293             while True:
   1294                 try:
-> 1295                     self.fp = io.open(file, filemode)
   1296                 except OSError:
   1297                     if filemode in modeDict:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/denoising-dirty-documents/train.zip'

## === cell 3
img_arr = mpimg.imread(os.path.join(extracting_path, "train", "107.png"))
h, w = img_arr.shape[:2]
print("Common Height:", h, "- Common Width:", w)
print("dtype:", img_arr.dtype)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/65057198.py in <cell line: 0>()
      1 # Determine a common shape from the first training image
----> 2 img_arr = mpimg.imread(os.path.join(extracting_path, "train", "107.png"))
      3 h, w = img_arr.shape[:2]
      4 print("Common Height:", h, "- Common Width:", w)
      5 print("dtype:", img_arr.dtype)

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in imread(fname, format)
   1561             "``np.array(PIL.Image.open(urllib.request.urlopen(url)))``."
   1562             )
-> 1563     with img_open(fname) as image:
   1564         return (_pil_png_to_float_array(image)
   1565                 if isinstance(image, PIL.PngImagePlugin.PngImageFile) else

/usr/local/lib/python3.11/dist-packages/PIL/ImageFile.py in __init__(self, fp, filename)
    133         if is_path(fp):
    134             # filename
--> 135             self.fp = open(fp, "rb")
    136             self.filename = os.fspath(fp)
    137             self._exclusive_fp = True

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/107.png'

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




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3200689519.py in <cell line: 0>()
----> 1 image_names = os.listdir(os.path.join(extracting_path, "train"))
      2 data_size = len(image_names)
      3 X = np.zeros([data_size, 2], dtype=np.uint16)
      4 for i in tqdm(range(data_size)):
      5     image_name = image_names[i]

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train'

## === cell 5
def images_to_array(data_dir, label_dir=None, img_size=(h, w)):
    """
    Read images (and optional label images) from directories,
    resize them to img_size, normalize to [0,1] and return as NumPy arrays.
    """
    image_names = sorted(os.listdir(data_dir))
    n = len(image_names)
    X_arr = np.zeros([n, img_size[0], img_size[1], 1], dtype=np.float32)

    for i in tqdm(range(n)):
        img_path = os.path.join(data_dir, image_names[i])
        img = load_img(img_path, color_mode="grayscale", target_size=img_size)
        arr = img_to_array(img) / 255.0  # shape (h, w, 1)
        X_arr[i] = arr

    if label_dir:
        label_names = sorted(os.listdir(label_dir))
        n_lbl = len(label_names)
        y_arr = np.zeros([n_lbl, img_size[0], img_size[1], 1], dtype=np.float32)

        for i in tqdm(range(n_lbl)):
            lbl_path = os.path.join(label_dir, label_names[i])
            lbl = load_img(lbl_path, color_mode="grayscale", target_size=img_size)
            arr = img_to_array(lbl) / 255.0
            y_arr[i] = arr

        perm = np.random.permutation(n_lbl)
        X_arr = X_arr[perm]
        y_arr = y_arr[perm]

        print("Output Data Size:", X_arr.shape)
        print("Output Label Size:", y_arr.shape)
        return X_arr, y_arr

    print("Output Data Size:", X_arr.shape)
    return X_arr




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2536420790.py in <cell line: 0>()
----> 1 def images_to_array(data_dir, label_dir=None, img_size=(h, w)):
      2     """
      3     Read images (and optional label images) from directories,
      4     resize them to img_size, normalize to [0,1] and return as NumPy arrays.
      5     """

NameError: name 'h' is not defined

## === cell 6
X, y = images_to_array(
    os.path.join(extracting_path, "train"),
    os.path.join(extracting_path, "train_cleaned"),
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2666857409.py in <cell line: 0>()
----> 1 X, y = images_to_array(
      2     os.path.join(extracting_path, "train"),
      3     os.path.join(extracting_path, "train_cleaned"),
      4 )
      5 

NameError: name 'images_to_array' is not defined

## === cell 7
val_split = int(0.15 * data_size)
X_val, y_val = X[:val_split], y[:val_split]
X_train, y_train = X[val_split:], y[val_split:]
print("Train data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3119646351.py in <cell line: 0>()
----> 1 val_split = int(0.15 * data_size)
      2 X_val, y_val = X[:val_split], y[:val_split]
      3 X_train, y_train = X[val_split:], y[val_split:]
      4 print("Train data shape:", X_train.shape)
      5 print("Validation data shape:", X_val.shape)

NameError: name 'data_size' is not defined

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
/tmp/ipykernel_11/3396667151.py in <cell line: 0>()
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
/tmp/ipykernel_11/2881513842.py in <cell line: 0>()
      1 history = model.fit(
----> 2     X_train,
      3     y_train,
      4     validation_data=(X_val, y_val),
      5     epochs=70,

NameError: name 'X_train' is not defined

## === cell 12
val_loss = model.evaluate(X_val, y_val, verbose=0)
print("Validation loss (MSE):", val_loss)
print("Validation loss (RMSE):", np.sqrt(val_loss))




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2047278541.py in <cell line: 0>()
----> 1 val_loss = model.evaluate(X_val, y_val, verbose=0)
      2 print("Validation loss (MSE):", val_loss)
      3 print("Validation loss (RMSE):", np.sqrt(val_loss))
      4 
      5 

NameError: name 'X_val' is not defined

## === cell 13
test_image_names = sorted(os.listdir(os.path.join(extracting_path, "test")))
test_size = len(test_image_names)
X_test = np.zeros((test_size, h, w, 1), dtype=np.float32)

for i in tqdm(range(test_size)):
    img_path = os.path.join(extracting_path, "test", test_image_names[i])
    img = load_img(img_path, color_mode="grayscale", target_size=(h, w))
    arr = img_to_array(img) / 255.0
    X_test[i] = arr

print("Test data shape:", X_test.shape)
print("Test sample dtype:", X_test.dtype)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3299162985.py in <cell line: 0>()
----> 1 test_image_names = sorted(os.listdir(os.path.join(extracting_path, "test")))
      2 test_size = len(test_image_names)
      3 X_test = np.zeros((test_size, h, w, 1), dtype=np.float32)
      4 
      5 for i in tqdm(range(test_size)):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/test'

## === cell 14
yh_test = model.predict(X_test, verbose=0)  # shape (test_size, h, w, 1)
yh_test = yh_test.squeeze(axis=-1)  # shape (test_size, h, w)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4026544853.py in <cell line: 0>()
----> 1 yh_test = model.predict(X_test, verbose=0)  # shape (test_size, h, w, 1)
      2 yh_test = yh_test.squeeze(axis=-1)  # shape (test_size, h, w)
      3 
      4 

NameError: name 'X_test' is not defined

## === cell 15
f, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(X_test[0, :, :, 0], cmap="gray")
ax[0].axis("off")
ax[1].imshow(yh_test[0], cmap="gray")
ax[1].axis("off")
plt.show()




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2662022856.py in <cell line: 0>()
      1 f, ax = plt.subplots(1, 2, figsize=(20, 10))
----> 2 ax[0].imshow(X_test[0, :, :, 0], cmap="gray")
      3 ax[0].axis("off")
      4 ax[1].imshow(yh_test[0], cmap="gray")
      5 ax[1].axis("off")

NameError: name 'X_test' is not defined

## === cell 16
submit_vector = []
for img in yh_test:
    submit_vector.extend(img.ravel().astype(float).tolist())

print("Total predictions collected:", len(submit_vector))




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2765188650.py in <cell line: 0>()
      1 submit_vector = []
----> 2 for img in yh_test:
      3     submit_vector.extend(img.ravel().astype(float).tolist())
      4 
      5 print("Total predictions collected:", len(submit_vector))

NameError: name 'yh_test' is not defined

## === cell 17
sample_csv = pd.read_csv(os.path.join(extracting_path, "sampleSubmission.csv"))
print("Sample submission size:", sample_csv.shape[0])
sample_csv.head(5)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/700945672.py in <cell line: 0>()
----> 1 sample_csv = pd.read_csv(os.path.join(extracting_path, "sampleSubmission.csv"))
      2 print("Sample submission size:", sample_csv.shape[0])
      3 sample_csv.head(5)
      4 
      5 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/sampleSubmission.csv'

## === cell 18
print("Sum of predicted pixels (sanity check):", sum(submit_vector))




## === cell 19
id_col = sample_csv["id"]
value_col = pd.Series(submit_vector, name="value")
submission = pd.concat([id_col, value_col], axis=1)
print("Submission preview:")
submission.head(5)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3291263743.py in <cell line: 0>()
----> 1 id_col = sample_csv["id"]
      2 value_col = pd.Series(submit_vector, name="value")
      3 submission = pd.concat([id_col, value_col], axis=1)
      4 print("Submission preview:")
      5 submission.head(5)

NameError: name 'sample_csv' is not defined

## === cell 20
submission_path = os.path.join(extracting_path, "Cleared.csv")
submission.to_csv(submission_path, index=False)
print("Submission saved to", submission_path)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2043622332.py in <cell line: 0>()
      1 submission_path = os.path.join(extracting_path, "Cleared.csv")
----> 2 submission.to_csv(submission_path, index=False)
      3 print("Submission saved to", submission_path)

NameError: name 'submission' is not defined
