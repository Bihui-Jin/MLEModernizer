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
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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
tf_keras==2.18.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Target score

0.61732

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio
import torch
from torch.utils import data
from tqdm.notebook import tqdm, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list):
        self.root_path = root_path
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        file_id = self.file_list[index]
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")
        image = np.array(imageio.imread(image_path), dtype=np.uint8)
        mask = np.array(imageio.imread(mask_path), dtype=np.uint8)
        return image, mask




## === cell 2
train_mask = pd.read_csv("../input/train.csv")
depth = pd.read_csv("../input/depths.csv")
train_path = "../input/train/"

file_list = list(train_mask["id"].values)
dataset = TGSSaltDataset(train_path, file_list)




## === cell 3
def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title("Image")
    axarr[1].set_title("Mask")




## === cell 4
for i in range(5):
    image, mask = dataset[np.random.randint(0, len(dataset))]
    plot2x2Array(image, mask)



## === cell 5
plt.figure(figsize=(6, 6))
plt.hist(depth["z"], bins=50)
plt.title("Depth distribution")




## === cell 6
def rleToMask(rleString, height, width):
    rows, cols = height, width
    try:
        rleNumbers = [int(numstring) for numstring in rleString.split(" ")]
        rlePairs = np.array(rleNumbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)
        for index, length in rlePairs:
            index -= 1
            img[index : index + length] = 255
        img = img.reshape(cols, rows)
        img = img.T
    except:
        img = np.zeros((cols, rows))
    return img




## === cell 7
def salt_proportion(imgArray):
    try:
        unique, counts = np.unique(imgArray, return_counts=True)
        return counts[1] / 10201.0
    except:
        return 0.0




## === cell 8
merged = train_mask.merge(depth, how="left")
merged.head()



## === cell 9
im_width = 128
im_height = 128
border = 5
im_chan = 1  # using a single grayscale channel
n_features = 1
path_train = "../input/train/"
path_test = "../input/test/"



## === cell 10
ids = ["1f1cc6b3a4", "5b7c160d0d", "6c40978ddf", "7dfdf6eeb8", "7e5a6e5013"]
plt.figure(figsize=(30, 15))
for j, img_name in enumerate(ids):
    q = j + 1
    img = load_img("../input/train/images/" + img_name + ".png", grayscale=True)
    img_mask = load_img("../input/train/masks/" + img_name + ".png", grayscale=True)
    img = np.array(img)
    img_cumsum = (np.float32(img) - img.mean()).cumsum(axis=0)
    img_mask = np.array(img_mask)
    plt.subplot(1, 3 * (1 + len(ids)), q * 3 - 2)
    plt.imshow(img, cmap="seismic")
    plt.subplot(1, 3 * (1 + len(ids)), q * 3 - 1)
    plt.imshow(img_cumsum, cmap="seismic")
    plt.subplot(1, 3 * (1 + len(ids)), q * 3)
    plt.imshow(img_mask)
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2004679787.py in <cell line: 0>()
      3 for j, img_name in enumerate(ids):
      4     q = j + 1
----> 5     img = load_img("../input/train/images/" + img_name + ".png", grayscale=True)
      6     img_mask = load_img("../input/train/masks/" + img_name + ".png", grayscale=True)
      7     img = np.array(img)

TypeError: load_img() got an unexpected keyword argument 'grayscale'

## === cell 11
train_ids = next(os.walk(path_train + "images"))[2]
test_ids = next(os.walk(path_test + "images"))[2]



## === cell 12
X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=bool)
print("Getting and resizing train images and masks ... ")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = load_img(os.path.join(path_train, "images", id_))
    x = img_to_array(img)[:, :, 0]  # single channel
    x = resize(x, (im_height, im_width, 1), mode="constant", preserve_range=True)
    X_train[n] = x
    mask = img_to_array(load_img(os.path.join(path_train, "masks", id_)))[:, :, 0]
    Y_train[n] = resize(
        mask, (im_height, im_width, 1), mode="constant", preserve_range=True
    )

print("Done!")




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4058653663.py in <cell line: 0>()
      2 Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=bool)
      3 print("Getting and resizing train images and masks ... ")
----> 4 sys.stdout.flush()
      5 for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
      6     img = load_img(os.path.join(path_train, "images", id_))

NameError: name 'sys' is not defined

## === cell 13
def mean_iou(y_true, y_pred):
    thresholds = tf.range(0.5, 1.0, 0.05)
    ious = []
    for t in thresholds:
        y_pred_t = tf.cast(y_pred > t, tf.int32)
        y_true_i = tf.cast(y_true, tf.int32)
        m = tf.keras.metrics.MeanIoU(num_classes=2)
        m.update_state(y_true_i, y_pred_t)
        ious.append(m.result())
    return tf.reduce_mean(ious)




## === cell 14
inputs = Input((im_height, im_width, im_chan))
s = Lambda(lambda x: x / 255.0)(inputs)

c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(s)
c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(16, (3, 3), activation="relu", padding="same")(p1)
c2 = Conv2D(16, (3, 3), activation="relu", padding="same")(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(32, (3, 3), activation="relu", padding="same")(p2)
c3 = Conv2D(32, (3, 3), activation="relu", padding="same")(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(p3)
c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(128, (3, 3), activation="relu", padding="same")(p4)
c5 = Conv2D(128, (3, 3), activation="relu", padding="same")(c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation="relu", padding="same")(u6)
c6 = Conv2D(64, (3, 3), activation="relu", padding="same")(c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation="relu", padding="same")(u7)
c7 = Conv2D(32, (3, 3), activation="relu", padding="same")(c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation="relu", padding="same")(u8)
c8 = Conv2D(16, (3, 3), activation="relu", padding="same")(c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding="same")(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation="relu", padding="same")(u9)
c9 = Conv2D(8, (3, 3), activation="relu", padding="same")(c9)

outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[mean_iou])
model.summary()



## === cell 15
callback = EarlyStopping(patience=12, verbose=1, restore_best_weights=True)
checkpointer = ModelCheckpoint("model-tgs-salt-1.h5", verbose=1, save_best_only=True)
results = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    batch_size=32,
    epochs=20,
    callbacks=[callback, checkpointer],
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_55/226337482.py in <cell line: 0>()
      1 callback = EarlyStopping(patience=12, verbose=1, restore_best_weights=True)
      2 checkpointer = ModelCheckpoint("model-tgs-salt-1.h5", verbose=1, save_best_only=True)
----> 3 results = model.fit(
      4     X_train,
      5     Y_train,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/4251139191.py in mean_iou(y_true, y_pred)
      2     thresholds = tf.range(0.5, 1.0, 0.05)
      3     ious = []
----> 4     for t in thresholds:
      5         y_pred_t = tf.cast(y_pred > t, tf.int32)
      6         y_true_i = tf.cast(y_true, tf.int32)

OperatorNotAllowedInGraphError: Iterating over a symbolic `tf.Tensor` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.

## === cell 16
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.uint8)
sizes_test = []
print("Getting and resizing test images ... ")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = load_img(os.path.join(path_test, "images", id_))
    x = img_to_array(img)[:, :, 0]
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(x, (im_height, im_width, 1), mode="constant", preserve_range=True)
    X_test[n] = x
print("Done!")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1660100119.py in <cell line: 0>()
      2 sizes_test = []
      3 print("Getting and resizing test images ... ")
----> 4 sys.stdout.flush()
      5 for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
      6     img = load_img(os.path.join(path_test, "images", id_))

NameError: name 'sys' is not defined

## === cell 17
model = load_model("model-tgs-salt-1.h5", custom_objects={"mean_iou": mean_iou})
preds_test = model.predict(X_test, verbose=1)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3203565380.py in <cell line: 0>()
----> 1 model = load_model("model-tgs-salt-1.h5", custom_objects={"mean_iou": mean_iou})
      2 preds_test = model.predict(X_test, verbose=1)
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model-tgs-salt-1.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 18
preds_test_upsampled = []
for i in tnrange(len(preds_test)):
    up = resize(
        np.squeeze(preds_test[i]),
        (sizes_test[i][0], sizes_test[i][1]),
        mode="constant",
        preserve_range=True,
    )
    preds_test_upsampled.append(up)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1808790907.py in <cell line: 0>()
      1 preds_test_upsampled = []
----> 2 for i in tnrange(len(preds_test)):
      3     up = resize(
      4         np.squeeze(preds_test[i]),
      5         (sizes_test[i][0], sizes_test[i][1]),

NameError: name 'preds_test' is not defined

## === cell 19
def RLenc(img, order="F", format=True):
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1
    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1
    if r != 0:
        runs.append((pos, r))
    if format:
        return " ".join(f"{p} {l}" for p, l in runs)
    else:
        return runs


pred_dict = {
    fn[:-4]: RLenc(np.round(preds_test_upsampled[i]))
    for i, fn in tqdm(enumerate(test_ids), total=len(test_ids))
}



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/425661453.py in <cell line: 0>()
     21 
     22 
---> 23 pred_dict = {
     24     fn[:-4]: RLenc(np.round(preds_test_upsampled[i]))
     25     for i, fn in tqdm(enumerate(test_ids), total=len(test_ids))

/tmp/ipykernel_55/425661453.py in <dictcomp>(.0)
     22 
     23 pred_dict = {
---> 24     fn[:-4]: RLenc(np.round(preds_test_upsampled[i]))
     25     for i, fn in tqdm(enumerate(test_ids), total=len(test_ids))
     26 }

IndexError: list index out of range

## === cell 20
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.names = ["id"]
sub.columns = ["rle_mask"]
sub.to_csv("submission.csv")

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3934167994.py in <cell line: 0>()
----> 1 sub = pd.DataFrame.from_dict(pred_dict, orient="index")
      2 sub.index.names = ["id"]
      3 sub.columns = ["rle_mask"]
      4 sub.to_csv("submission.csv")

NameError: name 'pred_dict' is not defined
