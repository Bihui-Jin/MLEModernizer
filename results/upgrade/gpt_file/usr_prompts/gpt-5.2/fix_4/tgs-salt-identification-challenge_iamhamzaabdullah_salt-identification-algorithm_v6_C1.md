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

0.64274

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.026) has done: 'I fix the TensorFlow/Keras import crash by pinning protobuf to a compatible version at runtime (a common Kaggle TF/protobuf mismatch) before importing TensorFlow/Keras. Then I fix the `mean_iou` metric dtype issue by casting inputs to `float32` before thresholding, so training can start. Next I ensure `ModelCheckpoint` actually writes a model file by giving it an explicit `.keras` path in `/kaggle/working` and by loading that exact path for inference. Finally, I make the test prediction → upsample → RLE pipeline robust by ensuring lengths align and by iterating over `test_ids` consistently, producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio
import torch
from torch.utils import data
from tqdm import tqdm
from skimage.transform import resize

import subprocess

try:
    import google.protobuf  # noqa: F401
    import importlib
    import pkg_resources

    pb_ver = pkg_resources.get_distribution("protobuf").version
    if tuple(int(x) for x in pb_ver.split(".")[:2]) >= (5, 0):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        importlib.invalidate_caches()
except Exception:
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"])

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import (
    Input,
    Lambda,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import load_img, img_to_array

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)




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
BASE_INPUT = "/kaggle/input/tgs-salt-identification-challenge"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input"  # fallback if dataset is mounted flat

train_mask = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
depth = pd.read_csv(os.path.join(BASE_INPUT, "depths.csv"))
train_path = os.path.join(BASE_INPUT, "train")

file_list = list(train_mask["id"].values)
dataset = TGSSaltDataset(train_path, file_list)

print("BASE_INPUT:", BASE_INPUT)
print("Train images path exists:", os.path.exists(os.path.join(train_path, "images")))




## === cell 3
def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1, 2, figsize=(8, 4))
    axarr[0].imshow(image, cmap="gray")
    axarr[1].imshow(mask, cmap="gray")
    axarr[0].grid(False)
    axarr[1].grid(False)
    axarr[0].set_title("Image")
    axarr[1].set_title("Mask")
    plt.show()




## === cell 4
for i in range(3):
    image, mask = dataset[np.random.randint(0, len(dataset))]
    plot2x2Array(image, mask)



## === cell 5
plt.figure(figsize=(6, 6))
plt.hist(depth["z"], bins=50)
plt.title("Depth distribution")
plt.show()




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
    except Exception:
        img = np.zeros((cols, rows), dtype=np.uint8)
    return img




## === cell 7
def salt_proportion(imgArray):
    try:
        unique, counts = np.unique(imgArray, return_counts=True)
        return counts[1] / 10201.0
    except Exception:
        return 0.0




## === cell 8
merged = train_mask.merge(depth, how="left", on="id")
merged.head()



## === cell 9
im_width = 128
im_height = 128
im_chan = 1

path_train = os.path.join(BASE_INPUT, "train")
path_test = os.path.join(BASE_INPUT, "test")



## === cell 10
ids = ["1f1cc6b3a4", "5b7c160d0d", "6c40978ddf", "7dfdf6eeb8", "7e5a6e5013"]
plt.figure(figsize=(30, 6))
for j, img_name in enumerate(ids):
    q = j + 1
    img = load_img(
        os.path.join(path_train, "images", img_name + ".png"), color_mode="grayscale"
    )
    img_mask = load_img(
        os.path.join(path_train, "masks", img_name + ".png"), color_mode="grayscale"
    )

    img = np.array(img)
    img_cumsum = (np.float32(img) - img.mean()).cumsum(axis=0)
    img_mask = np.array(img_mask)

    plt.subplot(1, 3 * (1 + len(ids)), q * 3 - 2)
    plt.imshow(img, cmap="seismic")
    plt.axis("off")
    plt.subplot(1, 3 * (1 + len(ids)), q * 3 - 1)
    plt.imshow(img_cumsum, cmap="seismic")
    plt.axis("off")
    plt.subplot(1, 3 * (1 + len(ids)), q * 3)
    plt.imshow(img_mask, cmap="gray")
    plt.axis("off")
plt.show()



## === cell 11
train_ids = next(os.walk(os.path.join(path_train, "images")))[2]
test_ids = next(os.walk(os.path.join(path_test, "images")))[2]

print("Num train images:", len(train_ids))
print("Num test images:", len(test_ids))



## === cell 12
X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.uint8)

print("Getting and resizing train images and masks ...")
sys.stdout.flush()

for n, id_ in tqdm(list(enumerate(train_ids)), total=len(train_ids)):
    img = load_img(os.path.join(path_train, "images", id_), color_mode="grayscale")
    x = img_to_array(img)  # (H,W,1)
    x = resize(
        x,
        (im_height, im_width, 1),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    X_train[n] = x.astype(np.uint8)

    mask = load_img(os.path.join(path_train, "masks", id_), color_mode="grayscale")
    mask = img_to_array(mask)  # (H,W,1)
    mask = resize(
        mask,
        (im_height, im_width, 1),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    Y_train[n] = (mask > 127).astype(np.uint8)

print("Done!")
print(
    "X_train:", X_train.shape, X_train.dtype, "Y_train:", Y_train.shape, Y_train.dtype
)




## === cell 13
def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true > 0.5, tf.int32)

    thresholds = tf.constant(np.arange(0.5, 1.0, 0.05), dtype=tf.float32)

    def iou_at_t(t):
        y_pred_t = tf.cast(y_pred > t, tf.int32)
        intersection = tf.reduce_sum(
            tf.cast(tf.logical_and(y_true == 1, y_pred_t == 1), tf.float32)
        )
        union = tf.reduce_sum(
            tf.cast(tf.logical_or(y_true == 1, y_pred_t == 1), tf.float32)
        )
        return tf.where(tf.equal(union, 0.0), 1.0, intersection / union)

    ious = tf.map_fn(iou_at_t, thresholds, fn_output_signature=tf.float32)
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
MODEL_PATH = "/kaggle/working/model-tgs-salt-1.keras"
callback = EarlyStopping(patience=12, verbose=1, restore_best_weights=True)
checkpointer = ModelCheckpoint(MODEL_PATH, verbose=1, save_best_only=True)

perm = np.random.RandomState(42).permutation(len(X_train))
X_train_shuf = X_train[perm]
Y_train_shuf = Y_train[perm]

results = model.fit(
    X_train_shuf,
    Y_train_shuf,
    validation_split=0.1,
    batch_size=8,
    epochs=12,
    callbacks=[callback, checkpointer],
)

print("Training complete. Model exists:", os.path.exists(MODEL_PATH))



## === cell 16
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.uint8)
sizes_test = []

print("Getting and resizing test images ...")
sys.stdout.flush()

for n, id_ in tqdm(list(enumerate(test_ids)), total=len(test_ids)):
    img = load_img(os.path.join(path_test, "images", id_), color_mode="grayscale")
    x = img_to_array(img)  # (H,W,1)
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(
        x,
        (im_height, im_width, 1),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    X_test[n] = x.astype(np.uint8)

print("Done!")
print("X_test:", X_test.shape, X_test.dtype)



## === cell 17
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, custom_objects={"mean_iou": mean_iou}, compile=False)
else:
    print("WARNING: Checkpoint not found; using in-memory trained model.")

split_ix = int(X_train_shuf.shape[0] * 0.9)
preds_train = model.predict(X_train_shuf[:split_ix], verbose=1)
preds_val = model.predict(X_train_shuf[split_ix:], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

print("preds_test shape:", preds_test.shape)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3270996.py in <cell line: 0>()
      1 if os.path.exists(MODEL_PATH):
----> 2     model = load_model(MODEL_PATH, custom_objects={"mean_iou": mean_iou}, compile=False)
      3 else:
      4     print("WARNING: Checkpoint not found; using in-memory trained model.")
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    187 
    188     if is_keras_zip or is_keras_dir or is_hf:
--> 189         return saving_lib.load_model(
    190             filepath,
    191             custom_objects=custom_objects,

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_lib.py in load_model(filepath, custom_objects, compile, safe_mode)
    365             )
    366         with open(filepath, "rb") as f:
--> 367             return _load_model_from_fileobj(
    368                 f, custom_objects, compile, safe_mode
    369             )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_lib.py in _load_model_from_fileobj(fileobj, custom_objects, compile, safe_mode)
    442             config_json = f.read()
    443 
--> 444         model = _model_from_config(
    445             config_json, custom_objects, compile, safe_mode
    446         )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_lib.py in _model_from_config(config_json, custom_objects, compile, safe_mode)
    431     # Construct the model from the configuration file in the archive.
    432     with ObjectSharingScope():
--> 433         model = deserialize_keras_object(
    434             config_dict, custom_objects, safe_mode=safe_mode
    435         )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    716     with custom_obj_scope, safe_mode_scope:
    717         try:
--> 718             instance = cls.from_config(inner_config)
    719         except TypeError as e:
    720             raise TypeError(

/usr/local/lib/python3.11/dist-packages/keras/src/models/model.py in from_config(cls, config, custom_objects)
    580             from keras.src.models.functional import functional_from_config
    581 
--> 582             return functional_from_config(
    583                 cls, config, custom_objects=custom_objects
    584             )

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in functional_from_config(cls, config, custom_objects)
    549     # First, we create all layers and enqueue nodes to be processed
    550     for layer_data in functional_config["layers"]:
--> 551         process_layer(layer_data)
    552 
    553     # Then we process nodes in order of layer depth.

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in process_layer(layer_data)
    521             )
    522         else:
--> 523             layer = serialization_lib.deserialize_keras_object(
    524                 layer_data, custom_objects=custom_objects
    525             )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    716     with custom_obj_scope, safe_mode_scope:
    717         try:
--> 718             instance = cls.from_config(inner_config)
    719         except TypeError as e:
    720             raise TypeError(

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/lambda_layer.py in from_config(cls, config, custom_objects, safe_mode)
    188             and fn_config["class_name"] == "__lambda__"
    189         ):
--> 190             cls._raise_for_lambda_deserialization("function", safe_mode)
    191             inner_config = fn_config["config"]
    192             fn = python_utils.func_load(

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/lambda_layer.py in _raise_for_lambda_deserialization(arg_name, safe_mode)
    170     def _raise_for_lambda_deserialization(arg_name, safe_mode):
    171         if safe_mode:
--> 172             raise ValueError(
    173                 "The `{arg_name}` of this `Lambda` layer is a Python lambda. "
    174                 "Deserializing it is unsafe. If you trust the source of the "

ValueError: The `{arg_name}` of this `Lambda` layer is a Python lambda. Deserializing it is unsafe. If you trust the source of the config artifact, you can override this error by passing `safe_mode=False` to `from_config()`, or calling `keras.config.enable_unsafe_deserialization().

## === cell 18
preds_test_upsampled = []
for i in tqdm(range(len(test_ids))):
    up = resize(
        np.squeeze(preds_test[i]),
        (sizes_test[i][0], sizes_test[i][1]),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    preds_test_upsampled.append(up)

print("Example upsampled shape:", preds_test_upsampled[0].shape)
print(
    "len(preds_test_upsampled):",
    len(preds_test_upsampled),
    "len(test_ids):",
    len(test_ids),
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/456768286.py in <cell line: 0>()
      2 for i in tqdm(range(len(test_ids))):
      3     up = resize(
----> 4         np.squeeze(preds_test[i]),
      5         (sizes_test[i][0], sizes_test[i][1]),
      6         mode="constant",

NameError: name 'preds_test' is not defined

## === cell 19
ix = random.randint(0, len(preds_train_t) - 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.squeeze(X_train_shuf[ix]), cmap="gray")
plt.title("Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(np.squeeze(Y_train_shuf[ix]).astype(np.float32), cmap="gray")
plt.title("Mask true")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(preds_train_t[ix]).astype(np.float32), cmap="gray")
plt.title("Mask pred")
plt.axis("off")
plt.show()




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3063195284.py in <cell line: 0>()
----> 1 ix = random.randint(0, len(preds_train_t) - 1)
      2 plt.figure(figsize=(12, 4))
      3 plt.subplot(1, 3, 1)
      4 plt.imshow(np.squeeze(X_train_shuf[ix]), cmap="gray")
      5 plt.title("Image")

NameError: name 'preds_train_t' is not defined

## === cell 20
def RLenc(img, order="F", format=True):
    bytes_ = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1
    for c in bytes_:
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
        pos += r
        r = 0

    if format:
        return " ".join([f"{rr[0]} {rr[1]}" for rr in runs])
    else:
        return runs


pred_dict = {}
for i, fn in enumerate(test_ids):
    img_id = fn[:-4]
    m = (preds_test_upsampled[i] > 0.5).astype(np.uint8)
    pred_dict[img_id] = RLenc(m)

sub = pd.DataFrame({"id": list(pred_dict.keys()), "rle_mask": list(pred_dict.values())})

sample_sub = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))
sub = sample_sub[["id"]].merge(sub, on="id", how="left")
sub["rle_mask"] = sub["rle_mask"].fillna("")

sub_path = "/kaggle/working/submission.csv"
sub.to_csv(sub_path, index=False)
print(sub.head())
print("Wrote submission.csv to:", sub_path, "with shape:", sub.shape)
print("Exists:", os.path.exists(sub_path))

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3610745232.py in <cell line: 0>()
     28 for i, fn in enumerate(test_ids):
     29     img_id = fn[:-4]
---> 30     m = (preds_test_upsampled[i] > 0.5).astype(np.uint8)
     31     pred_dict[img_id] = RLenc(m)
     32 

IndexError: list index out of range
