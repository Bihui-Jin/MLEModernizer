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
pillow==11.3.0
protobuf==6.33.0
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0
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

0.42183

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.2654) has done: 'I fix the environment/runtime breakages by removing the notebook-only magic, correcting/defensive-importing Keras/TensorFlow pieces that changed (the protobuf error, `concatenate`, and the TF1-style `mean_iou`). I also correct the dataset paths to the actual Kaggle layout you provided (`/kaggle/input/...`) so images load and the pipeline can train and predict end-to-end. To keep core logic intact, the U-Net architecture, loss, and overall training flow remain the same; only the metric implementation is updated to a TF2-safe version so model compilation and saving/loading work. Finally, I ensure the submission is written as a valid `submission.csv` with `id,rle_mask` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.3125) has done: 'I fix the protobuf/TensorFlow import crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow (the current code sets it too late). I also make the custom MeanIoU metric properly reset between epochs/batches by implementing the correct `reset_state()` method name for TF2, which avoids metric state leakage that can destabilize training/selection. Finally, I keep the model, data pipeline, and RLE core logic intact, but add a minimal post-processing threshold (0.5) instead of `np.round` to better match the intended mask binarization and improve the score toward the target.'
- What this solution (achieved 0.3432) has done: 'You’re hitting a TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`) before the code can run, so the main fix is to force the pure-Python protobuf implementation *early enough* and also disable C++ protobuf in a way that works reliably in Kaggle kernels. After that, I keep the model and training logic intact, but fix two correctness issues that can depress IoU: masks are being loaded as RGB+green-channel (wrong for binary masks) and then treated as soft labels; we load masks as grayscale and binarize them. Finally, I make the RLE encoder handle empty masks cleanly and ensure the submission is always written as a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.5221) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow-related import happens* and by explicitly disabling the C++ protobuf backend, which is what triggers `MessageFactory.GetPrototype` in this environment. This is a runtime-only fix and does not change the model/training logic or submission semantics. After that, we keep the U-Net, loss, training loop, and post-processing intact, and ensure the script always reaches the CSV write step with the correct `id,rle_mask` columns and ordering. This should restore end-to-end execution and keep your current scoring behavior (with potential minor stability improvements from the fixed environment).'
- What this solution (achieved 0.5221) has done: 'The crash happens before training due to an incompatibility between TensorFlow 2.18 and the forced pure-Python protobuf runtime; the `MessageFactory.GetPrototype` error is triggered by those environment overrides. To fix end-to-end execution, I remove the protobuf env-forcing and instead set deterministic seeds/flags only (score-neutral) while keeping the U-Net, loss, training loop, and post-processing unchanged. I also add small defensive guards to ensure the RLE encoder always returns a valid string (including empty masks) and that test image paths exist before loading, so the script reliably reaches the CSV write. Since your current score (0.5221) is already above the target (0.42183) and within the ±10% tolerance band, I avoid any score-changing modeling/training modifications.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from skimage.transform import resize

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

pd.set_option("display.max_colwidth", 100)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

gc.collect()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/tgs-salt-identification-challenge"

train_images_dir = os.path.join(DATA_ROOT, "train", "images")
train_masks_dir = os.path.join(DATA_ROOT, "train", "masks")
test_images_dir = os.path.join(DATA_ROOT, "test", "images")

depths_path = os.path.join(DATA_ROOT, "depths.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(train_images_dir), f"Missing: {train_images_dir}"
assert os.path.isdir(train_masks_dir), f"Missing: {train_masks_dir}"
assert os.path.isdir(test_images_dir), f"Missing: {test_images_dir}"
assert os.path.exists(depths_path), f"Missing: {depths_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"



## === cell 2
Train_Image_name = sorted(os.listdir(train_images_dir))
Test_Image_name = sorted(os.listdir(test_images_dir))

Train_Image_path, Train_Mask_path, Train_id = [], [], []
for fn in Train_Image_name:
    if not fn.lower().endswith(".png"):
        continue
    Train_Image_path.append(os.path.join(train_images_dir, fn))
    Train_Mask_path.append(os.path.join(train_masks_dir, fn))
    Train_id.append(fn.split(".")[0])

Test_Image_path, Test_id = [], []
for fn in Test_Image_name:
    if not fn.lower().endswith(".png"):
        continue
    Test_Image_path.append(os.path.join(test_images_dir, fn))
    Test_id.append(fn.split(".")[0])

df_Train_path = pd.DataFrame(
    {
        "id": Train_id,
        "Train_Image_path": Train_Image_path,
        "Train_Mask_path": Train_Mask_path,
    }
)
df_Test_path = pd.DataFrame({"id": Test_id, "Test_Image_path": Test_Image_path})

df_depths = pd.read_csv(depths_path)
df_sub = pd.read_csv(sample_sub_path)

df_Train_path = df_Train_path.merge(df_depths, on="id", how="left")
df_Test_path = df_Test_path.merge(df_depths, on="id", how="left")

df_Test_path = df_sub[["id"]].merge(df_Test_path, on="id", how="left")

print(df_Train_path.shape, df_Test_path.shape)
df_Train_path.head()



## === cell 3
from tqdm import tqdm


def read_image(paths, img_height, img_width, img_chan):
    pixel = np.zeros((len(paths), img_height, img_width, img_chan), dtype=np.float32)
    for n, p in tqdm(list(enumerate(paths)), total=len(paths)):
        img = load_img(p, color_mode="rgb")
        x = img_to_array(img)[:, :, 1]  # keep original logic: use green channel
        x = resize(x, (img_height, img_width), mode="constant", preserve_range=True)
        x = np.expand_dims(x, axis=-1)
        x = x / 255.0
        pixel[n] = x
    return pixel


def read_mask(paths, img_height, img_width, img_chan):
    pixel = np.zeros((len(paths), img_height, img_width, img_chan), dtype=np.float32)
    for n, p in tqdm(list(enumerate(paths)), total=len(paths)):
        img = load_img(p, color_mode="grayscale")
        x = img_to_array(img)[:, :, 0]
        x = resize(x, (img_height, img_width), mode="constant", preserve_range=True)
        x = np.expand_dims(x, axis=-1)
        x = x / 255.0
        x = (x > 0.5).astype(np.float32)
        pixel[n] = x
    return pixel


def read_image2(paths, img_height, img_width, img_chan):
    pixel = np.zeros((len(paths), img_height, img_width, img_chan), dtype=np.float32)
    sizes_test = []
    for n, p in tqdm(list(enumerate(paths)), total=len(paths)):
        if not isinstance(p, str) or (not os.path.exists(p)):
            raise FileNotFoundError(f"Missing test image path at index {n}: {p}")
        img = load_img(p, color_mode="rgb")
        x0 = img_to_array(img)[:, :, 1]
        sizes_test.append([x0.shape[0], x0.shape[1]])
        x = resize(x0, (img_height, img_width), mode="constant", preserve_range=True)
        x = np.expand_dims(x, axis=-1)
        x = x / 255.0
        pixel[n] = x
    return pixel, sizes_test


img_height = 128
img_width = 128
img_chan = 1

Train_Image_pixel = read_image(
    df_Train_path.Train_Image_path.values, img_height, img_width, img_chan
)
Train_Mask_pixel = read_mask(
    df_Train_path.Train_Mask_path.values, img_height, img_width, img_chan
)

print("Train Image shape: ", Train_Image_pixel.shape)
print("Train Mask shape: ", Train_Mask_pixel.shape)

Test_Image_pixel, sizes_test = read_image2(
    df_Test_path.Test_Image_path.values, img_height, img_width, img_chan
)
print("Test Image shape: ", Test_Image_pixel.shape)



## === cell 4
_ = array_to_img(Train_Image_pixel[0])



## === cell 5
_ = array_to_img(Train_Mask_pixel[0])



## === cell 6
_ = array_to_img(Test_Image_pixel[0])




## === cell 7
class MeanIoUThresholded(tf.keras.metrics.Metric):
    def __init__(self, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.thresholds = np.arange(0.5, 1.0, 0.05)
        self._ious = [tf.keras.metrics.MeanIoU(num_classes=2) for _ in self.thresholds]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.cast(y_true > 0.5, tf.int32)
        for t, m in zip(self.thresholds, self._ious):
            y_pred_bin = tf.cast(y_pred > t, tf.int32)
            m.update_state(y_true_bin, y_pred_bin)

    def result(self):
        vals = [m.result() for m in self._ious]
        return tf.reduce_mean(tf.stack(vals))

    def reset_state(self):
        for m in self._ious:
            m.reset_state()

    def reset_states(self):
        self.reset_state()


def mean_iou(y_true, y_pred):
    if not hasattr(mean_iou, "_metric"):
        mean_iou._metric = MeanIoUThresholded()
    mean_iou._metric.update_state(y_true, y_pred)
    return mean_iou._metric.result()




## === cell 8
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + K.epsilon()) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()
    )




## === cell 9
X = Train_Image_pixel
Y = Train_Mask_pixel
test = Test_Image_pixel

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.20, random_state=SEED
)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(test.shape)



## === cell 10
inputs = Input((img_height, img_width, img_chan))

c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(inputs)
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
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=[MeanIoUThresholded()]
)

model.summary()



## === cell 11
earlystopper = EarlyStopping(patience=2, verbose=1)
checkpointer = ModelCheckpoint("model-tgs-salt-1.keras", verbose=1, save_best_only=True)

results = model.fit(
    X,
    Y,
    validation_split=0.1,
    batch_size=8,
    epochs=2,
    callbacks=[earlystopper, checkpointer],
    verbose=1,
)



## === cell 12
if os.path.exists("model-tgs-salt-1.keras"):
    model = load_model(
        "model-tgs-salt-1.keras",
        custom_objects={"MeanIoUThresholded": MeanIoUThresholded},
    )
else:
    print("Warning: checkpoint not found, using in-memory model.")

pred_train = model.predict(X_train, verbose=1)
pred_val = model.predict(X_val, verbose=1)
preds_test = model.predict(test, verbose=1)



## === cell 13
preds_test_upsampled = []
for i in tqdm(range(len(preds_test)), total=len(preds_test)):
    preds_test_upsampled.append(
        resize(
            np.squeeze(preds_test[i]),
            (sizes_test[i][0], sizes_test[i][1]),
            mode="constant",
            preserve_range=True,
        )
    )




## === cell 14
def RLenc(img, order="F", format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    returns run length as a string (if format is True) per competition rules
    """
    if img is None:
        return "" if format else []
    img = np.asarray(img)
    if img.ndim != 2:
        img = np.squeeze(img)
    if img.size == 0:
        return "" if format else []

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

    if format:
        return " ".join([f"{rr[0]} {rr[1]}" for rr in runs])
    else:
        return runs


PRED_THRESH = 0.5

pred_dict = {}
for i, img_id in tqdm(
    list(enumerate(df_Test_path["id"].values)), total=len(df_Test_path)
):
    mask = (preds_test_upsampled[i] > PRED_THRESH).astype(np.uint8)
    enc = RLenc(mask)
    pred_dict[img_id] = enc if enc is not None else ""



## === cell 15
sub = pd.DataFrame({"id": list(pred_dict.keys()), "rle_mask": list(pred_dict.values())})

sub = df_sub[["id"]].merge(sub, on="id", how="left")
sub["rle_mask"] = sub["rle_mask"].fillna("")

sub.head()



## === cell 16
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", sub.shape)
print(sub.head())
