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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-image==0.25.2
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

0.64136

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.1301) has done: 'I add the missing `reset_state()` call (the correct TensorFlow API) and move the protobuf compatibility shim into the first cell so the script runs without errors. No other logic is changed, preserving the original model and post‑processing while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.1301) has done: 'I replace the single‑channel extraction with the average of all image channels (so the model sees the full grayscale information) and train a bit longer (30 epochs) to let the network learn better representations. These minimal changes keep the architecture untouched while providing richer input data and more training time, which should raise the mean‑average‑precision toward the target score.'
- What this solution (achieved 0.5221) has done: 'I keep the model and training pipeline unchanged but adjust the post‑processing threshold used when converting the predicted probability masks to binary masks. Instead of the default `np.round` (threshold 0.5), I apply a slightly lower threshold (0.4) that usually yields larger predicted regions and better overlap with the ground‑truth masks, which should increase the mean‑average‑precision toward the target score. This minimal change preserves the core logic while improving the evaluation metric.'
- What this solution (achieved 0.5221) has done: 'I slightly lower the binary‐mask threshold to 0.38 (a modest change that usually raises overlap with the ground truth) and add a simple test‑time augmentation: predict on the original test image and on its horizontal flip, then average the two predictions before applying the threshold. This keeps the model architecture and training unchanged while giving a small expected boost toward the target MAP score.'
- What this solution (achieved 0.5221) has done: 'I slightly lower the binary‑mask threshold (to 0.35) and add an extra test‑time augmentation (vertical flip) that is averaged with the original and horizontally‑flipped predictions. This keeps the model and training pipeline unchanged while providing a modest boost in segmentation quality, moving the MAP score closer to the target.'
- What this solution (achieved 0.5221) has done: 'I added a safeguard when loading the model checkpoint: the script now checks whether the weight file `model-tgs-salt-1.h5` exists before calling `load_weights`. If the file is missing (e.g., no improvement was saved during training), it skips the load and continues with the current model, preventing the FileNotFoundError and ensuring a submission CSV is produced.'
- What this solution (achieved 0.5221) has done: 'I raise the early‑stopping patience from 7 to 15 and extend the training run to 80 epochs (while keeping the same early‑stop condition). This lets the model train longer when it is still improving, which is a minimal change expected to raise the MAP score toward the target without altering the architecture or post‑processing.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):
        if hasattr(message_factory.MessageFactory, "GetMessageClass"):
            message_factory.MessageFactory.GetPrototype = (
                lambda self, descriptor: self.GetMessageClass(descriptor)
            )
        else:
            message_factory.MessageFactory.GetPrototype = (
                lambda self, descriptor: descriptor
            )
except Exception:
    pass




## === cell 1
import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tqdm import tqdm, tnrange
from skimage.transform import resize
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Lambda,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf

miou_metric = tf.keras.metrics.MeanIoU(num_classes=2)

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
TRAIN_PATH = "../input/train"
TEST_PATH = "../input/test"

print(os.listdir("../input"))

train_ids = next(os.walk("../input/train/images"))[2]
test_ids = next(os.walk("../input/test/images"))[2]

X_train = np.zeros(
    (len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8
)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = load_img(os.path.join(TRAIN_PATH, "images", id_))
    x = img_to_array(img).mean(axis=2)  # shape (H, W)
    x = resize(x, (IMG_HEIGHT, IMG_WIDTH, 1), mode="constant", preserve_range=True)
    X_train[n] = x

    mask = img_to_array(load_img(os.path.join(TRAIN_PATH, "masks", id_))).mean(axis=2)
    Y_train[n] = resize(
        mask, (IMG_HEIGHT, IMG_WIDTH, 1), mode="constant", preserve_range=True
    )

print("Done!")


def mean_iou_metric(y_true, y_pred):
    """Custom Mean IoU metric across thresholds 0.5‑0.95."""
    thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95
    ious = []
    y_true_i = tf.cast(y_true, tf.int32)
    for t in thresholds:
        y_pred_i = tf.cast(y_pred > t, tf.int32)
        miou_metric.reset_state()  # corrected API call
        miou_metric.update_state(y_true_i, y_pred_i)
        ious.append(miou_metric.result())
    return tf.reduce_mean(ious)


inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255.0)(inputs)  # Normalization

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
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[mean_iou_metric])
model.summary()

earlystopper = EarlyStopping(patience=15, verbose=1)
checkpointer = ModelCheckpoint("model-tgs-salt-1.h5", verbose=1, save_best_only=True)

results = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    batch_size=8,
    epochs=80,  # extended training epochs
    callbacks=[earlystopper, checkpointer],
)

weight_path = "model-tgs-salt-1.h5"
if os.path.isfile(weight_path):
    model.load_weights(weight_path)
else:
    print("Checkpoint not found; using the model from the final epoch.")

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print("Getting and resizing test images ... ")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = load_img(os.path.join(TEST_PATH, "images", id_))
    x = img_to_array(img).mean(axis=2)
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(x, (IMG_HEIGHT, IMG_WIDTH, 1), mode="constant", preserve_range=True)
    X_test[n] = x
print("Done!")

preds_orig = model.predict(X_test, verbose=1)

X_test_hflip = np.flip(X_test, axis=2)  # flip width dimension
preds_hflip = model.predict(X_test_hflip, verbose=1)
preds_hflip = np.flip(preds_hflip, axis=2)

X_test_vflip = np.flip(X_test, axis=1)  # flip height dimension
preds_vflip = model.predict(X_test_vflip, verbose=1)
preds_vflip = np.flip(preds_vflip, axis=1)

preds_test = (preds_orig + preds_hflip + preds_vflip) / 3.0

preds_test_upsampled = []
for i in tnrange(len(preds_test)):
    preds_test_upsampled.append(
        resize(
            np.squeeze(preds_test[i]),
            (sizes_test[i][0], sizes_test[i][1]),
            mode="constant",
            preserve_range=True,
        )
    )


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
        z = ""
        for rr in runs:
            z += f"{rr[0]} {rr[1]} "
        return z[:-1]
    else:
        return runs


THRESHOLD = 0.30

pred_dict = {
    fn[:-4]: RLenc((preds_test_upsampled[i] > THRESHOLD).astype(np.uint8))
    for i, fn in tqdm(enumerate(test_ids), total=len(test_ids))
}
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.names = ["id"]
sub.columns = ["rle_mask"]
sub.to_csv("submission.csv", index=True)
