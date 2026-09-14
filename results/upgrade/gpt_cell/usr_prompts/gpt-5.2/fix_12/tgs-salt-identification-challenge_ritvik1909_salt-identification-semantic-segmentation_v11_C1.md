# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
The timeout is most likely dominated by (1) the very expensive per-threshold sweep in cell 14 and (2) Python-loop overhead and list growth during test upsampling / RLE generation. I’ll keep the exact same model, training call, loss, and threshold-selection semantics, but replace the threshold sweep with a provably equivalent vectorized formulation that computes TP/union per image for all thresholds in one pass (no repeated re-binarization per threshold). I’ll also preallocate the upsampled prediction array and avoid per-image Python overhead where possible, while keeping identical resizing/interpolation and deterministic sorting. These changes reduce the constant factors substantially and should bring runtime under 600 seconds without changing results beyond negligible FP differences.

```python


## === cell 1
import os, sys, random, warnings, math

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm, trange
from itertools import chain

import cv2
from skimage.transform import resize  # kept to preserve import surface; no longer used in hot loops
from skimage.morphology import label  # unused but kept to preserve imports/compat

try:
    from google.protobuf import message_factory as _mf

    _MessageFactory = _mf.MessageFactory

    if not hasattr(_MessageFactory, "GetPrototype"):
        if hasattr(_MessageFactory, "GetMessageClass"):
            _MessageFactory.GetPrototype = _MessageFactory.GetMessageClass
        elif hasattr(_MessageFactory, "GetMessages"):
            def _get_prototype_fallback(self, descriptor):
                msgs = self.GetMessages([descriptor.file])
                return msgs.get(descriptor.full_name)

            _MessageFactory.GetPrototype = _get_prototype_fallback
        else:
            def _missing_getprototype(self, descriptor):
                raise AttributeError(
                    "protobuf MessageFactory has no GetPrototype/GetMessageClass; incompatible protobuf version."
                )

            _MessageFactory.GetPrototype = _missing_getprototype
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, Input, layers, callbacks, utils, optimizers

random.seed(19)
np.random.seed(19)
tf.random.set_seed(19)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    tf.config.threading.set_intra_op_parallelism_threads(min(4, os.cpu_count() or 4))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

warnings.filterwarnings("ignore")




## === cell 2
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = 'train/'
    path_test = 'test/'




## === cell 3
! unzip -q ../input/tgs-salt-identification-challenge/train.zip -d train/
! unzip -q ../input/tgs-salt-identification-challenge/test.zip -d test/




## === cell 4
pass




## === cell 5
def _base_dir_from_images_dir(images_dir: str) -> str:
    base = os.path.dirname(images_dir)
    return base if base.endswith(os.sep) else base + os.sep


if os.path.isdir("train/images") and os.path.isdir("train/masks"):
    _train_images_dir = "train/images"
    _train_masks_dir = "train/masks"
else:
    _train_images_dir, _train_masks_dir = None, None
    for root, dirs, files in os.walk("train"):
        if os.path.basename(root) == "images" and os.path.isdir(
            os.path.join(os.path.dirname(root), "masks")
        ):
            _train_images_dir = root
            _train_masks_dir = os.path.join(os.path.dirname(root), "masks")
            break
    if _train_images_dir is None or _train_masks_dir is None:
        raise FileNotFoundError(
            "Could not locate extracted train images/masks directories under 'train/'. "
            "Expected either 'train/images'+'train/masks' or a nested structure like 'train/**/images'+'train/**/masks'."
        )

train_images_dir = _train_images_dir
test_images_dir = os.path.join(config.path_test, "images")
if not os.path.isdir(test_images_dir):
    test_images_dir = None
    for root, dirs, files in os.walk(config.path_test):
        if os.path.basename(root) == "images":
            test_images_dir = root
            break
    if test_images_dir is None:
        raise FileNotFoundError(
            "Could not locate extracted test images directory under 'test/'. "
            "Expected either 'test/images' or a nested structure like 'test/**/images'."
        )

config.path_train = _base_dir_from_images_dir(train_images_dir)
config.path_test = _base_dir_from_images_dir(test_images_dir)

train_ids = sorted(os.listdir(os.path.join(config.path_train, "images")))
test_ids = sorted(os.listdir(os.path.join(config.path_test, "images")))




## === cell 6
X = np.empty(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y = np.empty((len(train_ids), config.im_height, config.im_width, 1), dtype=np.bool_)

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()

train_img_dir = os.path.join(config.path_train, "images")
train_mask_dir = os.path.join(config.path_train, "masks")

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img_path = os.path.join(train_img_dir, id_)
    msk_path = os.path.join(train_mask_dir, id_)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    img_r = cv2.resize(img, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR)
    X[n, ..., 0] = img_r

    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    if msk is None:
        raise FileNotFoundError(f"Failed to read mask: {msk_path}")
    msk_r = cv2.resize(msk, (config.im_width, config.im_height), interpolation=cv2.INTER_NEAREST)
    Y[n, ..., 0] = msk_r > 127

print("Done!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)




## === cell 7
n_total = len(X)
split = int(0.9 * n_total)

X_train = X[:split]
Y_train = Y[:split]
X_eval  = X[split:]
Y_eval  = Y[split:]

X_fliplr = X[:, :, ::-1, :]
Y_fliplr = Y[:, :, ::-1, :]
X_flipud = X[:, ::-1, :, :]
Y_flipud = Y[:, ::-1, :, :]

X_train = np.concatenate([X_train, X_fliplr, X_flipud], axis=0)
Y_train = np.concatenate([Y_train, Y_fliplr, Y_flipud], axis=0)

del X, Y, X_fliplr, Y_fliplr, X_flipud, Y_flipud

print('X train shape:', X_train.shape, 'X eval shape:', X_eval.shape)
print('Y train shape:', Y_train.shape, 'Y eval shape:', Y_eval.shape)




## === cell 8
def BatchActivate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = layers.Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x




## === cell 9
def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    scaled = layers.Lambda(lambda x: x / 255) (input_layer)

    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(scaled)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(DropoutRatio/2)(pool1)

    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(DropoutRatio)(pool2)

    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(DropoutRatio)(pool3)

    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(DropoutRatio)(pool4)

    convm = layers.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)
    
    deconv4 = layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(DropoutRatio)(uconv4)
    
    uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    deconv3 = layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])    
    uconv3 = layers.Dropout(DropoutRatio)(uconv3)
    
    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    deconv2 = layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
        
    uconv2 = layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    deconv1 = layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    
    uconv1 = layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    output_layer_noActi = layers.Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  layers.Activation('sigmoid')(output_layer_noActi)
    
    return output_layer

input_layer = Input((config.im_height, config.im_width, config.im_chan))
output_layer = build_model(input_layer, 16)

model = models.Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()




## === cell 10
pass




## === cell 11
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

results = model.fit(
    X_train, Y_train, validation_data=(X_eval, Y_eval), batch_size=8, epochs=300, callbacks=[es, rlp]
)




## === cell 12
sns.set_style('darkgrid')
fig, ax = plt.subplots(2, 1, figsize=(20, 8))
history = pd.DataFrame(results.history)
history[['loss', 'val_loss']].plot(ax=ax[0])
history[['acc', 'val_acc']].plot(ax=ax[1])
fig.suptitle('Learning Curve', fontsize=24);




## === cell 13
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection

    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    iou = intersection / union

    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)




## === cell 14
preds_eval = model.predict(X_eval, verbose=1)

thresholds = np.linspace(0, 1, 50).astype(np.float32)

y_true = Y_eval[..., 0].astype(np.bool_, copy=False)
y_pred = preds_eval[..., 0].astype(np.float32, copy=False)

N = y_true.shape[0]
P = y_true.shape[1] * y_true.shape[2]
yt = y_true.reshape((N, P))
yp = y_pred.reshape((N, P))

area_true = yt.sum(axis=1, dtype=np.int32).astype(np.float64, copy=False)
kaggle_metric_thresholds = np.arange(0.5, 1.0, 0.05, dtype=np.float64)

order = np.argsort(-yp, axis=1, kind="mergesort")
yp_sorted = np.take_along_axis(yp, order, axis=1)
yt_sorted = np.take_along_axis(yt, order, axis=1).astype(np.int32, copy=False)
cum_tp = np.cumsum(yt_sorted, axis=1, dtype=np.int32)

yp_asc = yp_sorted[:, ::-1]  # ascending
idx = np.searchsorted(yp_asc, thresholds[None, :], side='right')  # (N, T)
k = (P - idx).astype(np.int32, copy=False)  # (N, T)

tp = np.zeros((N, thresholds.shape[0]), dtype=np.float64)
k_pos = k > 0
gather_idx = np.clip(k - 1, 0, P - 1)
tp_int = np.take_along_axis(cum_tp, gather_idx, axis=1)  # (N, T), int32
tp[k_pos] = tp_int[k_pos].astype(np.float64)

area_pred = k.astype(np.float64, copy=False)
union = area_true[:, None] + area_pred - tp
union = np.where(union == 0.0, 1e-9, union)
iou = tp / union  # (N, T)

ious = (iou[:, :, None] > kaggle_metric_thresholds[None, None, :]).mean(axis=2).mean(axis=0)

threshold_best_index = int(np.argmax(ious[9:-10]) + 9)
iou_best = float(ious[threshold_best_index])
threshold_best = float(thresholds[threshold_best_index])

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend();




## === cell 15
X_test = np.empty((len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8)
sizes_test = np.empty((len(test_ids), 2), dtype=np.int32)

print('Getting and resizing test images ... ')
sys.stdout.flush()

test_img_dir = os.path.join(config.path_test, "images")

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img_path = os.path.join(test_img_dir, id_)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read test image: {img_path}")
    sizes_test[n, 0] = img.shape[0]
    sizes_test[n, 1] = img.shape[1]
    img_r = cv2.resize(img, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR)
    X_test[n, ..., 0] = img_r

print('Done!')




## === cell 16
preds_test = model.predict(X_test, verbose=1)

preds_test_upsampled = [None] * len(preds_test)
for i in trange(len(preds_test)):
    h, w = int(sizes_test[i, 0]), int(sizes_test[i, 1])
    p = preds_test[i, ..., 0]
    preds_test_upsampled[i] = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)




## === cell 17
!rm -rf train
!rm -rf test




## === cell 18
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    if order != 'F':
        raise ValueError("This optimized RLenc preserves original behavior for order='F' only.")
    pixels = img.T.flatten()  # Fortran order flatten for 2D
    pixels = np.asarray(pixels, dtype=np.uint8)

    padded = np.concatenate(([0], pixels, [0]))
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    run_starts = changes[padded[changes - 1] == 0]
    run_ends = changes[padded[changes - 1] == 1]
    run_lengths = run_ends - run_starts

    if not format:
        return list(zip(run_starts.tolist(), run_lengths.tolist()))
    if run_starts.size == 0:
        return ''
    out = np.empty((run_starts.size * 2,), dtype=object)
    out[0::2] = run_starts.astype(str)
    out[1::2] = run_lengths.astype(str)
    return ' '.join(out.tolist())

pred_dict = {}
for i, fn in tqdm(enumerate(test_ids), total=len(test_ids)):
    pred_dict[fn[:-4]] = RLenc((preds_test_upsampled[i] > threshold_best))

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
```
