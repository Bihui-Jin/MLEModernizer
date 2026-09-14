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

# 5. Target score

0.80965

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, random, warnings, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm, trange
import cv2
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import img_to_array, load_img

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras import models, Input, layers, callbacks, utils, optimizers
except Exception as e:
    tf = None
    print("TensorFlow import failed, will use fallback Otsu segmentation:", e)

warnings.filterwarnings("ignore")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = "train/"
    path_test = "test/"




## === cell 2
import zipfile, pathlib

base_input = pathlib.Path("input/tgs-salt-identification-challenge")
train_zip = base_input / "train.zip"
test_zip = base_input / "test.zip"

if train_zip.is_file():
    with zipfile.ZipFile(train_zip, "r") as zip_ref:
        zip_ref.extractall("train")
if test_zip.is_file():
    with zipfile.ZipFile(test_zip, "r") as zip_ref:
        zip_ref.extractall("test")



## === cell 3
train_image_dir = os.path.join(config.path_train, "images")
test_image_dir = os.path.join(config.path_test, "images")

if os.path.isdir(train_image_dir):
    train_ids = next(os.walk(train_image_dir))[2]
else:
    train_ids = []
    print("Train image directory not found; proceeding without training data.")

if os.path.isdir(test_image_dir):
    test_ids = next(os.walk(test_image_dir))[2]
else:
    test_ids = []
    print("Test image directory not found; cannot create submission.")
    sys.exit(0)



## --- ERROR in cell 3, traceback:
An exception has occurred, use %tb to see the full traceback.

SystemExit: 0


## === cell 4
if train_ids:
    X = np.zeros(
        (len(train_ids), config.im_height, config.im_width, config.im_chan),
        dtype=np.uint8,
    )
    Y = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)
    print("Loading and resizing training images and masks...")
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        x = img_to_array(
            load_img(
                os.path.join(config.path_train, "images", id_), color_mode="grayscale"
            )
        )
        x = resize(
            x,
            (config.im_height, config.im_width, config.im_chan),
            mode="constant",
            preserve_range=True,
        )
        X[n] = x
        mask = img_to_array(
            load_img(
                os.path.join(config.path_train, "masks", id_), color_mode="grayscale"
            )
        )
        Y[n] = resize(
            mask,
            (config.im_height, config.im_width, 1),
            mode="constant",
            preserve_range=True,
        )
    print("Training data loaded.")
else:
    X = np.empty((0, config.im_height, config.im_width, config.im_chan), dtype=np.uint8)
    Y = np.empty((0, config.im_height, config.im_width, 1), dtype=bool)



## === cell 5
if X.shape[0] > 0:
    split_idx = int(0.9 * len(X))
    X_train = X[:split_idx]
    Y_train = Y[:split_idx]
    X_eval = X[split_idx:]
    Y_eval = Y[split_idx:]

    X_train = np.append(X_train, [np.fliplr(x) for x in X], axis=0)
    Y_train = np.append(Y_train, [np.fliplr(x) for x in Y], axis=0)
    X_train = np.append(X_train, [np.flipud(x) for x in X], axis=0)
    Y_train = np.append(Y_train, [np.flipud(x) for x in Y], axis=0)
else:
    X_train = X_eval = Y_train = Y_eval = np.empty(
        (0, config.im_height, config.im_width, config.im_chan)
    )

del X, Y

print("Shapes -> X_train:", X_train.shape, "X_eval:", X_eval.shape)



## === cell 6
if tf is not None and X_train.shape[0] > 0:

    def BatchActivate(x):
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def convolution_block(
        x, filters, size, strides=(1, 1), padding="same", activation=True
    ):
        x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
        if activation:
            x = BatchActivate(x)
        return x

    def residual_block(blockInput, num_filters=16, batch_activate=False):
        x = BatchActivate(blockInput)
        x = convolution_block(x, num_filters, (3, 3))
        x = convolution_block(x, num_filters, (3, 3), activation=False)
        x = layers.Add()([x, blockInput])
        if batch_activate:
            x = BatchActivate(x)
        return x

    def build_model(input_layer, start_neurons, DropoutRatio=0.5):
        scaled = layers.Lambda(lambda x: x / 255.0)(input_layer)

        conv1 = layers.Conv2D(start_neurons, (3, 3), activation=None, padding="same")(
            scaled
        )
        conv1 = residual_block(conv1, start_neurons)
        conv1 = residual_block(conv1, start_neurons, True)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)
        pool1 = layers.Dropout(DropoutRatio / 2)(pool1)

        conv2 = layers.Conv2D(
            start_neurons * 2, (3, 3), activation=None, padding="same"
        )(pool1)
        conv2 = residual_block(conv2, start_neurons * 2)
        conv2 = residual_block(conv2, start_neurons * 2, True)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)
        pool2 = layers.Dropout(DropoutRatio)(pool2)

        conv3 = layers.Conv2D(
            start_neurons * 4, (3, 3), activation=None, padding="same"
        )(pool2)
        conv3 = residual_block(conv3, start_neurons * 4)
        conv3 = residual_block(conv3, start_neurons * 4, True)
        pool3 = layers.MaxPooling2D((2, 2))(conv3)
        pool3 = layers.Dropout(DropoutRatio)(pool3)

        conv4 = layers.Conv2D(
            start_neurons * 8, (3, 3), activation=None, padding="same"
        )(pool3)
        conv4 = residual_block(conv4, start_neurons * 8)
        conv4 = residual_block(conv4, start_neurons * 8, True)
        pool4 = layers.MaxPooling2D((2, 2))(conv4)
        pool4 = layers.Dropout(DropoutRatio)(pool4)

        convm = layers.Conv2D(
            start_neurons * 16, (3, 3), activation=None, padding="same"
        )(pool4)
        convm = residual_block(convm, start_neurons * 16)
        convm = residual_block(convm, start_neurons * 16, True)

        deconv4 = layers.Conv2DTranspose(
            start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
        )(convm)
        uconv4 = layers.concatenate([deconv4, conv4])
        uconv4 = layers.Dropout(DropoutRatio)(uconv4)
        uconv4 = layers.Conv2D(
            start_neurons * 8, (3, 3), activation=None, padding="same"
        )(uconv4)
        uconv4 = residual_block(uconv4, start_neurons * 8)
        uconv4 = residual_block(uconv4, start_neurons * 8, True)

        deconv3 = layers.Conv2DTranspose(
            start_neurons * 4, (3, 3), strides=(2, 2), padding="same"
        )(uconv4)
        uconv3 = layers.concatenate([deconv3, conv3])
        uconv3 = layers.Dropout(DropoutRatio)(uconv3)
        uconv3 = layers.Conv2D(
            start_neurons * 4, (3, 3), activation=None, padding="same"
        )(uconv3)
        uconv3 = residual_block(uconv3, start_neurons * 4)
        uconv3 = residual_block(uconv3, start_neurons * 4, True)

        deconv2 = layers.Conv2DTranspose(
            start_neurons * 2, (3, 3), strides=(2, 2), padding="same"
        )(uconv3)
        uconv2 = layers.concatenate([deconv2, conv2])
        uconv2 = layers.Dropout(DropoutRatio)(uconv2)
        uconv2 = layers.Conv2D(
            start_neurons * 2, (3, 3), activation=None, padding="same"
        )(uconv2)
        uconv2 = residual_block(uconv2, start_neurons * 2)
        uconv2 = residual_block(uconv2, start_neurons * 2, True)

        deconv1 = layers.Conv2DTranspose(
            start_neurons, (3, 3), strides=(2, 2), padding="same"
        )(uconv2)
        uconv1 = layers.concatenate([deconv1, conv1])
        uconv1 = layers.Dropout(DropoutRatio)(uconv1)
        uconv1 = layers.Conv2D(start_neurons, (3, 3), activation=None, padding="same")(
            uconv1
        )
        uconv1 = residual_block(uconv1, start_neurons)
        uconv1 = residual_block(uconv1, start_neurons, True)

        output_layer_noActi = layers.Conv2D(1, (1, 1), padding="same", activation=None)(
            uconv1
        )
        output_layer = layers.Activation("sigmoid")(output_layer_noActi)
        return output_layer

    input_layer = Input((config.im_height, config.im_width, config.im_chan))
    output_layer = build_model(input_layer, 16)
    model = models.Model(input_layer, output_layer)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    print("Model built successfully.")
else:
    model = None
    print("Skipping model creation (TensorFlow unavailable or no training data).")



## === cell 7
if model is not None:
    utils.plot_model(model, expand_nested=True, show_shapes=True)



## === cell 8
if model is not None and X_train.shape[0] > 0:
    es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
    rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

    results = model.fit(
        X_train,
        Y_train,
        validation_data=(X_eval, Y_eval),
        batch_size=8,
        epochs=50,
        callbacks=[es, rlp],
        verbose=2,
    )
else:
    results = None
    print("Training skipped.")



## === cell 9
if results is not None:
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    history = pd.DataFrame(results.history)
    history[["loss", "val_loss"]].plot(ax=ax[0])
    history[["acc", "val_acc"]].plot(ax=ax[1])
    fig.suptitle("Learning Curve", fontsize=24)
    plt.show()
else:
    print("No training history to display.")




## === cell 10
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection

    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    iou = intersection / union

    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1
        false_positives = np.sum(matches, axis=0) == 0
        false_negatives = np.sum(matches, axis=1) == 0
        tp = np.sum(true_positives)
        fp = np.sum(false_positives)
        fn = np.sum(false_negatives)
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
        metric.append(iou_metric(y_true_in[batch], y_pred_in[batch]))
    return np.mean(metric)




## === cell 11
if model is not None and X_eval.shape[0] > 0:
    preds_eval = model.predict(X_eval, verbose=1)

    thresholds = np.linspace(0, 1, 50)
    ious = np.array(
        [iou_metric_batch(Y_eval, np.int32(preds_eval > th)) for th in tqdm(thresholds)]
    )

    best_idx = np.argmax(ious[9:-10]) + 9
    iou_best = ious[best_idx]
    threshold_best = thresholds[best_idx]

    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title(f"Threshold vs IoU (best={threshold_best:.3f}, IoU={iou_best:.3f})")
    plt.legend()
    plt.show()
else:
    threshold_best = 0.5
    print("Using default threshold:", threshold_best)



## === cell 12
X_test = np.zeros(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = []
print("Loading and resizing test images...")
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img_path = os.path.join(config.path_test, "images", id_)
    x = img_to_array(load_img(img_path, color_mode="grayscale"))
    sizes_test.append([x.shape[0], x.shape[1]])  # original size
    x = resize(
        x,
        (config.im_height, config.im_width, config.im_chan),
        mode="constant",
        preserve_range=True,
    )
    X_test[n] = x
print("Done loading test data.")



## === cell 13
if model is not None:
    preds_test = model.predict(X_test, verbose=1)
else:
    preds_test = []
    for n in range(len(test_ids)):
        up = resize(
            np.squeeze(X_test[n]),
            (sizes_test[n][0], sizes_test[n][1]),
            mode="constant",
            preserve_range=True,
        )
        up_uint8 = (up * 255).astype(np.uint8)
        _, mask = cv2.threshold(up_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preds_test.append(mask / 255.0)  # normalize to [0,1] like model output
    preds_test = np.array(preds_test)

preds_test_upsampled = []
for i in trange(len(preds_test)):
    up = resize(
        np.squeeze(preds_test[i]),
        (sizes_test[i][0], sizes_test[i][1]),
        mode="constant",
        preserve_range=True,
    )
    preds_test_upsampled.append(up)




## === cell 14
def RLenc(img, order="F", format=True):
    """
    Run‑length encoding.
    img – binary mask (2‑D array)
    order – 'F' for Fortran‑style column‑wise flattening
    format – if True returns a formatted string
    """
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
        return " ".join(f"{s} {l}" for s, l in runs)
    return runs


pred_dict = {
    fn[:-4]: RLenc(np.round(preds_test_upsampled[i] > threshold_best))
    for i, fn in tqdm(enumerate(test_ids), total=len(test_ids))
}



## === cell 15
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.names = ["id"]
sub.columns = ["rle_mask"]
sub.to_csv("submission.csv", index=True)
print("Submission file 'submission.csv' written with", sub.shape[0], "rows.")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/196034110.py in <cell line: 0>()
      2 sub = pd.DataFrame.from_dict(pred_dict, orient="index")
      3 sub.index.names = ["id"]
----> 4 sub.columns = ["rle_mask"]
      5 sub.to_csv("submission.csv", index=True)
      6 print("Submission file 'submission.csv' written with", sub.shape[0], "rows.")

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __setattr__(self, name, value)
   6311         try:
   6312             object.__getattribute__(self, name)
-> 6313             return object.__setattr__(self, name, value)
   6314         except AttributeError:
   6315             pass

properties.pyx in pandas._libs.properties.AxisProperty.__set__()

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _set_axis(self, axis, labels)
    812         """
    813         labels = ensure_index(labels)
--> 814         self._mgr.set_axis(axis, labels)
    815         self._clear_item_cache()
    816 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in set_axis(self, axis, new_labels)
    236     def set_axis(self, axis: AxisInt, new_labels: Index) -> None:
    237         # Caller is responsible for ensuring we have an Index object.
--> 238         self._validate_set_axis(axis, new_labels)
    239         self.axes[axis] = new_labels
    240 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/base.py in _validate_set_axis(self, axis, new_labels)
     96 
     97         elif new_len != old_len:
---> 98             raise ValueError(
     99                 f"Length mismatch: Expected axis has {old_len} elements, new "
    100                 f"values have {new_len} elements"

ValueError: Length mismatch: Expected axis has 0 elements, new values have 1 elements
