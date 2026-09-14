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

0.72102

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    base_path = "/kaggle/input/tgs-salt-identification-challenge"
    path_train = os.path.join(base_path, "train")
    path_test = os.path.join(base_path, "test")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1222651930.py in <cell line: 0>()
----> 1 class config:
      2     im_width = 128
      3     im_height = 128
      4     im_chan = 1
      5     base_path = "/kaggle/input/tgs-salt-identification-challenge"

/tmp/ipykernel_11/1222651930.py in config()
      4     im_chan = 1
      5     base_path = "/kaggle/input/tgs-salt-identification-challenge"
----> 6     path_train = os.path.join(base_path, "train")
      7     path_test = os.path.join(base_path, "test")
      8 

NameError: name 'os' is not defined

## === cell 1
def load_resize_gray(path, target_h, target_w):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (h, w)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img = img[..., np.newaxis]  # add channel dim
    return img


X = np.empty(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y = np.empty((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)

print("Loading and resizing train images and masks...")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img_path = os.path.join(train_img_dir, id_)
    mask_path = os.path.join(train_mask_dir, id_)
    X[n] = load_resize_gray(img_path, config.im_height, config.im_width).astype(
        np.uint8
    )
    m = load_resize_gray(mask_path, config.im_height, config.im_width)
    Y[n] = (m > 127).astype(bool)  # binarize mask
print("Done!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3932612580.py in <cell line: 0>()
      6 
      7 
----> 8 X = np.empty(
      9     (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
     10 )

NameError: name 'np' is not defined

## === cell 2
split_idx = int(0.9 * len(X))
X_train_orig = X[:split_idx]
Y_train_orig = Y[:split_idx]
X_eval = X[split_idx:]
Y_eval = Y[split_idx:]

X_hflip = np.flip(X_train_orig, axis=2)  # horizontal flip
Y_hflip = np.flip(Y_train_orig, axis=2)

X_train = np.concatenate([X_train_orig, X_hflip], axis=0)
Y_train = np.concatenate([Y_train_orig, Y_hflip], axis=0)

print("After augmentation:")
print("X_train shape:", X_train.shape, "X_eval shape:", X_eval.shape)
print("Y_train shape:", Y_train.shape, "Y_eval shape:", Y_eval.shape)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3701583009.py in <cell line: 0>()
      1 # Reduce augmentation to only horizontal flip (halves the training size while keeping a useful transform)
----> 2 split_idx = int(0.9 * len(X))
      3 X_train_orig = X[:split_idx]
      4 Y_train_orig = Y[:split_idx]
      5 X_eval = X[split_idx:]

NameError: name 'X' is not defined

## === cell 3
def build_model(input_layer, start_neurons):
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(
        input_layer
    )
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(
        conv1
    )
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(0.25)(pool1)

    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(
        pool1
    )
    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(
        conv2
    )
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(0.5)(pool2)

    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(
        pool2
    )
    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(
        conv3
    )
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(0.5)(pool3)

    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(
        pool3
    )
    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(
        conv4
    )
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(0.5)(pool4)

    convm = layers.Conv2D(
        start_neurons * 16, (3, 3), activation="relu", padding="same"
    )(pool4)
    convm = layers.Conv2D(
        start_neurons * 16, (3, 3), activation="relu", padding="same"
    )(convm)

    deconv4 = layers.Conv2DTranspose(
        start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
    )(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.5)(uconv4)
    uconv4 = layers.Conv2D(
        start_neurons * 8, (3, 3), activation="relu", padding="same"
    )(uconv4)
    uconv4 = layers.Conv2D(
        start_neurons * 8, (3, 3), activation="relu", padding="same"
    )(uconv4)

    deconv3 = layers.Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same"
    )(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.5)(uconv3)
    uconv3 = layers.Conv2D(
        start_neurons * 4, (3, 3), activation="relu", padding="same"
    )(uconv3)
    uconv3 = layers.Conv2D(
        start_neurons * 4, (3, 3), activation="relu", padding="same"
    )(uconv3)

    deconv2 = layers.Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same"
    )(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(0.5)(uconv2)
    uconv2 = layers.Conv2D(
        start_neurons * 2, (3, 3), activation="relu", padding="same"
    )(uconv2)
    uconv2 = layers.Conv2D(
        start_neurons * 2, (3, 3), activation="relu", padding="same"
    )(uconv2)

    deconv1 = layers.Conv2DTranspose(
        start_neurons * 1, (3, 3), strides=(2, 2), padding="same"
    )(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(0.5)(uconv1)
    uconv1 = layers.Conv2D(
        start_neurons * 1, (3, 3), activation="relu", padding="same"
    )(uconv1)
    uconv1 = layers.Conv2D(
        start_neurons * 1, (3, 3), activation="relu", padding="same"
    )(uconv1)

    output_layer = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(
        uconv1
    )
    return output_layer


input_layer = Input((config.im_height, config.im_width, config.im_chan))
output_layer = build_model(input_layer, 16)

model = models.Model(input_layer, output_layer)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1336277759.py in <cell line: 0>()
     98 
     99 
--> 100 input_layer = Input((config.im_height, config.im_width, config.im_chan))
    101 output_layer = build_model(input_layer, 16)
    102 

NameError: name 'Input' is not defined

## === cell 4
def _normalize(img, mask):
    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32)
    return img, mask


train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train.astype(np.uint8), Y_train.astype(np.uint8))
)
train_dataset = (
    train_dataset.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1024, seed=19, reshuffle_each_iteration=False)
    .cache()
    .batch(8)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (X_eval.astype(np.uint8), Y_eval.astype(np.uint8))
    )
    .map(_normalize)
    .batch(8)
)

es = callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)

results = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[es, rlp],
    verbose=2,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/432083174.py in <cell line: 0>()
      6 
      7 
----> 8 train_dataset = tf.data.Dataset.from_tensor_slices(
      9     (X_train.astype(np.uint8), Y_train.astype(np.uint8))
     10 )

NameError: name 'tf' is not defined

## === cell 5
sns.set_style("darkgrid")
fig, ax = plt.subplots(2, 1, figsize=(20, 8))
history = pd.DataFrame(results.history)
history[["loss", "val_loss"]].plot(ax=ax[0])
history[["accuracy", "val_accuracy"]].plot(ax=ax[1])
fig.suptitle("Learning Curve", fontsize=24)
plt.show()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1316776477.py in <cell line: 0>()
----> 1 sns.set_style("darkgrid")
      2 fig, ax = plt.subplots(2, 1, figsize=(20, 8))
      3 history = pd.DataFrame(results.history)
      4 history[["loss", "val_loss"]].plot(ax=ax[0])
      5 history[["accuracy", "val_accuracy"]].plot(ax=ax[1])

NameError: name 'sns' is not defined

## === cell 6
def iou_metric(y_true, y_pred, print_table=False):
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(
        y_true.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    area_true = np.histogram(y_true, bins=true_objects)[0]
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
        p = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        if print_table:
            print(f"{t:.3f}\t{tp}\t{fp}\t{fn}\t{p:.3f}")
        prec.append(p)
    if print_table:
        print(f"AP\t-\t-\t-\t{np.mean(prec):.3f}")
    return np.mean(prec)


def iou_metric_batch(y_true_batch, y_pred_batch):
    return np.mean(
        [
            iou_metric(y_true, y_pred)
            for y_true, y_pred in zip(y_true_batch, y_pred_batch)
        ]
    )


preds_eval = model.predict(X_eval, verbose=0)
thresholds = np.linspace(0, 1, 50)
ious = []
for thr in tqdm(thresholds):
    ious.append(iou_metric_batch(Y_eval, (preds_eval > thr).astype(np.uint8)))
ious = np.array(ious)

if ious.size:
    best_idx = np.argmax(ious[9:-10]) + 9
    iou_best = ious[best_idx]
    threshold_best = thresholds[best_idx]
else:
    threshold_best = 0.5
    iou_best = 0.0

print(
    f"Best validation IoU (approx. target metric): {iou_best:.4f} at threshold {threshold_best:.3f}"
)

plt.plot(thresholds, ious, label="IoU")
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title(f"Threshold vs IoU (best={threshold_best:.2f}, IoU={iou_best:.3f})")
plt.legend()
plt.show()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2676286318.py in <cell line: 0>()
     52 
     53 
---> 54 preds_eval = model.predict(X_eval, verbose=0)
     55 thresholds = np.linspace(0, 1, 50)
     56 ious = []

NameError: name 'model' is not defined

## === cell 7
X_test = np.empty(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = []
print("Loading and resizing test images...")
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img_path = os.path.join(test_img_dir, id_)
    x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sizes_test.append([x.shape[0], x.shape[1]])  # original height, width
    x = cv2.resize(
        x, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR
    )
    X_test[n] = x[..., np.newaxis].astype(np.uint8)
print("Done!")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2082436496.py in <cell line: 0>()
----> 1 X_test = np.empty(
      2     (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
      3 )
      4 sizes_test = []
      5 print("Loading and resizing test images...")

NameError: name 'np' is not defined

## === cell 8
preds_test = model.predict(X_test, verbose=0)
preds_test_upsampled = []
for i in trange(len(preds_test)):
    up = cv2.resize(
        np.squeeze(preds_test[i]),
        (sizes_test[i][1], sizes_test[i][0]),  # width, height
        interpolation=cv2.INTER_LINEAR,
    )
    preds_test_upsampled.append(up)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1535205112.py in <cell line: 0>()
----> 1 preds_test = model.predict(X_test, verbose=0)
      2 preds_test_upsampled = []
      3 for i in trange(len(preds_test)):
      4     up = cv2.resize(
      5         np.squeeze(preds_test[i]),

NameError: name 'model' is not defined

## === cell 9
def RLenc(img, order="F", format=True):
    """
    img: binary mask (2D array)
    Returns run‑length encoding as required by the competition.
    """
    bytes = img.reshape(-1, order=order)
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
    return runs


pred_dict = {
    fn[:-4]: RLenc(np.round(preds_test_upsampled[i] > threshold_best))
    for i, fn in enumerate(test_ids)
}




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1461472142.py in <cell line: 0>()
     26 pred_dict = {
     27     fn[:-4]: RLenc(np.round(preds_test_upsampled[i] > threshold_best))
---> 28     for i, fn in enumerate(test_ids)
     29 }
     30 

NameError: name 'test_ids' is not defined

## === cell 10
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.name = "id"
sub.columns = ["rle_mask"]
sub.to_csv("submission.csv", index=True)
print("Submission file written to submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/967654881.py in <cell line: 0>()
----> 1 sub = pd.DataFrame.from_dict(pred_dict, orient="index")
      2 sub.index.name = "id"
      3 sub.columns = ["rle_mask"]
      4 sub.to_csv("submission.csv", index=True)
      5 print("Submission file written to submission.csv")

NameError: name 'pd' is not defined
