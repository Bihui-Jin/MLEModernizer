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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scipy==1.15.3
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

0.63693

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I fixed the protobuf import issue, corrected the depth‑array reshaping after augmentation, simplified the image loading to guarantee a single channel, adjusted the model architecture so the output size matches the input size, and reordered the cells so all variables are defined before they are used. These changes remove the runtime errors and produce a valid `output_6.csv` submission ready for Kaggle.'
- What this solution (achieved 0.5221) has done: 'Implemented fixes and modest enhancements:
- Set protobuf implementation before importing TensorFlow to avoid the `MessageFactory` error.
- Integrated depth information as a second input channel for both train and test data.
- Adjusted the model input shape to accommodate the new 2‑channel input.
- Minor clean‑up of variable handling while preserving original architecture and training routine.'
- What this solution (achieved 0.5221) has done: 'Implemented three key fixes:  
1. Set the protobuf implementation flag **before any imports** to eliminate the `MessageFactory` error.  
2. Loaded and merged test‑set depth information so the model receives the same normalized depth channel during inference (instead of a zero channel).  
3. Trained the network a bit longer (20 epochs) with a slightly deeper architecture to boost the validation score toward the target.'
- What this solution (achieved 0.5221) has done: 'Implemented three key enhancements while keeping the original model structure:
1. Filled missing depth values in the test set with the mean training depth to avoid NaNs.
2. Added vertical‑flip augmentation (in addition to the existing horizontal flip) and dynamically repeated the depth channel to match the expanded training set.
3. Adjusted the binary‑mask threshold from 0.5 to 0.45 to improve recall, which typically raises the mean‑average‑precision metric.'
- What this solution (achieved 0.5221) has done: 'The updates fix the protobuf import error, add test‑time augmentation averaging (horizontal and vertical flips) to boost prediction quality, increase training epochs to 30 for better learning, and lower the binary mask threshold to 0.4, which together should raise the score toward the target while keeping the original model architecture unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

BASE_PATH = "../input/tgs-salt-identification-challenge"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
df_depths = pd.read_csv(os.path.join(BASE_PATH, "depths.csv"))
df_depths["z"] = df_depths["z"] / np.max(df_depths["z"])
df_train = df_train.merge(
    df_depths.rename(columns={"z": "depth"}), on="id", how="inner"
)
print(df_train.head())
print("Train rows:", df_train.shape[0])




## === cell 2
test_img_path = os.path.join(BASE_PATH, "test/images")
test_ids = [
    os.path.splitext(f)[0]
    for f in os.listdir(test_img_path)
    if f.lower().endswith(".png")
]
df_test = pd.DataFrame({"id": test_ids})
df_test = df_test.merge(df_depths.rename(columns={"z": "depth"}), on="id", how="left")
df_test["depth"] = df_test["depth"].fillna(df_depths["depth"].mean())
print(df_test.head())
print("Test rows:", df_test.shape[0])




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'depth'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/118855558.py in <cell line: 0>()
      9 df_test = df_test.merge(df_depths.rename(columns={"z": "depth"}), on="id", how="left")
     10 # Fill any missing depth values with the training mean
---> 11 df_test["depth"] = df_test["depth"].fillna(df_depths["depth"].mean())
     12 print(df_test.head())
     13 print("Test rows:", df_test.shape[0])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'depth'

## === cell 3
def load_image(path):
    """Load a PNG image as a numpy array (grayscale)."""
    return np.array(Image.open(path).convert("L"))


train_img_path = os.path.join(BASE_PATH, "train/images")
train_mask_path = os.path.join(BASE_PATH, "train/masks")

df_train["images"] = [
    load_image(f"{train_img_path}/{idx}.png") for idx in df_train["id"]
]
df_train["masks"] = [
    np.array(Image.open(f"{train_mask_path}/{idx}.png")) for idx in df_train["id"]
]

print("Sample train image shape:", df_train["images"].iloc[0].shape)

df_train["images"] = df_train["images"].apply(
    lambda x: (x[..., np.newaxis] / 255.0).astype(np.float32)
)
df_train["masks"] = df_train["masks"].apply(
    lambda x: (x[..., np.newaxis] / x.max()).round().astype(int)
)

print("After processing, image shape:", df_train["images"].iloc[0].shape)
print("After processing, mask shape:", df_train["masks"].iloc[0].shape)




## === cell 4
df_test["images"] = [load_image(f"{test_img_path}/{idx}.png") for idx in df_test["id"]]
df_test["images"] = df_test["images"].apply(
    lambda x: (x[..., np.newaxis] / 255.0).astype(np.float32)
)

print("Sample test image shape:", df_test["images"].iloc[0].shape)




## === cell 5
train_x = np.stack(df_train["images"].values)  # (N,101,101,1)
train_y = np.stack(df_train["masks"].values)  # (N,101,101,1)
test_x = np.stack(df_test["images"].values)  # (M,101,101,1)

print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)
print("test_x shape :", test_x.shape)




## === cell 6
train_x_hflip = np.flip(train_x, axis=2)
train_y_hflip = np.flip(train_y, axis=2)

train_x_vflip = np.flip(train_x, axis=1)
train_y_vflip = np.flip(train_y, axis=1)

train_x = np.concatenate([train_x, train_x_hflip, train_x_vflip], axis=0)
train_y = np.concatenate([train_y, train_y_hflip, train_y_vflip], axis=0)

print("After augmentation, train_x shape:", train_x.shape)
print("After augmentation, train_y shape:", train_y.shape)




## === cell 7
num_train = train_x.shape[0]
repeat_factor = num_train // df_train.shape[0]  # should be 3 after augmentation
depth_vals = np.tile(df_train["depth"].values, repeat_factor)

depth_np = (
    np.repeat(depth_vals[:, None, None, None], 101 * 101, axis=1)
    .reshape(num_train, 101, 101, 1)
    .astype(np.float32)
)

train_x = np.concatenate([train_x, depth_np], axis=-1)  # (N,101,101,2)
print("train_x with depth shape:", train_x.shape)

depth_test = df_test["depth"].values.astype(np.float32)
depth_test_np = np.repeat(depth_test[:, None, None, None], 101 * 101, axis=1).reshape(
    df_test.shape[0], 101, 101, 1
)

test_x = np.concatenate([test_x, depth_test_np], axis=-1)  # (M,101,101,2)
print("test_x with depth shape:", test_x.shape)




## === cell 8
X_tr, X_val, y_tr, y_val = train_test_split(
    train_x, train_y, test_size=0.1, random_state=42
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(101, 101, 2)),  # image + depth
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(
    X_tr, y_tr, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=2
)


def iou_metric(pred, true):
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return 1.0 if union == 0 else intersection / union


val_pred_prob = model.predict(X_val, batch_size=32)

thresholds = np.arange(0.30, 0.56, 0.05)
best_thr = 0.5
best_score = -1.0
for thr in thresholds:
    val_pred_bin = (val_pred_prob > thr).astype(int)
    ious = [iou_metric(val_pred_bin[i], y_val[i]) for i in range(val_pred_bin.shape[0])]
    mean_iou = np.mean(ious)
    if mean_iou > best_score:
        best_score = mean_iou
        best_thr = thr
print(f"Best threshold on validation set: {best_thr:.2f} (mean IoU ≈ {best_score:.4f})")




## === cell 9
y_pred_orig = model.predict(test_x, batch_size=32)

test_x_hflip = np.flip(test_x, axis=2)
y_pred_hflip = model.predict(test_x_hflip, batch_size=32)
y_pred_hflip = np.flip(y_pred_hflip, axis=2)

test_x_vflip = np.flip(test_x, axis=1)
y_pred_vflip = model.predict(test_x_vflip, batch_size=32)
y_pred_vflip = np.flip(y_pred_vflip, axis=1)

y_pred = (y_pred_orig + y_pred_hflip + y_pred_vflip) / 3.0
print("Prediction shape after TTA averaging:", y_pred.shape)




## === cell 10
def post_process(x):
    """
    Convert binary mask array to run‑length encoding strings.
    Empty masks are encoded as "-1" as required by the competition.
    """
    num_files = x.shape[0]
    Height, Width = x.shape[1], x.shape[2]
    pic_size = Height * Width
    main_list = []
    for i in range(num_files):
        pic = x[i].ravel()
        s = ""
        length = 0
        start = 0
        for j in range(pic_size):
            if j == pic_size - 1:
                if pic[j] == 1 and length == 0:
                    s += f" {j+1} 1"
                elif pic[j] == 1 and length != 0:
                    s += f" {start} {length+1}"
            else:
                if pic[j] == 1:
                    length += 1
                    if length == 1:
                        start = j + 1
                elif pic[j] == 0 and length > 0:
                    s += f" {start} {length}"
                    length = 0
        s = s.strip()
        if not s:  # no mask pixels
            s = "-1"
        main_list.append(s)
    return main_list


y_bin = (y_pred > best_thr).astype(int)
new_y_pred = post_process(y_bin)
print("Processed predictions, count:", len(new_y_pred))




## === cell 11
submission = pd.DataFrame({"id": df_test["id"], "rle_mask": new_y_pred})
submission.to_csv("output_6.csv", index=False)
print("Submission saved to output_6.csv")

## --- ERROR in outputing the csv:
Invalid submission: Expected all items in `submission` to be strings, but found non-string items!
