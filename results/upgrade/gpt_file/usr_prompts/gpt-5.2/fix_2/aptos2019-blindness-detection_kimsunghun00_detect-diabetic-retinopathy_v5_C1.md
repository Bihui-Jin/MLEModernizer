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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

-0.087519

# 6. Current score

0.01907

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.01907) has done: 'I fix the environment-breaking notebook-only syntax (`%matplotlib inline`, shell `!ls`) and update tqdm usage so the script runs as a .py in Kaggle. I resolve the TensorFlow/protobuf `to_categorical` crash by switching to `tf.keras` consistently, and I remove the broken `imblearn` oversampling step (it fails due to sklearn version mismatch) while keeping the same CNN/training approach. I also fix the missing `model.fit(...)` call (currently an indented fragment) and update the checkpoint monitor to `val_accuracy` so weights are actually saved/loaded. Finally, I ensure test predictions align exactly to `test.csv` `id_code` order (not `os.listdir`/glob order) so the submission has the correct ids and passes Kaggle validation.'

# 9. Code solution

## === cell 0
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

pd.set_option("display.max_rows", 10)

np.random.seed(123)
tf.random.set_seed(123)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_data_folder = "/kaggle/input/aptos2019-blindness-detection"
train_data_folder = os.path.join(base_data_folder, "train_images")

print("Base folder exists:", os.path.exists(base_data_folder))
print("Train images folder exists:", os.path.exists(train_data_folder))
print("Base folder listing (first 20):", sorted(os.listdir(base_data_folder))[:20])



## === cell 2
train_files_names = sorted(os.listdir(train_data_folder))
train_files_names[:5]



## === cell 3
train_df = pd.read_csv(os.path.join(base_data_folder, "train.csv"))
train_df.head()



## === cell 4
train_images = []
missing = 0
for id_code in tqdm(train_df["id_code"].values, desc="Loading train images"):
    fp = os.path.join(train_data_folder, f"{id_code}.png")
    image_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
    if image_bgr is None:
        missing += 1
        continue
    image_resized = cv2.resize(image_bgr, dsize=(0, 0), fx=0.12, fy=0.12)
    train_images.append(image_resized)

print("Loaded train images:", len(train_images), "Missing:", missing)



## === cell 5
if missing == 0:
    labels = train_df.copy()
else:
    kept = []
    for id_code in train_df["id_code"].values:
        if os.path.exists(os.path.join(train_data_folder, f"{id_code}.png")):
            kept.append(id_code)
    labels = train_df[train_df["id_code"].isin(kept)].reset_index(drop=True)

y_data = labels["diagnosis"].astype(int)
labels.head()



## === cell 6
labels["diagnosis"].hist()
print(labels["diagnosis"].value_counts())




## === cell 7
def crop_image_from_gray(img, tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img > tol

    if not mask.any():
        return img

    img1 = img[:, :, 0][np.ix_(mask.any(axis=1), mask.any(axis=0))]
    img2 = img[:, :, 1][np.ix_(mask.any(axis=1), mask.any(axis=0))]
    img3 = img[:, :, 2][np.ix_(mask.any(axis=1), mask.any(axis=0))]
    img = np.stack([img1, img2, img3], axis=-1)
    return img




## === cell 8
def circle_crop(img):
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(
        img, dsize=(largest_side, largest_side), interpolation=cv2.INTER_CUBIC
    )

    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    background = np.zeros(shape=(height, width), dtype=np.uint8)
    cv2.circle(background, (x, y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=background)
    return img




## === cell 9
pic_num = 43
img = train_images[pic_num]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis("off")



## === cell 10
img1 = crop_image_from_gray(img_rgb)
plt.imshow(img1)
plt.axis("off")



## === cell 11
img2 = circle_crop(img_rgb)
plt.imshow(img2)
plt.axis("off")



## === cell 12
X_data = []
for image in tqdm(train_images, desc="Preprocessing train images"):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    circle_img = circle_crop(img_rgb)
    image_resized = cv2.resize(
        circle_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC
    )
    X_data.append(image_resized)

X_data = np.array(X_data).reshape(-1, 224, 224, 3)
print(X_data.shape)




## === cell 13
def values_in_mask(X):
    background = np.zeros(shape=(224, 224), dtype=np.uint8)
    circle_mask = cv2.circle(background, (112, 112), 110, 1, thickness=-1)

    dim1 = X[:, :, 0]
    dim2 = X[:, :, 1]
    dim3 = X[:, :, 2]

    circle_locations = circle_mask == 1
    R = dim1[circle_locations]
    G = dim2[circle_locations]
    B = dim3[circle_locations]
    return R, G, B




## === cell 14
def min_max_scaler_rgb(X):
    R, G, B = values_in_mask(X)

    dim1 = X[:, :, 0].astype("float32")
    dim2 = X[:, :, 1].astype("float32")
    dim3 = X[:, :, 2].astype("float32")

    min_R, min_G, min_B = np.min(R), np.min(G), np.min(B)
    max_R, max_G, max_B = np.max(R), np.max(G), np.max(B)

    denom_R = (max_R - min_R) if (max_R - min_R) != 0 else 1.0
    denom_G = (max_G - min_G) if (max_G - min_G) != 0 else 1.0
    denom_B = (max_B - min_B) if (max_B - min_B) != 0 else 1.0

    img_R = (dim1 - min_R) / denom_R
    img_G = (dim2 - min_G) / denom_G
    img_B = (dim3 - min_B) / denom_B

    img_R = np.clip(img_R, 0.0, 1.0)
    img_G = np.clip(img_G, 0.0, 1.0)
    img_B = np.clip(img_B, 0.0, 1.0)

    img = np.stack([img_R, img_G, img_B], axis=-1)
    return img




## === cell 15
def min_max_scaler_gray(X):
    denom = np.max(X) - np.min(X)
    denom = denom if denom != 0 else 1.0
    img = (X - np.min(X)) / denom
    return img




## === cell 16
fig = plt.figure(figsize=(14, 8))
for idx, image in enumerate(train_images[:10]):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig.add_subplot(2, 5, idx + 1)
    plt.imshow(image_rgb)
    plt.title("Label:{0}".format(labels["diagnosis"].iloc[idx]))
    plt.xlabel(labels["id_code"].iloc[idx])
    plt.tight_layout()



## === cell 17
fig = plt.figure(figsize=(14, 8))
for idx, image in enumerate(X_data[:10]):
    fig.add_subplot(2, 5, idx + 1)
    plt.imshow(image)
    plt.title("Label:{0}".format(labels["diagnosis"].iloc[idx]))
    plt.xlabel(labels["id_code"].iloc[idx])
    plt.tight_layout()



## === cell 18
del train_images



## === cell 19
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X_data, y_data, test_size=0.2, stratify=y_data, random_state=123
)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)



## === cell 20
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_valid_onehot = tf.keras.utils.to_categorical(y_valid, num_classes=5)

print(y_train_onehot.shape)
print(y_valid_onehot.shape)



## === cell 21
plt.hist(y_train)
plt.hist(y_valid)
plt.title("Train and Validation set Distribution")
plt.legend(["Train", "Validation"])
plt.show()



## === cell 22
X_resampled = X_train
y_resampled = y_train
y_resampled_onehot = y_train_onehot

print(X_resampled.shape, y_resampled_onehot.shape)



## === cell 23
plt.hist(y_resampled)
plt.hist(y_valid)
plt.title("Train and Validation set Distribution (No oversampling in environment)")
plt.legend(["Train", "Validation"], loc="right")
plt.show()



## === cell 24
del X_data, X_train



## === cell 25
model = models.Sequential()



## === cell 26
model.add(
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        input_shape=(224, 224, 3),
        name="Conv1-1",
    )
)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool1"))



## === cell 27
model.add(
    layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv2-1",
    )
)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool2"))



## === cell 28
model.add(
    layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv3-1",
    )
)
model.add(
    layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv3-2",
    )
)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool3"))



## === cell 29
model.add(
    layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv4-1",
    )
)
model.add(
    layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv4-2",
    )
)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool4"))



## === cell 30
model.add(
    layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv5-1",
    )
)
model.add(
    layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="Conv5-2",
    )
)
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool5"))



## === cell 31
model.add(layers.Flatten())



## === cell 32
model.add(layers.Dense(256, activation="relu", name="Dense1"))
model.add(layers.Dropout(0.3))



## === cell 33
model.add(layers.Dense(256, activation="relu", name="Dense2"))
model.add(layers.Dropout(0.3))



## === cell 34
model.add(layers.Dense(5, activation="softmax", name="Final"))



## === cell 35
model.summary()



## === cell 36
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 37
callback_list = [
    ModelCheckpoint(
        filepath="cnn_checkpoint.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1,
    )
]



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3300929495.py in <cell line: 0>()
      1 # Fix: monitor name should match tf.keras metric key ('val_accuracy'), so checkpoint is saved.
      2 callback_list = [
----> 3     ModelCheckpoint(
      4         filepath="cnn_checkpoint.h5",
      5         monitor="val_accuracy",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=cnn_checkpoint.h5

## === cell 38
history = model.fit(
    X_resampled,
    y_resampled_onehot,
    batch_size=200,
    epochs=100,
    validation_data=(X_valid, y_valid_onehot),
    callbacks=callback_list,
    verbose=2,
)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2039269899.py in <cell line: 0>()
      6     epochs=100,
      7     validation_data=(X_valid, y_valid_onehot),
----> 8     callbacks=callback_list,
      9     verbose=2,
     10 )

NameError: name 'callback_list' is not defined

## === cell 39
epochs_arr = np.arange(1, len(history.history["loss"]) + 1)

plt.plot(epochs_arr, history.history["loss"], label="Training")
plt.plot(epochs_arr, history.history["val_loss"], label="Validation")
plt.title("Loss History Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(epochs_arr, history.history["accuracy"], label="Training")
plt.plot(epochs_arr, history.history["val_accuracy"], label="Validation")
plt.title("Accuracy History Plot")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/844684027.py in <cell line: 0>()
----> 1 epochs_arr = np.arange(1, len(history.history["loss"]) + 1)
      2 
      3 plt.plot(epochs_arr, history.history["loss"], label="Training")
      4 plt.plot(epochs_arr, history.history["val_loss"], label="Validation")
      5 plt.title("Loss History Plot")

NameError: name 'history' is not defined

## === cell 40
model.save("cnn_model.h5")



## === cell 41
restored_model = load_model("cnn_model.h5")
if os.path.exists("cnn_checkpoint.h5"):
    restored_model.load_weights("cnn_checkpoint.h5")
else:
    print(
        "Warning: cnn_checkpoint.h5 not found; using last-epoch weights in cnn_model.h5"
    )



## === cell 42
restored_model.evaluate(X_valid, y_valid_onehot, verbose=0)



## === cell 43
y_pred = np.argmax(restored_model.predict(X_valid, verbose=0), axis=1)

print("Predict:", y_pred[:10])
print("Validation:", np.array(y_valid[:10]))



## === cell 44
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=y_valid, y_pred=y_pred)

print("Confusion Matrix")
print(cm)
print()
print("Shape :", cm.shape)
print("Accurcy: {0:.2f}%".format(np.trace(cm) / np.sum(cm) * 100))



## === cell 45
from sklearn.metrics import classification_report

print(
    classification_report(
        y_valid,
        y_pred,
        digits=4,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
    )
)



## === cell 46
test_data_folder = os.path.join(base_data_folder, "test_images")
print("Test images folder exists:", os.path.exists(test_data_folder))



## === cell 47
test_df = pd.read_csv(os.path.join(base_data_folder, "test.csv"))
test_df.head()



## === cell 48
X_test = []
test_ids = test_df["id_code"].values
missing_test = 0

for id_code in tqdm(test_ids, desc="Loading test images"):
    fp = os.path.join(test_data_folder, f"{id_code}.png")
    image_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
    if image_bgr is None:
        missing_test += 1
        X_test.append(np.zeros((224, 224, 3), dtype=np.uint8))
        continue

    image_resized = cv2.resize(image_bgr, dsize=(0, 0), fx=0.12, fy=0.12)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    circle_img = circle_crop(image_rgb)
    image_resized2 = cv2.resize(
        circle_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC
    )
    X_test.append(image_resized2)

X_test = np.array(X_test)
print("X_test:", X_test.shape, "Missing test:", missing_test)



## === cell 49
preds = np.argmax(restored_model.predict(X_test, verbose=0), axis=1)
print("Predicted (first 10):", preds[:10])



## === cell 50
submission = pd.DataFrame({"id_code": test_ids, "diagnosis": preds.astype(int)})
submission.head()



## === cell 51
submission = submission.sort_values("id_code").reset_index(drop=True)
submission.head()



## === cell 52
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())



## === cell 53
sample_sub = pd.read_csv(os.path.join(base_data_folder, "sample_submission.csv"))
print("sample_submission shape:", sample_sub.shape)
print(
    "id_code match (as sets):", set(sample_sub["id_code"]) == set(submission["id_code"])
)
print(
    "id_code match (sorted equality):",
    sample_sub.sort_values("id_code")["id_code"].values.tolist()
    == submission["id_code"].values.tolist(),
)
