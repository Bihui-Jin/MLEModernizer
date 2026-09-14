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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.2577

# 6. Current score

0.18759

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.41031) has done: 'Diagnosis: The crash happens because `model.predict_classes()` was removed from modern `tf.keras`/Keras APIs, so `Sequential` no longer provides that method. The correct replacement is to call `model.predict()` to get class probabilities/logits and then take `argmax` over the class axis to get class indices. This preserves the same inference semantics (highest-probability class) that `predict_classes()` used.  

Patch summary: In cell 22, replace the removed `predict_classes` call with `np.argmax(model.predict(...), axis=1)` and keep the output as an integer label vector compatible with cell 23.  

Updated cells: Only cell 22 is modified.  

Compatibility notes for cell k+1: `poll` remains a 1D NumPy array of predicted class integers, so `freeman['label'] = poll` works unchanged.  

Assumptions: The model outputs a `(N, 5)` softmax probability array for `Xt`, so `argmax(axis=1)` matches the original intended behavior.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.41031) is better than the target (0.2577) for a higher-is-better metric, so to move closer to the target we should deliberately (but legitimately) reduce performance with the smallest, safest change. The most minimal way is to reduce training signal by training for fewer epochs while keeping the exact same model, loss, data pipeline, and submission semantics. I also add fixed random seeds to make the resulting score stable/reproducible around the new level rather than drifting between runs. The submission file path/name and required columns stay unchanged.'
- What this solution (achieved 0.05568) has done: 'Your current score (0.05531) is far below the target (0.2577), so we should make a minimal, legitimate change that increases accuracy without changing the core CNN or loss. The biggest issue is that the cassava dataset is highly class-imbalanced, and your current training treats all classes equally; adding `class_weight` to `model.fit()` keeps the same architecture/training loop but gives minority classes more learning signal, typically lifting accuracy substantially. I also add a simple internal validation accuracy printout after training (using the same `validation_split`) so you can quickly see whether the change moved you upward before submitting. The submission formatting and prediction semantics remain unchanged.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.05568) is far below the target (0.2577), so we should make a small, legitimate change that improves generalization without changing the CNN architecture or loss. The biggest issue is the current training set-up: it loads all images into RAM and uses `validation_split` on already-shuffled-in-memory arrays, which can be unstable and can also lead to a weak training signal with the large batch size. I keep the same model and training loop semantics, but (1) add a stratified train/validation split (same 15%) and (2) train on a shuffled training subset while validating on a held-out subset, which typically improves accuracy reliably. The submission generation remains identical (argmax of softmax) and still writes a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'We need to raise accuracy from 0.11584 toward 0.2577 (higher-is-better), and we’re far below target, so we should make a small, legitimate improvement without changing the CNN architecture, loss, or overall training loop. The most impactful minimal fix here is correcting the input pipeline: your images are being loaded with `target_size=(H, W, 3)` which is not the intended API (it should be `(H, W)`), and that can silently distort loading/resize behavior and hurt training/inference consistency. I change `load_img(..., target_size=(img_height, img_width))` in both train and test loading so the model sees properly resized RGB images as intended, keeping everything else the same. This should improve generalization and move the score upward toward the target while preserving core logic and producing the same `submission.csv` format.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) is below the target (0.2577) for a higher-is-better metric, so we should make a small, legitimate change that improves accuracy without changing the CNN architecture, loss, or overall training loop. The most impactful minimal fix here is to correct the training label type: currently `y` is a pandas Series with dtype `object`/`int64` inconsistencies across steps, and we also mix Series/NumPy in ways that can silently affect stratification and class weights. I make `y` a clean `np.int32` array once, reuse it consistently for splitting and class weights, and ensure `class_weight` includes all 5 classes in a stable way. This keeps the same model, epochs, batch size, and inference (`argmax`), but typically improves training stability and moves accuracy upward toward the target.'
- What this solution (achieved 0.61323) has done: 'We need to increase accuracy from 0.11584 toward 0.2577 (higher-is-better), and we’re still far below target, so we make the smallest legitimate training-quality improvements without changing the CNN architecture, loss, or inference semantics. The biggest issue is the very large batch size (1000) for Adam on this small CNN, which commonly harms convergence/generalization; reducing it to a still-reasonable size (128) keeps the same training loop and epochs but usually improves accuracy. To avoid any accidental train/test preprocessing mismatch, we also force RGB mode in both train and test image loading (same resizing/normalization as before). The submission format and argmax prediction remain identical and still write `submission.csv`.'
- What this solution (achieved 0.11099) has done: 'Your current score (0.61323) is well above the target (0.2577) for a higher-is-better metric, so we should deliberately and safely reduce performance with the smallest change that keeps the same model, loss, and inference semantics. The most minimal lever is to reduce training signal by training for fewer epochs while leaving the architecture, optimizer, batch size, class weighting, and preprocessing unchanged. To keep the resulting score stable (so it doesn’t swing unpredictably), we preserve the deterministic seed/determinism settings already in your code. The submission generation stays identical (argmax over predicted probabilities) and still writes a valid `submission.csv`.'
- What this solution (achieved 0.18759) has done: 'Your current score (0.11099) is below the target (0.2577) for a higher-is-better metric, so we should make a small legitimate accuracy improvement without changing the CNN architecture, loss, or prediction semantics. The most minimal high-impact fix is to train for a bit longer: your script currently trains for only 1 epoch, which is typically underfitting for Cassava even with class weights. I increase epochs to 3 and update the plotting cell accordingly, keeping the same optimizer, batch size, class_weight, split, preprocessing, and argmax submission logic. This should move accuracy upward toward the target band while keeping runtime reasonable and producing the same valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
)
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import keras
from tqdm import tqdm

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 1
pass



## === cell 2
df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
samplesub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)



## === cell 3
df.head()



## === cell 4
samplesub.head()



## === cell 5
img_height = 100
img_width = 100

X = []

for i in tqdm(range(df.shape[0])):
    path = (
        "/kaggle/input/cassava-leaf-disease-classification/train_images/"
        + df["image_id"][i]
    )
    img = image.load_img(path, target_size=(img_height, img_width), color_mode="rgb")
    img = image.img_to_array(img)
    img = img / 255.0
    X.append(img)

X = np.array(X)



## === cell 6
pass



## === cell 7
pass



## === cell 8
X.shape



## === cell 9
df.label.unique()



## === cell 10
y = df["label"].to_numpy(dtype=np.int32)



## === cell 11
pass



## === cell 12
model = Sequential()
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        input_shape=X[0].shape,
    )
)
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=5, activation="softmax"))

model.summary()



## === cell 13
model.compile(
    optimizer="Adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)



## === cell 14
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y,
)

n_classes = 5
counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
total = float(len(y_train))
class_weight = {}
for c in range(n_classes):
    if counts[c] > 0:
        class_weight[c] = total / (n_classes * counts[c])
    else:
        class_weight[c] = 1.0  # safe fallback; shouldn't happen with stratify

EPOCHS = 3

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=EPOCHS,
    batch_size=128,
    verbose=1,
    validation_data=(X_val, y_val),
    class_weight=class_weight,
    shuffle=True,
)



## === cell 15
pass



## === cell 16
import matplotlib.pyplot as plt



## === cell 17
epoch_range = range(1, EPOCHS + 1)
plt.plot(epoch_range, history.history["sparse_categorical_accuracy"])
plt.plot(epoch_range, history.history["val_sparse_categorical_accuracy"])
plt.title("Model_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "val"], loc="upper left")
plt.show()

plt.plot(epoch_range, history.history["loss"])
plt.plot(epoch_range, history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "val"], loc="upper left")
plt.show()



## === cell 18
pass



## === cell 19
pass



## === cell 20
img_height = 100
img_width = 100

Xt = []

for i in tqdm(range(samplesub.shape[0])):
    path = (
        "/kaggle/input/cassava-leaf-disease-classification/test_images/"
        + samplesub["image_id"][i]
    )
    img = image.load_img(path, target_size=(img_height, img_width), color_mode="rgb")
    img = image.img_to_array(img)
    img = img / 255.0
    Xt.append(img)

Xt = np.array(Xt)



## === cell 21
poll = np.argmax(model.predict(Xt, verbose=0), axis=1)



## === cell 22
freeman = samplesub.copy()
freeman["label"] = poll.astype(int)

freeman.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", freeman.shape)
print(freeman.head())
