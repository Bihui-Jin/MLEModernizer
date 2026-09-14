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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
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
protobuf==6.33.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.7038

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import os
os.chdir("../input")


## === cell 1
meta_data = pd.read_csv("train.csv")
meta_data.head()


## === cell 2
train_dir = "train/train"
test_dir =  "test/test"
os.listdir(train_dir)[:5]
print(len(os.listdir(train_dir)))


## === cell 3
import os

import sys, subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_gen = ImageDataGenerator(
    rescale=1 / 255,
    horizontal_flip=True,
    height_shift_range=0.2,
    width_shift_range=0.2,
    brightness_range=[0.2, 1.2],
)
valid_gen = ImageDataGenerator(rescale=1 / 255)

meta_data.has_cactus = meta_data.has_cactus.astype(str)

split_idx = int(0.9 * len(meta_data))
split_idx = max(1, min(split_idx, len(meta_data) - 1))

train_generator = train_gen.flow_from_dataframe(
    dataframe=meta_data[:split_idx],
    target_size=(32, 32),
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
)

valid_generator = valid_gen.flow_from_dataframe(
    dataframe=meta_data[split_idx:],
    target_size=(32, 32),
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
)


## === cell 4
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19

base_model = VGG19(input_shape=(32,32,3),include_top=False, weights="imagenet")


## === cell 5
base_model.summary()


## === cell 6

for layer in base_model.layers:
    layer.trainable = False
    

last_layer = base_model.get_layer("block5_pool")
last_output = last_layer.output


extend = keras.layers.Flatten()(last_output)
extend = keras.layers.Dense(1024, activation="relu")(extend)
extend = keras.layers.Dropout(0.2)(extend)
extend = keras.layers.Dense(512, activation = "relu") (extend)
extend = keras.layers.Dropout(0.2)(extend)
extend = keras.layers.Dense(256, activation = "relu") (extend)
extend = keras.layers.Dropout(0.2)(extend)
extend = keras.layers.Dense(1, activation="sigmoid")(extend)


model = keras.models.Model(base_model.input, extend)


model.compile(loss = "binary_crossentropy",
             optimizer="adam",
             metrics=["acc"])

model.summary()


## === cell 7
model.fit(train_generator, validation_data=valid_generator, verbose=1, epochs=10)


## === cell 8
history = model.history


## === cell 9
acc = history.history["acc"]
loss = history.history["loss"]
val_acc = history.history["val_acc"]
val_loss = history.history["val_loss"]
epochs = range(len(acc))


## === cell 10
import matplotlib.pyplot as plt


plt.plot(epochs, acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.axis([0, 4, 0.7, 1])
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.figure()


## === cell 12
os.listdir("test/test")[:5]


## === cell 13
import cv2
images = []

for image in os.listdir("test/test"):
    images.append( cv2.imread("test/test/" + image))


## === cell 14
import numpy as np

processed = []
for img in images:
    if img is None:
        continue
    if img.shape[:2] != (32, 32):
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    processed.append(img)

image = np.stack(processed, axis=0)


## === cell 15
image.resize(4000, 32, 32, 3)
image.shape


## === cell 16
prediction = model.predict(image)
prediction.resize(4000)


## === cell 17
import numpy as np

test_ids = sorted(os.listdir(test_dir))
preds = np.asarray(prediction).reshape(-1)[: len(test_ids)]

sub = pd.DataFrame({"id": test_ids, "has_cactus": preds})


## === cell 18
sub.head()


## === cell 19
os.listdir("..")


## === cell 20
sub.to_csv("../working/samplesubmission.csv", index=False)
