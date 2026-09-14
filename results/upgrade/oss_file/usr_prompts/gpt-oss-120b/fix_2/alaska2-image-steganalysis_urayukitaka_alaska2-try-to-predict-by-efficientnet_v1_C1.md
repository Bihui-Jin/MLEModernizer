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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.8

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

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.574

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as l
from tensorflow.keras import applications as apps



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except Exception:  # fallback to default strategy
    strategy = tf.distribute.get_strategy()



## === cell 2
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = (512, 512)
BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"



## === cell 3
categories = ["Test", "JUNIWARD", "JMiPOD", "Cover", "UERD"]
rows = []
for cat in categories:
    dir_path = os.path.join(BASE_PATH, cat)
    for fname in os.listdir(dir_path):
        rows.append({"cate": cat, "name": fname, "path": os.path.join(dir_path, fname)})
df = pd.DataFrame(rows)




## === cell 4
def cate_label(row):
    return 0 if row["cate"] == "Cover" else 1


Test_df = df[df["cate"] == "Test"].copy().reset_index(drop=True)
Train_df = df[df["cate"] != "Test"].copy().reset_index(drop=True)
Train_df["flg"] = Train_df.apply(cate_label, axis=1)



## === cell 5
Train_df = Train_df.sample(1000, random_state=42).reset_index(drop=True)

X = Train_df["path"].values
y = Train_df["flg"].values
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2539759468.py in <cell line: 0>()
      4 X = Train_df["path"].values
      5 y = Train_df["flg"].values
----> 6 X_train, X_val, y_train, y_val = train_test_split(
      7     X, y, test_size=0.2, random_state=10, stratify=y
      8 )

NameError: name 'train_test_split' is not defined

## === cell 6
X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3284567222.py in <cell line: 0>()
      1 # Convert to NumPy arrays (already numpy, but keep for clarity)
----> 2 X_train = np.array(X_train)
      3 X_val = np.array(X_val)
      4 y_train = np.array(y_train)
      5 y_val = np.array(y_val)

NameError: name 'X_train' is not defined

## === cell 7
AUTOTUNE = tf.data.experimental.AUTOTUNE


def decode_image(filename, label=None, image_size=IMAGE_SIZE):
    bits = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(bits, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, image_size)
    if label is None:
        return img
    return img, label


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:
        return image
    return image, label




## === cell 8
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(decode_image, num_parallel_calls=AUTOTUNE)
    .map(data_augment, num_parallel_calls=AUTOTUNE)
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(decode_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_paths = Test_df["path"].values
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda x: decode_image(x, label=None), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3828025891.py in <cell line: 0>()
      1 train_ds = (
----> 2     tf.data.Dataset.from_tensor_slices((X_train, y_train))
      3     .map(decode_image, num_parallel_calls=AUTOTUNE)
      4     .map(data_augment, num_parallel_calls=AUTOTUNE)
      5     .shuffle(1024)

NameError: name 'X_train' is not defined

## === cell 9
with strategy.scope():
    model = tf.keras.Sequential(
        [
            apps.EfficientNetB3(
                input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                weights="imagenet",
                include_top=False,
            ),
            l.GlobalAveragePooling2D(),
            l.Dense(1, activation="sigmoid"),
        ]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])



## === cell 10
steps_per_epoch = len(X_train) // BATCH_SIZE
EPOCHS = 5  # modest number for the sampled data
history = model.fit(
    train_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=val_ds
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1413565228.py in <cell line: 0>()
----> 1 steps_per_epoch = len(X_train) // BATCH_SIZE
      2 EPOCHS = 5  # modest number for the sampled data
      3 history = model.fit(
      4     train_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=val_ds
      5 )

NameError: name 'X_train' is not defined

## === cell 11
preds = model.predict(test_ds, verbose=1).reshape(-1)

sample_sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
sample_sub["Label"] = preds
sample_sub.to_csv("submission.csv", index=False)

print("Submission saved to submission.csv")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/11531706.py in <cell line: 0>()
      1 # Predict on the test set
----> 2 preds = model.predict(test_ds, verbose=1).reshape(-1)
      3 
      4 # Load the sample submission file and write predictions
      5 sample_sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

NameError: name 'test_ds' is not defined
