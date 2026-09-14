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
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.0463500884061631

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def intialize_accel(hardware):
    """
    Initialise TPU or GPU strategy.
    """
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    return tf.distribute.get_strategy()


strategy = intialize_accel("TPU")
AUTO = tf.data.experimental.AUTOTUNE




## === cell 2
DIR = "../input/hotel-id-2021-fgvc8/"
TRAIN_IMG_DIR = os.path.join(DIR, "train_images")
TEST_IMG_DIR = os.path.join(DIR, "test_images")

train_df = pd.read_csv(os.path.join(DIR, "train.csv"))
train_df = train_df.drop_duplicates(subset=["image"])
print("Unique chains:", train_df.chain.nunique())
print("Unique hotels:", train_df.hotel_id.nunique())
print("Training samples:", train_df.shape[0])




## === cell 3
le = preprocessing.LabelEncoder()
train_df["label"] = le.fit_transform(train_df["hotel_id"])
Classes = train_df["label"].nunique()
print("Number of classes:", Classes)




## === cell 4
size = (200, 200)
Channels = 3
input_shape = (size[0], size[1], Channels)


def image_process(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    return img, label


def augment(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label




## === cell 5
train_paths = (
    TRAIN_IMG_DIR + "/" + train_df["chain"].astype(str) + "/" + train_df["image"]
).values
train_labels = train_df["label"].values

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.1, random_state=SEED, stratify=train_labels
)

BATCH_SIZE = 64  # larger batch reduces steps per epoch without altering logic

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(image_process, num_parallel_calls=AUTO)
train_ds = train_ds.map(augment, num_parallel_calls=AUTO)
train_ds = train_ds.shuffle(1024, seed=SEED).batch(BATCH_SIZE).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(image_process, num_parallel_calls=AUTO)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/1555474831.py in <cell line: 0>()
      5 train_labels = train_df["label"].values
      6 
----> 7 train_paths, val_paths, train_labels, val_labels = train_test_split(
      8     train_paths, train_labels, test_size=0.1, random_state=SEED, stratify=train_labels
      9 )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2581         cv = CVClass(test_size=n_test, train_size=n_train, random_state=random_state)
   2582 
-> 2583         train, test = next(cv.split(X=arrays[0], y=stratify))
   2584 
   2585     return list(

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
   1687         """
   1688         X, y, groups = indexable(X, y, groups)
-> 1689         for train, test in self._iter_indices(X, y, groups):
   1690             yield train, test
   1691 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _iter_indices(self, X, y, groups)
   2076         class_counts = np.bincount(y_indices)
   2077         if np.min(class_counts) < 2:
-> 2078             raise ValueError(
   2079                 "The least populated class in y has only 1"
   2080                 " member, which is too few. The minimum"

ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.

## === cell 6
def create_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=x
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu", dtype="float32")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(
        x
    )
    return tf.keras.Model(inputs, outputs)




## === cell 7
def compile_model(model, lr=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model




## === cell 8
EPOCHS = 2
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=2, mode="max", verbose=1
)

with strategy.scope():
    model = create_model(input_shape, Classes)
    model = compile_model(model, lr=1e-3)

print("Starting training …")
model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[reduce_lr],
    verbose=1,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2050175367.py in <cell line: 0>()
     10 print("Starting training …")
     11 model.fit(
---> 12     train_ds,
     13     epochs=EPOCHS,
     14     validation_data=val_ds,

NameError: name 'train_ds' is not defined

## === cell 9
test_paths = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
test_names = [os.path.basename(p) for p in test_paths]

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def decode_only_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, size)
    return img


test_ds = test_ds.map(lambda p: tf.io.read_file(p), num_parallel_calls=AUTO)
test_ds = test_ds.map(decode_only_bytes, num_parallel_calls=AUTO)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)

pred_probs = model.predict(test_ds, verbose=1)

top5_idx = np.argsort(-pred_probs, axis=1)[:, :5]  # descending order

flat_idx = top5_idx.ravel()
flat_hotel = le.inverse_transform(flat_idx)
top5_hotel = flat_hotel.reshape(top5_idx.shape)

submission = pd.DataFrame(
    {"image": test_names, "hotel_id": [" ".join(map(str, ids)) for ids in top5_hotel]}
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/446686152.py in <cell line: 0>()
     14 test_ds = test_ds.map(lambda p: tf.io.read_file(p), num_parallel_calls=AUTO)
     15 test_ds = test_ds.map(decode_only_bytes, num_parallel_calls=AUTO)
---> 16 test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)
     17 
     18 pred_probs = model.predict(test_ds, verbose=1)

NameError: name 'BATCH_SIZE' is not defined

## === cell 10
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/701193129.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 submission.to_csv(submission_path, index=False)
      3 print(f"Submission written to {submission_path}")

NameError: name 'submission' is not defined
