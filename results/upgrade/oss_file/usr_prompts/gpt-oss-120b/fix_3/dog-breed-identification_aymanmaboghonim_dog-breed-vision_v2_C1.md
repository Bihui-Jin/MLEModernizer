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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.88512

# 6. Current score

5.1415

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.1415) has done: 'I replace the TensorFlow‑Hub MobileNet layer with the built‑in Keras MobileNetV2 model, fix the stratified split to use class indices instead of one‑hot vectors, and adjust the imports accordingly. These changes resolve the import error, the train‑test split error, and the model construction error, allowing the pipeline to run end‑to‑end and produce a valid .csv submission while keeping the original workflow intact.'

# 9. Code solution

## === cell 0
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Sequential

from sklearn.model_selection import train_test_split



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print("TF version:", tf.__version__)
print(
    "GPU",
    (
        "available (YESS!!!!)"
        if tf.config.list_physical_devices("GPU")
        else "not available :("
    ),
)



## === cell 2
labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")



## === cell 3
labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10))
plt.title("Breed distribution in training set")
plt.show()



## === cell 4
train_path = "../input/dog-breed-identification/train/"
filenames = [os.path.join(train_path, fname + ".jpg") for fname in labels_csv["id"]]
print(f"Found {len(filenames)} training images.")



## === cell 5
if len(os.listdir(train_path)) == len(filenames):
    print("Filenames match actual amount of files!")
else:
    print("Filenames do NOT match actual amount of files, check the target directory.")



## === cell 6
unique_breeds = np.unique(labels_csv["breed"])
breed_to_index = {b: i for i, b in enumerate(unique_breeds)}
y_indices = labels_csv["breed"].map(breed_to_index).values
y_onehot = to_categorical(y_indices, num_classes=len(unique_breeds))

print(f"Number of unique breeds: {len(unique_breeds)}")
print(f"One‑hot label shape: {y_onehot.shape}")



## === cell 7
NUM_IMAGES = 1000
X_subset = np.array(filenames[:NUM_IMAGES])
y_subset_onehot = y_onehot[:NUM_IMAGES]
y_subset_indices = y_indices[:NUM_IMAGES]

X_train, X_val, y_train, y_val = train_test_split(
    X_subset,
    y_subset_onehot,
    test_size=0.2,
    random_state=42,
    stratify=y_subset_indices,  # use class indices for stratification
)

print(f"Train/validation split: {len(X_train)} / {len(X_val)}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2093109848.py in <cell line: 0>()
      4 y_subset_indices = y_indices[:NUM_IMAGES]
      5 
----> 6 X_train, X_val, y_train, y_val = train_test_split(
      7     X_subset,
      8     y_subset_onehot,

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

## === cell 8
IMG_SIZE = 224


def process_image(image_path):
    """Read an image file, decode, resize and normalize to [0,1]."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image


def get_image_label(image_path, label):
    """Return a tuple (image_tensor, one‑hot_label)."""
    image = process_image(image_path)
    return image, label




## === cell 9
BATCH_SIZE = 32


def create_data_batches(
    x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False
):
    """Create tf.data.Dataset batches for training, validation or test."""
    if test_data:
        data = tf.data.Dataset.from_tensor_slices(tf.constant(x))
        data = data.map(process_image).batch(batch_size)
        return data
    elif valid_data:
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(x), tf.constant(y, dtype=tf.float32))
        )
        data = data.map(get_image_label).batch(batch_size)
        return data
    else:
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(x), tf.constant(y, dtype=tf.float32))
        )
        data = data.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)
        data = data.map(get_image_label).batch(batch_size)
        return data


train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1244041396.py in <cell line: 0>()
     25 
     26 
---> 27 train_data = create_data_batches(X_train, y_train)
     28 val_data = create_data_batches(X_val, y_val, valid_data=True)
     29 

NameError: name 'X_train' is not defined

## === cell 10
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
OUTPUT_SHAPE = len(unique_breeds)


def create_model():
    print("Building model with Keras MobileNetV2 backbone...")
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=INPUT_SHAPE
    )
    base_model.trainable = False  # freeze backbone
    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(OUTPUT_SHAPE, activation="softmax", name="predictions"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


model = create_model()
model.summary()




## === cell 11
def create_tensorboard_callback():
    logdir = os.path.join("./logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(log_dir=logdir)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)



## === cell 12
NUM_EPOCHS = 5  # keep small for quick execution


def train_model():
    tb_cb = create_tensorboard_callback()
    model.fit(
        train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
        callbacks=[tb_cb, early_stopping],
        verbose=2,
    )
    return model


model = train_model()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3848097035.py in <cell line: 0>()
     14 
     15 
---> 16 model = train_model()
     17 

/tmp/ipykernel_11/3848097035.py in train_model()
      5     tb_cb = create_tensorboard_callback()
      6     model.fit(
----> 7         train_data,
      8         epochs=NUM_EPOCHS,
      9         validation_data=val_data,

NameError: name 'train_data' is not defined

## === cell 13
sample_sub_path = "../input/dog-breed-identification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
breed_columns = list(sample_sub.columns)[1:]  # exclude the 'id' column



## === cell 14
test_path = "../input/dog-breed-identification/test/"
test_filenames = [
    os.path.join(test_path, fname)
    for fname in os.listdir(test_path)
    if fname.lower().endswith(".jpg")
]
print(f"Found {len(test_filenames)} test images.")

test_data = create_data_batches(test_filenames, test_data=True)



## === cell 15
test_predictions = model.predict(test_data, verbose=1)
print(f"Test predictions shape: {test_predictions.shape}")



## === cell 16
preds_df = pd.DataFrame(test_predictions, columns=unique_breeds)
preds_df = preds_df[breed_columns]  # re‑order columns to match submission format
preds_df.insert(
    0,
    "id",
    [os.path.splitext(os.path.basename(p))[0] for p in test_filenames],
)

print("Submission head:")
print(preds_df.head())



## === cell 17
submission_path = "./full_submission_mobilenetv2.csv"
preds_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
