# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import EfficientNetB7

tqdm.pandas()


## === cell 1
strategy = tf.distribute.get_strategy()


## === cell 2
BASE_PATH = "../input/plant-pathology-2020-fgvc7"
IMAGE_PATH = os.path.join(BASE_PATH, "images")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

sub = pd.read_csv(SUB_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)




## === cell 3
def init_grabcut_mask(h, w):
    mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = cv2.GC_PR_FGD
    mask[2 * h // 5 : 3 * h // 5, 2 * w // 5 : 2 * w // 5] = cv2.GC_FGD
    return mask


def remove_background(image, h=136, w=205):
    orig_image = image
    image = cv2.resize(image, (w, h))
    mask = init_grabcut_mask(h, w)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgm, fgm, 1, cv2.GC_INIT_WITH_MASK)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    h0, w0 = orig_image.shape[:2]
    mask_binary = cv2.resize(mask_binary, (w0, h0))
    result = cv2.bitwise_and(orig_image, orig_image, mask=mask_binary)
    return result




## === cell 4
def rotate(x: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(x)[:-1]
    x = tf.image.rot90(
        x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )
    return tf.image.resize(x, shape)


def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x


def color(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def zoom(x: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(x)[:-1]
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        crops = tf.image.crop_and_resize(
            [img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=shape
        )
        return crops[
            tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)
        ]

    choice = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
    x = tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))
    return x




## === cell 5
def get_data_generators(preprocess=True, augment=True, IMAGE_SIZE=(408, 615), nfolds=5):
    def load_image(image_id):
        file_path = f"{image_id}.jpg"
        img = cv2.imread(os.path.join(IMAGE_PATH, file_path))
        img = cv2.resize(img, IMAGE_SIZE[::-1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if preprocess:
            img = remove_background(img)
        return img

    print("Preprocessing training images...")
    train_images = np.stack(train_data["image_id"].apply(load_image).values)
    plt.imshow(train_images[0])
    labels = train_data[["healthy", "multiple_diseases", "rust", "scab"]]
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, np.stack(labels.values))
    )

    def map_func(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = (tf.cast(label, tf.float32) + 0.01) / 1.04
        return image, label

    dataset = dataset.map(
        map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True
    )

    if augment:
        augmentations = [flip, color, rotate, zoom]
        for f in augmentations:
            dataset = dataset.map(
                lambda x, y: (f(x), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=True,
            )
        dataset = dataset.map(
            lambda x, y: (tf.clip_by_value(x, 0, 1), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=True,
        )

    fold_size = len(train_data) // nfolds
    folds = []
    for idx in range(nfolds):
        fold = dataset.take(fold_size)
        folds.append(fold)
        dataset = dataset.skip(fold_size)

    print("Preprocessing test images...")
    test_images = np.stack(test_data["image_id"].apply(load_image).values)
    test = tf.data.Dataset.from_tensor_slices(test_images)
    test = test.map(
        lambda image: tf.cast(image, tf.float32) / 255.0,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False,
    )
    test = test.batch(32).prefetch(2)

    return folds, test




## === cell 6
folds, test = get_data_generators(preprocess=False, augment=True)




## === cell 7
def get_model():
    inputs = keras.Input(shape=(408, 615, 3))
    x = EfficientNetB7(
        include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
    )(inputs)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(4, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model




## === cell 8
with strategy.scope():
    model = get_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )




## === cell 9
def get_train_val_split(val_fold):
    train_folds = folds[:val_fold] + folds[val_fold + 1 :]
    train = train_folds[0]
    for f in train_folds[1:]:
        train = train.concatenate(f)
    val = folds[val_fold]
    return train, val


splits = [get_train_val_split(i) for i in range(len(folds))]
train, val = splits[0]
for t, v in splits[1:]:
    train = train.concatenate(t)
    val = val.concatenate(v)

train = train.repeat().batch(32).prefetch(2)
val = val.repeat().batch(32).prefetch(2)

history = model.fit(
    train,
    steps_per_epoch=47,
    epochs=50,
    validation_data=val,
    validation_steps=12,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6
        ),
        keras.callbacks.CSVLogger("log.csv", separator=";"),
    ],
)


## === cell 10
test_pr = model.predict(test, verbose=1)
sub.loc[:, "healthy":] = test_pr
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
sub.head()

## --- ERROR in outputing the csv:
Invalid submission: Expected submission to have 183 rows but got 4
