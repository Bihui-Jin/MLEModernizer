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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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
import tensorflow as tf



## === cell 1
import os, random, math, re, multiprocessing
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0, DenseNet201
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

num_cpus = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(num_cpus)
tf.config.threading.set_intra_op_parallelism_threads(num_cpus)

tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)



## === cell 2
strategy = tf.distribute.get_strategy()
print("REPLICAS:", strategy.num_replicas_in_sync)

BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGE_DIR = os.path.join(BASE_PATH, "images")

EPOCHS = 5  # modest number to keep runtime low
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE



## === cell 3
img_size = 224  # smaller size for faster training


def format_path(st):
    return os.path.join(IMAGE_DIR, f"{st}.jpg")




## === cell 4
train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
sub_df = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

train_paths = train_df["image_id"].apply(format_path).values
test_paths = test_df["image_id"].apply(format_path).values
train_labels = train_df.loc[:, "healthy":].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2020, stratify=train_labels
)


def load_images(paths):
    imgs = []
    for p in paths:
        bits = tf.io.read_file(p)
        img = tf.image.decode_jpeg(bits, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (img_size, img_size))
        imgs.append(img.numpy())
    return np.stack(imgs)


train_images = load_images(train_paths)
valid_images = load_images(valid_paths)
test_images = load_images(test_paths)




## === cell 5
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(bits, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
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




## === cell 6
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .map(data_augment, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_images).batch(BATCH_SIZE).prefetch(AUTO)
)




## === cell 7
def get_model(backbone):
    base = backbone(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    outputs = Dense(train_labels.shape[1], activation="softmax")(base.output)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model




## === cell 8
with strategy.scope():
    model1 = get_model(EfficientNetB0)
    model2 = get_model(DenseNet201)

model1.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=1,
)

model2.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 9
best_alpha = 0.52
print("Generating predictions...")
prob1 = model1.predict(test_dataset, verbose=1)
prob2 = model2.predict(test_dataset, verbose=1)
probabilities = best_alpha * prob1 + (1 - best_alpha) * prob2

sub_df.loc[:, "healthy":] = probabilities
sub_df.to_csv("submission.csv", index=False)
print("Saved submission.csv")
sub_df.head()
