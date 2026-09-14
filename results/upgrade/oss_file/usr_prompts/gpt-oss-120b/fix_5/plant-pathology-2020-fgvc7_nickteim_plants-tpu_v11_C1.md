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
import numpy as np, pandas as pd, os, random, re, math
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, mixed_precision
from tensorflow.keras.applications import EfficientNetB7
from matplotlib import pyplot as plt

tf.random.set_seed(2020)
np.random.seed(2020)
random.seed(2020)

try:
    mixed_precision.set_global_policy("mixed_float16")
except Exception as e:
    print("Mixed precision not set:", e)

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__)




## === cell 1
BASE_PATH = os.getenv("KAGGLE_INPUT", "../input")
BASE_PATH = os.path.join(BASE_PATH, "plant-pathology-2020-fgvc7")

train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

IMG_DIR = os.path.join(BASE_PATH, "images")
train_paths = (
    train["image_id"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg")).values
)
test_paths = test["image_id"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg")).values

train_labels = train.loc[:, "healthy":].values.astype(np.float32)

np.random.shuffle(train_paths)  # shuffle in‑place for reproducibility
val_frac = 0.1
val_size = int(len(train_paths) * val_frac)
train_paths_split = train_paths[val_size:]
val_paths_split = train_paths[:val_size]
train_labels_split = train_labels[val_size:]
val_labels_split = train_labels[:val_size]




## === cell 2
strategy = tf.distribute.get_strategy()  # works on CPU/GPU and TPU

nb_classes = 4
BATCH_SIZE = 8 * strategy.num_replicas_in_sync  # 8 on TPU, 1 on CPU/GPU
img_size = 768
EPOCHS = 40
AUTO = tf.data.AUTOTUNE




## === cell 3
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    return image, label


def data_augment(image, label=None, seed=2020):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    image = tf.image.adjust_saturation(image, 2.0)  # moderate saturation
    image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)
    if label is None:
        return image
    return image, label


train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths_split, train_labels_split))
    .map(decode_image, num_parallel_calls=AUTO)
    .cache("/tmp/train_cache")
    .map(data_augment, num_parallel_calls=AUTO)
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((val_paths_split, val_labels_split))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda f: decode_image(f, None), num_parallel_calls=AUTO)
    .cache("/tmp/test_cache")
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




## === cell 4
LR_START = 1e-5
LR_MAX = 1e-4 * strategy.num_replicas_in_sync
LR_MIN = 1e-5
LR_RAMPUP_EPOCHS = 15
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = 0.8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        ) + LR_MIN
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)




## === cell 5
def get_model():
    base = EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = True
    model = models.Sequential(
        [base, layers.Flatten(), layers.Dense(nb_classes, activation="sigmoid")]
    )
    return model




## === cell 6
with strategy.scope():
    model = get_model()
model.compile(
    optimizer=optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)




## === cell 7
history = model.fit(
    train_dataset,
    steps_per_epoch=int(np.ceil(train_labels_split.shape[0] / BATCH_SIZE)),
    validation_data=val_dataset,
    validation_steps=int(np.ceil(val_labels_split.shape[0] / BATCH_SIZE)),
    epochs=EPOCHS,
    callbacks=[lr_callback],
    verbose=2,
)




## === cell 8
probs = model.predict(test_dataset, verbose=0)




## === cell 9
sub.loc[:, "healthy":] = probs
sub.to_csv("submission.csv", index=False)
print("Saved submission to submission.csv")
print(sub.head())
