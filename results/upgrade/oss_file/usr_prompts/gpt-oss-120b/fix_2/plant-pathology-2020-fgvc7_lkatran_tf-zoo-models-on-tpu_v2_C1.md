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

## === cell 1
import math, os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



## === cell 2
strategy = tf.distribute.get_strategy()
print("REPLICAS:", strategy.num_replicas_in_sync)

DATA_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
AUTO = tf.data.AUTOTUNE

EPOCHS = 15
BATCH_SIZE = 8 * strategy.num_replicas_in_sync




## === cell 3
def format_path(st):
    return os.path.join(DATA_PATH, "images", f"{st}.jpg")




## === cell 4
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

train_paths = train["image_id"].apply(format_path).values
test_paths = test["image_id"].apply(format_path).values

train_labels = train.loc[:, "healthy":].values.astype(np.float32)



## === cell 5
f, ax = plt.subplots(3, 6, figsize=(18, 7))
ax = ax.flatten()
for i in range(18):
    img_path = os.path.join(DATA_PATH, "images", f"Train_{i}.jpg")
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        ax[i].imshow(img)
        ax[i].axis("off")
plt.close(f)



## === cell 6
img_size = 224  # EfficientNetB0 works well with 224×224
nb_classes = 4


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    return (image, label) if label is not None else image


def data_augment(image, label=None, seed=42):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    return (image, label) if label is not None else image




## === cell 7
train_paths_split, valid_paths_split, train_labels_split, valid_labels_split = (
    train_test_split(
        train_paths, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
)

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths_split, train_labels_split))
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .map(data_augment, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths_split, valid_labels_split))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)



## === cell 8
LR_START = 1e-5
LR_MAX = 1e-4 * strategy.num_replicas_in_sync
LR_MIN = 1e-5
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 2
LR_EXP_DECAY = 0.8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * (
            LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)
        ) + LR_MIN
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

rng = list(range(EPOCHS))
y = [lrfn(x) for x in rng]
plt.figure()
plt.plot(rng, y)
plt.title("Learning rate schedule")
plt.xlabel("epoch")
plt.ylabel("lr")
plt.close()




## === cell 9
def get_model():
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output
    output = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=output)


with strategy.scope():
    model = get_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
model.summary()



## === cell 10
history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_paths_split) // BATCH_SIZE,
    validation_data=valid_dataset,
    validation_steps=len(valid_paths_split) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[lr_callback],
    verbose=2,
)




## === cell 11
def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:
        plt.figure(figsize=(10, 5))
    ax = plt.subplot(subplot)
    ax.plot(training, label="train")
    ax.plot(validation, label="valid")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(title)
    ax.legend()


display_training_curves(
    history.history["loss"], history.history["val_loss"], "Loss", 211
)
display_training_curves(
    history.history["categorical_accuracy"],
    history.history["val_categorical_accuracy"],
    "Accuracy",
    212,
)
plt.close("all")



## === cell 12
model_path = "efficientnet_b0.h5"
model.save(model_path)



## === cell 13
probs = model.predict(test_dataset, verbose=0)
sub.loc[:, "healthy":] = probs
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
