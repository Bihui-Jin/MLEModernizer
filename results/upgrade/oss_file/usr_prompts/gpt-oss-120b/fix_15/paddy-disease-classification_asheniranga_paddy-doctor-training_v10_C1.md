# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import cv2
import albumentations as A
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

tf.config.optimizer.set_jit(True)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU memory growth error:", e)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

train_meta_data = "/kaggle/input/paddy-disease-classification/train.csv"
train_data_dir = "/kaggle/input/paddy-disease-classification/train_images"
test_data_dir = "/kaggle/input/paddy-disease-classification/test_images"

train_df = pd.read_csv(train_meta_data)
class_names = sorted(train_df["label"].unique())
class_indices = {name: idx for idx, name in enumerate(class_names)}

epochs = 15  # was 30
lr = 1e-4
valid_split = 0.2
input_size = 224  # unchanged
batch_size = 64  # was 32, larger batch reduces steps/epoch
classes = len(class_names)
optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
loss = tf.keras.losses.CategoricalCrossentropy()
initializer = tf.keras.initializers.HeUniform()




## === cell 1
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor="val_loss", restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=5, monitor="val_loss", factor=0.5, verbose=1
)




## === cell 2
def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x, anchors_y = [], []
        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_x:
                anchors_x.append(rv)
        for _ in range(patches):
            rv = np.random.randint(0, image.shape[1])
            if rv not in anchors_y:
                anchors_y.append(rv)
        for x, y in zip(anchors_x, anchors_y):
            image[x : x + patch_size, y : y + patch_size, :] = 0
    return image


def random_gaus_blur(image):
    if random.choice([True, False]):
        return cv2.GaussianBlur(image, (7, 7), 0)
    return image


def random_displacement(image):
    if random.choice([True, False]):
        ax = random.choice([0, 1])
        slices = np.split(image, 8, axis=ax)
        np.random.shuffle(slices)
        return np.row_stack(slices) if ax == 0 else np.column_stack(slices)
    return image


def center_crop_and_random_augmentations_fn(image):
    """
    Operates on a NumPy array (as provided by tf.numpy_function).
    Performs a safe random crop/rescale to `input_size` and several
    lightweight augmentations.
    """
    h, w, _ = image.shape
    if h > input_size and w > input_size:
        top = np.random.randint(0, h - input_size + 1)
        left = np.random.randint(0, w - input_size + 1)
        image = image[top : top + input_size, left : left + input_size, :]

    image = random_cutout(image, 8, 16)
    image = random_displacement(image)
    image = random_gaus_blur(image)

    image = tf.image.random_brightness(image, 0.2).numpy()
    image = tf.image.random_contrast(image, 0.5, 2.0).numpy()
    image = tf.image.random_saturation(image, 0.75, 1.25).numpy()
    image = tf.image.random_hue(image, 0.1).numpy()
    return image.astype(np.float32)  # ensure float32 for downstream pipeline


def test_time_augmentation_fn(image):
    """
    Light TTA that works on NumPy arrays.
    """
    h, w, _ = image.shape
    if h > input_size and w > input_size:
        top = np.random.randint(0, h - input_size + 1)
        left = np.random.randint(0, w - input_size + 1)
        image = image[top : top + input_size, left : left + input_size, :]

    image = tf.image.random_brightness(image, 0.2).numpy()
    image = tf.image.random_contrast(image, 0.5, 2.0).numpy()
    return image.astype(np.float32)


def cv2_resize(img, size):
    return cv2.resize(img, size)




## === cell 3
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=valid_split,
    subset="training",
    seed=42,
    image_size=(input_size, input_size),
    batch_size=batch_size,
    label_mode="categorical",
    class_names=class_names,
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=valid_split,
    subset="validation",
    seed=42,
    image_size=(input_size, input_size),
    batch_size=batch_size,
    label_mode="categorical",
    class_names=class_names,
)


def _augment_batch(image_batch, label_batch):
    """
    Apply the same per‑image augmentation used before, but now on a whole
    batch to minimise Python‑level overhead.
    """

    def augment_batch(np_imgs):
        aug_imgs = [center_crop_and_random_augmentations_fn(img) for img in np_imgs]
        return np.stack(aug_imgs).astype(np.float32)

    aug = tf.numpy_function(
        func=augment_batch,
        inp=[image_batch],
        Tout=tf.float32,
    )
    aug.set_shape([None, input_size, input_size, 3])
    aug = aug / 255.0
    return aug, label_batch


def _augment_batch_no_label(image_batch):
    def augment_batch(np_imgs):
        aug_imgs = [center_crop_and_random_augmentations_fn(img) for img in np_imgs]
        return np.stack(aug_imgs).astype(np.float32)

    aug = tf.numpy_function(
        func=augment_batch,
        inp=[image_batch],
        Tout=tf.float32,
    )
    aug.set_shape([None, input_size, input_size, 3])
    aug = aug / 255.0
    return aug


train_ds = raw_train_ds.map(
    _augment_batch, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_ds = raw_val_ds.map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)




## === cell 4
train_batch = next(iter(train_ds))[0]
val_batch = next(iter(val_ds))[0]
print(
    "Train batch shape:", train_batch.shape, "Validation batch shape:", val_batch.shape
)




## === cell 5
base_model = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(input_size, input_size, 3),
    pooling="avg",
)

model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.Dense(classes, activation="softmax"),
    ]
)




## === cell 6
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])




## === cell 7
model.summary()




## === cell 8
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)




## === cell 9
tta_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=valid_split,
    subset="validation",
    seed=42,
    image_size=(input_size, input_size),
    batch_size=batch_size,
    label_mode=None,
)

tta_dataset = tta_dataset.map(
    _augment_batch_no_label, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)




## === cell 10
eve_encodings = np.zeros((len(raw_val_ds.file_paths), classes))
for _ in range(3):
    enc = model.predict(tta_dataset, verbose=0)
    eve_encodings += enc
eve_encodings /= 3




## === cell 11
pred_classes = np.argmax(eve_encodings, axis=1)
true_classes = np.array(
    [class_indices[os.path.basename(os.path.dirname(p))] for p in raw_val_ds.file_paths]
)
print("Validation accuracy:", accuracy_score(true_classes, pred_classes))




## === cell 12
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["accuracy"])),
    y=history.history["accuracy"],
    label="train",
)
sns.lineplot(
    x=range(len(history.history["val_accuracy"])),
    y=history.history["val_accuracy"],
    label="validation",
)
plt.show()




## === cell 13
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["loss"])), y=history.history["loss"], label="train"
)
sns.lineplot(
    x=range(len(history.history["val_loss"])),
    y=history.history["val_loss"],
    label="validation",
)
plt.show()




## === cell 14
pd.DataFrame(history.history).to_csv("model_effnetb4_history.csv", index=False)




## === cell 15
model.save("model_effnet_b4.hdf5")
model.save_weights("model_effnet_b4.weights.h5")




## === cell 16
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    labels=None,
    image_size=(input_size, input_size),
    batch_size=batch_size,
    shuffle=False,
)
test_file_paths = test_dataset.file_paths




## === cell 17
def normalize_batch(batch):
    batch = tf.cast(batch, tf.float32) / 255.0
    return batch


test_dataset = test_dataset.map(lambda x: normalize_batch(x))




## === cell 18
test_encodings = np.zeros((len(test_file_paths), classes))
for _ in range(3):
    enc = model.predict(test_dataset, verbose=0)
    test_encodings += enc
test_encodings /= 3




## === cell 19
predict_max = np.argmax(test_encodings, axis=1)




## === cell 20
inverse_map = {v: k for k, v in class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]




## === cell 21
files = [os.path.basename(p) for p in test_file_paths]
submission = pd.DataFrame({"image_id": files, "label": predictions})
submission.to_csv("model_submission_v17.csv", index=False)
print(submission.head())
