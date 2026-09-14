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

No external packages required in the script and installed.

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
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTHONHASHSEED", "2021")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    _CPU = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _CPU))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
DATA_DIR = "../input/paddy-disease-classification/train_images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2

train_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_raw.class_names
num_classes = len(class_names)
class_indices = {name: i for i, name in enumerate(class_names)}
inv_class = {i: name for name, i in class_indices.items()}

print("num_classes:", num_classes)
print("class_indices:", class_indices)



## === cell 2
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.0139, fill_mode="reflect", seed=SEED),  # ~5 degrees
        layers.RandomZoom(
            height_factor=0.3, width_factor=0.3, fill_mode="reflect", seed=SEED
        ),
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=SEED),
    ],
    name="data_augmentation",
)

rescale = layers.Rescaling(1.0 / 255.0, name="rescale")


@tf.function
def _decode_and_rescale(images, labels):
    images = tf.cast(images, tf.float32)
    images = rescale(images)
    return images, labels


@tf.function
def _augment(images, labels):
    images = data_augmentation(images, training=True)
    return images, labels


train_cache_path = os.path.join(".", "tfds_train_cache")
val_cache_path = os.path.join(".", "tfds_val_cache")

train_ds = (
    train_raw.map(_decode_and_rescale, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache(train_cache_path)
    .map(_augment, num_parallel_calls=AUTOTUNE, deterministic=True)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_raw.map(_decode_and_rescale, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache(val_cache_path)
    .prefetch(AUTOTUNE)
)

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True

train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)



## === cell 3
pass



## === cell 4
model = tf.keras.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())

model.add(layers.Dense(2048, activation="relu"))
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()



## === cell 5
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
    steps_per_execution=16,
)



## === cell 6
lr_scheduler = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.8, patience=10, verbose=1
)

save_best = tf.keras.callbacks.ModelCheckpoint(
    "Model.weights.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)



## === cell 7
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[save_best, lr_scheduler],
    verbose=1,
)



## === cell 8
if os.path.exists("Model.weights.h5"):
    model.load_weights("Model.weights.h5")




## === cell 9
model.evaluate(val_ds, verbose=1)



## === cell 10
test_path = "../input/paddy-disease-classification/test_images"

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    test_path,
    labels=None,
    label_mode=None,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_files = list(test_ds_raw.file_paths)


@tf.function
def _prep_test(images):
    images = tf.cast(images, tf.float32)
    images = rescale(images)
    return images


test_ds = test_ds_raw.map(
    _prep_test, num_parallel_calls=AUTOTUNE, deterministic=True
).prefetch(AUTOTUNE)



## === cell 11
predict = model.predict(test_ds, verbose=1)



## === cell 12
predicted_class_indices = np.argmax(predict, axis=1)
predictions = [inv_class[int(k)] for k in predicted_class_indices]



## === cell 13
image_ids = [os.path.basename(f) for f in test_files]

results = pd.DataFrame({"image_id": image_ids, "label": predictions})

sample_path = "../input/paddy-disease-classification/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    results = sample[["image_id"]].merge(results, on="image_id", how="left")
    if results["label"].isna().any():
        mode_label = pd.Series(predictions).mode().iloc[0]
        results["label"] = results["label"].fillna(mode_label)

results.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", results.shape)
print(results.head())



## === cell 14
results["label"].value_counts()



## === cell 15
results
