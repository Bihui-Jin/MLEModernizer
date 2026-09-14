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

# 5. Target score

0.9089861751152074

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.06341) has done: 'The timeout is dominated by (1) extremely heavy per-step CPU image augmentation/decoding with a single-threaded `ImageDataGenerator`, and (2) an over-parameterized `Flatten + Dense(2048,1024,...)` head on 224×224 inputs that makes each training step expensive for 50 epochs. To keep the exact same model/training logic and accuracy semantics, the main speedups are: enable TensorFlow graph/XLA compilation for the existing model, turn on multi-worker prefetching in the Keras generators (pure I/O pipeline speedup), and remove notebook-only debug cells that iterate/visualize batches (they cost extra time but do not affect training/inference outputs). These changes reduce wall-clock time without changing architecture, loss, optimizer, epochs, or data used.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

SEED = 2021
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
def _train_preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = rescale(images)
    images = data_augmentation(images, training=True)
    return images, labels


@tf.function
def _val_preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = rescale(images)
    return images, labels


train_cache_path = os.path.join(".", "tfds_train_cache")
val_cache_path = os.path.join(".", "tfds_val_cache")

train_ds = (
    train_raw.map(_train_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache(train_cache_path)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_raw.map(_val_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache(val_cache_path)
    .prefetch(AUTOTUNE)
)

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.autotune_buffers = True

train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2891201530.py in <cell line: 0>()
     54 options.experimental_optimization.map_parallelization = True
     55 options.experimental_optimization.parallel_batch = True
---> 56 options.experimental_optimization.autotune_buffers = True
     57 
     58 train_ds = train_ds.with_options(options)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

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
    "Model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2915751117.py in <cell line: 0>()
      5 # CHANGE (timeout): save only weights instead of the whole model each time validation improves.
      6 # This preserves "best checkpoint" semantics and final accuracy, but avoids expensive full-model serialization.
----> 7 save_best = tf.keras.callbacks.ModelCheckpoint(
      8     "Model.h5",
      9     monitor="val_accuracy",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=Model.h5

## === cell 7
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[save_best, lr_scheduler],
    verbose=1,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/258537718.py in <cell line: 0>()
      3     validation_data=val_ds,
      4     epochs=50,
----> 5     callbacks=[save_best, lr_scheduler],
      6     verbose=1,
      7 )

NameError: name 'save_best' is not defined

## === cell 8
if os.path.exists("Model.h5"):
    model.load_weights("Model.h5")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
    steps_per_execution=16,
)



## === cell 9
model.evaluate(val_ds, verbose=1)



## === cell 10
test_path = "../input/paddy-disease-classification/test_images"

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    labels=None,
    label_mode=None,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


@tf.function
def _prep_test(images):
    images = tf.cast(images, tf.float32)
    images = rescale(images)
    return images


test_ds = test_ds.map(
    _prep_test, num_parallel_calls=AUTOTUNE, deterministic=True
).prefetch(AUTOTUNE)

test_files = list(test_ds.file_paths)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1264219486.py in <cell line: 0>()
     25 # CHANGE (correctness + avoids extra work): use file_paths from the dataset to guarantee identical ordering
     26 # to the batches yielded by image_dataset_from_directory (no need for separate glob/sort reconciliation).
---> 27 test_files = list(test_ds.file_paths)
     28 

AttributeError: '_PrefetchDataset' object has no attribute 'file_paths'

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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4152617880.py in <cell line: 0>()
----> 1 image_ids = [os.path.basename(f) for f in test_files]
      2 
      3 results = pd.DataFrame({"image_id": image_ids, "label": predictions})
      4 
      5 sample_path = "../input/paddy-disease-classification/sample_submission.csv"

NameError: name 'test_files' is not defined

## === cell 14
results["label"].value_counts()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/936236964.py in <cell line: 0>()
----> 1 results["label"].value_counts()
      2 

NameError: name 'results' is not defined

## === cell 15
results

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/962467392.py in <cell line: 0>()
----> 1 results

NameError: name 'results' is not defined
