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
import random
import math
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

print("TF:", tf.__version__)



## === cell 1
train_meta_data = "../input/paddy-disease-classification/train.csv"
train_data_dir = "../input/paddy-disease-classification/train_images"
epochs = 100
lr = 1e-4
valid_split = 0.2
input_size = 224
batch_size = 32
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.categorical_crossentropy



## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=15, monitor="val_loss", restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=5, monitor="val_loss", factor=0.75, verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_chp.keras", monitor="val_loss", verbose=1, save_best_only=True
)



## === cell 3
get_transform = None



## === cell 4
meta = pd.read_csv(train_meta_data)
class_names = sorted(meta["label"].unique().tolist())
print("Detected classes:", class_names, "count:", len(class_names))

AUTOTUNE = tf.data.AUTOTUNE

tfdata_options = tf.data.Options()
tfdata_options.experimental_deterministic = True
try:
    tfdata_options.threading.private_threadpool_size = max(
        4, (os.cpu_count() or 4) // 2
    )
except Exception:
    pass
try:
    tfdata_options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    tfdata_options.experimental_optimization.map_parallelization = True
    tfdata_options.experimental_optimization.parallel_batch = True
except Exception:
    pass

train_raw = tf.keras.utils.image_dataset_from_directory(
    train_data_dir + "/",
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(input_size, input_size),
    shuffle=True,
    seed=SEED,
    validation_split=valid_split,
    subset="training",
    interpolation="bilinear",
).with_options(tfdata_options)

valid_raw = tf.keras.utils.image_dataset_from_directory(
    train_data_dir + "/",
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(input_size, input_size),
    shuffle=False,
    seed=SEED,
    validation_split=valid_split,
    subset="validation",
    interpolation="bilinear",
).with_options(tfdata_options)

augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED),
        tf.keras.layers.RandomRotation(factor=10.0 / 360.0, seed=SEED),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=SEED
        ),
    ],
    name="augment",
)


def _norm_only(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def _prep_valid(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


train_card = int(train_raw.cardinality().numpy())
valid_card = int(valid_raw.cardinality().numpy())

steps_per_epoch = None
validation_steps = None

valid_ds = (
    valid_raw.map(_prep_valid, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
).with_options(tfdata_options)

train_base = (
    train_raw.map(_norm_only, num_parallel_calls=AUTOTUNE).cache()
).with_options(tfdata_options)

train_ds = (
    train_base.map(
        lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)
).with_options(tfdata_options)

class_indices = {name: i for i, name in enumerate(class_names)}
print("num_classes:", len(class_names))
print("train batches:", train_card, "valid batches:", valid_card)



## === cell 5
print("Train element_spec:", train_ds.element_spec)
print("Valid element_spec:", valid_ds.element_spec)



## === cell 6
pass



## === cell 7
pass



## === cell 8
meta.head()



## === cell 9
pass



## === cell 10
pass



## === cell 11
back_bone = tf.keras.applications.Xception(
    weights="imagenet", input_shape=(input_size, input_size, 3), include_top=False
)
back_bone.summary()



## === cell 12
pass



## === cell 13
back_bone.trainable = False

input_layer = Input(shape=(input_size, input_size, 3))
x = back_bone(input_layer, training=False)
x = GlobalAveragePooling2D()(x)
output_layer = Dense(len(class_names), activation="softmax")(x)

model = Model(input_layer, output_layer)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"], jit_compile=True)



## === cell 14
model.summary()



## === cell 15
if os.path.exists("best_chp.keras"):
    try:
        model = tf.keras.models.load_model("best_chp.keras", compile=False)
        model.compile(
            optimizer=optimizer, loss=loss, metrics=["accuracy"], jit_compile=True
        )
        print("Loaded existing checkpoint: best_chp.keras")
    except Exception as e:
        print(
            "Checkpoint exists but could not be loaded, continuing from scratch. Error:",
            repr(e),
        )

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr, checkpoint],
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 16
model.evaluate(valid_ds, steps=validation_steps, verbose=1)



## === cell 17
pass



## === cell 18
pass



## === cell 19
temp_hist = pd.DataFrame(history.history)
temp_hist.to_csv("model_xception_history.csv", index=False)
temp_hist.head()



## === cell 20
model.save("model_xception.keras")



## === cell 21
model.save_weights("model_xception_weights.weights.h5")



## === cell 22
test_loc = "../input/paddy-disease-classification/test_images"

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=test_loc,
    labels=None,
    label_mode=None,
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(input_size, input_size),
    shuffle=False,
    interpolation="bilinear",
).with_options(tfdata_options)


def _prep_test(images):
    images = tf.cast(images, tf.float32) / 255.0
    return images


test_ds = (
    test_ds.map(_prep_test, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
).with_options(tfdata_options)



## === cell 23
train_datagen = type("Obj", (), {})()
train_datagen.class_indices = class_indices
train_datagen.num_classes = len(class_names)
train_datagen.class_names = class_names
train_datagen.class_indices



## === cell 24
pred_probs = model.predict(test_ds, verbose=1)
predict_max = np.argmax(pred_probs, axis=1)

inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
predictions = [inverse_map[int(k)] for k in predict_max]

files = [os.path.basename(p) for p in test_ds.file_paths]

sub = pd.DataFrame({"image_id": files, "label": predictions})

sample = pd.read_csv("../input/paddy-disease-classification/sample_submission.csv")
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")

if sub["label"].isna().any():
    most_common_label = meta["label"].mode().iloc[0]
    sub["label"] = sub["label"].fillna(most_common_label)

sub.to_csv("model_submission_v5.csv", index=False)
sub.head()
