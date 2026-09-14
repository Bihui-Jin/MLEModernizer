# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-image==0.25.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Code solution

## === cell 0
import os


import numpy as np
import pandas as pd

for d in [
    "/kaggle/input/plant-seedlings-classification/train",
    "/kaggle/input/plant-seedlings-classification/test",
]:
    print(d, "exists:", os.path.exists(d))



## === cell 1
print("Skipping train.csv generation; using flow_from_directory directly for speed.")



## === cell 2
print("Skipping train DataFrame preview for speed.")



## === cell 3
print("Class distribution print skipped; will infer classes from generator.")



## === cell 4
import matplotlib.pyplot as plt

print("Pie plot skipped for runtime.")



## === cell 5
print("Random image visualization skipped for runtime.")



## === cell 6
print("Image size histogram skipped for runtime.")



## === cell 7
import random
import tensorflow as tf
from math import exp

from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow:", tf.__version__)

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 8
import math

file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
batch_size = 32
val_split = 0.2
image_size = (256, 256)
PROYECT_FOLDER_TRAIN = "/kaggle/input/plant-seedlings-classification/train/"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Standardize for ResNetV2
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    rescale=0.9,
    vertical_flip=True,
    horizontal_flip=True,
    validation_split=val_split,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=val_split,
)

train_generator = train_datagen.flow_from_directory(
    PROYECT_FOLDER_TRAIN,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    seed=seed,
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    PROYECT_FOLDER_TRAIN,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    seed=seed,
    shuffle=True,
)

num_classes = train_generator.num_classes
print("Detected num_classes:", num_classes)
print("Class indices:", train_generator.class_indices)

steps_per_epoch = math.ceil(train_generator.n / train_generator.batch_size)
validation_steps = math.ceil(val_generator.n / val_generator.batch_size)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)


def gen_to_dataset(gen, num_classes=None):
    if gen.class_mode is None:
        output_signature = (
            tf.TensorSpec(
                shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32
            ),
        )

        def _iter():
            for x in gen:
                yield (x,)

    else:
        output_signature = (
            tf.TensorSpec(
                shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
        )

        def _iter():
            for x, y in gen:
                yield x, y

    ds = tf.data.Dataset.from_generator(_iter, output_signature=output_signature)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = gen_to_dataset(train_generator, num_classes=num_classes)
val_ds = gen_to_dataset(val_generator, num_classes=num_classes)



## === cell 9
input_shape_c = (image_size[0], image_size[1], 3)
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)



## === cell 10
for layer in base_model.layers:
    if layer.name == "conv5_block1_1_conv":
        break
    layer.trainable = False

pre_trained_model = Sequential()
pre_trained_model.add(base_model)
pre_trained_model.add(layers.Flatten())
pre_trained_model.add(layers.Dense(512, activation="relu"))
pre_trained_model.add(Dropout(0.5))
pre_trained_model.add(BatchNormalization())
pre_trained_model.add(layers.Dense(num_classes, activation="softmax"))
pre_trained_model.summary()



## === cell 11
epochs = 200

print("[INFO]: Compiling the model...")
pre_trained_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * exp(-0.1)


annealer = LearningRateScheduler(scheduler)

earlystop = EarlyStopping(
    patience=5,
    monitor="val_loss",
    restore_best_weights=False,
)

modelsave = ModelCheckpoint(
    filepath=file + ".h5",
    save_best_only=True,
    monitor="val_loss",
    verbose=1,
)

print("[INFO]: Entrenando la red...")

H_pre = pre_trained_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[annealer, earlystop, modelsave],
)



## === cell 12
print("[INFO]: Training curves plot skipped for runtime.")



## === cell 13
print("Skipping test.csv generation; using flow_from_directory directly for speed.")



## === cell 14
print("Skipping test DataFrame preview for speed.")



## === cell 15
batch_size = 32
image_size = (256, 256)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    "/kaggle/input/plant-seedlings-classification/test",
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,
)

test_ds = gen_to_dataset(test_generator, num_classes=None)



## === cell 16
best_path = file + ".h5"
if os.path.exists(best_path):
    pre_trained_model.load_weights(best_path)

predicted_class = pre_trained_model.predict(
    test_ds,
    steps=math.ceil(test_generator.n / test_generator.batch_size),
    verbose=1,
)

print("n_classes_model:", predicted_class.shape[1])
print("n_test_images:", predicted_class.shape[0])

predicted_class_number = np.argmax(predicted_class, axis=1)
print("n_preds:", len(predicted_class_number))

idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
print("Train class_indices:", train_generator.class_indices)



## === cell 17
sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

pred_files = [os.path.basename(fn) for fn in test_generator.filenames]
pred_species = [idx_to_class[int(i)] for i in predicted_class_number]
pred_map = dict(zip(pred_files, pred_species))

submission = sample_sub.copy()
submission["file"] = submission["file"].astype(str)
submission["species"] = submission["file"].map(pred_map)

if submission["species"].isna().any():
    fallback = idx_to_class.get(0, list(idx_to_class.values())[0])
    submission["species"] = submission["species"].fillna(fallback)

submission = submission[["file", "species"]]
submission["species"] = submission["species"].astype(str)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print(submission.shape)

check = pd.read_csv(out_path)
print(check.columns.tolist())
print(check.head())
print("Any NA species:", check["species"].isna().any())
print("Unique predicted species:", sorted(check["species"].unique())[:20])
print("n_rows:", len(check))
print("n_files_unique:", check["file"].nunique())
