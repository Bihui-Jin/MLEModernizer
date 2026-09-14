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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd



## === cell 1
train_root = "/kaggle/input/plant-seedlings-classification/train"
classes = sorted(
    d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))
)
dataFrameTrain = pd.DataFrame(
    {
        "specie": classes,
        "count": [len(os.listdir(os.path.join(train_root, c))) for c in classes],
    }
)
print((int(dataFrameTrain["count"].sum()), 3))  # mimic prior "shape-like" output
print(dataFrameTrain.head())



## === cell 2
print(dataFrameTrain.describe(include="all"))



## === cell 3
print(f"Number of classes: {len(classes)}")
datos_classes = dataFrameTrain.set_index("specie")[["count"]]
print(datos_classes)



## === cell 4
import tensorflow as tf
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
from math import exp

print("TF version:", tf.__version__)
tf.keras.utils.set_random_seed(42)

tf.config.optimizer.set_jit(False)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 5
file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"

batch_size = 64
seed = 42
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
    preprocessing_function=preprocess_input, validation_split=val_split
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
print("Detected num_classes from generator:", num_classes)
print("class_indices:", train_generator.class_indices)

train_ds = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
    ),
)
val_ds = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
    ),
)

val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)



## === cell 6
input_shape_c = (image_size[0], image_size[1], 3)
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)



## === cell 7
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



## === cell 8
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
)

modelsave = ModelCheckpoint(filepath=file + ".keras", save_best_only=True, verbose=1)

print("[INFO]: Entrenando la red...")
H_pre = pre_trained_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    epochs=epochs,
    callbacks=[annealer, earlystop, modelsave],
)



## === cell 9
test_root = "/kaggle/input/plant-seedlings-classification/test"
print(
    "Test images:",
    len([f for f in os.listdir(test_root) if f.lower().endswith(".png")]),
)



## === cell 10
test_files = sorted(
    [f for f in os.listdir(test_root) if os.path.isfile(os.path.join(test_root, f))]
)
dataFrameTest = pd.DataFrame(
    {"path": [os.path.join(test_root, f) for f in test_files], "file": test_files}
)
print(dataFrameTest.shape)
print(dataFrameTest.head())
print(dataFrameTest["path"].iloc[0])



## === cell 11
batch_size = 64
seed = 42
image_size = (256, 256)
PROYECT_FOLDER_TEST = "/kaggle/input/plant-seedlings-classification/"

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    PROYECT_FOLDER_TEST,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,
    seed=seed,
    classes=["test"],
)

test_ds = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_signature=tf.TensorSpec(
        shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32
    ),
).prefetch(tf.data.AUTOTUNE)



## === cell 12
predicted_class = pre_trained_model.predict(
    test_ds,
    steps=len(test_generator),
    verbose=1,
)
predicted_class = predicted_class[: test_generator.n]

print("Num classes:", predicted_class.shape[1])
print("Num predictions:", predicted_class.shape[0])

predicted_class_number = np.argmax(predicted_class, axis=1)
print("Num predicted labels:", len(predicted_class_number))

idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
classes_ordered = [idx_to_class[i] for i in range(len(idx_to_class))]
print("Classes ordered:", classes_ordered)



## === cell 13
submission_path = "/kaggle/working/submission.csv"
sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)

test_files_from_gen = [os.path.basename(p) for p in test_generator.filenames]
pred_species_from_gen = [classes_ordered[i] for i in predicted_class_number]
pred_map = dict(zip(test_files_from_gen, pred_species_from_gen))

sub_df = sample[["file"]].copy()
sub_df["species"] = sub_df["file"].map(pred_map)

if sub_df["species"].isna().any():
    fallback_class = classes_ordered[0]
    sub_df["species"] = sub_df["species"].fillna(fallback_class)

sub_df = sub_df[["file", "species"]]
sub_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print("Our submission shape:", sub_df.shape, "columns:", sub_df.columns.tolist())
print(sub_df.head())
print("Submission saved at:", submission_path)
print("NaNs in species:", int(sub_df["species"].isna().sum()))
print("Unique predicted species:", sub_df["species"].nunique())
