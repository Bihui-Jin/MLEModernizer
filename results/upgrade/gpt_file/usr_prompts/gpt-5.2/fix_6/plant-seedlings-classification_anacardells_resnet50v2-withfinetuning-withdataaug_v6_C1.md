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

print("Input root exists:", os.path.exists("/kaggle/input"))
print(
    "Train dir exists:",
    os.path.exists("/kaggle/input/plant-seedlings-classification/train"),
)
print(
    "Test dir exists:",
    os.path.exists("/kaggle/input/plant-seedlings-classification/test"),
)



## === cell 1
TRAIN_ROOT = "/kaggle/input/plant-seedlings-classification/train"
print("Train root:", TRAIN_ROOT)
print("Skipping train.csv generation (unused by training/prediction).")



## === cell 2
classes = sorted(
    [d for d in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, d))]
)
print(f"Number of classes: {len(classes)}")
print("Classes:", classes)



## === cell 3
print("Skipping plots for runtime.")



## === cell 4
print("Skipping sample image visualization for runtime.")



## === cell 5
print("Skipping image-size histogram for runtime.")



## === cell 6
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

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

print("TensorFlow:", tf.__version__)
print("Using tf.keras:", tf.keras.__name__)



## === cell 7
file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
batch_size = 32
seed = 42
val_split = 0.2
image_size = (256, 256)
PROYECT_FOLDER_TRAIN = "/kaggle/input/plant-seedlings-classification/train/"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Standardize for ResNet
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

print("Class indices:", train_generator.class_indices)
num_classes = len(train_generator.class_indices)
print("num_classes:", num_classes)


def _paths_and_onehot_from_directory_iterator(di):
    filepaths = np.asarray(di.filepaths)
    y_int = np.asarray(di.classes, dtype=np.int32)
    y = tf.one_hot(y_int, depth=num_classes, dtype=tf.float32)
    return filepaths, y


train_filepaths, train_y = _paths_and_onehot_from_directory_iterator(train_generator)
val_filepaths, val_y = _paths_and_onehot_from_directory_iterator(val_generator)


def _make_dataset(filepaths, y, augment: bool, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, y))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(filepaths), seed=seed, reshuffle_each_iteration=True
        )

    def _load_and_preprocess(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(
            img, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True
        )
        img = img * 255.0
        return img, label

    ds = ds.map(
        _load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )

    if augment:
        def _apply_idg_batch(images, labels):
            def _np_apply(x):
                out = np.empty_like(x)
                for i in range(x.shape[0]):
                    out[i] = train_datagen.random_transform(x[i])
                    out[i] = train_datagen.standardize(out[i])
                return out

            images = tf.numpy_function(_np_apply, [images], Tout=tf.float32)
            images.set_shape([None, image_size[0], image_size[1], 3])
            return images, labels

        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(
            _apply_idg_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
    else:
        def _standardize(img, label):
            img = preprocess_input(img)
            return img, label

        ds = ds.map(
            _standardize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
        ds = ds.batch(batch_size, drop_remainder=True)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = _make_dataset(train_filepaths, train_y, augment=True, shuffle=True)
val_ds = _make_dataset(val_filepaths, val_y, augment=False, shuffle=False)

train_steps = train_generator.n // train_generator.batch_size
val_steps = val_generator.n // val_generator.batch_size
print("train_steps:", train_steps, "val_steps:", val_steps)



## === cell 8
input_shape_c = (image_size[0], image_size[1], 3)
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)



## === cell 9
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



## === cell 10
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

print("[INFO]: Training the network...")

H_pre = pre_trained_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=epochs,
    callbacks=[annealer, earlystop, modelsave],
)



## === cell 11
print("[INFO]: Training finished. Skipping curve plots for runtime.")
print("Epochs run:", len(H_pre.history.get("loss", [])))



## === cell 12
csv_testfile = "/kaggle/working/test.csv"
print("Skipping test.csv generation for runtime. (Not needed.)")



## === cell 13
print("Skipping loading test.csv for runtime.")



## === cell 14
test_batch_size = 32
seed = 42
image_size = (256, 256)
PROYECT_FOLDER_TEST = "/kaggle/input/plant-seedlings-classification/"

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    PROYECT_FOLDER_TEST,
    target_size=image_size,
    color_mode="rgb",
    batch_size=test_batch_size,
    classes=["test"],
    shuffle=False,
)

list_of_files = test_generator.filenames
print("Example test filename from generator:", list_of_files[0])
print("Num test files:", len(list_of_files))



## === cell 15
predicted_class = pre_trained_model.predict(
    test_generator,
    verbose=1,
)



## === cell 16
predicted_class_number = np.argmax(predicted_class, axis=1)

idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
classes_ordered = [idx_to_class[i] for i in range(len(idx_to_class))]

print("Ordered classes:", classes_ordered[:5], "...", len(classes_ordered))
print("Predicted class indices sample:", predicted_class_number[:10])



## === cell 17
sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"

sample = pd.read_csv(sample_path)
if "file" not in sample.columns or "species" not in sample.columns:
    raise ValueError(f"Unexpected sample submission columns: {sample.columns.tolist()}")

pred_files = [os.path.basename(f) for f in list_of_files]
pred_species = [classes_ordered[i] for i in predicted_class_number]
pred_df = pd.DataFrame({"file": pred_files, "species": pred_species})

submission = sample[["file"]].merge(pred_df, on="file", how="left")
if submission["species"].isna().any():
    missing = submission.loc[submission["species"].isna(), "file"].head(10).tolist()
    raise ValueError(f"Missing predictions for some files, e.g.: {missing}")

submission = submission[["file", "species"]]

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.shape)
print(submission.head())
print("Submission columns:", submission.columns.tolist())



## === cell 18
dataFrameResults = pd.read_csv("/kaggle/working/submission.csv")
print(dataFrameResults.shape)
print(dataFrameResults.head())
print("Columns:", dataFrameResults.columns.tolist())

if list(dataFrameResults.columns) != ["file", "species"]:
    raise ValueError("Invalid submission columns; expected exactly ['file','species'].")

print("Done. Submission is ready at /kaggle/working/submission.csv")
