# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

3.12

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
pillow==11.3.0
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
tqdm==4.67.1

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

# 5. Target score

0.85516

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, math, re
import numpy as np
import pandas as pd

GCS_DS_PATH = None



## === cell 1
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf
        from packaging.version import Version

        if Version(google.protobuf.__version__) >= Version("5.0.0"):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass


_ensure_protobuf_compatible()

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator



## === cell 2
print(tf.__version__)




## === cell 3
def preprocesar_features(base_model, generator):
    generator.reset()

    outputs_x = []
    outputs_y = []

    for i in tqdm(range(n_batch)):
        batch_x, batch_y = next(generator)
        outputs_x.append(base_model.predict_on_batch(batch_x))
        outputs_y.append(batch_y)

    outputs_x = np.vstack(outputs_x)
    outputs_y = np.vstack(outputs_y)
    print(f"Outputs shape: {outputs_x.shape}, {outputs_y.shape}")
    return outputs_x, outputs_y


def load_images_in_batches(
    path,
    batch_size,
    process_input_function,
    valid_ext=(".png", ".jpg", ".jpeg", ".bmp"),
):
    entries = sorted(os.listdir(path))
    filepaths = []
    filenames = []
    for name in entries:
        full = os.path.join(path, name)
        if os.path.isfile(full) and name.lower().endswith(valid_ext):
            filepaths.append(full)
            filenames.append(name)

    batch_images = []
    batch_filenames = []
    for i, (fp, fn) in enumerate(zip(filepaths, filenames)):
        img = Image.open(fp).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.asarray(img_resized, np.float32)
        batch_images.append(img_array)
        batch_filenames.append(fn)

        if len(batch_images) == batch_size or i == len(filepaths) - 1:
            batch_images_array = np.array(batch_images)
            batch_images_array = process_input_function(batch_images_array)
            yield batch_images_array, batch_filenames
            batch_images, batch_filenames = [], []


def predict_files(path, model, batch_size, process_input_function):
    y_pred = []
    y_probs = []
    all_filenames = []
    for batch_images, filenames in load_images_in_batches(
        path, batch_size, process_input_function
    ):
        y = model.predict(batch_images, verbose=0)
        y_pred.extend(np.argmax(y, axis=1).tolist())
        y_probs.extend(y.tolist())
        all_filenames.extend(filenames)
    return y_pred, y_probs, all_filenames


def save_results(results, filenames):
    all_labels = []
    for i, cls_idx in enumerate(results):
        all_labels.append(lbl_dict[cls_idx])

    df_output = pd.DataFrame({"file": filenames, "species": all_labels})
    df_output.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", df_output.shape)
    print(df_output.head())




## === cell 4
img_size = 224
batch_size = 64
train_path = "/kaggle/input/plant-seedlings-classification/train"
test_path = "/kaggle/input/plant-seedlings-classification/test"



## === cell 5
from tensorflow.keras.applications.resnet import (
    ResNet152,
    preprocess_input as preprocess_input_resnet152,
)

img_size = 224
batch_size = 64

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input_resnet152,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
    subset="training",
)

validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False,
    subset="validation",
)



## === cell 6
n_batch = math.ceil(train_generator.samples / batch_size)



## === cell 7
lbl_dict = {v: k for k, v in train_generator.class_indices.items()}
print("Num classes:", train_generator.num_classes)
print("Example mapping (index->label):", dict(list(lbl_dict.items())[:5]))



## === cell 8
base_resnet152 = ResNet152(
    include_top=False, input_shape=(img_size, img_size, 3), pooling="avg"
)
base_resnet152.trainable = False
base_resnet152.summary()



## === cell 9
cls_resnet152 = Sequential()
cls_resnet152.add(Dense(512, activation="relu", input_shape=(2048,)))
cls_resnet152.add(Dropout(0.5))
cls_resnet152.add(Dense(128, activation="relu"))
cls_resnet152.add(Dense(train_generator.num_classes, activation="softmax"))

cls_resnet152.compile(
    loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"]
)
cls_resnet152.summary()



## === cell 10
final_resnet152_model = Sequential()
final_resnet152_model.add(base_resnet152)
final_resnet152_model.add(cls_resnet152)

final_resnet152_model.compile(
    loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"]
)
final_resnet152_model.summary()



## === cell 11
print("GPUs disponibles:", tf.config.list_physical_devices("GPU"))



## === cell 12
outputs_x, outputs_y = preprocesar_features(base_resnet152, train_generator)



## === cell 13
log_cls_resnet152 = cls_resnet152.fit(
    outputs_x, outputs_y, epochs=20, batch_size=256, verbose=True
)



## === cell 14
final_resnet152_model.evaluate(validation_generator, verbose=1)



## === cell 15
y_pred, y_probs, all_filenames = predict_files(
    test_path, final_resnet152_model, 64, preprocess_input_resnet152
)
print("Predictions:", len(y_pred), "Filenames:", len(all_filenames))
print("First filenames:", all_filenames[:5])



## === cell 16
save_results(y_pred, all_filenames)
