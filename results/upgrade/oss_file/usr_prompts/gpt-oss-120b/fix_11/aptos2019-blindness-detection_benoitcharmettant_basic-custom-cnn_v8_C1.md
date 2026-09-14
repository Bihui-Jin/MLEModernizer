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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

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
pillow==11.3.0
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.467311

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.01298) has done: 'I fixed the import error for `ImageDataGenerator`, switched deprecated `fit_generator`/`predict_generator` calls to their modern equivalents, changed the final activation to `softmax` (better for multiclass cross‑entropy), removed the erroneous test generator creation, and updated the prediction block to work on the resized test images. These changes eliminate the runtime crashes and allow the script to produce a valid `submission.csv` which should improve the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.56323) has done: 'I fixed the import error by explicitly importing TensorFlow, corrected the `steps_per_epoch` and `validation_steps` to be integers, and removed the aggressive down‑sampling so the model trains on the full dataset (which should raise the quadratic weighted kappa toward the target). All other logic is unchanged.'
- What this solution (achieved 0.5741) has done: 'I removed the direct TensorFlow import that caused a protobuf‑related `AttributeError`. TensorFlow is still accessed indirectly via the Keras imports later, so functionality remains unchanged while eliminating the crash. No other logic is altered, preserving the existing model and training pipeline, which already scores above the target.'
- What this solution (achieved 0.52453) has done: 'I remove the unused `to_categorical` import that triggers a protobuf `AttributeError` during the initial import of Keras utilities. This fixes the runtime error while keeping all model logic unchanged; the current score already exceeds the target, so no further model tweaks are needed. The rest of the pipeline remains intact, ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.0) has done: 'Implemented a fix for the TensorFlow import error by switching to Keras‑only utilities for loading images. Replaced the `ImageDataGenerator` workflow with `image_dataset_from_directory`, which avoids the problematic TensorFlow import and still provides correctly shaped, batched, and labeled data. Updated the training call to use the new dataset objects, removing explicit step arguments. These changes resolve the runtime crash, ensure a valid `submission.csv` is generated, and keep the model architecture and training logic unchanged, preserving the already‑above‑target score.'
- What this solution (achieved 0.0) has done: 'I fixed the crash caused by the incompatible `image_dataset_from_directory` import. The code now uses TensorFlow’s implementation (`tf.keras.utils.image_dataset_from_directory`), which avoids the protobuf `MessageFactory` error while keeping the original data pipeline and model unchanged. No other logic is altered, so the training and prediction flow remain the same and a proper `submission.csv` file is produced.'
- What this solution (achieved 0.0) has done: 'I replaced the failing TensorFlow `image_dataset_from_directory` call with a simple NumPy loader that reads the resized images directly from the folder structure, builds training‑validation arrays and one‑hot labels, and then feeds those arrays to `model.fit`.  This eliminates the protobuf error, keeps the original model unchanged, and lets the script produce a proper `submission.csv` so the score can be evaluated.'
- What this solution (achieved 0.0) has done: 'Implemented a lightweight one‑hot encoder to replace the failing `keras.utils.to_categorical` import, eliminating the protobuf‑related error. The custom `to_categorical` uses NumPy eye indexing and keeps the rest of the pipeline untouched, ensuring the script runs end‑to‑end and writes a proper `submission.csv`. This minimal fix resolves the runtime crash while preserving the original model and training logic, allowing the Kaggle submission to be generated and evaluated.'
- What this solution (achieved 0.0) has done: 'Implemented a minimal fix by switching all Keras imports to TensorFlow‑Keras (`tensorflow.keras`) which avoids the protobuf `MessageFactory` error that halted execution. The rest of the pipeline—including data handling, model architecture, training, and submission generation—remains unchanged, ensuring the script now runs end‑to‑end and produces a valid `submission.csv` file.'
- What this solution (achieved 0.0) has done: 'Implemented minimal fixes to resolve the protobuf import error by removing direct TensorFlow imports and switching to the pure Keras API. Updated model and callback imports to use `keras` instead of `tensorflow.keras`, ensuring the script runs end‑to‑end and produces a valid `submission.csv` while keeping the original model architecture and training logic intact.'

# 9. Code solution

## === cell 0
import os
import shutil
import gc
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

DATA_PATH = "../input/"
TRAIN_PATH = os.path.join(DATA_PATH, "train_images/")

print("Data directories:", os.listdir(DATA_PATH))



## === cell 1
base_tile_dir = os.path.join(DATA_PATH, "train_images/")
df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.png"))})
df["id"] = df.path.map(lambda x: os.path.splitext(os.path.basename(x))[0])

labels = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
labels = labels.rename(columns={"id_code": "id", "diagnosis": "label"})
df_data = df.merge(labels, on="id")
df_data = shuffle(df_data.reset_index(drop=True))
print(df_data.head())



## === cell 2
SAMPLE_SIZE = 150  # retained for reference; not used after removing sampling
print(df_data.label.value_counts())

y = df_data["label"]
df_train, df_val = train_test_split(
    df_data, test_size=0.10, random_state=101, stratify=y
)

train_path = "base_dir/train"
valid_path = "base_dir/valid"
test_path = os.path.join(DATA_PATH, "test_images")
for fold in [train_path, valid_path]:
    for subf in ["0", "1", "2", "3", "4"]:
        os.makedirs(os.path.join(fold, subf), exist_ok=True)



## === cell 3
df_data.set_index("id", inplace=True)



## === cell 4
IMAGE_SIZE = 96
for image in tqdm(df_train["id"].values, desc="Resize train"):
    fname = f"{image}.png"
    label = str(df_data.loc[image, "label"])
    src = os.path.join(TRAIN_PATH, fname)
    dst = os.path.join(train_path, label, fname)
    pil_im = Image.open(src)
    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))
    resized_image.save(dst)

for image in tqdm(df_val["id"].values, desc="Resize val"):
    fname = f"{image}.png"
    label = str(df_data.loc[image, "label"])
    src = os.path.join(TRAIN_PATH, fname)
    dst = os.path.join(valid_path, label, fname)
    pil_im = Image.open(src)
    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))
    resized_image.save(dst)




## === cell 5
def to_categorical(labels, num_classes=None):
    labels = np.asarray(labels, dtype=int)
    if num_classes is None:
        num_classes = np.max(labels) + 1
    return np.eye(num_classes, dtype=np.float32)[labels]


def load_dataset_from_dir(base_dir):
    images, lbls = [], []
    for label_str in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        for img_path in glob(os.path.join(label_dir, "*.png")):
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            images.append(img)
            lbls.append(int(label_str))
    X = np.array(images, dtype=np.float32)
    y = to_categorical(lbls, num_classes=5)
    return X, y


X_train, y_train = load_dataset_from_dir(train_path)
X_val, y_val = load_dataset_from_dir(valid_path)

print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)



## === cell 6
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Activation,
    Conv2D,
    MaxPool2D,
)
from keras.optimizers import Adam

kernel_size = (3, 3)
pool_size = (2, 2)
first_filters = 32
second_filters = 64
third_filters = 128
dropout_conv = 0.3
dropout_dense = 0.5

model = Sequential(
    [
        Conv2D(
            first_filters,
            kernel_size,
            activation="relu",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        ),
        Conv2D(first_filters, kernel_size, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPool2D(pool_size=pool_size),
        Dropout(dropout_conv),
        Conv2D(second_filters, kernel_size, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(second_filters, kernel_size, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPool2D(pool_size=pool_size),
        Dropout(dropout_conv),
        Conv2D(third_filters, kernel_size, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(third_filters, kernel_size, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPool2D(pool_size=pool_size),
        Dropout(dropout_conv),
        Flatten(),
        Dense(256, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_dense),
        Dense(5, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(0.01), loss="categorical_crossentropy", metrics=["accuracy"]
)
print("Model compiled successfully.")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystopper = EarlyStopping(
    monitor="val_loss", patience=2, verbose=1, restore_best_weights=True
)
reducel = ReduceLROnPlateau(monitor="val_loss", patience=1, verbose=1, factor=0.1)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=13,
    batch_size=32,
    shuffle=True,
    callbacks=[reducel, earlystopper],
    verbose=2,
)



## === cell 8
y_pred_keras = None  # placeholder kept for compatibility



## === cell 9
base_test_dir = os.path.join(DATA_PATH, "test_images")
test_files = glob(os.path.join(base_test_dir, "*.png"))

os.makedirs("test_resized", exist_ok=True)
for image in tqdm(test_files, desc="Resize test"):
    fname = os.path.basename(image)
    dst = os.path.join("test_resized", fname)
    pil_im = Image.open(image)
    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))
    resized_image.save(dst)

test_files = glob(os.path.join("test_resized", "*.png"))
submission = pd.DataFrame()
file_batch = 20
max_idx = len(test_files)

for idx in range(0, max_idx, file_batch):
    batch_files = test_files[idx : idx + file_batch]
    test_df = pd.DataFrame({"path": batch_files})
    test_df["id_code"] = test_df.path.map(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    test_df["image"] = test_df["path"].map(imread)
    K_test = np.stack(test_df["image"].values)
    K_test = (K_test - K_test.mean()) / K_test.std()
    predictions = model.predict(K_test, verbose=0)
    pred = np.argmax(predictions, axis=1)
    test_df["diagnosis"] = pred
    submission = pd.concat(
        [submission, test_df[["id_code", "diagnosis"]]], ignore_index=True
    )

print("Submission preview:")
print(submission.head())



## === cell 10
shutil.rmtree(train_path, ignore_errors=True)
shutil.rmtree(valid_path, ignore_errors=True)
shutil.rmtree("test_resized", ignore_errors=True)
submission.to_csv("submission.csv", index=False, header=True)
print("submission.csv written:", os.path.getsize("submission.csv"), "bytes")



## === cell 11
df = pd.read_csv("submission.csv")
print(df["diagnosis"].value_counts())
