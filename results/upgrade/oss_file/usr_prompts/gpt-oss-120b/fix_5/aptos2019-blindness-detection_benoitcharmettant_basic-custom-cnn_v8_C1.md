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

0.52453

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.01298) has done: 'I fixed the import error for `ImageDataGenerator`, switched deprecated `fit_generator`/`predict_generator` calls to their modern equivalents, changed the final activation to `softmax` (better for multiclass cross‑entropy), removed the erroneous test generator creation, and updated the prediction block to work on the resized test images. These changes eliminate the runtime crashes and allow the script to produce a valid `submission.csv` which should improve the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.56323) has done: 'I fixed the import error by explicitly importing TensorFlow, corrected the `steps_per_epoch` and `validation_steps` to be integers, and removed the aggressive down‑sampling so the model trains on the full dataset (which should raise the quadratic weighted kappa toward the target). All other logic is unchanged.'
- What this solution (achieved 0.5741) has done: 'I removed the direct TensorFlow import that caused a protobuf‑related `AttributeError`. TensorFlow is still accessed indirectly via the Keras imports later, so functionality remains unchanged while eliminating the crash. No other logic is altered, preserving the existing model and training pipeline, which already scores above the target.'
- What this solution (achieved 0.52453) has done: 'I remove the unused `to_categorical` import that triggers a protobuf `AttributeError` during the initial import of Keras utilities. This fixes the runtime error while keeping all model logic unchanged; the current score already exceeds the target, so no further model tweaks are needed. The rest of the pipeline remains intact, ensuring a valid `submission.csv` is produced.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import shutil
from tqdm import tqdm
from PIL import Image


DATA_PATH = "../input/"
TRAIN_PATH = DATA_PATH + "train_images/"

print(os.listdir(DATA_PATH))

from glob import glob
from skimage.io import imread
import gc

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



## === cell 1
base_tile_dir = DATA_PATH + "train_images/"

df = pd.DataFrame({"path": glob(base_tile_dir + "/*.png")})

df["id"] = df.path.map(lambda x: x.split("/")[3].split(".")[0])

labels = pd.read_csv(DATA_PATH + "train.csv")
labels = labels.rename(index=str, columns={"id_code": "id", "diagnosis": "label"})

df_data = df.merge(labels, on="id")

df_data = shuffle(df_data.reset_index(drop=True))

df_data.head()


## === cell 2
SAMPLE_SIZE = 150  # retained for reference; not used after removing sampling

df_data.head()
print(df_data.label.value_counts())

y = df_data["label"]
df_train, df_val = train_test_split(
    df_data, test_size=0.10, random_state=101, stratify=y
)

train_path = "base_dir/train"
valid_path = "base_dir/valid"
test_path = "../input/test_images"  # path to original test images
for fold in [train_path, valid_path]:
    for subf in ["0", "1", "2", "3", "4"]:
        os.makedirs(os.path.join(fold, subf), exist_ok=True)


## === cell 3
df_data.set_index("id", inplace=True)
df_data.head()


## === cell 4
IMAGE_SIZE = 96

for image in tqdm(df_train["id"].values):
    fname = image + ".png"
    label = str(df_data.loc[image, "label"])  # get the label for a certain image
    src = os.path.join(TRAIN_PATH, fname)
    dst = os.path.join(train_path, label, fname)

    pil_im = Image.open(src)
    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))
    resized_image.save(dst)

for image in tqdm(df_val["id"].values):
    fname = image + ".png"
    label = str(df_data.loc[image, "label"])  # get the label for a certain image
    src = os.path.join(TRAIN_PATH, fname)
    dst = os.path.join(valid_path, label, fname)

    pil_im = Image.open(src)
    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))
    resized_image.save(dst)


## === cell 5
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32
val_batch_size = 32

train_steps = int(np.ceil(num_train_samples / train_batch_size))
val_steps = int(np.ceil(num_val_samples / val_batch_size))

datagen = ImageDataGenerator(
    preprocessing_function=lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x,
    horizontal_flip=True,
    vertical_flip=True,
)

train_gen = datagen.flow_from_directory(
    train_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=train_batch_size,
    class_mode="categorical",
)

val_gen = datagen.flow_from_directory(
    valid_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=val_batch_size,
    class_mode="categorical",
)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam

kernel_size = (3, 3)
pool_size = (2, 2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.5

model = Sequential()
model.add(
    Conv2D(
        first_filters,
        kernel_size,
        activation="relu",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
)
model.add(Conv2D(first_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(5, activation="softmax"))  # softmax for multiclass

model.compile(Adam(0.01), loss="categorical_crossentropy", metrics=["accuracy"])

print("Done !")


## === cell 7
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystopper = EarlyStopping(
    monitor="val_loss", patience=2, verbose=1, restore_best_weights=True
)
reducel = ReduceLROnPlateau(monitor="val_loss", patience=1, verbose=1, factor=0.1)

history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=13,
    callbacks=[reducel, earlystopper],
)


## === cell 8
y_pred_keras = None  # placeholder kept for compatibility


## === cell 9
base_test_dir = "../input/test_images/"

test_files = glob(os.path.join(base_test_dir, "*.png"))

os.makedirs("test_resized/", exist_ok=True)

for image in tqdm(test_files):
    fname = os.path.basename(image)
    src = image
    dst = os.path.join("test_resized/", fname)
    pil_im = Image.open(src)
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

submission.head()


## === cell 10
shutil.rmtree(train_path, ignore_errors=True)
shutil.rmtree(valid_path, ignore_errors=True)
shutil.rmtree("test_resized/", ignore_errors=True)
submission.to_csv("submission.csv", index=False, header=True)


## === cell 11
df = pd.read_csv("submission.csv")
df["diagnosis"].value_counts()
