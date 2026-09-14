# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
import warnings

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")




## === cell 1
class SimpleEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch + 1}: " + ", ".join(
            [
                f"{k}={v:.4f}"
                for k, v in logs.items()
                if isinstance(v, (int, float, np.floating))
            ]
        )
        print(msg)




## === cell 2
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 3
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPUs available:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU available. Using CPU.")



## === cell 4
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"



## === cell 5
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
sample_sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")




## === cell 6
def get_image_path(id_code, is_train=True):
    ext = ".png"
    if is_train:
        return os.path.join(TRAIN_IMG_DIR, id_code + ext)
    else:
        return os.path.join(TEST_IMG_DIR, id_code + ext)




## === cell 7
train_df["filepath"] = TRAIN_IMG_DIR + "/" + train_df["id_code"].astype(str) + ".png"
test_df["filepath"] = TEST_IMG_DIR + "/" + test_df["id_code"].astype(str) + ".png"



## === cell 8
IMG_SIZE = 299



## === cell 9
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_transform = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    rotation_range=180,
    shear_range=20,
    zoom_range=(0.8, 1.2),
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="reflect",
)



## === cell 10
valid_transform = ImageDataGenerator()




## === cell 11
def load_and_preprocess_image(path, transform=None):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Image not found at path: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    if transform is not None:
        image = transform.random_transform(image)
        image = transform.standardize(image)

    image = image.astype(np.float32)
    image = preprocess_input(image)  # correct preprocessing for InceptionResNetV2
    return image




## === cell 12
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=32,
        transform=None,
        is_train=True,
        num_classes=5,
        shuffle=True,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.transform = transform
        self.is_train = bool(is_train)
        self.num_classes = int(num_classes)
        self.shuffle = bool(shuffle)

        self.filepaths = self.df["filepath"].to_numpy(dtype=object)
        if self.is_train:
            y_int = self.df["diagnosis"].to_numpy(dtype=np.int32)
            self.labels_oh = tf.keras.utils.to_categorical(
                y_int, num_classes=self.num_classes
            ).astype(np.float32)
        else:
            self.labels_oh = None

        self.indexes = np.arange(len(self.df), dtype=np.int32)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_idx = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        bs = batch_idx.shape[0]

        images = np.empty((bs, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        if self.is_train:
            labels = self.labels_oh[batch_idx]
            for i, j in enumerate(batch_idx):
                images[i] = load_and_preprocess_image(
                    self.filepaths[j], transform=self.transform
                )
            return images, labels
        else:
            for i, j in enumerate(batch_idx):
                images[i] = load_and_preprocess_image(
                    self.filepaths[j], transform=self.transform
                )
            return images




## === cell 13
train_df_split, valid_df_split = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["diagnosis"]
)

BATCH_SIZE = 32
train_gen = DataGenerator(
    train_df_split, batch_size=BATCH_SIZE, transform=train_transform, is_train=True
)
valid_gen = DataGenerator(
    valid_df_split,
    batch_size=BATCH_SIZE,
    transform=valid_transform,
    is_train=True,
    shuffle=False,
)



## === cell 14
try:
    base_model = InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
except Exception as e:
    print(
        f"Warning: could not load imagenet weights due to: {e}\nFalling back to random initialization."
    )
    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

for layer in base_model.layers:
    layer.trainable = True



## === cell 15
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(100)(x)
x = Dropout(0.3)(x)
predictions = Dense(5, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 16
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)
earlystop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-7
)



## === cell 17
EPOCHS = 1

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=valid_gen,
    callbacks=[SimpleEpochLogger(), checkpoint, earlystop, reduce_lr],
    verbose=1,
)



## === cell 18
if os.path.exists("best_model.keras"):
    model = tf.keras.models.load_model("best_model.keras")



## === cell 19
test_gen = DataGenerator(
    test_df,
    batch_size=BATCH_SIZE,
    transform=valid_transform,
    is_train=False,
    shuffle=False,
)

preds = model.predict(
    test_gen,
    verbose=1,
)
test_df["diagnosis"] = np.argmax(preds, axis=1).astype(int)



## === cell 20
submission_csv = "submission.csv"

sub = sample_sub[["id_code"]].copy()
sub = sub.merge(test_df[["id_code", "diagnosis"]], on="id_code", how="left")

sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

sub.to_csv(submission_csv, index=False)
print(f"Submission file saved as {submission_csv}")
print(sub.head())
