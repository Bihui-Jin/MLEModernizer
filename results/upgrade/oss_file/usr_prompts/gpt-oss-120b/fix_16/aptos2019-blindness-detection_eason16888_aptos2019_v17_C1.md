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

3.10

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, proto):
            return proto.__class__

        _mf.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass  # if protobuf is not available, TensorFlow will raise later

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import gc
from tqdm import tqdm
import concurrent.futures  # used for parallel preprocessing

BASE_DIR = "/kaggle/input/aptos2019-blindness-detection"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 20  # unchanged
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## === cell 1
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df["diagnosis"] = train_df["diagnosis"].astype(str)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["diagnosis"],
    random_state=SEED,
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df["filename"] = train_df["id_code"] + ".png"
val_df["filename"] = val_df["id_code"] + ".png"


def _load_and_preprocess(path):
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return load_ben_color(img)


def _prepare_dataset(df, folder):
    paths = [os.path.join(folder, fname) for fname in df["filename"]]
    N = len(paths)
    X = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, img in enumerate(
            tqdm(
                ex.map(_load_and_preprocess, paths),
                total=N,
                desc="Pre‑processing",
            )
        ):
            X[i] = img
    y = tf.keras.utils.to_categorical(df["diagnosis"].astype(int).values, num_classes=5)
    return X, y


train_images_path = os.path.join(BASE_DIR, "train_images")
val_images_path = os.path.join(BASE_DIR, "train_images")  # same folder

X_train, y_train = _prepare_dataset(train_df, train_images_path)
X_val, y_val = _prepare_dataset(val_df, val_images_path)

del train_df, val_df
gc.collect()




## === cell 3
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.2,
)

val_datagen = ImageDataGenerator()  # no augmentation for validation

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_generator = val_datagen.flow(
    X_val,
    y_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
)




## === cell 4
base_model = DenseNet121(
    weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers[:-20]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
output = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1,
)

train_generator.close()
val_generator.close()
gc.collect()




## === cell 5
test_ids = test_df["id_code"].values
test_images_path = os.path.join(BASE_DIR, "test_images")

test_predictions = np.empty(len(test_ids), dtype="int32")


def _load_test_image(img_id):
    img_path = os.path.join(test_images_path, f"{img_id}.png")
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return load_ben_color(img)


for start_idx in tqdm(range(0, len(test_ids), BATCH_SIZE), desc="Predicting"):
    end_idx = min(start_idx + BATCH_SIZE, len(test_ids))
    batch_ids = test_ids[start_idx:end_idx]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        batch_imgs = list(ex.map(_load_test_image, batch_ids))

    batch_array = np.stack(batch_imgs, axis=0)  # (batch, H, W, C)
    preds = model.predict(batch_array, verbose=0)
    expected = np.dot(preds, np.arange(5))
    pred_classes = np.rint(expected).astype(int)
    pred_classes = np.clip(pred_classes, 0, 4)
    test_predictions[start_idx:end_idx] = pred_classes




## === cell 6
submission_path = os.path.join(BASE_DIR, "sample_submission.csv")
submission = pd.read_csv(submission_path)
submission["diagnosis"] = test_predictions
submission.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_predictions, return_counts=True)
print(dict(zip(unique, counts)))
print("Submission file written to submission.csv")
