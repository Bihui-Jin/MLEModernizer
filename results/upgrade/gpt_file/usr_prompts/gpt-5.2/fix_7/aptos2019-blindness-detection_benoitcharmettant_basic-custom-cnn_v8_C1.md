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

0.65416

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.60914) has done: 'I fix the runtime/import errors caused by mixing legacy `keras` APIs with Keras 3 by switching to `tf_keras` equivalents and replacing removed `fit_generator/predict_generator` calls with `fit/predict`. I also make the data-path handling robust for the Kaggle filesystem you showed (so it finds `../input/aptos2019-blindness-detection/...` if `../input/train_images/` isn’t present). To move your score toward the target (your current score is far below), I keep the same CNN and training loop but change the final layer from `sigmoid` to `softmax` (consistent with `categorical_crossentropy`) and improve the inference preprocessing to match training (per-image standardization and correct id extraction), which should substantially improve QWK while preserving the core approach. Finally, I ensure a valid `submission.csv` with exactly `id_code,diagnosis` for all test rows, in the same order as `test.csv`.'
- What this solution (achieved 0.61305) has done: 'I fix the crash happening at import time (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by forcing TF/Keras to use the pure‑Python protobuf implementation before importing `tf_keras`, which is a common Kaggle environment incompatibility. I also make the submission creation more robust by ensuring predictions are generated for exactly the test set (in `test.csv` order) and that `diagnosis` is written as integers with the correct column names. These changes are execution/stability fixes and should be score-neutral (or very close) relative to your existing approach and model. The rest of the model/training logic is kept intact.'
- What this solution (achieved 0.6473) has done: 'I fix the import-time protobuf crash by setting the additional env flags that force the pure‑Python protobuf implementation and disable the C++ fast path before *any* TF/Keras import. I also remove the deprecated/unused generator prediction on the validation directory (it was using `steps=len(df_val)` which is incorrect and can hang/overrun) since it’s not used for submission and can waste time or fail. Finally, I keep the model/training/inference logic the same, but make the image standardization numerically safe (avoid division by zero) in both the generator and manual test preprocessing to prevent NaNs that can destabilize predictions.'
- What this solution (achieved 0.67087) has done: 'I fix the import-time protobuf crash by ensuring the pure-Python protobuf implementation is forced *before* any TensorFlow/Keras-related import and by removing an invalid env var that can still trigger the `MessageFactory.GetPrototype` path. I also add a small compatibility fallback to import `tf_keras` only after setting those flags (and fail fast with a clear message if it still can’t load), which is execution/stability focused and score-neutral. Since your current score (0.6473) is already well above the target (0.467311), I not change the model, training procedure, or prediction post-processing to avoid moving the score further away. The rest of the pipeline (paths, generators, training, inference, and writing `submission.csv`) is preserved.'
- What this solution (achieved 0.64355) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing the pure‑Python protobuf implementation *and* disabling the C++ implementation via the standard `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`/`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` pair before any TF/Keras import, and by ensuring we import `tf_keras` only after that. I keep the model/training/inference logic unchanged (score-neutral intent, since your current score is already above target), but I add a small path cleanup to avoid accidental reuse of stale resized directories across runs. Finally, I ensure the submission is always written as `submission.csv` with the exact required columns and row order from `test.csv`.'
- What this solution (achieved 0.65416) has done: 'I fix the import-time protobuf crash (`MessageFactory.GetPrototype`) by pinning protobuf to the pure‑Python implementation and forcing TF/Keras to use the Python protobuf backend before *any* protobuf/TF imports (including `skimage`), which is the root cause of the current failure. I also make the `tf_keras` import more robust by falling back to `tensorflow.keras` if needed, without changing the model/training/inference logic (so score should stay in the same ballpark and not be intentionally optimized further since you’re already above the target). Finally, I keep the exact submission format and row order from `test.csv` and ensure `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import shutil
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

try:
    import tf_keras as keras  # noqa: F401
except Exception:
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Failed to import both tf_keras and tensorflow.keras even after forcing "
            "pure-Python protobuf. Cannot continue."
        ) from e

from skimage.io import imread

DATA_PATH = "../input/"
ALT_DATA_PATH = os.path.join(DATA_PATH, "aptos2019-blindness-detection")


def pick_existing_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


TRAIN_CSV = pick_existing_path(
    os.path.join(DATA_PATH, "train.csv"), os.path.join(ALT_DATA_PATH, "train.csv")
)
TEST_CSV = pick_existing_path(
    os.path.join(DATA_PATH, "test.csv"), os.path.join(ALT_DATA_PATH, "test.csv")
)
SAMPLE_SUB = pick_existing_path(
    os.path.join(DATA_PATH, "sample_submission.csv"),
    os.path.join(ALT_DATA_PATH, "sample_submission.csv"),
)

TRAIN_IMG_DIR = pick_existing_path(
    os.path.join(DATA_PATH, "train_images"), os.path.join(ALT_DATA_PATH, "train_images")
)
TEST_IMG_DIR = pick_existing_path(
    os.path.join(DATA_PATH, "test_images"), os.path.join(ALT_DATA_PATH, "test_images")
)

print("Resolved paths:")
print("TRAIN_CSV:", TRAIN_CSV)
print("TEST_CSV :", TEST_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR :", TEST_IMG_DIR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_tile_dir = TRAIN_IMG_DIR

df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.png"))})
df["id"] = df["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])

labels = pd.read_csv(TRAIN_CSV)
labels = labels.rename(columns={"id_code": "id", "diagnosis": "label"})

df_data = df.merge(labels, on="id")
df_data.head()



## === cell 2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

SAMPLE_SIZE = 150

df_0 = df_data[df_data["label"] == 0].sample(SAMPLE_SIZE * 3, random_state=101)
df_1 = df_data[df_data["label"] == 1].sample(SAMPLE_SIZE * 2, random_state=101)
df_2 = df_data[df_data["label"] == 2].sample(SAMPLE_SIZE * 2, random_state=101)
df_3 = df_data[df_data["label"] == 3].sample(SAMPLE_SIZE, random_state=101)
df_4 = df_data[df_data["label"] == 4].sample(200, random_state=101)

df_data = shuffle(
    pd.concat([df_0, df_1, df_2, df_3, df_4], axis=0).reset_index(drop=True)
)

print(df_data.head())
print(df_data.label.value_counts())

y = df_data["label"]
df_train, df_val = train_test_split(
    df_data, test_size=0.10, random_state=101, stratify=y
)

train_path = "base_dir/train"
valid_path = "base_dir/valid"

if os.path.exists("base_dir"):
    shutil.rmtree("base_dir")

for fold in [train_path, valid_path]:
    for subf in ["0", "1", "2", "3", "4"]:
        os.makedirs(os.path.join(fold, subf), exist_ok=True)



## === cell 3
df_data.set_index("id", inplace=True)
df_data.head()



## === cell 4
IMAGE_SIZE = 96

for image_id in tqdm(df_train["id"].values, desc="Resizing train"):
    fname = image_id + ".png"
    label = str(df_data.loc[image_id, "label"])
    src = os.path.join(TRAIN_IMG_DIR, fname)
    dst = os.path.join(train_path, label, fname)

    pil_im = Image.open(src).convert("RGB")
    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))
    resized_image.save(dst)

for image_id in tqdm(df_val["id"].values, desc="Resizing valid"):
    fname = image_id + ".png"
    label = str(df_data.loc[image_id, "label"])
    src = os.path.join(TRAIN_IMG_DIR, fname)
    dst = os.path.join(valid_path, label, fname)

    pil_im = Image.open(src).convert("RGB")
    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))
    resized_image.save(dst)



## === cell 5
from tf_keras.preprocessing.image import ImageDataGenerator

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32
val_batch_size = 32

train_steps = int(np.ceil(num_train_samples / train_batch_size))
val_steps = int(np.ceil(num_val_samples / val_batch_size))


def safe_standardize(x):
    x = x.astype(np.float32, copy=False)
    s = x.std()
    if not np.isfinite(s) or s < 1e-6:
        return x
    return (x - x.mean()) / s


datagen = ImageDataGenerator(
    preprocessing_function=safe_standardize,
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

test_gen = datagen.flow_from_directory(
    valid_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
)



## === cell 6
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tf_keras.layers import Conv2D, MaxPool2D
from tf_keras.optimizers import Adam

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

model.add(Dense(5, activation="softmax"))

model.compile(Adam(0.01), loss="categorical_crossentropy", metrics=["accuracy"])
print("Done !")



## === cell 7
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
    verbose=1,
)



## === cell 8
print("Skip unused validation generator prediction to avoid step mismatch.")



## === cell 9
base_test_dir = TEST_IMG_DIR
test_files = glob(os.path.join(base_test_dir, "*.png"))

if os.path.exists("valid"):
    shutil.rmtree("valid")
os.makedirs("valid", exist_ok=True)

for image_path in tqdm(test_files, desc="Resizing test"):
    fname = os.path.basename(image_path)
    src = os.path.join(base_test_dir, fname)
    dst = os.path.join("valid", fname)

    pil_im = Image.open(src).convert("RGB")
    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))
    resized_image.save(dst)

test_files = sorted(glob(os.path.join("valid", "*.png")))

test_df_ids = pd.read_csv(TEST_CSV)
id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in test_files}

missing = [i for i in test_df_ids["id_code"].values if i not in id_to_path]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images after resizing. Example: {missing[:5]}"
    )

paths_in_order = [id_to_path[i] for i in test_df_ids["id_code"].values]

submission = pd.DataFrame({"id_code": test_df_ids["id_code"].values, "diagnosis": 0})

file_batch = 20
max_idx = len(paths_in_order)

for idx in range(0, max_idx, file_batch):
    batch_paths = paths_in_order[idx : idx + file_batch]
    batch_imgs = []
    for p in batch_paths:
        img = imread(p)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        batch_imgs.append(img)

    K_test = np.stack(batch_imgs).astype(np.float32)

    for i in range(K_test.shape[0]):
        s = K_test[i].std()
        if np.isfinite(s) and s >= 1e-6:
            K_test[i] = (K_test[i] - K_test[i].mean()) / s

    predictions = model.predict(K_test, verbose=0)
    pred_labels = np.argmax(predictions, axis=1).astype(int)

    submission.loc[idx : idx + len(pred_labels) - 1, "diagnosis"] = pred_labels

submission["diagnosis"] = submission["diagnosis"].astype(int).clip(0, 4)
submission.head()



## === cell 10
for p in [train_path, valid_path, "valid"]:
    if os.path.exists(p):
        shutil.rmtree(p)

submission.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 11
df_check = pd.read_csv("submission.csv")
print(df_check.head())
print(df_check.columns.tolist())
print(df_check["diagnosis"].value_counts(dropna=False))
print("Rows:", len(df_check))
