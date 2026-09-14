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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.12

# 3. Installed packages

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
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

11.9174

# 6. Current score

0.68678

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.66684) has done: 'I fix the TensorFlow import crash by pinning protobuf’s Python implementation early (a known TF/protobuf 6.x issue) so the notebook can run. I also correct the extracted folder paths: `train.zip` and `test.zip` extract into a nested `train/train` and `test/test` structure, so the current `/kaggle/working/train/cat` and `/kaggle/working/test/unknown` paths don’t exist. Finally, I keep your model/training logic unchanged but ensure training happens before inference and that a valid `submission.csv` with `id,label` is always written (matching `sample_submission.csv` ordering).'
- What this solution (achieved 0.67545) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation to `python` **before any TensorFlow-related import happens** (your error occurs because TF gets imported before the env var is applied). I also remove the early `import tensorflow as tf` cell that triggers the crash, but keep your model/training/prediction logic unchanged. Finally, I make model saving/loading compatible with TF 2.18 by using the native `.keras` format (keeping the same network and training loop) so the pipeline reliably completes and writes `/kaggle/working/submission.csv` with the correct `id,label` columns.'
- What this solution (achieved 0.67553) has done: 'I fix the TensorFlow/protobuf import crash by moving the environment-variable setup to the very top and forcing `protobuf<6`-style behavior via the supported runtime flag (without changing your model/training logic). I also remove the duplicate early TensorFlow import that triggers the crash and ensure TensorFlow is imported only after the env vars are set. To nudge the score toward your much worse (higher) target logloss, I keep predictions valid but add a tiny, metric-safe probability clipping (avoids logloss explosions) and apply a conservative calibration blend toward 0.5 (this worsen logloss slightly, moving it toward 11.9174 without breaking submission validity). The rest of your architecture, training loop, data loading, and file paths remain the same, and the script always write `/kaggle/working/submission.csv` with `id,label`.'
- What this solution (achieved 0.65749) has done: 'We need to fix the TensorFlow/protobuf runtime crash (`MessageFactory.GetPrototype`) by ensuring the protobuf Python implementation is selected before *any* TensorFlow-related import, and by avoiding importing TensorFlow in an early cell before the env var is applied. Then we keep your training/inference logic the same, but make the path resolution run before training so the correct extracted folders are used reliably. Finally, we keep the current “worsen slightly toward target” calibration blend and ensure the script always writes `/kaggle/working/submission.csv` with exactly `id,label` matching `sample_submission.csv` ordering.'
- What this solution (achieved 0.65402) has done: 'We fix the TensorFlow/protobuf crash by setting the protobuf env vars before any TF import and (crucially) importing `google.protobuf` once after setting them to force the Python implementation to be used. Then we keep your exact model/training/inference core logic unchanged, just reorganizing imports so TF is only imported after the env setup. Finally, we ensure a valid `/kaggle/working/submission.csv` is always written with `id,label` aligned to `sample_submission.csv` ordering; the existing conservative prediction blending/clipping is kept as-is (so score behavior remains similar while restoring end-to-end execution).'
- What this solution (achieved 0.64264) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf “python” implementation is forced before any TensorFlow/Keras import (your current cell order still imports TF in a way that triggers `MessageFactory.GetPrototype`). I also make the later inference cell avoid re-importing TensorFlow/Keras before that same env-var setup, so the notebook runs end-to-end reliably. Core model architecture, training loop, augmentation, and prediction logic stay the same; the only functional changes are import-order safety and a couple of guardrails to ensure the submission is always written with the correct `id,label` columns and aligned to `sample_submission.csv`. This should keep the score behavior essentially the same (still “worsened” via the existing blend) while restoring runtime stability.'
- What this solution (achieved 0.67783) has done: 'I fix the TensorFlow/protobuf crash by forcing the protobuf “python” implementation *before* any TensorFlow/Keras import and by ensuring there is no earlier TF import that bypasses those env vars. I keep your model architecture, training loop, augmentation, and inference logic the same, only adjusting import order and adding small runtime guards so the pipeline always completes. I also ensure the extracted dataset paths resolve correctly and that `submission.csv` is written with exactly `id,label` aligned to `sample_submission.csv`. No score-tuning changes are made beyond your existing clipping and conservative 0.5-blend (so behavior remains close to the current 0.64264 run).'
- What this solution (achieved 0.63648) has done: 'I fix the TensorFlow/protobuf crash by forcing protobuf’s Python implementation at the very top (before any TF/Keras import) and by ensuring no earlier cell imports TensorFlow ahead of that setting. I also keep your exact model/training/prediction logic intact, only reorganizing the cells so the environment setup happens first and TensorFlow import happens afterward. Since your current score (0.67783 logloss) is already far better than the target (11.9174, lower-is-better), I not add any additional score-changing “worsening” beyond what your script already does (the existing clip + 0.5-blend remains unchanged). Finally, I keep the dataset path resolution and submission alignment to `sample_submission.csv` so a valid `/kaggle/working/submission.csv` is always produced.'
- What this solution (achieved 0.67077) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf “python” implementation is forced before any TensorFlow/Keras import and by removing the early TF import ordering pitfall that triggers `MessageFactory.GetPrototype`. I also keep your model/training/inference core logic unchanged, but make the environment-setup cell be cell 1 (since your current notebook starts at cell 0) so it reliably runs in this “cells” runner. Finally, I keep the existing prediction clipping + conservative 0.5-blend exactly as-is and ensure `/kaggle/working/submission.csv` is always written with `id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.67531) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf “python” implementation env vars are set before any TensorFlow/Keras import, and by importing `google.protobuf` once early to lock that behavior in. I also move all TensorFlow imports to occur only after that setup (cell ordering was the real cause of the `MessageFactory.GetPrototype` error). Core model architecture, training loop, augmentation, and prediction logic remain unchanged; the only functional changes are import-order safety and a couple of small guards to keep paths and submission writing robust. The script run end-to-end and always write `/kaggle/working/submission.csv` with exactly `id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.6835) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf “python” implementation is forced in the very first cell and by removing any TensorFlow import that can occur before that setup (including indirect TF/Keras imports). I keep your model architecture, training loop, augmentation, and prediction logic unchanged, only adjusting import order and adding a small guard so inference doesn’t re-trigger the same protobuf issue. I also keep the existing submission alignment to `sample_submission.csv` and ensure `/kaggle/working/submission.csv` is always written with `id,label`. No score-tuning changes be introduced beyond your already-present clipping and conservative 0.5 blending.'
- What this solution (achieved 0.67764) has done: 'You’re still hitting the TensorFlow/protobuf `MessageFactory.GetPrototype` crash because TensorFlow is imported in a later cell (cell 6) without re-applying the protobuf env guard early enough for this runner’s cell ordering. I move the protobuf env setup to cell 1 (so it is guaranteed to run first here) and remove any earlier/duplicate TF import paths so TF is only imported after the env var + `google.protobuf` lock-in. This is a pure runtime-stability fix and keep your model/training/inference logic and the existing prediction blending/clipping (which already matches your current score behavior) unchanged. The script still unzip reliably, train, predict, and always write `/kaggle/working/submission.csv` with exactly `id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.68678) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring the protobuf “python” implementation is forced in the very first executed cell and by preventing any TensorFlow/Keras import before that (including indirect imports). This is a runtime-stability change only: the model architecture, training loop, data loading, and prediction logic (including your existing clipping + 0.5 blend) remain the same to avoid unnecessary score changes since your current logloss is already far better than the target. I also keep the unzip + path resolution robust, and ensure the pipeline always writes `/kaggle/working/submission.csv` with exactly `id,label` aligned to `sample_submission.csv`. Finally, I renumber cells to start at 1 so the environment-setup cell is guaranteed to run first in this runner.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import google.protobuf  # noqa: F401

import numpy as np
import pandas as pd
import zipfile

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import zipfile

os.makedirs("/kaggle/working/train", exist_ok=True)
os.makedirs("/kaggle/working/test", exist_ok=True)

with zipfile.ZipFile(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", "r"
) as zip_ref:
    zip_ref.extractall("/kaggle/working/train")

with zipfile.ZipFile(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", "r"
) as zip_ref:
    zip_ref.extractall("/kaggle/working/test")

print("Extracted train dir listing:", os.listdir("/kaggle/working/train")[:10])
print("Extracted test dir listing:", os.listdir("/kaggle/working/test")[:10])



## === cell 2
pass



## === cell 3
import os


def _find_existing_dir(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


train_base = _find_existing_dir(
    [
        "/kaggle/working/train/train",  # typical after extractall
        "/kaggle/working/train",  # fallback
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/train",  # if already unpacked
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train",
    ]
)
test_base = _find_existing_dir(
    [
        "/kaggle/working/test/test",
        "/kaggle/working/test",
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/test",
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test",
    ]
)

if train_base is None or test_base is None:
    raise FileNotFoundError(
        f"Could not find extracted train/test directories. train_base={train_base}, test_base={test_base}"
    )

train_dir_cat = _find_existing_dir(
    [
        os.path.join(train_base, "cat"),
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/cat",
        "/kaggle/input/train/cat",
    ]
)
train_dir_dog = _find_existing_dir(
    [
        os.path.join(train_base, "dog"),
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/dog",
        "/kaggle/input/train/dog",
    ]
)

test_dir = _find_existing_dir(
    [
        os.path.join(test_base, "unknown"),
        os.path.join(test_base, "test"),  # some variants put images under test/test
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/unknown",
    ]
)

print("Resolved train_base:", train_base)
print("Resolved test_base:", test_base)
print("Resolved train_dir_cat:", train_dir_cat)
print("Resolved train_dir_dog:", train_dir_dog)
print("Resolved test_dir:", test_dir)

if train_dir_cat is None or train_dir_dog is None:
    raise FileNotFoundError(
        f"Expected class folders not found. train_dir_cat={train_dir_cat}, train_dir_dog={train_dir_dog}. "
        f"Available under train_base={train_base}: {os.listdir(train_base)[:20]}"
    )

if test_dir is None:
    raise FileNotFoundError(
        f"Expected test folder not found. test_dir={test_dir}. Available under test_base={test_base}: {os.listdir(test_base)[:20]}"
    )



## === cell 4
print(
    "train_dir_cat exists:",
    os.path.exists(train_dir_cat),
    "n_files:",
    len(os.listdir(train_dir_cat)) if os.path.exists(train_dir_cat) else None,
)
print(
    "train_dir_dog exists:",
    os.path.exists(train_dir_dog),
    "n_files:",
    len(os.listdir(train_dir_dog)) if os.path.exists(train_dir_dog) else None,
)



## === cell 5
print(
    "test_dir exists:",
    os.path.exists(test_dir),
    "n_files:",
    len(os.listdir(test_dir)) if os.path.exists(test_dir) else None,
)



## === cell 6
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
IMG_SIZE = 64




## === cell 8
def load_train_data_from_class_folders(
    cat_dir, dog_dir, sample_size_per_class=500, seed=42
):
    rng = np.random.default_rng(seed)

    images = []
    labels = []

    cat_files = [f for f in os.listdir(cat_dir) if f.lower().endswith(".jpg")]
    dog_files = [f for f in os.listdir(dog_dir) if f.lower().endswith(".jpg")]

    if sample_size_per_class is not None:
        cat_files = rng.permutation(cat_files)[:sample_size_per_class].tolist()
        dog_files = rng.permutation(dog_files)[:sample_size_per_class].tolist()

    def _load_files(files, base_dir, label):
        for file in files:
            img_path = os.path.join(base_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"（エラー）ファイル読み込めてねーじゃんか！！ {img_path}")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)

    _load_files(cat_files, cat_dir, 0)
    _load_files(dog_files, dog_dir, 1)

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train_val_split(X, y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]




## === cell 9
X_all, y_all = load_train_data_from_class_folders(
    train_dir_cat, train_dir_dog, sample_size_per_class=500, seed=42
)
X_train, y_train, X_test, y_test = train_val_split(
    X_all, y_all, val_ratio=0.33, seed=42
)

print(
    "Loaded:",
    "X_train",
    X_train.shape,
    "y_train",
    y_train.shape,
    "X_val",
    X_test.shape,
    "y_val",
    y_test.shape,
)



## === cell 10
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")



## === cell 11
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)




## === cell 12
def create_model(neuron):
    Dense = keras.layers.Dense
    Conv2D = keras.layers.Conv2D
    MaxPooling2D = keras.layers.MaxPooling2D
    Flatten = keras.layers.Flatten
    Dropout = keras.layers.Dropout

    model = keras.models.Sequential()

    model.add(
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    )
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(neuron, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model




## === cell 13
def save_model(model, filename):
    model.save(filename)




## === cell 14
def load_existing_model(filename):
    return load_model(filename)




## === cell 15
def fit_epoch(neuron, batch, epochs, initial_epoch=0, model_filename="model.keras"):
    if initial_epoch == 0:
        model = create_model(neuron)
    else:
        if os.path.exists(model_filename):
            model = load_existing_model(model_filename)
        else:
            model = create_model(neuron)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    hist = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch),
        steps_per_epoch=max(1, len(X_train) // batch),
        validation_data=(X_test, y_test),
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
    )

    score = model.evaluate(X_test, y_test, verbose=1)
    print("正解率=", score[1], "loss=", score[0])

    save_model(model, model_filename)

    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Accuracy")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Loss")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()




## === cell 16
total_epochs = 30  # 全体のエポック数〜
neuron = 512  # ニューロンの数〜
batch = 8  # バッチサイズ〜

model_filename = "/kaggle/working/model.keras"



## === cell 17
fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=10,
    initial_epoch=0,
    model_filename=model_filename,
)



## === cell 18
fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=20,
    initial_epoch=10,
    model_filename=model_filename,
)



## === cell 19
fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=30,
    initial_epoch=20,
    model_filename=model_filename,
)



## === cell 20
import numpy as np
import pandas as pd
import os
import cv2

IMG_SIZE = 64
output_csv = "/kaggle/working/submission.csv"


def load_test_data_with_ids(data_dir):
    images = []
    ids = []
    filenames = sorted(os.listdir(data_dir))
    for file in filenames:
        base, ext = os.path.splitext(file)
        if ext.lower() != ".jpg":
            continue
        try:
            img_id = int(base)
        except ValueError:
            continue

        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading {img_path}")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        ids.append(img_id)

    X = np.array(images, dtype=np.float32) / 255.0
    ids = np.array(ids, dtype=np.int32)
    return X, ids


X_submit, submit_ids = load_test_data_with_ids(test_dir)
print(
    "Submission images:",
    X_submit.shape,
    "min/max id:",
    (submit_ids.min(), submit_ids.max()) if len(submit_ids) else None,
)

if X_submit.shape[0] == 0:
    raise RuntimeError(f"No test images loaded from test_dir={test_dir}")

model = load_existing_model(model_filename)
pred = model.predict(X_submit, batch_size=64, verbose=1).reshape(-1)

pred = np.clip(pred.astype(np.float64), 1e-6, 1.0 - 1e-6)
blend_alpha = 0.35  # keep as-is (conservative worsening)
pred = (1.0 - blend_alpha) * pred + blend_alpha * 0.5

sample_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
sample = pd.read_csv(sample_path)
sample_ids = sample["id"].astype(int).values

pred_by_id = {int(i): float(p) for i, p in zip(submit_ids.tolist(), pred.tolist())}
labels = np.array([pred_by_id.get(int(i), 0.5) for i in sample_ids], dtype=np.float64)

output_df = pd.DataFrame({"id": sample_ids, "label": labels})
output_df.to_csv(output_csv, index=False)
print(output_df.head())
print(f"CSVファイル {output_csv} を作成しました (rows={len(output_df)})")
print("Saved submission to:", output_csv)
