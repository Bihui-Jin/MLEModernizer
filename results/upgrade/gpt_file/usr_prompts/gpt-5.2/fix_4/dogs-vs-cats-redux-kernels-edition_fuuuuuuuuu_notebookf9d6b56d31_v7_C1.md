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

12.75741

# 6. Current score

0.65638

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.64817) has done: 'I fix the root cause of the FileNotFoundError by making the train/test directory discovery robust to the actual zip extraction structure, so `load_data()` and `load_test_data()` can always find images. I also ensure the training split cell runs before any training calls so `X_train/y_train` exist, eliminating the downstream NameErrors without changing the model/training logic. Finally, I guarantee a valid `submission.csv` is written with the required `id,label` columns by aligning predictions to `sample_submission.csv` and filling any missing ids with 0.5.'
- What this solution (achieved 0.65638) has done: 'The crash comes from a protobuf/TensorFlow incompatibility triggered by the protobuf “MessageFactory.GetPrototype” API change; to keep the core training/prediction logic unchanged, I fix it by forcing the pure-Python protobuf implementation before importing TensorFlow. I also make the zip extraction idempotent and ensure we always locate the correct flat `train/` and `test/` image folders, so data loading can’t silently point at the wrong nested directory. Finally, I keep your existing model/training exactly as-is but make submission generation robust: always align to `sample_submission.csv`, ensure `id` parsing works, and write `/kaggle/working/submission.csv` with the required `id,label` columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import zipfile
import os

os.makedirs("/kaggle/working/train", exist_ok=True)
os.makedirs("/kaggle/working/test", exist_ok=True)

train_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"

if len(os.listdir("/kaggle/working/train")) == 0:
    with zipfile.ZipFile(train_zip, "r") as zip_ref:
        zip_ref.extractall("/kaggle/working/train")

if len(os.listdir("/kaggle/working/test")) == 0:
    with zipfile.ZipFile(test_zip, "r") as zip_ref:
        zip_ref.extractall("/kaggle/working/test")

print("Extracted train dirs:", os.listdir("/kaggle/working/train")[:10])
print("Extracted test dirs:", os.listdir("/kaggle/working/test")[:10])



## === cell 2
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _ensure_protobuf_compat():
    try:
        import google.protobuf
        from google.protobuf import __version__ as pb_ver

        print(
            "protobuf version:",
            pb_ver,
            "implementation:",
            os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
        )
    except Exception as e:
        print("Warning: protobuf import issue:", e)


_ensure_protobuf_compat()



## === cell 3
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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
def _find_dir_with_jpgs(root, must_contain_subdirs=None):
    """
    Find the first directory under `root` that contains .jpg files.
    If must_contain_subdirs is provided (list), require those subdirs exist under the directory.
    """
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        if must_contain_subdirs is not None:
            ok = all(
                os.path.isdir(os.path.join(dirpath, d)) for d in must_contain_subdirs
            )
            if not ok:
                continue
        if any(f.lower().endswith(".jpg") for f in filenames):
            return dirpath
    return None


preferred_train = "/kaggle/working/train/train"
preferred_test = "/kaggle/working/test/test"

train_dir = (
    preferred_train
    if os.path.isdir(preferred_train)
    else _find_dir_with_jpgs("/kaggle/working/train")
)
test_dir = (
    preferred_test
    if os.path.isdir(preferred_test)
    else _find_dir_with_jpgs("/kaggle/working/test")
)

print(
    "Using train_dir:",
    train_dir,
    "exists:",
    os.path.isdir(train_dir) if train_dir else None,
)
print(
    "Using test_dir:",
    test_dir,
    "exists:",
    os.path.isdir(test_dir) if test_dir else None,
)

if train_dir is None:
    raise FileNotFoundError(
        "Could not locate any training .jpg files under /kaggle/working/train"
    )
if test_dir is None:
    raise FileNotFoundError(
        "Could not locate any test .jpg files under /kaggle/working/test"
    )



## === cell 5
IMG_SIZE = 64




## === cell 6
def load_data(data_dir, sample_size=1000):
    images = []
    labels = []
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
    files = files[:sample_size]
    for file in files:
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            label = 1 if "dog" in file.lower() else 0
            labels.append(label)
        else:
            print(f"error {img_path}")
    return np.array(images, dtype=np.float32) / 255.0, np.array(labels, dtype=np.int64)


def load_test_data(data_dir):
    images = []
    filenames = [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
    filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
    for file in filenames:
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
        else:
            print(f"Error reading {img_path}")
    return np.array(images, dtype=np.float32) / 255.0, filenames




## === cell 7
X_all, y_all = load_data(train_dir, sample_size=1000)

rng = np.random.RandomState(42)
idx = np.arange(len(X_all))
rng.shuffle(idx)
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[val_idx], y_all[val_idx]

print("Train:", X_train.shape, y_train.shape, "Val:", X_test.shape, y_test.shape)



## === cell 8
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)




## === cell 9
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




## === cell 10
def save_model(model, filename):
    model.save(filename)




## === cell 11
def load_existing_model(filename):
    return load_model(filename)




## === cell 12
def fit_epoch(neuron, batch, epochs, initial_epoch=0, model_filename="model.keras"):
    if initial_epoch == 0 or (not os.path.exists(model_filename)):
        model = create_model(neuron)
    else:
        model = load_existing_model(model_filename)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    steps = max(1, len(X_train) // batch)
    hist = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch, shuffle=True),
        steps_per_epoch=steps,
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
    plt.legend(["train", "val"], loc="upper left")
    plt.show()

    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Loss")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()

    return model




## === cell 13
neuron = 512
batch = 8
model_filename = "/kaggle/working/model.keras"



## === cell 14
model = fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=10,
    initial_epoch=0,
    model_filename=model_filename,
)



## === cell 15
model = fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=20,
    initial_epoch=10,
    model_filename=model_filename,
)



## === cell 16
model = fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=30,
    initial_epoch=20,
    model_filename=model_filename,
)



## === cell 17
import pandas as pd
import numpy as np
import os

output_csv = "/kaggle/working/submission.csv"

X_submit, submit_filenames = load_test_data(test_dir)

model = load_model(model_filename)
pred = model.predict(X_submit, verbose=1).reshape(-1)

pred = np.clip(pred, 1e-7, 1 - 1e-7)

sub = (
    pd.DataFrame(
        {
            "id": [int(os.path.splitext(f)[0]) for f in submit_filenames],
            "label": pred.astype(np.float64),
        }
    )
    .sort_values("id")
    .reset_index(drop=True)
)

sample_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"

if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    sub = sample[["id"]].merge(sub, on="id", how="left")
    sub["label"] = sub["label"].fillna(0.5)

sub.to_csv(output_csv, index=False)
print(f"CSVファイル {output_csv} を作成しました")
print(sub.head())
print("Rows:", len(sub), "Cols:", list(sub.columns))
print("label range:", float(sub["label"].min()), float(sub["label"].max()))
