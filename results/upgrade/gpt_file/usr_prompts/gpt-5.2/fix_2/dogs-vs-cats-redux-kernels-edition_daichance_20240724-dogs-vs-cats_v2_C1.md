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

13.47586

# 6. Current score

0.67962

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.67962) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure‑Python protobuf implementation before importing TensorFlow (this is a common Kaggle TF2.18+ issue). Then I correct the extracted dataset paths by locating the actual train/test image folders after unzip (your code currently points to non-existent `/kaggle/working/train/train` and `/kaggle/working/test/test`). Finally, I generate a valid `submission.csv` with the required `id,label` columns using probabilities (not hard 0/1 thresholds) and ensure test IDs are sorted numerically to align with Kaggle’s expected format.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import zipfile

with zipfile.ZipFile(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", "r"
) as zip_ref:
    zip_ref.extractall("/kaggle/working/train")

with zipfile.ZipFile(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", "r"
) as zip_ref:
    zip_ref.extractall("/kaggle/working/test")



## === cell 2
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import shutil
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import csv

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
from pathlib import Path


def find_train_test_dirs():
    train_root = Path("/kaggle/working/train")
    test_root = Path("/kaggle/working/test")

    train_candidates = [
        train_root / "train" / "train",
        train_root / "train",
        train_root,
    ]
    test_candidates = [
        test_root / "test" / "test",
        test_root / "test",
        test_root,
    ]

    def pick_dir(cands, must_have_jpg=True):
        for d in cands:
            if d.exists() and d.is_dir():
                if not must_have_jpg:
                    return str(d)
                if any(p.suffix.lower() == ".jpg" for p in d.iterdir()):
                    return str(d)
        base = cands[-1]
        for d in base.rglob("*"):
            if d.is_dir() and any(p.suffix.lower() == ".jpg" for p in d.iterdir()):
                return str(d)
        raise FileNotFoundError(f"Could not find image directory under {base}")

    return pick_dir(train_candidates), pick_dir(test_candidates)


train_dir, test_dir = find_train_test_dirs()
print("Resolved train_dir:", train_dir)
print("Resolved test_dir:", test_dir)



## === cell 4
IMG_SIZE = 64




## === cell 5
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
            print(f"（エラー）ファイル読み込み不可 {img_path}")
    return np.array(images, dtype=np.float32) / 255.0, np.array(labels, dtype=np.int32)




## === cell 6
X_train, y_train = load_data(train_dir, sample_size=1000)
X_test, y_test = load_data(
    train_dir, sample_size=200
)  # validation from train, not test folder
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_test.shape, "y_val:", y_test.shape)



## === cell 7
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)




## === cell 8
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




## === cell 9
def save_model(model, filename):
    model.save(filename)




## === cell 10
def load_existing_model(filename):
    return load_model(filename)




## === cell 11
def fit_epoch(neuron, batch, epochs, initial_epoch=0, model_filename="model.h5"):
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
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Loss")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()




## === cell 12
neuron = 512
batch = 8
model_filename = "/kaggle/working/model.h5"



## === cell 13
fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=10,
    initial_epoch=0,
    model_filename=model_filename,
)



## === cell 14
import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras.models import load_model

output_csv = "/kaggle/working/submission.csv"


def load_test_data(data_dir):
    filenames = [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
    filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
    images = []
    for file in filenames:
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
        else:
            print(f"Error reading {img_path}")
    return np.array(images, dtype=np.float32) / 255.0, filenames


X_submit, test_filenames = load_test_data(test_dir)
model = load_model(model_filename)

pred = model.predict(X_submit, verbose=0).reshape(-1)

pred = np.clip(pred, 1e-6, 1 - 1e-6)

submission = pd.DataFrame(
    {"id": [int(os.path.splitext(f)[0]) for f in test_filenames], "label": pred}
)
submission.to_csv(output_csv, index=False)
print(f"Submission saved to: {output_csv}")
print(submission.head())
print("Rows:", len(submission), "Columns:", submission.columns.tolist())
