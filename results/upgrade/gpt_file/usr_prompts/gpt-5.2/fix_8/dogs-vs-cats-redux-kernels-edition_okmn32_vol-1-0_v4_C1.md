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

14.10584

# 6. Current score

8.00747

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 8.00747) has done: 'I fix the two root causes preventing end-to-end execution: (1) the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, and (2) incorrect extracted folder paths by dynamically resolving where `train.zip` and `test.zip` unpacked. Then I ensure training variables (`X_train2`, etc.) are created before calling `fit_epoch`, and ensure the inference step points to the correct test image directory and always writes a valid `submission.csv` with `id,label`. These changes keep your core CNN, augmentation, and training loop intact while making the pipeline run reliably in the Kaggle environment. Since no valid score was produced yet, the focus is correctness and generating a valid submission file.'
- What this solution (achieved 8.00747) has done: 'I fix the TensorFlow import crash (`MessageFactory`/protobuf mismatch) by setting a safe protobuf implementation before importing TensorFlow, which is the direct cause of the runtime error. I also ensure paths and extraction stay robust as you already intended, and keep your CNN/training loop unchanged. Because your current score (8.00747, lower-is-better) is already much better than the target (14.10584), I won’t make any model/training changes that could further improve the score; the goal here is correctness and stable end-to-end submission creation. Finally, I keep the submission format exactly `id,label` and ensure IDs are sorted and written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 8.00747) has done: 'I fix the TensorFlow/protobuf import crash by removing the environment override that forces the pure-Python protobuf implementation, which is what triggers the `MessageFactory.GetPrototype` error in this Kaggle environment. I also make the zip extraction path resolution more robust so we consistently train on the extracted `train/` JPGs and infer on the extracted numeric `test/` JPGs, without changing your CNN/training loop. Since your current score (8.00747, lower-is-better) is already much better than the target (14.10584), I not make any model/training changes that could further improve it; changes are limited to stability/correctness and guaranteed submission writing. Finally, I ensure the produced `/kaggle/working/submission.csv` always has exactly the required `id,label` columns with sorted integer ids.'
- What this solution (achieved 8.00747) has done: 'I fix the TensorFlow/protobuf crash by ensuring we do not force the pure-Python protobuf runtime and by importing TensorFlow in a clean environment (this is the direct cause of the `MessageFactory.GetPrototype` error). I also make the zip extraction and directory resolution robust so `train_dir` points to the folder with `cat.*.jpg`/`dog.*.jpg` and `test_dir` points to numeric-id JPGs, without changing your CNN, augmentation, or training loop. Since your current score (8.00747) is already better than the target (14.10584) for a lower-is-better metric, I avoid any model/training changes that could improve the score further and focus on stability and correct submission creation. Finally, I guarantee a valid `/kaggle/working/submission.csv` with exactly `id,label`, sorted by integer id, and with safe probability clipping.'
- What this solution (achieved 8.00747) has done: 'I fix the TensorFlow/protobuf import crash by not forcing the `cpp` protobuf implementation (this environment can’t import `google.protobuf.pyext._message`), and instead forcing the pure-Python implementation before importing TensorFlow. Then I ensure `tensorflow.keras` symbols (`keras`, `ImageDataGenerator`, `load_model`) are always defined by importing TensorFlow successfully once and reusing it across cells, which resolves the downstream `NameError`s. Finally, I keep your CNN, augmentation, and training loop intact, and make sure we always write a valid `/kaggle/working/submission.csv` with `id,label` sorted by id (score-neutral, correctness-only).'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import os as _os

for dirname, _, filenames in _os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(_os.path.join(dirname, filename))
print("... (listing truncated)")



## === cell 1
import zipfile
import os

os.makedirs("/kaggle/working/train", exist_ok=True)
os.makedirs("/kaggle/working/test", exist_ok=True)


def _has_jpgs(root):
    for dpath, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(".jpg"):
                return True
    return False


train_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"

if not _has_jpgs("/kaggle/working/train"):
    with zipfile.ZipFile(train_zip, "r") as zip_ref:
        zip_ref.extractall("/kaggle/working/train")

if not _has_jpgs("/kaggle/working/test"):
    with zipfile.ZipFile(test_zip, "r") as zip_ref:
        zip_ref.extractall("/kaggle/working/test")

print("Extracted train contents sample:", os.listdir("/kaggle/working/train")[:10])
print("Extracted test contents sample:", os.listdir("/kaggle/working/test")[:10])



## === cell 2
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def resolve_image_dir(root_dir, expected_numeric=False):
    """
    Find a directory under root_dir that contains image files.
    If expected_numeric=True, prefer directories containing numeric .jpg filenames.
    """
    candidates = []
    for dpath, dnames, fnames in os.walk(root_dir):
        jpgs = [f for f in fnames if f.lower().endswith(".jpg")]
        if not jpgs:
            continue
        if expected_numeric:
            numeric_cnt = sum(os.path.splitext(f)[0].isdigit() for f in jpgs)
            candidates.append((numeric_cnt, len(jpgs), dpath))
        else:
            candidates.append((len(jpgs), dpath))
    if not candidates:
        raise FileNotFoundError(f"No .jpg files found under: {root_dir}")

    if expected_numeric:
        candidates.sort(reverse=True)  # max numeric_cnt, then max total
        best = candidates[0][2]
    else:
        candidates.sort(reverse=True)  # max jpg count
        best = candidates[0][1]
    return best


train_dir = resolve_image_dir("/kaggle/working/train", expected_numeric=False)
test_dir = resolve_image_dir("/kaggle/working/test", expected_numeric=True)

print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)
print(
    "Train files:",
    len([f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")]),
)
print(
    "Test files :",
    len([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]),
)



## === cell 4
IMG_SIZE = 64




## === cell 5
def load_data(data_dir, sample_size=1000):
    images = []
    labels = []
    files = [f for f in sorted(os.listdir(data_dir)) if f.lower().endswith(".jpg")][
        :sample_size
    ]

    for file in files:
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            label = 1 if "dog" in file else 0
            labels.append(label)
        else:
            print(f"error {img_path}")
    return np.array(images, dtype=np.float32) / 255.0, np.array(
        labels, dtype=np.float32
    )




## === cell 6
X_train, y_train = load_data(train_dir, sample_size=1000)

val_size = 200
X_val = X_train[:val_size]
y_val = y_train[:val_size]
X_train2 = X_train[val_size:]
y_train2 = y_train[val_size:]

print("X_train2:", X_train2.shape, "X_val:", X_val.shape, "y_train2:", y_train2.shape)



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
def fit_epoch(neuron, batch, epochs, initial_epoch=0, model_filename="model.keras"):
    if initial_epoch == 0 or (not os.path.exists(model_filename)):
        model = create_model(neuron)
        initial_epoch = 0
    else:
        model = load_existing_model(model_filename)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    hist = model.fit(
        datagen.flow(X_train2, y_train2, batch_size=batch, shuffle=True),
        steps_per_epoch=max(1, len(X_train2) // batch),
        validation_data=(X_val, y_val),
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
    )
    score = model.evaluate(X_val, y_val, verbose=1)
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




## === cell 12
total_epochs = 30
neuron = 512
batch = 8
model_filename = "/kaggle/working/model.keras"



## === cell 13
fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=10,
    initial_epoch=0,
    model_filename=model_filename,
)



## === cell 14
fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=20,
    initial_epoch=10,
    model_filename=model_filename,
)



## === cell 15
fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=30,
    initial_epoch=20,
    model_filename=model_filename,
)



## === cell 16
import numpy as np
import pandas as pd
import os
import cv2

IMG_SIZE = 64
model_filename = "/kaggle/working/model.keras"
output_csv = "/kaggle/working/submission.csv"


def load_test_data(data_dir):
    images = []
    ids = []
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
    try:
        filenames = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        filenames = sorted(files)

    for file in filenames:
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            stem = os.path.splitext(file)[0]
            ids.append(int(stem) if stem.isdigit() else stem)
        else:
            print(f"Error reading {img_path}")

    X = np.array(images, dtype=np.float32) / 255.0
    return X, np.array(ids)


X_test_submit, test_ids = load_test_data(test_dir)

model = load_model(model_filename)

predictions = model.predict(X_test_submit, batch_size=32, verbose=1).reshape(-1)

predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

output_df = pd.DataFrame({"id": test_ids.astype(int), "label": predictions})
output_df = output_df.sort_values("id")
output_df.to_csv(output_csv, index=False)

print(f"CSVファイル {output_csv} を作成しました")
print(output_df.head())
print("Rows:", len(output_df))
print(
    "Saved:",
    os.path.exists(output_csv),
    "Size:",
    os.path.getsize(output_csv) if os.path.exists(output_csv) else None,
)
