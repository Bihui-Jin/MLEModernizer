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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

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

19.19287

# 6. Current score

0.69314

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.69314) has done: 'I fixed the broken file paths, added a safe search for the dataset folder, and guarded against empty image lists that caused the train‑test split to fail. The script now correctly locates the training, test, and sample‑submission files, loads the images, trains a tiny model (if TensorFlow is available), and writes a valid `submission.csv` with either model predictions or a 0.5 baseline.'
- What this solution (achieved 0.69314) has done: 'The update adds a protobuf compatibility flag before importing TensorFlow to stop the `MessageFactory` AttributeError, and keeps the import safely wrapped so that if TensorFlow cannot be loaded the script falls back to the baseline 0.5 predictions (which already gives a low log‑loss). No other logic is altered, preserving the original model and data handling while ensuring a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os, re, glob, pathlib
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        Dropout,
        Flatten,
        Conv2D,
        MaxPooling2D,
        Dense,
        Activation,
        BatchNormalization,
    )
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        ReduceLROnPlateau,
        EarlyStopping,
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception as e:
    tf = None
    print("TensorFlow import failed; proceeding with baseline predictions.", e)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def find_dataset_root():
    possible_roots = [
        pathlib.Path("/kaggle/input"),
        pathlib.Path("./input"),
        pathlib.Path("./data"),
        pathlib.Path("."),
    ]
    for root in possible_roots:
        candidate = root / "dogs-vs-cats-redux-kernels-edition"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Dataset root 'dogs-vs-cats-redux-kernels-edition' not found."
    )


DATA_ROOT = find_dataset_root()
TRAIN_ROOT = DATA_ROOT / "train"
TEST_ROOT = DATA_ROOT / "test" / "unknown"

print(f"Dataset root found at: {DATA_ROOT}")

train_images = sorted(
    glob.glob(os.path.join(TRAIN_ROOT, "**", "*.jpg"), recursive=True),
    key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)],
)
test_images = sorted(
    glob.glob(os.path.join(TEST_ROOT, "**", "*.jpg"), recursive=True),
    key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)],
)

print(f"Found {len(train_images)} training images, {len(test_images)} test images.")

ROWS, COLS, CHANNELS = 150, 150, 3


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    return img


def prep_data(images, label_from_path=False, max_images=None):
    X, y = [], []
    for i, image_file in enumerate(tqdm(images, desc="Reading images")):
        if max_images is not None and i >= max_images:
            break
        image = read_image(image_file)
        X.append(image)
        if label_from_path:
            y.append(1 if "dog" in os.path.basename(image_file).lower() else 0)
        else:
            y.append(0)  # placeholder
    return np.array(X), np.array(y)


print("Processing test images")
X_test_raw, _ = prep_data(test_images, label_from_path=False)
print(f"Test : {len(X_test_raw)} images, shape {X_test_raw.shape[1:]}")

if tf is not None:
    print("Processing training images")
    X_all, y_all = prep_data(train_images, label_from_path=True, max_images=8000)
    print(f"Train: {len(X_all)} images, shape {X_all.shape[1:]}")




## === cell 2
def locate_sample_submission():
    usual = DATA_ROOT / "sample_submission.csv"
    if usual.is_file():
        return usual
    matches = list(pathlib.Path(".").rglob("sample_submission.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError("sample_submission.csv not found.")


sample_submission_path = locate_sample_submission()
sample_df = pd.read_csv(sample_submission_path, dtype={"id": str})
submission_ids = sample_df["id"].tolist()

baseline_pred = 0.5
submission_df = pd.DataFrame(
    {"id": submission_ids, "label": [baseline_pred] * len(submission_ids)}
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Baseline submission file written to {submission_path}")




## === cell 3
if tf is not None and "X_all" in globals() and len(X_all) > 0:
    X_all = X_all.astype("float32") / 255.0
    X_test_raw = X_test_raw.astype("float32") / 255.0

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=1, stratify=y_all
    )

    batch_size = 100
    epochs = 2

    train_datagen = ImageDataGenerator(rescale=1.0)
    val_datagen = ImageDataGenerator(rescale=1.0)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    validation_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    def build_model(N_Filters=32):
        input_layer = Input(shape=(ROWS, COLS, CHANNELS), name="InputLayer")
        x = Conv2D(N_Filters, (3, 3), padding="same", activation="relu")(input_layer)
        x = Conv2D(N_Filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(N_Filters * 2, (3, 3), padding="valid", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(N_Filters * 4, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(N_Filters * 8, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(N_Filters * 8, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(N_Filters * 8, activation="relu")(x)
        x = Dropout(0.5)(x)

        output = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=RMSprop(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    model = build_model()

    check_point = ModelCheckpoint(
        "BestModel.keras",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )
    lr_reduce = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.1, min_delta=0.0001, patience=3, verbose=1
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(X_val) // batch_size,
        epochs=epochs,
        callbacks=[lr_reduce, check_point],
    )

    preds = model.predict(X_test_raw, batch_size=batch_size, verbose=0).flatten()
    submission_df["label"] = preds
    submission_df.to_csv(submission_path, index=False)
    print("Submission file updated with model predictions.")
else:
    print(
        "TensorFlow not available or no training data – baseline submission already written."
    )
