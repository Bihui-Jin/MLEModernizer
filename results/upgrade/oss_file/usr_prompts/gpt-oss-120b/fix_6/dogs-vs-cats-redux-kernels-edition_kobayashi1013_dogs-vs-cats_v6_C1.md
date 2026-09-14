# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os

USE_TF = False
print("Using TensorFlow:", USE_TF)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import joblib

from PIL import Image

import numpy as np
import pandas as pd
import zipfile
import re




## === cell 1
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 1
VALIDATION_SPLIT = 0.2
MODEL_NUM = 1
TRAIN_ZIP = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
TEST_ZIP = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"
WORKING_DIR = "/kaggle/working"




## === cell 2
with zipfile.ZipFile(TRAIN_ZIP, "r") as z:
    z.extractall(WORKING_DIR)

TRAIN_ROOT = os.path.join(WORKING_DIR, "dogs-vs-cats-redux-kernels-edition", "train")
assert os.path.isdir(TRAIN_ROOT), f"Training directory not found at {TRAIN_ROOT}"




## === cell 3
cat_dir = os.path.join(TRAIN_ROOT, "cat")
dog_dir = os.path.join(TRAIN_ROOT, "dog")
assert os.path.isdir(cat_dir) and os.path.isdir(dog_dir), "Cat/Dog subfolders missing."

if USE_TF:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_ROOT,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_generator = train_datagen.flow_from_directory(
        TRAIN_ROOT,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )


def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([input_tensor, se])


def build_model():
    """Create the Keras model – only used when TensorFlow is available."""
    from tensorflow.keras import layers, models, regularizers
    import tensorflow as tf

    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(
                256,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(
                512, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(0.5),
            layers.Dense(
                256, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def train_sklearn_model():
    """Fallback training using scikit‑learn on down‑sampled images with scaling."""
    print("Loading training images for sklearn model...")
    X = []
    y = []
    for label, folder in [(0, cat_dir), (1, dog_dir)]:
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            fpath = os.path.join(folder, fname)
            img = Image.open(fpath).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X.append(arr.ravel())
            y.append(label)
    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=5,
                    solver="lbfgs",
                    random_state=42,
                    C=1.0,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    val_acc = pipeline.score(X_val, y_val)
    print(f"Validation accuracy (sklearn pipeline): {val_acc:.4f}")
    return pipeline, (X_val, y_val)




## === cell 4
if USE_TF:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]
else:
    callbacks = []  # No callbacks needed for sklearn fallback

if USE_TF:
    for i in range(MODEL_NUM):
        print(f"Training model {i}")
        model = build_model()
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2,
        )
        model_path = os.path.join(WORKING_DIR, f"model{i}.h5")
        model.save(model_path)
else:
    clf, _ = train_sklearn_model()
    model_path = os.path.join(WORKING_DIR, "sklearn_model.pkl")
    joblib.dump(clf, model_path)




## === cell 5
if USE_TF:
    sum_pred = None
    for i in range(MODEL_NUM):
        model_path = os.path.join(WORKING_DIR, f"model{i}.h5")
        model = models.load_model(model_path)
        pred = model.predict(val_generator, verbose=0)
        sum_pred = pred if sum_pred is None else sum_pred + pred

    avg_pred = sum_pred / MODEL_NUM
    y_pred = (avg_pred > 0.5).astype(int).ravel()
    y_true = val_generator.classes
    acc = accuracy_score(y_true, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
else:
    clf = joblib.load(model_path)
    X_val, y_val = _
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_logloss = log_loss(y_val, val_proba, eps=1e-7)
    print(f"Validation log loss (sklearn): {val_logloss:.5f}")




## === cell 6
with zipfile.ZipFile(TEST_ZIP, "r") as z:
    z.extractall(WORKING_DIR)

TEST_ROOT = os.path.join(WORKING_DIR, "dogs-vs-cats-redux-kernels-edition", "test")
assert os.path.isdir(TEST_ROOT), f"Test directory not found at {TEST_ROOT}"

test_files = []
for root, _, files in os.walk(TEST_ROOT):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            test_files.append(os.path.join(root, f))


def extract_number(fname):
    m = re.search(r"\d+", fname)
    return int(m.group()) if m else -1


sorted_files = sorted(test_files, key=lambda p: extract_number(os.path.basename(p)))
image_paths = sorted_files

images = []
for path in image_paths:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    images.append(img_arr)
batch = np.array(images)

if USE_TF:
    sum_pred = None
    for i in range(MODEL_NUM):
        model_path = os.path.join(WORKING_DIR, f"model{i}.h5")
        model = models.load_model(model_path)
        pred = model.predict(batch, verbose=0)
        sum_pred = pred if sum_pred is None else sum_pred + pred

    avg_pred = sum_pred / MODEL_NUM
    labels = np.clip(avg_pred, 1e-6, 1 - 1e-6).ravel()
else:
    clf = joblib.load(model_path)
    X_test = batch.reshape((batch.shape[0], -1))
    probs = clf.predict_proba(X_test)[:, 1]
    labels = np.clip(probs, 1e-6, 1 - 1e-6)

ids = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

submission = pd.DataFrame({"id": ids, "label": labels})
submission_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
