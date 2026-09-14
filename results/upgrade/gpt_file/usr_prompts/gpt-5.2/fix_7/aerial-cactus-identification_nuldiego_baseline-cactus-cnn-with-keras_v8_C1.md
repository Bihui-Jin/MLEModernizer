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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

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
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.5068

# 6. Current score

0.99733

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99623) has done: 'I fix the runtime errors caused by mixing Keras 3 APIs with deprecated `keras.preprocessing` / `fit_generator` / `predict_generator`, and I point the image directories to the actual competition folder structure under `/kaggle/input/aerial-cactus-identification/`. I keep your CNN architecture and overall training approach the same, but switch the data pipeline to `tf.data` + `tf.io` so it works reliably in this environment. I also fix the plotting keys (`accuracy` vs `acc`) and ensure the submission is written with the correct columns and a `.csv` suffix.'
- What this solution (achieved 0.99738) has done: 'I fix the runtime crash that happens immediately on importing TensorFlow/Keras (the `MessageFactory.GetPrototype` protobuf incompatibility), because it prevents any training or submission generation. The smallest stable fix in Kaggle is to force the pure-Python protobuf implementation before importing TensorFlow, which avoids that specific C++ protobuf API mismatch. Since your current score (0.99623) is already far above the target (0.5068) and within the allowed “move toward target” direction (you’re better than target), I not change model/training behavior; the patch is intended to be score-neutral and only restore end-to-end execution and CSV output. I also keep your paths and submission format unchanged.'
- What this solution (achieved 0.99551) has done: 'I fix the immediate runtime crash caused by a protobuf/TensorFlow incompatibility by forcing the pure-Python protobuf implementation and ensuring it is applied before TensorFlow is imported. To make this robust in Kaggle, I also set the related `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` and clear any pre-imported `google.protobuf` modules (so the setting actually takes effect). These changes are score-neutral (they don’t alter your model, training loop, data pipeline, or inference) and are only to restore end-to-end execution so a valid `.csv` submission is written. Everything else (architecture, tf.data loading, epochs, and submission format/path) remain the same.'
- What this solution (achieved 0.9955) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by ensuring TensorFlow is imported before `tf_keras` (and by keeping the protobuf environment override applied before any TF/protobuf-related import). This is a minimal, score-neutral change that restores end-to-end execution and submission creation. I also keep the existing model, training loop, tf.data pipeline, and submission formatting unchanged to avoid moving the score further away from your target. Finally, I keep paths as-is and ensure the submission is written as a `.csv` with the required columns.'
- What this solution (achieved 0.99722) has done: 'I fix the protobuf/TensorFlow import crash by forcing TensorFlow to use the pure-Python protobuf implementation and (critically) making sure no protobuf modules are imported before that setting takes effect. This is the earliest blocker preventing the notebook from running end-to-end and writing a submission. I keep the model, data pipeline, training loop, and prediction logic unchanged to avoid moving your already-above-target score further away from the target. The output still be a valid `submission_baseline.csv` with the required `id,has_cactus` columns.'
- What this solution (achieved 0.99733) has done: 'I fix the protobuf/TensorFlow import crash that prevents the notebook from running by adding a safe, minimal fallback: try importing `tensorflow`/`tf_keras`, and if the `MessageFactory.GetPrototype` error occurs, forcibly downgrade `protobuf` to a compatible version via `pip` and restart the Python process once. This change is purely to unblock execution and should be score-neutral (no model/data/training logic changes). I also keep the original paths, tf.data pipeline, training loop, and submission formatting intact so it still writes a valid `submission_baseline.csv` with `id,has_cactus`. Finally, I keep seeding as-is for determinism.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")


def _import_tf_stack():
    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            sys.modules.pop(m, None)

    import numpy as np
    import pandas as pd

    import tensorflow as tf
    import tf_keras as keras
    from tf_keras import layers, models

    return np, pd, tf, keras, layers, models


try:
    np, pd, tf, keras, layers, models = _import_tf_stack()
except AttributeError as e:
    if "MessageFactory" in str(e) and "GetPrototype" in str(e):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                "protobuf==4.25.3",
            ]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
    raise

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
]
DATA_ROOT = next(
    (p for p in DATA_ROOT_CANDIDATES if os.path.exists(p)),
    "/kaggle/input/aerial-cactus-identification",
)

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
train_df["has_cactus"] = train_df["has_cactus"].astype(int)

validation_df = train_df.sample(n=int(0.4 * len(train_df)), random_state=SEED)
train_df = train_df[~train_df["id"].isin(validation_df["id"])].reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)

print("Train size:", len(train_df))
print("Valid size:", len(validation_df))
print("Validation label distribution:\n", validation_df["has_cactus"].value_counts())



## === cell 2
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.build()
model.summary()



## === cell 3
IMG_SIZE = (32, 32)
BATCH_SIZE = 20
AUTOTUNE = tf.data.AUTOTUNE


def _load_image_and_label(filename, label=None):
    img_bytes = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, (1,))
    return img, label


def make_dataset(df, images_dir, training=True):
    filepaths = tf.constant([os.path.join(images_dir, x) for x in df["id"].values])
    labels = tf.constant(df["has_cactus"].values, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(lambda p, y: _load_image_and_label(p, y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, TRAIN_DIR, training=True)
valid_ds = make_dataset(validation_df, TRAIN_DIR, training=False)



## === cell 4
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 5
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=valid_ds,
    verbose=2,
)



## === cell 6
import matplotlib.pyplot as plt


def plot_history(history):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if len(acc) > 0:
        plt.plot(epochs, acc, "b-", label="Training acc")
    if len(val_acc) > 0:
        plt.plot(epochs, val_acc, "r-", label="Validation acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "b-", label="Training loss")
    if len(val_loss) > 0:
        plt.plot(epochs, val_loss, "r-", label="Validation loss")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_history(history)



## === cell 7
sub_df = pd.read_csv(SAMPLE_SUB)
test_ids = sub_df["id"].values
test_filepaths = tf.constant([os.path.join(TEST_DIR, x) for x in test_ids])

test_ds = tf.data.Dataset.from_tensor_slices(test_filepaths)
test_ds = test_ds.map(_load_image_and_label, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(64).prefetch(AUTOTUNE)

y_pred = model.predict(test_ds, verbose=0).reshape(-1)
y_pred = np.clip(y_pred, 0.0, 1.0)

submission = pd.DataFrame({"id": test_ids, "has_cactus": y_pred})
submission_path = "submission_baseline.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())



## === cell 8
submission["has_cactus"].describe()
