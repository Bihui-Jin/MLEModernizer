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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.4861

# 6. Current score

0.99938

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99911) has done: 'I fixed the protobuf import issue, corrected TensorFlow‑API usage errors (weight‑file naming, metric keys, and deprecated functions), streamlined the data pipelines, and rewrote the test‑set prediction so it creates a proper `submission.csv` with the required `id,has_cactus` columns. The model architecture and training approach remain unchanged, only the bugs that prevented execution were addressed, enabling an end‑to‑end run that generate a valid submission file.'
- What this solution (achieved 0.99938) has done: 'I added two environment variables before importing TensorFlow to avoid the protobuf `GetPrototype` error, and made the test‑image path construction robust by trying both possible input locations. These minimal changes let the notebook run end‑to‑end and still produce the same high‑performing model and a correct `submission.csv` file.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_C_EXTENSION"] = "1"

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from IPython.display import Image, display



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print(tf.__version__)



## === cell 2
train_csv = pd.read_csv("../input/train.csv")
print(train_csv.describe())
print(train_csv.head())



## === cell 3
try:
    display(Image("../input/train/train/0004be2cfeaba1c0361d39e2b000257b.jpg"))
    display(Image("../input/train/train/000c8a36845c0208e833c79c1bffedd1.jpg"))
except Exception:
    pass



## === cell 4
filenames = ["../input/train/train/" + fname for fname in train_csv["id"].tolist()]
labels = train_csv["has_cactus"].astype(np.float32).tolist()

train_filenames, val_filenames, train_labels, val_labels = train_test_split(
    filenames, labels, train_size=0.9, random_state=420, stratify=labels
)

size_train = len(train_filenames)
size_val = len(val_filenames)



## === cell 5
IMAGE_SIZE = 96
BATCH_SIZE = 32


def _parse_fn(filename, label):
    image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename))
    image_normalized = (tf.cast(image_decoded, tf.float32) / 127.5) - 1.0
    image_resized = tf.image.resize(image_normalized, (IMAGE_SIZE, IMAGE_SIZE))
    return image_resized, tf.cast(label, tf.float32)




## === cell 6
train_data = (
    tf.data.Dataset.from_tensor_slices(
        (tf.constant(train_filenames), tf.constant(train_labels))
    )
    .map(_parse_fn)
    .shuffle(buffer_size=10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_data = (
    tf.data.Dataset.from_tensor_slices(
        (tf.constant(val_filenames), tf.constant(val_labels))
    )
    .map(_parse_fn)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 7
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
)
base_model.trainable = False

pool_layer = tf.keras.layers.GlobalMaxPooling2D()
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

model = tf.keras.Sequential([base_model, pool_layer, output_layer])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()



## === cell 8
num_epochs = 5
steps_per_epoch = size_train // BATCH_SIZE
validation_steps = size_val // BATCH_SIZE

history = model.fit(
    train_data,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 9
model.save_weights("weights_epoch_5.weights.h5")



## === cell 10
train_acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
train_loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

print(f"Final training accuracy: {train_acc[-1] if train_acc else 'N/A'}")
print(f"Final validation accuracy: {val_acc[-1] if val_acc else 'N/A'}")



## === cell 11
sample_sub = pd.read_csv("../input/sample_submission.csv")
test_ids = sample_sub["id"].tolist()

possible_prefixes = [
    "../input/test/",
    "../input/aerial-cactus-identification/test/",
]
test_filenames = []
for fname in test_ids:
    found = False
    for prefix in possible_prefixes:
        path = os.path.join(prefix, fname)
        if os.path.exists(path):
            test_filenames.append(path)
            found = True
            break
    if not found:
        test_filenames.append(os.path.join(possible_prefixes[0], fname))




## === cell 12
def _parse_test_fn(filename):
    image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename))
    image_normalized = (tf.cast(image_decoded, tf.float32) / 127.5) - 1.0
    image_resized = tf.image.resize(image_normalized, (IMAGE_SIZE, IMAGE_SIZE))
    return image_resized


test_dataset = (
    tf.data.Dataset.from_tensor_slices(tf.constant(test_filenames))
    .map(_parse_test_fn)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 13
pred_probs = model.predict(test_dataset, verbose=1)
pred_probs = np.ravel(pred_probs)  # flatten to 1‑D array



## === cell 14
submission_df = pd.DataFrame({"id": test_ids, "has_cactus": pred_probs})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission_df.shape}")
