# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

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
protobuf==6.33.0
scikit-image==0.25.2
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

0.8993

# 6. Current score

0.99665

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.99926) has done: 'I replace the missing `keras_preprocessing` import with the built‑in `tf.keras.preprocessing.image.ImageDataGenerator`, point the data loaders directly at the already‑extracted input folders, fix the generator and model compilation arguments, use a proper `.keras` checkpoint filename, switch to `model.fit` (instead of the removed `fit_generator`), and create a test generator that outputs the probability for the cactus class. These minimal fixes let the notebook run end‑to‑end and write a valid `submission.csv` while keeping the original architecture and training logic.'
- What this solution (achieved 0.99902) has done: 'I remove the failing `skimage.exposure` import and replace the histogram‑equalisation `preprocess` function with a lightweight identity version (the ImageDataGenerator already applies samplewise centering and normalization). This fixes the AttributeError while keeping the rest of the pipeline unchanged, allowing the script to run end‑to‑end and generate a valid `submission.csv`. No changes to model architecture or training logic are made, and the score remains unchanged (already above the target).'
- What this solution (achieved 0.99648) has done: 'Implemented fixes:
- Set protobuf implementation to pure‑Python to avoid the `MessageFactory` error.
- Removed the deprecated `random_rotation` augmentation that caused TensorFlow eager‑tensor issues.
- Cleaned test image list to include only `.jpg` files, preventing the directory‑load error.
- Kept all original logic, model architecture and training flow intact while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.99665) has done: 'Implemented a protobuf compatibility shim before any TensorFlow imports to restore the missing `GetPrototype` method, preventing the “MessageFactory has no attribute GetPrototype” error. This minimal fix allows the entire pipeline to run end‑to‑end and generate a valid `submission.csv` without altering model architecture or training logic.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import message_factory

if not hasattr(message_factory.MessageFactory, "GetPrototype"):

    def _get_prototype(self, descriptor):
        return self.GetMessageClass(descriptor)

    message_factory.MessageFactory.GetPrototype = _get_prototype

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename.endswith(".csv") or filename.endswith(".zip"):
            print(os.path.join(dirname, filename))




## === cell 1
BASE_PATH = "/kaggle/input/aerial-cactus-identification/"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

files_dataframe = pd.read_csv(TRAIN_CSV, dtype=str)
files_dataframe.head()




## === cell 2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class SimpleDataGenerator(tf.keras.utils.Sequence):
    """Minimal data generator for image classification."""

    def __init__(
        self,
        df,
        directory,
        batch_size=32,
        target_size=(32, 32),
        shuffle=True,
        augment=False,
        has_labels=True,
    ):
        self.df = df.reset_index(drop=True)
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.has_labels = has_labels

        self.filenames = self.df["id"].tolist()
        if self.has_labels:
            self.labels = to_categorical(
                self.df["has_cactus"].astype(int), num_classes=2
            )
        else:
            self.labels = None

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_files = [self.filenames[i] for i in batch_indexes]
        batch_images = []
        for fname in batch_files:
            img_path = os.path.join(self.directory, fname)
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=self.target_size
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            if self.augment:
                if np.random.rand() < 0.5:
                    img_array = tf.image.flip_left_right(img_array)
                if np.random.rand() < 0.5:
                    img_array = tf.image.flip_up_down(img_array)
            batch_images.append(img_array)
        X = np.stack(batch_images, axis=0)

        if self.has_labels:
            y = self.labels[batch_indexes]
            return X, y
        else:
            return X


train_df, val_df = train_test_split(
    files_dataframe,
    test_size=0.25,
    random_state=42,
    stratify=files_dataframe["has_cactus"],
)




## === cell 3
training_generator = SimpleDataGenerator(
    df=train_df,
    directory=TRAIN_IMG_DIR,
    batch_size=32,
    target_size=(32, 32),
    shuffle=True,
    augment=True,
    has_labels=True,
)

validation_generator = SimpleDataGenerator(
    df=val_df,
    directory=TRAIN_IMG_DIR,
    batch_size=32,
    target_size=(32, 32),
    shuffle=False,
    augment=False,
    has_labels=True,
)




## === cell 4
from tensorflow.keras import layers, models

model = models.Sequential(
    [
        layers.Conv2D(32, (5, 5), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(128, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()




## === cell 5
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.001, verbose=1
)

checkpoint_path = "/tmp/best_model.keras"
save_best_model = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

history = model.fit(
    training_generator,
    steps_per_epoch=len(training_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=12,
    callbacks=[reduce_lr, save_best_model],
    verbose=1,
)




## === cell 6
from tensorflow.keras.models import load_model

model = load_model(checkpoint_path)




## === cell 7
test_filenames = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
)
test_df = pd.DataFrame({"id": test_filenames})

test_generator = SimpleDataGenerator(
    df=test_df,
    directory=TEST_IMG_DIR,
    batch_size=32,
    target_size=(32, 32),
    shuffle=False,
    augment=False,
    has_labels=False,
)

pred_probs = model.predict(test_generator, verbose=1)
cactus_prob = pred_probs[:, 1]  # probability of having a cactus

submission = pd.DataFrame({"id": test_generator.filenames, "has_cactus": cactus_prob})
submission["id"] = submission["id"].apply(lambda x: os.path.basename(x))

submission.to_csv("submission.csv", index=False)
print("Saved submission to submission.csv, shape:", submission.shape)




## === cell 8
import shutil

for folder in ["training", "test"]:
    try:
        shutil.rmtree(folder)
    except OSError:
        pass
