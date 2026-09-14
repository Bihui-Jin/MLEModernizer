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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.4916

# 6. Current score

0.99978

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.9999) has done: 'I fix the environment/path issues so the script can actually find the competition data under `/kaggle/input/aerial-cactus-identification/`. Then I replace deprecated/broken Keras APIs (`keras.preprocessing`, `fit_generator`, `predict_generator`) with their supported equivalents in `tf_keras` while keeping the same model and training loop behavior. I also fix the generator setup for the test set (use the correct datagen, disable shuffling, and output probabilities—not hard 0/1—since the metric is AUC). Finally, I ensure a valid `submission.csv` is written with the exact required columns and row alignment.'
- What this solution (achieved 0.99999) has done: 'I fix the runtime crash coming from `tf_keras.preprocessing.image.ImageDataGenerator` by switching to `tensorflow.keras.preprocessing.image.ImageDataGenerator`, which is compatible with the Kaggle TensorFlow/protobuf stack and avoids the `MessageFactory.GetPrototype` error. I keep the same data pipeline behavior (same augmentations, same split/seed, same model, same training loop semantics) and only adjust imports to unblock execution. Since your current score (0.9999) is already well above the target (0.4916), I won’t introduce any score-improving changes; this patch is intended to be score-neutral while producing a valid `submission.csv`. The script still write `submission.csv` with the required `id,has_cactus` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.99997) has done: 'I fix the crash in the image generator by avoiding the TensorFlow/protobuf incompatibility that triggers `MessageFactory.GetPrototype`, while keeping the same augmentation and training/inference semantics. Concretely, I switch the data pipeline to `tf_keras.preprocessing.image.ImageDataGenerator` (and keep the model in `tf_keras`), plus add a small defensive fallback to `keras.preprocessing.image.ImageDataGenerator` if needed. I also ensure the dataframe column dtypes/values are correct for `flow_from_dataframe` and keep the submission writing exactly as required (`id,has_cactus` to `submission.csv`). Since your current score is already far above the target, these are score-neutral stability fixes only.'
- What this solution (achieved 0.99969) has done: 'I fix the runtime crash in the image generator by switching to the TensorFlow-bundled `tensorflow.keras.preprocessing.image.ImageDataGenerator`, which avoids the protobuf `MessageFactory.GetPrototype` incompatibility you’re hitting. This is a stability-only change that keeps the same data pipeline behavior (same augmentations, same splits/seed, same model/training loop) and should be essentially score-neutral—important since your current score is far above the target and we don’t want to improve further. I also add a small fallback import chain so the notebook runs across Kaggle environments, while keeping submission formatting and row alignment unchanged. The script run end-to-end and write `submission.csv` with the required `id,has_cactus` columns.'
- What this solution (achieved 0.99978) has done: 'I fix the runtime crash coming from `ImageDataGenerator` by avoiding the protobuf-incompatible import paths and using a safe PIL-based fallback generator if needed, while keeping your model, training loop, augmentations, and prediction semantics the same. This is a stability-focused change intended to be score-neutral (your current score is far above the target, so we avoid any performance-improving edits). I also ensure the generators always yield correctly-shaped float32 batches and that `submission.csv` is written with the exact required `id,has_cactus` columns and row alignment.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

BASE_DIR = "/kaggle/input/aerial-cactus-identification"
train_dir = os.path.join(BASE_DIR, "train")
test_dir = os.path.join(BASE_DIR, "test")

train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
test_df = sample_sub.copy()

print("Train images dir exists:", os.path.isdir(train_dir))
print("Test images dir exists:", os.path.isdir(test_dir))
print("train_df:", train_df.shape, "sample_sub:", sample_sub.shape)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
from sklearn.model_selection import train_test_split

X = train_df["id"].astype(str)
y = train_df["has_cactus"].astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

train_gen_df = pd.concat(
    [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
)
valid_gen_df = pd.concat(
    [X_valid.reset_index(drop=True), y_valid.reset_index(drop=True)], axis=1
)

train_gen_df.columns = ["id", "has_cactus"]
valid_gen_df.columns = ["id", "has_cactus"]

print(train_gen_df.shape, valid_gen_df.shape, y_train.mean(), y_valid.mean())



## === cell 2
import math

ImageDataGenerator = None
_import_errors = []


def _try_import_idg(module_path: str):
    mod = __import__(module_path, fromlist=["ImageDataGenerator"])
    return getattr(mod, "ImageDataGenerator")


for _path in (
    "tensorflow.keras.preprocessing.image",  # preferred
    "keras.preprocessing.image",  # sometimes works
    "tf_keras.preprocessing.image",  # last resort
):
    try:
        ImageDataGenerator = _try_import_idg(_path)
        print(f"Using ImageDataGenerator from: {_path}")
        break
    except Exception as e:
        _import_errors.append((_path, repr(e)))
        ImageDataGenerator = None

batch_size = 64

if ImageDataGenerator is not None:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0, horizontal_flip=True, vertical_flip=True
    )
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_gen_df,
        directory=train_dir,
        x_col="id",
        y_col="has_cactus",
        class_mode="raw",  # binary float labels
        batch_size=batch_size,
        target_size=(32, 32),
        shuffle=True,
        seed=SEED,
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    valid_gen = valid_datagen.flow_from_dataframe(
        dataframe=valid_gen_df,
        directory=train_dir,
        x_col="id",
        y_col="has_cactus",
        class_mode="raw",
        batch_size=batch_size,
        target_size=(32, 32),
        shuffle=False,
    )
else:
    print(
        "Falling back to PIL-based generators because ImageDataGenerator import failed:\n"
        + "\n".join([f"{p}: {e}" for p, e in _import_errors])
    )

    from PIL import Image
    import tf_keras
    from tf_keras.utils import Sequence

    class SimpleImageSequence(Sequence):
        def __init__(
            self,
            df: pd.DataFrame,
            directory: str,
            batch_size: int = 64,
            target_size=(32, 32),
            shuffle: bool = True,
            seed: int = 42,
            rescale: float = 1.0 / 255.0,
            hflip: bool = False,
            vflip: bool = False,
            with_labels: bool = True,
        ):
            self.df = df.reset_index(drop=True).copy()
            self.directory = directory
            self.batch_size = int(batch_size)
            self.target_size = tuple(target_size)
            self.shuffle = bool(shuffle)
            self.seed = int(seed)
            self.rescale = float(rescale) if rescale is not None else None
            self.hflip = bool(hflip)
            self.vflip = bool(vflip)
            self.with_labels = bool(with_labels)

            self.filenames = self.df["id"].astype(str).tolist()
            if self.with_labels:
                self.labels = self.df["has_cactus"].astype(np.float32).values
            else:
                self.labels = None

            self.indices = np.arange(len(self.filenames))
            self.rng = np.random.RandomState(self.seed)
            self.on_epoch_end()

        def __len__(self):
            return int(math.ceil(len(self.filenames) / self.batch_size))

        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.indices)

        def _load_one(self, fname: str):
            path = os.path.join(self.directory, fname)
            img = Image.open(path).convert("RGB")
            if img.size != (self.target_size[1], self.target_size[0]):
                img = img.resize(
                    (self.target_size[1], self.target_size[0]), resample=Image.BILINEAR
                )

            if self.hflip and (self.rng.rand() < 0.5):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.vflip and (self.rng.rand() < 0.5):
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

            arr = np.asarray(img, dtype=np.float32)
            if self.rescale is not None:
                arr *= self.rescale
            return arr

        def __getitem__(self, idx):
            batch_inds = self.indices[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]
            batch_x = np.stack(
                [self._load_one(self.filenames[i]) for i in batch_inds], axis=0
            ).astype(np.float32)
            if self.with_labels:
                batch_y = self.labels[batch_inds].astype(np.float32)
                return batch_x, batch_y
            return batch_x

    train_gen = SimpleImageSequence(
        df=train_gen_df,
        directory=train_dir,
        batch_size=batch_size,
        target_size=(32, 32),
        shuffle=True,
        seed=SEED,
        rescale=1.0 / 255.0,
        hflip=True,
        vflip=True,
        with_labels=True,
    )

    valid_gen = SimpleImageSequence(
        df=valid_gen_df,
        directory=train_dir,
        batch_size=batch_size,
        target_size=(32, 32),
        shuffle=False,
        seed=SEED,
        rescale=1.0 / 255.0,
        hflip=False,
        vflip=False,
        with_labels=True,
    )



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
from tf_keras import Sequential
from tf_keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
)

input_shape = (32, 32, 3)

model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(16, (3, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 4
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint

weights_path = "model.weights.h5"

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(
        filepath=weights_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    ),
]

steps_per_epoch = int(np.ceil(len(train_gen_df) / batch_size))
validation_steps = int(np.ceil(len(valid_gen_df) / batch_size))

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=100,
    verbose=1,
    callbacks=callbacks,
)

if not os.path.exists(weights_path):
    model.save_weights(weights_path)



## === cell 5
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

acc_key = "accuracy" if "accuracy" in history.history else "acc"
val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

plt.plot(history.history[acc_key], label=acc_key)
plt.plot(history.history[val_acc_key], label=val_acc_key)
plt.legend()
plt.show()



## === cell 6
model.load_weights(weights_path)

if ImageDataGenerator is not None:
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df.assign(id=test_df["id"].astype(str)),
        directory=test_dir,
        x_col="id",
        y_col=None,
        class_mode=None,
        batch_size=64,
        target_size=(32, 32),
        shuffle=False,
    )
else:
    test_gen = SimpleImageSequence(
        df=test_df.assign(id=test_df["id"].astype(str)),
        directory=test_dir,
        batch_size=64,
        target_size=(32, 32),
        shuffle=False,
        seed=SEED,
        rescale=1.0 / 255.0,
        hflip=False,
        vflip=False,
        with_labels=False,
    )

preds = model.predict(
    test_gen, steps=int(np.ceil(len(test_df) / 64)), verbose=1
).reshape(-1)

preds = preds[: len(test_df)]



## === cell 7
submission = pd.DataFrame(
    {"id": test_df["id"].values, "has_cactus": preds.astype(float)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Done!")
