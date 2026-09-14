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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
        input/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
            test/
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
            train/
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> working/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> working/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

# 5. Target score

0.99749

# 6. Current score

0.80614

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.15845) has done: 'Implemented a minimal, functional pipeline:
- Removed the strict GPU‑check that caused early termination.
- Replaced the TensorFlow‑Hub MobileNet model with the native `tf.keras.applications.MobileNetV2` (same architecture, avoids the KerasLayer import error).
- Built proper one‑hot labels from breed names.
- Fixed the data‑pipeline to cast images and labels to `float32`.
- Simplified callbacks (removed the failing TensorBoard path) and kept an early‑stopping callback.
- Trained the model on a small subset for speed, then re‑trained on the full dataset.
- Generated predictions for the test set, built the submission DataFrame matching the required column order, and wrote a valid `submission.csv`.'
- What this solution (achieved 0.80614) has done: 'I fix the TensorFlow import by forcing protobuf to use the pure‑Python implementation, correct the misuse of `Dataset.shuffle()` (the argument name should be positional), and adjust the cell numbering to start at 1 while preserving the original order. These changes remove the runtime errors, allow the model to train, and produce a proper `submission.csv` without altering the core modeling logic.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import datetime, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print("TensorFlow version:", tf.__version__)



## === cell 2
labels = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
print("Loaded", len(labels), "labels")



## === cell 3
unique_breeds = np.sort(labels["breed"].unique())
breed_to_idx = {b: i for i, b in enumerate(unique_breeds)}
num_classes = len(unique_breeds)
print("Number of breeds:", num_classes)



## === cell 4
train_dir = "/kaggle/input/dog-breed-identification/train/"
test_dir = "/kaggle/input/dog-breed-identification/test/"

train_files = [os.path.join(train_dir, f"{img_id}.jpg") for img_id in labels["id"]]
train_labels_idx = labels["breed"].map(breed_to_idx).values



## === cell 5
train_labels_onehot = tf.keras.utils.to_categorical(
    train_labels_idx, num_classes=num_classes
)



## === cell 6
IMG_SIZE = 224


def process_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0‑1
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img




## === cell 7
BATCH_SIZE = 32


def make_dataset(image_paths, labels_onehot=None, shuffle=False, repeat=False):
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    if labels_onehot is not None:
        lbl = tf.data.Dataset.from_tensor_slices(labels_onehot.astype(np.float32))
        ds = tf.data.Dataset.zip((ds, lbl))
    if shuffle:
        ds = ds.shuffle(len(image_paths))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    return ds




## === cell 8
from sklearn.model_selection import train_test_split

train_paths, val_paths, train_y, val_y = train_test_split(
    train_files,
    train_labels_onehot,
    test_size=0.1,
    random_state=42,
    stratify=train_labels_idx,
)

train_ds = make_dataset(train_paths, train_y, shuffle=True)
val_ds = make_dataset(val_paths, val_y, shuffle=False)




## === cell 9
def create_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # fine‑tune later if needed
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model




## === cell 10
model = create_model()
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=2, restore_best_weights=True
)
model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[early_stop])



## === cell 11
full_ds = make_dataset(train_files, train_labels_onehot, shuffle=True, repeat=False)
model.fit(full_ds, epochs=3, callbacks=[early_stop])  # few extra epochs



## === cell 12
test_files = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jpg")]
)
test_ds = make_dataset(test_files, shuffle=False)



## === cell 13
test_preds = model.predict(test_ds, verbose=1)  # shape (num_test, num_classes)



## === cell 14
sample_sub = pd.read_csv(
    "/kaggle/input/dog-breed-identification/sample_submission.csv", nrows=1
)
breed_columns = sample_sub.columns.tolist()[1:]  # skip 'id'
pred_df = pd.DataFrame(test_preds, columns=unique_breeds)
pred_df = pred_df[breed_columns]  # reorder to match submission format
pred_df.insert(0, "id", [os.path.basename(p)[:-4] for p in test_files])



## === cell 15
submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print("Submission written to:", submission_path)



## === cell 16
print("File size (bytes):", os.path.getsize(submission_path))
print(pred_df.head())
