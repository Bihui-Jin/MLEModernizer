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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.8115235457063733

# 6. Current score

0.272

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.272) has done: 'I remove the TensorFlow Addons import that’s causing the protobuf `MessageFactory` error, since it isn’t used anywhere in your pipeline. Then I fix the missing external pretrained model path by building the closest equivalent in-notebook (EfficientNetB0-style classifier) and training it on the provided `train_images`, so `preds` is always defined and the notebook runs end-to-end. Finally, I fix the submission label assignment bugs (`==` vs `=`, chained indexing) and make the threshold-to-label conversion robust while preserving the same “threshold then fallback to argmax” core post-processing idea, ensuring a valid `submission.csv` with the required `image,labels` columns is written.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
h_target = 384
w_target = 384
batch_size = 32



## === cell 3
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)

labels_df = pd.DataFrame(Y, columns=class_names)
print("Num classes:", len(class_names))
print("Classes:", class_names)

class_to_idx = {c: i for i, c in enumerate(class_names)}
healthy_idx = class_to_idx.get("healthy", None)
print("healthy_idx:", healthy_idx)



## === cell 4
idx = np.arange(len(train))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_frac = 0.1
n_val = int(len(train) * val_frac)
val_idx = idx[:n_val]
trn_idx = idx[n_val:]

train_trn = train.iloc[trn_idx].reset_index(drop=True)
train_val = train.iloc[val_idx].reset_index(drop=True)

Y_trn = Y[trn_idx]
Y_val = Y[val_idx]

print("Train/Val:", train_trn.shape, train_val.shape)



## === cell 5
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_trn,
    directory=TRAIN_IMG_DIR,
    x_col="image",
    y_col="labels",
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode="categorical",  # will override with provided labels via y vector below
    batch_size=batch_size,
    shuffle=True,
    seed=SEED,
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=train_val,
    directory=TRAIN_IMG_DIR,
    x_col="image",
    y_col="labels",
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=False,
)

test_generator = test_datagen.flow_from_dataframe(
    submissions,
    directory=TEST_IMG_DIR,
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode=None,
    shuffle=False,
    batch_size=batch_size,
)




## === cell 6
def multilabel_generator(base_gen, y_array):
    i = 0
    n = len(y_array)
    while True:
        x_batch = next(base_gen)
        bs = x_batch.shape[0]
        y_batch = y_array[i : i + bs]
        i += bs
        if i >= n:
            i = 0
        yield x_batch, y_batch.astype(np.float32)


train_ml_gen = multilabel_generator(train_generator, Y_trn)
val_ml_gen = multilabel_generator(val_generator, Y_val)

steps_per_epoch = int(np.ceil(len(train_trn) / batch_size))
val_steps = int(np.ceil(len(train_val) / batch_size))

print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)



## === cell 7
inputs = keras.Input(shape=(h_target, w_target, 3))
base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs,
    pooling="avg",
)
x = base.output
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(class_names), activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

base.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 8
history1 = model.fit(
    train_ml_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ml_gen,
    validation_steps=val_steps,
    epochs=2,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)

history2 = model.fit(
    train_ml_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ml_gen,
    validation_steps=val_steps,
    epochs=1,
    verbose=1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3268418950.py in <cell line: 0>()
      2 # Keep runtime under control: 2 phases, small epoch counts.
      3 # Phase 1: train head
----> 4 history1 = model.fit(
      5     train_ml_gen,
      6     steps_per_epoch=steps_per_epoch,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/3589955093.py in multilabel_generator(base_gen, y_array)
      7     while True:
      8         x_batch = next(base_gen)
----> 9         bs = x_batch.shape[0]
     10         y_batch = y_array[i : i + bs]
     11         i += bs

AttributeError: 'tuple' object has no attribute 'shape'

## === cell 9
preds = model.predict(test_generator, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])



## === cell 10
thresh = 0.2

out_labels = []
for i in range(len(submissions)):
    p = preds[i]
    chosen = np.where(p >= thresh)[0].tolist()

    if len(chosen) == 0:
        chosen = [int(np.argmax(p))]

    if healthy_idx is not None and healthy_idx in chosen and len(chosen) > 1:
        chosen = [j for j in chosen if j != healthy_idx]

    if healthy_idx is not None and int(np.argmax(p)) == healthy_idx:
        if np.max(np.delete(p, healthy_idx)) < thresh:
            chosen = [healthy_idx]

    label_str = " ".join([class_names[j] for j in chosen])
    out_labels.append(label_str)

submissions = submissions.copy()
submissions["labels"] = out_labels

submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)

print(submissions.head())
print("Wrote submission.csv with shape:", submissions.shape)
