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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

18.905466801224414

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Warning: could not enable_op_determinism(); continuing. Error:", repr(e))

DATA_DIR = "../input/petfinder-pawpularity-score"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
TEST_CSV_PATH = f"{DATA_DIR}/test.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train"
TEST_IMG_DIR = f"{DATA_DIR}/test"

IMG_SIZE = 224
BATCH_SIZE = 32

train_csv = pd.read_csv(TRAIN_CSV_PATH)

train_meta = train_csv.drop(columns=["Id", "Pawpularity"]).to_numpy(dtype=np.float32)
train_label = train_csv["Pawpularity"].to_numpy(dtype=np.float32)
train_img_paths = (TRAIN_IMG_DIR + "/" + train_csv["Id"].values + ".jpg").astype(str)

n = len(train_csv)
split = int(n * 0.95)

AUTOTUNE = tf.data.AUTOTUNE

DS_OPTIONS = tf.data.Options()
DS_OPTIONS.deterministic = True
try:
    DS_OPTIONS.experimental_optimization.map_parallelization = True
    DS_OPTIONS.experimental_optimization.parallel_batch = True
    DS_OPTIONS.experimental_optimization.autotune_buffers = True
    DS_OPTIONS.experimental_optimization.autotune_cpu_budget = True
except Exception as e:
    print(
        "Warning: could not set some dataset optimization options; continuing. Error:",
        repr(e),
    )


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(
        img_bytes, channels=3, dct_method="INTEGER_FAST", fancy_upscaling=False
    )
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


@tf.function
def _map_image_only(p):
    return _decode_resize_preprocess(p)


def make_image_dataset(paths, batch_size=BATCH_SIZE):
    paths = tf.convert_to_tensor(paths)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_map_image_only, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.with_options(DS_OPTIONS)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_dataset_from_arrays(
    x_img_emb, meta, labels=None, training=False, batch_size=BATCH_SIZE
):
    x_img_emb = tf.convert_to_tensor(x_img_emb)
    meta = tf.convert_to_tensor(meta)
    if labels is not None:
        labels = tf.convert_to_tensor(labels)

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices((x_img_emb, meta))
    else:
        ds = tf.data.Dataset.from_tensor_slices(((x_img_emb, meta), labels))

    ds = ds.with_options(DS_OPTIONS)

    if training:
        ds = ds.shuffle(
            buffer_size=min(int(x_img_emb.shape[0]), 2048),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


print("train_meta:", train_meta.shape, train_meta.dtype)
print("train_label:", train_label.shape, train_label.dtype)
print("n, split:", n, split)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
eff = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
eff.trainable = False

train_img_ds = make_image_dataset(train_img_paths[:split], batch_size=BATCH_SIZE)
val_img_ds = make_image_dataset(train_img_paths[split:], batch_size=BATCH_SIZE)

train_emb = eff.predict(train_img_ds, verbose=0)
val_emb = eff.predict(val_img_ds, verbose=0)

print("train_emb:", train_emb.shape, train_emb.dtype)
print("val_emb:", val_emb.shape, val_emb.dtype)

inputA = keras.Input(shape=(train_emb.shape[1],), name="image_emb")
x = layers.BatchNormalization()(inputA)

inputB = keras.Input(shape=(12,), name="meta")
combined = layers.Concatenate()([x, inputB])
combined = layers.Dense(1)(combined)

model = keras.Model(inputs=[inputA, inputB], outputs=combined)

model.summary()

model.compile(
    optimizer="rmsprop",
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError()],
)

train_ds = make_dataset_from_arrays(
    train_emb,
    train_meta[:split],
    train_label[:split],
    training=True,
    batch_size=BATCH_SIZE,
)
val_ds = make_dataset_from_arrays(
    val_emb,
    train_meta[split:],
    train_label[split:],
    training=False,
    batch_size=BATCH_SIZE,
)

train_steps = int(np.ceil(split / BATCH_SIZE))
val_steps = int(np.ceil((n - split) / BATCH_SIZE))
print("train_steps, val_steps:", train_steps, val_steps)

history = model.fit(
    train_ds,
    epochs=20,
    steps_per_epoch=train_steps,
    verbose=2,
)

results = model.evaluate(
    val_ds,
    steps=val_steps,
    verbose=0,
)
print("val loss, val rmse:", results)



## === cell 2
test_csv = pd.read_csv(TEST_CSV_PATH)
test_meta = test_csv.drop(columns=["Id"]).to_numpy(dtype=np.float32)
test_img_paths = (TEST_IMG_DIR + "/" + test_csv["Id"].values + ".jpg").astype(str)

test_img_ds = make_image_dataset(test_img_paths, batch_size=BATCH_SIZE)
test_emb = eff.predict(test_img_ds, verbose=0)

test_ds = make_dataset_from_arrays(
    test_emb,
    test_meta,
    labels=None,
    training=False,
    batch_size=BATCH_SIZE,
)

test_steps = int(np.ceil(len(test_csv) / BATCH_SIZE))

prediction = model.predict(test_ds, steps=test_steps, verbose=0).reshape(-1)
prediction = np.clip(prediction, 0.0, 100.0)

submission = pd.DataFrame({"Id": test_csv["Id"].values, "Pawpularity": prediction})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv columns:", list(submission.columns))

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4184908140.py in <cell line: 0>()
     17 test_steps = int(np.ceil(len(test_csv) / BATCH_SIZE))
     18 
---> 19 prediction = model.predict(test_ds, steps=test_steps, verbose=0).reshape(-1)
     20 prediction = np.clip(prediction, 0.0, 100.0)
     21 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    158     inputs = tree.flatten(inputs)
    159     if len(inputs) != len(input_spec):
--> 160         raise ValueError(
    161             f'Layer "{layer_name}" expects {len(input_spec)} input(s),'
    162             f" but it received {len(inputs)} input tensors. "

ValueError: Layer "functional" expects 2 input(s), but it received 1 input tensors. Inputs received: [<tf.Tensor 'data:0' shape=(32, 1280) dtype=float32>]
