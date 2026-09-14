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

20.576311563618383

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

DATA_ROOT = "/kaggle/input/petfinder-pawpularity-score"
for dirname, _, filenames in os.walk(DATA_ROOT):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

TRAIN_CSV_PATH = f"{DATA_ROOT}/train.csv"
TEST_CSV_PATH = f"{DATA_ROOT}/test.csv"
TRAIN_IMG_DIR = f"{DATA_ROOT}/train"
TEST_IMG_DIR = f"{DATA_ROOT}/test"

IMG_SIZE = (299, 299)  # InceptionV3 default
BATCH_SIZE = 32

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

train_df["Id"] = train_df["Id"].astype(str)
test_df["Id"] = test_df["Id"].astype(str)

train_df["img_path"] = TRAIN_IMG_DIR + "/" + train_df["Id"] + ".jpg"
test_df["img_path"] = TEST_IMG_DIR + "/" + test_df["Id"] + ".jpg"



## === cell 3


@tf.function
def decode_resize_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, fancy_upscaling=False)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # InceptionV3 preprocessing
    return img


@tf.function
def _map_train(path, y):
    return decode_resize_preprocess(path), y


def make_dataset(df, training=True, cache=False, cache_name="ds_cache"):
    paths_np = df["img_path"].values.astype("U")

    if training:
        y_np = df["Pawpularity"].astype(np.float32).values
        ds = tf.data.Dataset.from_tensor_slices((paths_np, y_np))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths_np)

    if training:
        ds = ds.shuffle(min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True)

    options = tf.data.Options()
    options.deterministic = False
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    def _read_map_train(p, y):
        return _map_train(p, y)

    if training:
        ds = ds.apply(
            tf.data.experimental.parallel_interleave(
                lambda p, y: tf.data.Dataset.from_tensors((p, y)).map(
                    _read_map_train, num_parallel_calls=tf.data.AUTOTUNE
                ),
                cycle_length=tf.data.AUTOTUNE,
                block_length=16,
                sloppy=True,
            )
        )
    else:
        ds = ds.apply(
            tf.data.experimental.parallel_interleave(
                lambda p: tf.data.Dataset.from_tensors(p).map(
                    decode_resize_preprocess, num_parallel_calls=tf.data.AUTOTUNE
                ),
                cycle_length=tf.data.AUTOTUNE,
                block_length=16,
                sloppy=True,
            )
        )

    if cache:
        cache_path = os.path.join(CACHE_DIR, cache_name)
        ds = ds.cache(cache_path)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 4
trn_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)

train_ds = make_dataset(trn_df, training=True, cache=True, cache_name="train_cache")
val_ds = make_dataset(val_df, training=True, cache=True, cache_name="val_cache")
test_ds = make_dataset(test_df, training=False, cache=True, cache_name="test_cache")

print(
    "Train size:", len(trn_df), "Valid size:", len(val_df), "Test size:", len(test_df)
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2879947203.py in <cell line: 0>()
      1 trn_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)
      2 
----> 3 train_ds = make_dataset(trn_df, training=True, cache=True, cache_name="train_cache")
      4 val_ds = make_dataset(val_df, training=True, cache=True, cache_name="val_cache")
      5 test_ds = make_dataset(test_df, training=False, cache=True, cache_name="test_cache")

/tmp/ipykernel_11/238608019.py in make_dataset(df, training, cache, cache_name)
     51     if training:
     52         # Convert to dataset of individual elements and interleave mapping
---> 53         ds = ds.apply(
     54             tf.data.experimental.parallel_interleave(
     55                 lambda p, y: tf.data.Dataset.from_tensors((p, y)).map(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in apply(self, transformation_func)
   2585       A new `Dataset` with the transformation applied as described above.
   2586     """
-> 2587     dataset = transformation_func(self)
   2588     if not isinstance(dataset, data_types.DatasetV2):
   2589       raise TypeError(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/experimental/ops/interleave_ops.py in _apply_fn(dataset)
     80 
     81   def _apply_fn(dataset):
---> 82     return readers.ParallelInterleaveDataset(dataset, map_func, cycle_length,
     83                                              block_length, sloppy,
     84                                              buffer_output_elements,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/readers.py in __init__(self, input_dataset, map_func, cycle_length, block_length, sloppy, buffer_output_elements, prefetch_input_elements, name)
    364     self._name = name
    365 
--> 366     variant_tensor = ged_ops.legacy_parallel_interleave_dataset_v2(
    367         self._input_dataset._variant_tensor,  # pylint: disable=protected-access
    368         self._map_func.function.captured_inputs,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_experimental_dataset_ops.py in legacy_parallel_interleave_dataset_v2(input_dataset, other_arguments, cycle_length, block_length, buffer_output_elements, prefetch_input_elements, f, output_types, output_shapes, deterministic, metadata, name)
   6746       return _result
   6747     except _core._NotOkStatusException as e:
-> 6748       _ops.raise_from_not_ok_status(e, name)
   6749     except _core._FallbackException:
   6750       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__LegacyParallelInterleaveDatasetV2_Targuments_0_device_/job:localhost/replica:0/task:0/device:CPU:0}} `cycle_length` must be > 0 [Op:LegacyParallelInterleaveDatasetV2] name: 

## === cell 5
base = InceptionV3(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base.trainable = False  # keep stable, fast, and within runtime

inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1, activation="linear")(x)

model = models.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    steps_per_execution=16,
)

model.summary()



## === cell 6
EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3781331062.py in <cell line: 0>()
      1 EPOCHS = 3
----> 2 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      3 

NameError: name 'train_ds' is not defined

## === cell 7
pred = model.predict(test_ds, verbose=1).reshape(-1)
pred = np.clip(pred, 0.0, 100.0)

submission = pd.DataFrame(
    {"Id": test_df["Id"].values, "Pawpularity": pred.astype(np.float32)}
)

assert submission.shape[0] == test_df.shape[0]
assert list(submission.columns) == ["Id", "Pawpularity"]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/692382764.py in <cell line: 0>()
----> 1 pred = model.predict(test_ds, verbose=1).reshape(-1)
      2 pred = np.clip(pred, 0.0, 100.0)
      3 
      4 submission = pd.DataFrame(
      5     {"Id": test_df["Id"].values, "Pawpularity": pred.astype(np.float32)}

NameError: name 'test_ds' is not defined
