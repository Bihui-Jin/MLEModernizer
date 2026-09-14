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

18.196449398872364

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split

print("TensorFlow:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df_path = "../input/petfinder-pawpularity-score/train.csv"
test_df_path = "../input/petfinder-pawpularity-score/test.csv"
train_data_path = "../input/petfinder-pawpularity-score/train"
test_data_path = "../input/petfinder-pawpularity-score/test"

checkpoint_filepath = "save_models2.weights.h5"
checkpoint_filepath2 = "save_models.weights.h5"

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 456
TARGET = "Pawpularity"
SEED = 88
BATCH_SIZE = 64
DROPOUT_RATE = 0.2
TEST_SIZE = 0.15
EPOCHS = 15
EPOCHS_2 = 10
DATA_SHAPE = 12

LEARNING_RATE = 1e-3
DECAY_STEPS = 100
DECAY_RATE = 0.96

os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception as _:
    pass

_cpu = os.cpu_count() or 2
tf.config.threading.set_intra_op_parallelism_threads(max(1, _cpu - 1))
tf.config.threading.set_inter_op_parallelism_threads(2)




## === cell 2
df_train = pd.read_csv(train_df_path)
df_test = pd.read_csv(test_df_path)

print("The shape of train dataset: ", df_train.shape)
print("The shape of test dataset: ", df_test.shape)
print(df_train.head(2))




## === cell 3
df_train["Path"] = train_data_path + "/" + df_train["Id"].astype(str) + ".jpg"
df_train["Filename"] = df_train["Id"].astype(str) + ".jpg"

df_test["Path"] = test_data_path + "/" + df_test["Id"].astype(str) + ".jpg"
df_test["Filename"] = df_test["Id"].astype(str) + ".jpg"

print(df_train.dtypes)




## === cell 4
_RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR


@tf.function
def _load_image_preprocess(path):
    jpeg = tf.io.read_file(path)
    image = tf.image.decode_jpeg(jpeg, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method=_RESIZE_METHOD)
    image = tf.cast(image, tf.float32)
    return tf.keras.applications.efficientnet.preprocess_input(image)


def _with_fast_deterministic_options(ds):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_slack = True
    return ds.with_options(options)


def _cache_path(name):
    cache_dir = os.path.join(".", "tfdata_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, name)


def _build_train_valid_ds(
    df, cache_key=None, training=False, repeat=False, augment_layer=None
):
    paths = df["Path"].to_numpy()
    meta = (
        df.drop(["Id", "Pawpularity", "Path", "Filename"], axis=1)
        .astype(np.float32)
        .to_numpy(copy=False)
    )
    y = df["Pawpularity"].astype(np.float32).to_numpy(copy=False)

    ds = tf.data.Dataset.from_tensor_slices((paths, meta, y))
    ds = _with_fast_deterministic_options(ds)

    def _decode_map(path, meta_row, y_val):
        img = _load_image_preprocess(path)
        return (img, meta_row), y_val

    ds = ds.map(_decode_map, num_parallel_calls=min(8, _cpu), deterministic=True)

    if cache_key is not None:
        ds = ds.cache(_cache_path(cache_key))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 2048),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    if augment_layer is not None:

        def _augment_map(inputs, y_val):
            img, meta_row = inputs
            img = augment_layer(img, training=True)
            return (img, meta_row), y_val

        ds = ds.map(_augment_map, num_parallel_calls=min(8, _cpu), deterministic=True)

    if repeat:
        ds = ds.repeat()

    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(min(4, _cpu))
    return ds


def _build_test_ds(df, cache_key=None):
    paths = df["Path"].to_numpy()
    meta = (
        df.drop(["Id", "Path", "Filename"], axis=1)
        .astype(np.float32)
        .to_numpy(copy=False)
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, meta))
    ds = _with_fast_deterministic_options(ds)

    def _map_fn(path, meta_row):
        img = _load_image_preprocess(path)
        return (img, meta_row)

    ds = ds.map(_map_fn, num_parallel_calls=min(8, _cpu), deterministic=True)

    if cache_key is not None:
        ds = ds.cache(_cache_path(cache_key))

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(min(4, _cpu))
    return ds




## === cell 5
train_df, valid_df = train_test_split(
    df_train, test_size=TEST_SIZE, shuffle=True, random_state=SEED
)

steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(valid_df) / BATCH_SIZE))

print("Train steps_per_epoch:", steps_per_epoch)
print("Valid validation_steps:", validation_steps)




## === cell 6
augementation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomWidth(factor=(0.2, 0.3)),
        tf.keras.layers.RandomRotation(factor=(-0.2, 0.3)),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomHeight(0.2),
    ],
    name="augmentation",
)

train_set = _build_train_valid_ds(
    train_df,
    cache_key="train_aftermap.cache",
    training=True,
    repeat=True,
    augment_layer=augementation,
)
valid_set = _build_train_valid_ds(
    valid_df,
    cache_key="valid_aftermap.cache",
    training=False,
    repeat=False,
    augment_layer=None,
)


def get_model():
    img_input = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image input")

    x = img_input

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = False  # preserve the intended "pretrained frozen" behavior

    x = backbone(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    img_output = tf.keras.layers.Dense(32, activation="relu")(x)
    img_model = tf.keras.Model(img_input, img_output, name="img_model")

    data_input = tf.keras.layers.Input(shape=(DATA_SHAPE,), name="data input")
    y = tf.keras.layers.Dense(64, activation="relu")(data_input)
    data_output = tf.keras.layers.Dense(32, activation="relu")(y)
    data_model = tf.keras.Model(data_input, data_output, name="meta_model")

    concat_layer = tf.keras.layers.Concatenate(name="concat_layer")(
        [img_model.output, data_model.output]
    )
    combined_dropout = tf.keras.layers.Dropout(DROPOUT_RATE)(concat_layer)
    combined_dense = tf.keras.layers.Dense(32, activation="relu")(combined_dropout)
    combined_batch = tf.keras.layers.BatchNormalization()(combined_dense)
    final_dropout = tf.keras.layers.Dropout(DROPOUT_RATE)(combined_batch)

    output_layer = tf.keras.layers.Dense(1, activation="relu")(final_dropout)

    model = tf.keras.Model(
        inputs=[img_model.input, data_model.input],
        outputs=output_layer,
        name="pawpularity_model",
    )
    return model


model = get_model()
model.summary()




## === cell 7
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    staircase=True,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    jit_compile=True,
)

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_rmse",
    mode="min",
    verbose=1,
    save_best_only=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_rmse", factor=0.5, patience=2, verbose=1, mode="min", min_lr=1e-6
)




## === cell 8
history = model.fit(
    train_set,
    validation_data=valid_set,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[model_checkpoint, reduce_lr],
    verbose=2,
)

if os.path.exists(checkpoint_filepath):
    model.load_weights(checkpoint_filepath)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1702704398.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_set,
      3     validation_data=valid_set,
      4     epochs=EPOCHS,
      5     steps_per_epoch=steps_per_epoch,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node IteratorGetNext defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/1702704398.py", line 1, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

Cannot batch tensors with different shapes in component 0. First element had shape [436,561,3] and element 32 had shape [421,591,3].
	 [[{{node IteratorGetNext}}]] [Op:__inference_multi_step_on_iterator_19045]

## === cell 9
test_data = _build_test_ds(df_test, cache_key="test_aftermap.cache")

prediction = model.predict(test_data, verbose=1)
prediction = np.asarray(prediction).reshape(-1)

prediction = np.clip(prediction, 0.0, 100.0)

sub = pd.DataFrame({"Id": df_test["Id"].values, "Pawpularity": prediction})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2503285685.py in <cell line: 0>()
      1 test_data = _build_test_ds(df_test, cache_key="test_aftermap.cache")
      2 
----> 3 prediction = model.predict(test_data, verbose=1)
      4 prediction = np.asarray(prediction).reshape(-1)
      5 

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

ValueError: Layer "pawpularity_model" expects 2 input(s), but it received 1 input tensors. Inputs received: [<tf.Tensor 'data:0' shape=(64, 456, 456, 3) dtype=float32>]
