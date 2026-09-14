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

geopandas==0.14.4
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

20.47794

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, importlib

try:
    import google.protobuf  # noqa: F401
    import google.protobuf.__version__ as _pbv  # type: ignore
except Exception:
    _pbv = None


def _version_tuple(v):
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except Exception:
        return (0, 0, 0)


if _pbv is None or _version_tuple(_pbv) >= (5, 0, 0):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    import google.protobuf

    importlib.reload(google.protobuf)

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn  # noqa: F401
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

tf.keras.utils.set_random_seed(42)



## === cell 1
train = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
sample_submission = pd.read_csv(
    "../input/petfinder-pawpularity-score/sample_submission.csv"
)



## === cell 2
train.head()



## === cell 3
train["file_path"] = train["Id"].apply(
    lambda identifier: "../input/petfinder-pawpularity-score/train/"
    + identifier
    + ".jpg"
)
test["file_path"] = test["Id"].apply(
    lambda identifier: "../input/petfinder-pawpularity-score/test/"
    + identifier
    + ".jpg"
)



## === cell 4
train.head()



## === cell 5
train["Pawpularity"].hist()



## === cell 6
tabular_columns = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]
image_size = 128
batch_size = 128




## === cell 7
def preprocess(image_url, tabular):
    image_string = tf.io.read_file(image_url)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.central_crop(image, 1.0)
    image = tf.image.resize(image, (image_size, image_size))

    tabular = tf.cast(tabular, tf.float32)
    y = tabular[0]
    x_tab = tabular[1:]
    return (image, x_tab), tf.cast(y, tf.float32)




## === cell 8
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))




## === cell 9
def block(x, filters, kernel_size, repetitions, pool_size=2, strides=2):
    for i in range(repetitions):
        x = tf.keras.layers.Conv2D(
            filters, kernel_size, activation="relu", padding="same"
        )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size, strides)(x)
    return x




## === cell 10
def get_model():
    image_inputs = tf.keras.Input((image_size, image_size, 3))
    tabular_inputs = tf.keras.Input((len(tabular_columns),))
    image_x = block(image_inputs, 8, 3, 2)
    image_x = block(image_x, 16, 3, 2)
    image_x = block(image_x, 32, 3, 2)
    image_x = block(image_x, 64, 3, 2)
    image_x = block(image_x, 128, 3, 2)
    image_x = tf.keras.layers.GlobalAveragePooling2D()(image_x)

    tabular_x = tf.keras.layers.Dense(16)(tabular_inputs)
    tabular_x = tf.keras.layers.Dense(16)(tabular_x)
    tabular_x = tf.keras.layers.Concatenate()([tabular_x, tabular_inputs])
    tabular_x = tf.keras.layers.Dense(16)(tabular_x)
    tabular_x = tf.keras.layers.Dense(16)(tabular_x)
    tabular_x = tf.keras.layers.Concatenate()([tabular_x, tabular_inputs])

    x = tf.keras.layers.Concatenate(axis=1)([image_x, tabular_x])
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=[image_inputs, tabular_inputs], outputs=[output])
    return model




## === cell 11
model = get_model()
try:
    tf.keras.utils.plot_model(model, show_shapes=True)
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 12
model.summary()



## === cell 13
image = np.random.normal(size=(1, image_size, image_size, 3)).astype(np.float32)
tabular = np.random.normal(size=(1, len(tabular_columns))).astype(np.float32)
print(image.shape, tabular.shape)
print(model((image, tabular)).shape)



## === cell 14
tf.keras.backend.clear_session()
models = []
historys = []
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
train_best_fold = True
best_fold = 4

for index, (train_indices, val_indices) in enumerate(kfold.split(train)):
    if train_best_fold and index != best_fold:
        continue

    x_train = train.loc[train_indices, "file_path"].values
    tabular_train = train.loc[
        train_indices, ["Pawpularity"] + tabular_columns
    ].values.astype(np.float32)
    x_val = train.loc[val_indices, "file_path"].values
    tabular_val = train.loc[
        val_indices, ["Pawpularity"] + tabular_columns
    ].values.astype(np.float32)

    checkpoint_path = "model_%d.weights.h5" % (index)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        min_delta=1e-4,
        patience=10,
        restore_best_weights=False,
        monitor="val_loss",
        mode="min",
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3, patience=2, min_lr=1e-7, monitor="val_loss", mode="min"
    )
    callbacks = [early_stop, checkpoint, reduce_lr]

    optimizer = tf.keras.optimizers.Adam(1e-3)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, tabular_train))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(512)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, tabular_val))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    model = get_model()
    model.compile(loss=rmse, optimizer=optimizer, metrics=["mae", "mape"])
    history = model.fit(
        train_ds, epochs=300, validation_data=val_ds, callbacks=callbacks, verbose=2
    )

    try:
        for metrics in [("loss", "val_loss"), ("mae", "val_mae"), ("mape", "val_mape")]:
            pd.DataFrame(history.history, columns=list(metrics)).plot()
            plt.show()
        if "lr" in history.history:
            pd.DataFrame(history.history, columns=["lr"]).plot()
            plt.show()
    except Exception as e:
        print("Plotting skipped:", repr(e))

    model.load_weights(checkpoint_path)
    historys.append(history)
    models.append(model)




## === cell 15
def preprocess_test_data(image_url, tabular):
    image_string = tf.io.read_file(image_url)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.central_crop(image, 1.0)
    image = tf.image.resize(image, (image_size, image_size))
    tabular = tf.cast(tabular, tf.float32)
    return (image, tabular)


test_ds = (
    tf.data.Dataset.from_tensor_slices(
        (test["file_path"].values, test[tabular_columns].values.astype(np.float32))
    )
    .map(preprocess_test_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

use_best_result = False

if len(models) == 0:
    raise RuntimeError(
        "No trained models found; training loop did not produce any models."
    )

if use_best_result:
    if train_best_fold:
        best_model = models[0]
    else:
        best_fold = 0
        best_score = 10e8
        for fold, history in enumerate(historys):
            for val_rmse in history.history.get("val_loss", []):
                if val_rmse < best_score:
                    best_score = val_rmse
                    best_fold = fold
        print("Best Score:%.5f Best Fold: %d" % (best_score, best_fold + 1))
        best_model = models[best_fold]
    results = best_model.predict(test_ds, verbose=0).reshape(-1)
else:
    total_results = []
    for model in models:
        total_results.append(model.predict(test_ds, verbose=0).reshape(-1))
    results = np.mean(np.stack(total_results, axis=0), axis=0).reshape(-1)

results = np.clip(results, 0.0, 100.0)

if len(results) != len(sample_submission):
    raise ValueError(
        f"Prediction length {len(results)} does not match submission length {len(sample_submission)}"
    )

sub = sample_submission.copy()
sub["Pawpularity"] = results.astype(np.float32)
sub = sub[["Id", "Pawpularity"]]
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2243707673.py in <cell line: 0>()
     45     total_results = []
     46     for model in models:
---> 47         total_results.append(model.predict(test_ds, verbose=0).reshape(-1))
     48     results = np.mean(np.stack(total_results, axis=0), axis=0).reshape(-1)
     49 

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

ValueError: Layer "functional" expects 2 input(s), but it received 1 input tensors. Inputs received: [<tf.Tensor 'data:0' shape=(128, 128, 128, 3) dtype=float32>]
