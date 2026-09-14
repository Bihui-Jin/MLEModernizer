# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, sys, subprocess
import numpy as np
import pandas as pd

try:
    import google.protobuf  # noqa: F401
    import pkgutil, importlib
    import pkg_resources

    pb_ver = pkg_resources.get_distribution("protobuf").version
    if int(pb_ver.split(".")[0]) >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import keras_tuner as kt

np.random.seed(42)
tf.random.set_seed(42)



## === cell 1
train = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
sample_submission = pd.read_csv(
    "../input/petfinder-pawpularity-score/sample_submission.csv"
)



## === cell 2
train.head()



## === cell 3
train["Pawpularity"].hist()



## === cell 4
batch_size = 128  # Batch Size
train_on_fold = None  # Which fold to train, None to train on all folds.
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




## === cell 5
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))




## === cell 6
def build_model(hp):
    inputs = tf.keras.layers.Input(shape=(len(tabular_columns),))
    width = hp.Choice("width", [8, 16, 32, 64])
    depth = hp.Choice("depth", [3, 6, 9, 12])
    activation = "relu"
    dropout = hp.Choice("dropout", [0.0, 0.1, 0.2])
    use_batch_norm = hp.Choice("use_batch_norm", [True, False])
    loss_function = hp.Choice("loss_function", ["mse", "rmse"])
    kernel_regularizer = hp.Choice("kernel_regularizer", ["none", "l1", "l2", "l1_l2"])

    actual_kernel_regularizer = None
    if kernel_regularizer == "l1":
        actual_kernel_regularizer = keras.regularizers.l1()
    if kernel_regularizer == "l2":
        actual_kernel_regularizer = keras.regularizers.l2()
    if kernel_regularizer == "l1_l2":
        actual_kernel_regularizer = keras.regularizers.l1_l2()

    x = inputs
    for i in range(depth):
        x = keras.layers.Dense(
            width,
            activation=activation,
            kernel_regularizer=actual_kernel_regularizer,
        )(x)
        if (i + 1) % 3 == 0:
            if dropout > 0:
                x = keras.layers.Dropout(dropout)(x)
            if use_batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Concatenate()([x, inputs])

    output = keras.layers.Dense(1, activation="relu")(x)
    model = keras.Model(inputs=inputs, outputs=output)

    adam = keras.optimizers.Adam(learning_rate=hp.Float("learing_rate", 1e-5, 5e-3))
    loss = "mse" if loss_function == "mse" else rmse
    model.compile(loss=loss, optimizer=adam, metrics=["mae", rmse])
    return model




## === cell 7
def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return x, y




## === cell 8
tf.keras.backend.clear_session()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

tuner = None
best_hp = None

for index, (train_indices, val_indices) in enumerate(kfold.split(train)):
    if train_on_fold is not None and train_on_fold != index:
        continue

    train_features = train.loc[train_indices, tabular_columns].to_numpy(np.float32)
    train_targets = train.loc[train_indices, ["Pawpularity"]].to_numpy(np.float32)
    val_features = train.loc[val_indices, tabular_columns].to_numpy(np.float32)
    val_targets = train.loc[val_indices, ["Pawpularity"]].to_numpy(np.float32)

    checkpoint_path = f"tuner_fold_{index}.weights.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, save_weights_only=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        min_delta=1e-4, patience=10, restore_best_weights=False
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3, patience=2, min_lr=1e-7
    )
    callbacks = [early_stop, checkpoint, reduce_lr]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_features, train_targets))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(512, seed=42)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_features, val_targets))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    tuner = kt.RandomSearch(
        build_model,
        objective=kt.Objective("val_rmse", direction="min"),
        max_trials=100,
        directory="directory",
        project_name="pawpularity_tabular",
        overwrite=True,
        seed=42,
    )
    tuner.search(
        train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks, verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    break



## === cell 9
best_model = tuner.get_best_models(1)[0]
try:
    keras.utils.plot_model(best_model, show_shapes=True)
except Exception:
    pass



## === cell 10
"""
Example best config (will vary by run):
 {'width': 64,
  'depth': 6,
  'dropout': 0.1,
  'use_batch_norm': 0,
  'loss_function': 'mse',
  'kernel_regularizer': 'none',
  'learing_rate': 0.0038644865099609653}
"""
best_hp.get_config()["values"]



## === cell 11
models = []
for index, (train_indices, val_indices) in enumerate(kfold.split(train)):
    if train_on_fold is not None and train_on_fold != index:
        continue

    train_features = train.loc[train_indices, tabular_columns].to_numpy(np.float32)
    train_targets = train.loc[train_indices, ["Pawpularity"]].to_numpy(np.float32)
    val_features = train.loc[val_indices, tabular_columns].to_numpy(np.float32)
    val_targets = train.loc[val_indices, ["Pawpularity"]].to_numpy(np.float32)

    checkpoint_path = f"model_{index}.weights.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, save_weights_only=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        min_delta=1e-4, patience=10, restore_best_weights=False
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3, patience=2, min_lr=1e-7
    )
    callbacks = [early_stop, checkpoint, reduce_lr]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_features, train_targets))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(512, seed=42)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_features, val_targets))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    model = tuner.hypermodel.build(best_hp)
    model.fit(
        train_ds, epochs=50, validation_data=val_ds, callbacks=callbacks, verbose=1
    )
    model.load_weights(checkpoint_path)
    models.append(model)




## === cell 12
def preprocess_test_data(x):
    x = tf.cast(x, tf.float32)
    return x




## === cell 13
test_features = test[tabular_columns].to_numpy(np.float32)
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_features)
    .map(preprocess_test_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

total_results = []
for model in models:
    preds = model.predict(test_ds, verbose=0).reshape(-1)
    total_results.append(preds)

results = np.mean(np.stack(total_results, axis=0), axis=0).reshape(-1)

results = np.clip(results, 1.0, 100.0)

assert len(results) == len(sample_submission), (len(results), len(sample_submission))

sample_submission["Pawpularity"] = results
sample_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())
