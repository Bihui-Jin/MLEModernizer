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

# 5. Target score

20.50304

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import keras_tuner as kt


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
sample_submission = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")


## === cell 2
train.head()


## === cell 3
train["Pawpularity"].hist()


## === cell 4
batch_size = 128 # Batch Size
train_on_fold = 4 # Which fold to train, None to train on all folds.
tabular_columns = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']


## === cell 5
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true -  y_pred) ** 2))


## === cell 6
def build_model(hp):
    inputs = tf.keras.layers.Input((len(tabular_columns)))
    width = hp.Choice('width', [8, 16, 32, 64])
    depth = hp.Choice('depth', [3, 6, 9, 12])
    activation = "relu"
    dropout = hp.Choice('dropout', [0.0, 0.1, 0.2])
    use_batch_norm = hp.Choice('use_batch_norm', [True, False])
    loss_function = hp.Choice("loss_function", ["mse", "rmse"])
    kernel_regularizer = hp.Choice("kernel_regularizer", ["none", "l1", "l2", "l1_l2"])
    acutal_kernel_regularizer = None
    if acutal_kernel_regularizer == "l1":
        acutal_kernel_regularizer = keras.regularizers.l1()
    if acutal_kernel_regularizer == "l2":
        acutal_kernel_regularizer = keras.regularizers.l2()
    if acutal_kernel_regularizer == "l1_l2":
        acutal_kernel_regularizer = keras.regularizers.l1_l2()
    for i in range(depth):
        if i == 0:
            x = inputs
           
        x = keras.layers.Dense(
            width, 
            activation=activation,
            kernel_regularizer=acutal_kernel_regularizer
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
    print(x, y)
    return x, y


## === cell 8
tf.keras.backend.clear_session()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for index, (train_indices, val_indices) in enumerate(kfold.split(train)):
    if train_on_fold is not None and train_on_fold != index:
        continue
    train_features = train.loc[train_indices, tabular_columns]
    train_targets = train.loc[train_indices, ["Pawpularity"]]
    val_features = train.loc[val_indices, tabular_columns]
    val_targets = train.loc[val_indices, ["Pawpularity"]]
    checkpoint_path = "model_%d.h5"%(index)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_best_only=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        min_delta=1e-4, 
        patience=10
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3,
        patience=2, 
        min_lr=1e-7
    )
    callbacks = [early_stop, checkpoint, reduce_lr]
    print(train_features.shape, train_targets.shape)
    print(val_features.shape, val_targets.shape)
    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_targets)).map(preprocess).shuffle(512).batch(batch_size).cache().prefetch(2)
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_targets)).map(preprocess).batch(batch_size).cache().prefetch(2)
    tuner = kt.RandomSearch(
        build_model,
        objective=kt.Objective("val_rmse", direction="min"),
        max_trials=100,
        directory="directory"
    )
    tuner.search(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)
    break


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2643297671.py in <cell line: 0>()
     27     train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_targets)).map(preprocess).shuffle(512).batch(batch_size).cache().prefetch(2)
     28     val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_targets)).map(preprocess).batch(batch_size).cache().prefetch(2)
---> 29     tuner = kt.RandomSearch(
     30         build_model,
     31         objective=kt.Objective("val_rmse", direction="min"),

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/tuners/randomsearch.py in __init__(self, hypermodel, objective, max_trials, seed, hyperparameters, tune_new_entries, allow_new_entries, max_retries_per_trial, max_consecutive_failed_trials, **kwargs)
    172             max_consecutive_failed_trials=max_consecutive_failed_trials,
    173         )
--> 174         super().__init__(oracle, hypermodel, **kwargs)
    175 

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/tuner.py in __init__(self, oracle, hypermodel, max_model_size, optimizer, loss, metrics, distribution_strategy, directory, project_name, logger, tuner_id, overwrite, executions_per_trial, **kwargs)
    120             )
    121 
--> 122         super().__init__(
    123             oracle=oracle,
    124             hypermodel=hypermodel,

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in __init__(self, oracle, hypermodel, directory, project_name, overwrite, **kwargs)
    130         else:
    131             # Only populate initial space if not reloading.
--> 132             self._populate_initial_space()
    133 
    134         # Run in distributed mode.

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in _populate_initial_space(self)
    190         self.hypermodel.declare_hyperparameters(hp)
    191         self.oracle.update_space(hp)
--> 192         self._activate_all_conditions()
    193 
    194     def search(self, *fit_args, **fit_kwargs):

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in _activate_all_conditions(self)
    147         hp = self.oracle.get_space()
    148         while True:
--> 149             self.hypermodel.build(hp)
    150             self.oracle.update_space(hp)
    151 

/tmp/ipykernel_11/354670894.py in build_model(hp)
      1 def build_model(hp):
----> 2     inputs = tf.keras.layers.Input((len(tabular_columns)))
      3     width = hp.Choice('width', [8, 16, 32, 64])
      4     depth = hp.Choice('depth', [3, 6, 9, 12])
      5     activation = "relu"

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/input_layer.py in Input(shape, batch_size, dtype, sparse, batch_shape, name, tensor, optional)
    189     ```
    190     """
--> 191     layer = InputLayer(
    192         shape=shape,
    193         batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/input_layer.py in __init__(self, shape, batch_size, dtype, sparse, batch_shape, input_tensor, optional, name, **kwargs)
     90 
     91             if shape is not None:
---> 92                 shape = backend.standardize_shape(shape)
     93                 batch_shape = (batch_size,) + shape
     94 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in standardize_shape(shape)
    560             raise ValueError("Undefined shapes are not supported.")
    561         if not hasattr(shape, "__iter__"):
--> 562             raise ValueError(f"Cannot convert '{shape}' to a shape.")
    563         if config.backend() == "tensorflow":
    564             if isinstance(shape, tf.TensorShape):

ValueError: Cannot convert '12' to a shape.

## === cell 9
best_model = tuner.get_best_models()[0]
keras.utils.plot_model(best_model, show_shapes=True)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2588444247.py in <cell line: 0>()
----> 1 best_model = tuner.get_best_models()[0]
      2 keras.utils.plot_model(best_model, show_shapes=True)

NameError: name 'tuner' is not defined

## === cell 10
"""
 {'width': 64,
 'depth': 6,
 'dropout': 0.1,
 'use_batch_norm': 0,
 'loss_function': 'mse',
 'kernel_regularizer': 'none',
 'learing_rate': 0.0038644865099609653}
"""
best_hp = tuner.get_best_hyperparameters()[0]
best_hp.get_config()["values"]


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/559563030.py in <cell line: 0>()
      8  'learing_rate': 0.0038644865099609653}
      9 """
---> 10 best_hp = tuner.get_best_hyperparameters()[0]
     11 best_hp.get_config()["values"]

NameError: name 'tuner' is not defined

## === cell 11
models = []
for index, (train_indices, val_indices) in enumerate(kfold.split(train)):
    if train_on_fold is not None and train_on_fold != index:
        continue
    train_features = train.loc[train_indices, tabular_columns]
    train_targets = train.loc[train_indices, ["Pawpularity"]]
    val_features = train.loc[val_indices, tabular_columns]
    val_targets = train.loc[val_indices, ["Pawpularity"]]
    checkpoint_path = "model_%d.h5"%(index)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_best_only=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        min_delta=1e-4, 
        patience=10
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3,
        patience=2, 
        min_lr=1e-7
    )
    callbacks = [early_stop, checkpoint, reduce_lr]
    print(train_features.shape, train_targets.shape)
    print(val_features.shape, val_targets.shape)
    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_targets)).map(preprocess).shuffle(512).batch(batch_size).cache().prefetch(2)
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_targets)).map(preprocess).batch(batch_size).cache().prefetch(2)
    model = tuner.hypermodel.build(best_hp)
    history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=callbacks)
    model.load_weights(checkpoint_path)
    models.append(model)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2936424571.py in <cell line: 0>()
     26     train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_targets)).map(preprocess).shuffle(512).batch(batch_size).cache().prefetch(2)
     27     val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_targets)).map(preprocess).batch(batch_size).cache().prefetch(2)
---> 28     model = tuner.hypermodel.build(best_hp)
     29     history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=callbacks)
     30     model.load_weights(checkpoint_path)

NameError: name 'tuner' is not defined

## === cell 12
def preprocess_test_data(x):
    x = tf.cast(x, tf.float32)
    print(x)
    return x


## === cell 13
test_ds = tf.data.Dataset.from_tensor_slices((test[tabular_columns])).map(preprocess_test_data).batch(batch_size).prefetch(2)


## === cell 14
if train_on_fold is not None:
    best_model = models[0]
    results = best_model.predict(test_ds).reshape(-1)
else:
    total_results = []
    for model in models:
        total_results.append(model.predict(test_ds).reshape(-1))
    results = np.mean(total_results, axis=0).reshape(-1)
sample_submission["Pawpularity"] = results
sample_submission.to_csv("submission.csv", index=False)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/51639747.py in <cell line: 0>()
      1 if train_on_fold is not None:
----> 2     best_model = models[0]
      3     results = best_model.predict(test_ds).reshape(-1)
      4 else:
      5     total_results = []

IndexError: list index out of range
