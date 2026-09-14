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

20.466488895709546

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
import matplotlib.pyplot as plt

import tf_keras as tf
from tf_keras import layers

from sklearn.model_selection import KFold

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/petfinder-pawpularity-score/train.csv"
test_path = "../input/petfinder-pawpularity-score/test.csv"

train_imgs_path = "../input/petfinder-pawpularity-score/train"
test_imgs_path = "../input/petfinder-pawpularity-score/test"



## === cell 2
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)



## === cell 3
train_ids = train_data["Id"].values
test_ids = test_data["Id"].values

train_data.drop(["Id"], axis=1, inplace=True)
test_data.drop(["Id"], axis=1, inplace=True)



## === cell 4
Y = train_data.pop("Pawpularity")



## === cell 5
Y = Y.values.astype(np.float32)
X = train_data.values.astype(np.float32)
X_test = test_data.values.astype(np.float32)




## === cell 6
def build_model(in_shape):
    model = tf.keras.Sequential()
    model.add(
        layers.Dense(
            32,
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeUniform(),
            input_shape=in_shape,
        )
    )
    model.add(
        layers.Dense(
            64, activation="relu", kernel_initializer=tf.keras.initializers.HeUniform()
        )
    )
    model.add(
        layers.Dense(
            32, activation="relu", kernel_initializer=tf.keras.initializers.HeUniform()
        )
    )
    model.add(
        layers.Dense(
            1, activation="relu", kernel_initializer=tf.keras.initializers.HeUniform()
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(delta=50.0),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    return model




## === cell 7
def scheduler(epoch, lr):
    print(f"EPOCH : {epoch} LEARNING RATE : {lr}")
    if epoch > 50:
        return (0.99 ** (epoch - 50)) * 0.001
    else:
        return lr


callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1844647233.py in <cell line: 0>()
      7 
      8 
----> 9 callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)
     10 

AttributeError: module 'tf_keras' has no attribute 'keras'

## === cell 8
EPOCH = 100
BATCH_SIZE = 32
N_SPLITS = 10

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

models = []
oof_rmse = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, Y)):
    print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = Y[train_idx], Y[valid_idx]

    model = build_model([X_train.shape[-1]])

    callback0 = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"BestModel{fold+1}.h5",
        monitor="val_root_mean_squared_error",
        save_best_only=True,
        verbose=1,
    )

    his = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        callbacks=[callback0, callback1],
        verbose=0,
    )

    best_model = tf.keras.models.load_model(f"BestModel{fold+1}.h5", compile=True)
    models.append(best_model)

    best_val_rmse = float(np.min(his.history["val_root_mean_squared_error"]))
    oof_rmse.append(best_val_rmse)
    print(f"Fold {fold+1} best val RMSE: {best_val_rmse:.5f}")

print(f"Mean of fold-best val RMSE: {np.mean(oof_rmse):.5f}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2268224149.py in <cell line: 0>()
     15     y_train, y_valid = Y[train_idx], Y[valid_idx]
     16 
---> 17     model = build_model([X_train.shape[-1]])
     18 
     19     # Save best per fold (local working directory)

/tmp/ipykernel_11/162987174.py in build_model(in_shape)
      1 def build_model(in_shape):
----> 2     model = tf.keras.Sequential()
      3     model.add(
      4         layers.Dense(
      5             32,

AttributeError: module 'tf_keras' has no attribute 'keras'

## === cell 9
test_preds = []
for model in models:
    pred = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)
    test_preds.append(pred)

test_pred = (np.sum(test_preds, axis=0) / len(test_preds)).reshape(-1)

test_pred = np.clip(test_pred, 0.0, 100.0)



## === cell 10
submission_file = pd.read_csv(
    "../input/petfinder-pawpularity-score/sample_submission.csv"
)

pred_df = pd.DataFrame({"Id": test_ids, "Pawpularity": test_pred})
submission_file = submission_file[["Id"]].merge(pred_df, on="Id", how="left")

submission_file.to_csv("submission.csv", index=False)
print(submission_file.head())
print("Wrote submission.csv with shape:", submission_file.shape)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1025896781.py in <cell line: 0>()
      4 
      5 # Ensure correct ordering by Id (robustness if any order differs)
----> 6 pred_df = pd.DataFrame({"Id": test_ids, "Pawpularity": test_pred})
      7 submission_file = submission_file[["Id"]].merge(pred_df, on="Id", how="left")
      8 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length
