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

3.9

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
pillow==11.3.0
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
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("PROTOCOL_BUFFERS_DISABLE_CPP_IMPLEMENTATION", None)

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

import matplotlib.pyplot as plt  # kept to preserve original imports

np.random.seed(42)
tf.random.set_seed(42)

try:
    _cpu_cnt = os.cpu_count() or 2
    _thr = max(1, min(8, _cpu_cnt))
    tf.config.threading.set_intra_op_parallelism_threads(_thr)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 1
df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
df_test = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
sample_sub = pd.read_csv(
    "/kaggle/input/petfinder-pawpularity-score/sample_submission.csv"
)

print(df.shape, df_test.shape)
print(df.columns.tolist())




## === cell 2
def image_generalize(image_path: tf.Tensor):
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)  # deterministic decode on CPU
    image = tf.image.resize(
        image, (128, 128), method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image = tf.cast(image, tf.float32) / 255.0
    return image


train_dir = "/kaggle/input/petfinder-pawpularity-score/train"
test_dir = "/kaggle/input/petfinder-pawpularity-score/test"

train_paths = tf.constant((train_dir + "/" + df["Id"].astype(str) + ".jpg").values)
ytrain = df["Pawpularity"].astype(np.float32).values
test_paths = tf.constant((test_dir + "/" + df_test["Id"].astype(str) + ".jpg").values)

BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE

val_size = int(len(df) * 0.2)
train_size = len(df) - val_size


def _map_train(path, y):
    return image_generalize(path), tf.cast(y, tf.float32)


base_ds = tf.data.Dataset.from_tensor_slices((train_paths, ytrain))

val_ds = base_ds.take(val_size)
train_ds = base_ds.skip(val_size)

cache_root = "/kaggle/working/tfds_cache_128_v1"
os.makedirs(cache_root, exist_ok=True)

train_cache = os.path.join(cache_root, "train")
val_cache = os.path.join(cache_root, "val")
test_cache = os.path.join(cache_root, "test")


def _safe_cache(ds, cache_path: str):
    """
    BUGFIX: file-based Dataset.cache requires the parent directory to exist and be writable.
    If anything goes wrong, fall back to in-memory caching for stability.
    """
    try:
        parent = os.path.dirname(cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return ds.cache(cache_path)
    except Exception as e:
        print(f"Warning: falling back to in-memory cache for {cache_path}: {repr(e)}")
        return ds.cache()


train_ds = train_ds.map(_map_train, num_parallel_calls=AUTO, deterministic=True)
val_ds = val_ds.map(_map_train, num_parallel_calls=AUTO, deterministic=True)

train_ds = _safe_cache(train_ds, train_cache)
val_ds = _safe_cache(val_ds, val_cache)

train_ds = train_ds.shuffle(
    buffer_size=train_size, seed=42, reshuffle_each_iteration=False
)

train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths).map(
    image_generalize, num_parallel_calls=AUTO, deterministic=True
)
test_ds = _safe_cache(test_ds, test_cache)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

print(
    "Prepared datasets:",
    f"train batches={int(np.ceil(train_size / BATCH_SIZE))},",
    f"val batches={int(np.ceil(val_size / BATCH_SIZE))},",
    f"test batches={int(np.ceil(len(df_test) / BATCH_SIZE))}",
)



## === cell 3
model = Sequential()
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        input_shape=(128, 128, 3),
    )
)
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(Conv2D(filters=8, kernel_size=(3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=1, activation="relu"))



## === cell 4
model.compile(
    loss="mse",
    optimizer="Adam",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae", "mape"],
    steps_per_execution=32,
)

history = model.fit(train_ds, validation_data=val_ds, epochs=15)



## === cell 5
pred = model.predict(test_ds, verbose=0)
pred = pred.reshape(-1).astype(np.float32)

pred = np.clip(pred, 0.0, 100.0)

final = pd.DataFrame({"Id": df_test["Id"].values, "Pawpularity": pred})
final = sample_sub[["Id"]].merge(final, on="Id", how="left")

if final["Pawpularity"].isna().any():
    missing = final.loc[final["Pawpularity"].isna(), "Id"].iloc[:5].tolist()
    raise ValueError(
        f"Submission has missing predictions for some Ids, e.g.: {missing}"
    )

final["Pawpularity"] = final["Pawpularity"].astype(np.float32)
final.to_csv("submission.csv", index=False)

print(final.head())
print("Wrote submission.csv with shape:", final.shape)
print("submission.csv path: submission.csv")
