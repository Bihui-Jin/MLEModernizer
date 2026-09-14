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

3.13

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

18.70930197642729

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

_cpu = os.cpu_count() or 2
tf.config.threading.set_intra_op_parallelism_threads(max(1, _cpu // 2))
tf.config.threading.set_inter_op_parallelism_threads(max(1, _cpu // 2))

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

BASE_DIR = "/kaggle/input/petfinder-pawpularity-score"
train_csv_path = os.path.join(BASE_DIR, "train.csv")
test_csv_path = os.path.join(BASE_DIR, "test.csv")
train_img_dir = os.path.join(BASE_DIR, "train")
test_img_dir = os.path.join(BASE_DIR, "test")

assert os.path.exists(train_csv_path), f"Missing {train_csv_path}"
assert os.path.exists(test_csv_path), f"Missing {test_csv_path}"
assert os.path.exists(train_img_dir), f"Missing {train_img_dir}"
assert os.path.exists(test_img_dir), f"Missing {test_img_dir}"

df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

meta_cols = df_test.columns.tolist()
meta_cols.remove("Id")  # 12 metadata cols
assert (
    len(meta_cols) == 12
), f"Expected 12 metadata cols, got {len(meta_cols)}: {meta_cols}"

print("Train shape:", df_train.shape, "Test shape:", df_test.shape)
print("Metadata columns:", meta_cols)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = (224, 224)

feature_extractor = ResNet101(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
feature_extractor.trainable = False

CACHE_DIR = "/kaggle/working/pawpularity_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_paths(tag: str):
    return (
        os.path.join(CACHE_DIR, f"{tag}_img_feats.npy"),
        os.path.join(CACHE_DIR, f"{tag}_meta_feats.npy"),
        os.path.join(CACHE_DIR, f"{tag}_preproc_img_cache"),  # tf.data cache directory
    )


@tf.function(reduce_retracing=True)
def _load_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


def extract_features_for_df(df, img_dir, batch_size=128, cache_tag=None):
    img_cache_path, meta_cache_path, preproc_cache_dir = (None, None, None)
    if cache_tag is not None:
        img_cache_path, meta_cache_path, preproc_cache_dir = _cache_paths(cache_tag)
        if os.path.exists(img_cache_path) and os.path.exists(meta_cache_path):
            feats_np = np.load(img_cache_path, mmap_mode=None).astype(
                np.float32, copy=False
            )
            metas = np.load(meta_cache_path, mmap_mode=None).astype(
                np.float32, copy=False
            )
            return feats_np, metas

    ids = df["Id"].astype(str).to_numpy()

    paths = np.char.add(np.char.add(img_dir + os.sep, ids), ".jpg").astype(object)

    metas = df[meta_cols].to_numpy(dtype=np.float32, copy=True)

    options = tf.data.Options()
    options.deterministic = True

    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
    ds = ds.map(
        _load_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache_tag is not None:
        ds = ds.cache(preproc_cache_dir)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    feats_np = feature_extractor.predict(ds, verbose=0).astype(np.float32, copy=False)

    if cache_tag is not None:
        np.save(img_cache_path, feats_np)
        np.save(meta_cache_path, metas)

    return feats_np, metas


FE_BATCH = 256

image_features, features = extract_features_for_df(
    df_train, train_img_dir, batch_size=FE_BATCH, cache_tag="train"
)
responses = df_train["Pawpularity"].to_numpy(dtype=np.float32)

image_features_test, features_test = extract_features_for_df(
    df_test, test_img_dir, batch_size=FE_BATCH, cache_tag="test"
)

print(
    "Extracted train image_features:",
    image_features.shape,
    "train meta:",
    features.shape,
    "y:",
    responses.shape,
)
print(
    "Extracted test  image_features:",
    image_features_test.shape,
    "test  meta:",
    features_test.shape,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/797885950.py in <cell line: 0>()
     84 FE_BATCH = 256
     85 
---> 86 image_features, features = extract_features_for_df(
     87     df_train, train_img_dir, batch_size=FE_BATCH, cache_tag="train"
     88 )

/tmp/ipykernel_11/797885950.py in extract_features_for_df(df, img_dir, batch_size, cache_tag)
     53     # Speed: avoid Python loop creating ~9k strings; NumPy char operations are faster and equivalent.
     54     # Produces identical paths like f"{img_dir}/{id}.jpg".
---> 55     paths = np.char.add(np.char.add(img_dir + os.sep, ids), ".jpg").astype(object)
     56 
     57     metas = df[meta_cols].to_numpy(dtype=np.float32, copy=True)

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U48' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 2
X_images_train, X_images_test, X_features_train, X_features_test, y_train, y_test = (
    train_test_split(
        image_features, features, responses, test_size=0.2, random_state=SEED
    )
)

image_input = Input(shape=(image_features.shape[1],))
x = Dense(256, activation="relu")(image_input)

feature_input = Input(shape=(12,))
y = Dense(64, activation="relu")(feature_input)

combined = Concatenate()([x, y])
z = Dense(256, activation="relu")(combined)
z = Dropout(0.2)(z)
z = Dense(512, activation="relu")(z)
z = Dropout(0.2)(z)
z = Dense(128, activation="relu")(z)
z = Dropout(0.2)(z)
z = Dense(48, activation="relu")(z)
z = Dropout(0.2)(z)
z = Dense(10, activation="relu")(z)
z = Dropout(0.3)(z)
output = Dense(1, activation="sigmoid")(z)

batch_size = 256
epochs = 1000
early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0000001,
    patience=20,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
    start_from_epoch=3,
)

y_train_100 = y_train / 100.0
y_test_100 = y_test / 100.0

model = Model(inputs=[image_input, feature_input], outputs=output)

model.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mae"], steps_per_execution=50
)

X_images_train = np.ascontiguousarray(X_images_train, dtype=np.float32)
X_features_train = np.ascontiguousarray(X_features_train, dtype=np.float32)
X_images_test = np.ascontiguousarray(X_images_test, dtype=np.float32)
X_features_test = np.ascontiguousarray(X_features_test, dtype=np.float32)
y_train_100 = np.ascontiguousarray(y_train_100, dtype=np.float32)
y_test_100 = np.ascontiguousarray(y_test_100, dtype=np.float32)

n_train = X_images_train.shape[0]
val_size = int(np.floor(0.2 * n_train))
train_size = n_train - val_size

X_img_tr, X_img_val = X_images_train[:train_size], X_images_train[train_size:]
X_meta_tr, X_meta_val = X_features_train[:train_size], X_features_train[train_size:]
y_tr, y_val = y_train_100[:train_size], y_train_100[train_size:]

train_ds = tf.data.Dataset.from_tensor_slices(((X_img_tr, X_meta_tr), y_tr))
train_ds = train_ds.shuffle(
    buffer_size=train_size, seed=SEED, reshuffle_each_iteration=True
)

train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(((X_img_val, X_meta_val), y_val))
val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[early_stop],
    verbose=1,
)

test_ds = tf.data.Dataset.from_tensor_slices(
    ((X_images_test, X_features_test), y_test_100)
)
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

loss, mae = model.evaluate(test_ds, verbose=0)
print(f"Test Loss: {loss}, Test MAE: {mae}")
print("RMSE:", (np.sqrt(loss) * 100.0))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2395325523.py in <cell line: 0>()
      1 X_images_train, X_images_test, X_features_train, X_features_test, y_train, y_test = (
      2     train_test_split(
----> 3         image_features, features, responses, test_size=0.2, random_state=SEED
      4     )
      5 )

NameError: name 'image_features' is not defined

## === cell 3
pass




## === cell 4
pred_submit_ds = tf.data.Dataset.from_tensor_slices(
    (image_features_test, features_test)
)
pred_submit_ds = pred_submit_ds.batch(256, drop_remainder=False).prefetch(
    tf.data.AUTOTUNE
)

prediction = model.predict(pred_submit_ds, verbose=0).astype(np.float32, copy=False)
prediction = prediction * 100.0
prediction = np.clip(prediction, 0.0, 100.0)

results_df = pd.DataFrame(
    {"Id": df_test["Id"].values, "Pawpularity": prediction.flatten()}
)
results_df.to_csv("submission.csv", index=False)
print("Results saved to 'submission.csv'")
print(results_df.head())
print("Submission shape:", results_df.shape)
assert results_df.shape[0] == df_test.shape[0]
assert list(results_df.columns) == ["Id", "Pawpularity"]
assert results_df["Id"].isna().sum() == 0
assert results_df["Pawpularity"].isna().sum() == 0

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/423696910.py in <cell line: 0>()
      1 pred_submit_ds = tf.data.Dataset.from_tensor_slices(
----> 2     (image_features_test, features_test)
      3 )
      4 pred_submit_ds = pred_submit_ds.batch(256, drop_remainder=False).prefetch(
      5     tf.data.AUTOTUNE

NameError: name 'image_features_test' is not defined
