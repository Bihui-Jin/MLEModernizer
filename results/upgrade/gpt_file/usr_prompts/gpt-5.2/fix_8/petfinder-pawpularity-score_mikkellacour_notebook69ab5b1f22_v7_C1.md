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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

tf_data_options = tf.data.Options()
tf_data_options.experimental_deterministic = True

print("TF version:", tf.__version__)



## === cell 1
BASE_PATH = "/kaggle/input/petfinder-pawpularity-score"
train_csv = os.path.join(BASE_PATH, "train.csv")
test_csv = os.path.join(BASE_PATH, "test.csv")
train_img_dir = os.path.join(BASE_PATH, "train")
test_img_dir = os.path.join(BASE_PATH, "test")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

meta_cols = [
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

assert "Id" in df_train.columns and "Pawpularity" in df_train.columns
assert all(c in df_train.columns for c in meta_cols)
assert all(c in df_test.columns for c in meta_cols)

print(df_train.shape, df_test.shape)



## === cell 2
backbone = ResNet101(
    weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
)


@tf.function(reduce_retracing=True)
def _load_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


@tf.function(reduce_retracing=True)
def _backbone_forward(batch_imgs):
    return backbone(batch_imgs, training=False)


def extract_image_features(ids, image_dir, batch_size=256, cache_prefix=None):
    paths = [os.path.join(image_dir, f"{img_id}.jpg") for img_id in ids]
    n = len(paths)
    feat_dim = int(backbone.output_shape[-1])

    if cache_prefix is not None:
        ids_path = f"{cache_prefix}_ids.npy"
        feats_path = f"{cache_prefix}_feats.dat"
        shape_path = f"{cache_prefix}_shape.npy"

        if (
            os.path.exists(ids_path)
            and os.path.exists(feats_path)
            and os.path.exists(shape_path)
        ):
            cached_ids = np.load(ids_path, allow_pickle=True).tolist()
            cached_shape = tuple(np.load(shape_path).tolist())
            if cached_ids == ids and cached_shape == (n, feat_dim):
                mm = np.memmap(
                    feats_path, mode="r", dtype=np.float32, shape=cached_shape
                )
                return ids, np.asarray(mm)

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths))
    ds = ds.with_options(tf_data_options)
    ds = ds.map(
        _load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    if cache_prefix is not None:
        feats_path = f"{cache_prefix}_feats.dat"
        feats_mm = np.memmap(
            feats_path, mode="w+", dtype=np.float32, shape=(n, feat_dim)
        )
    else:
        feats_mm = np.empty((n, feat_dim), dtype=np.float32)

    write_pos = 0

    def _write_feats_to_memmap(batch_feats_np):
        nonlocal write_pos
        b = batch_feats_np.shape[0]
        feats_mm[write_pos : write_pos + b] = batch_feats_np
        write_pos += b
        return np.int64(0)

    for batch_imgs in ds:
        batch_feats = _backbone_forward(batch_imgs)
        batch_feats_np = batch_feats.numpy()  # float32 already
        _write_feats_to_memmap(batch_feats_np)

    if cache_prefix is not None:
        np.save(f"{cache_prefix}_ids.npy", np.asarray(ids, dtype=object))
        np.save(f"{cache_prefix}_shape.npy", np.asarray([n, feat_dim], dtype=np.int64))
        feats_mm.flush()
        feats_arr = np.asarray(
            np.memmap(feats_path, mode="r", dtype=np.float32, shape=(n, feat_dim))
        )
        return ids, feats_arr

    return ids, np.asarray(feats_mm)


train_ids = df_train["Id"].tolist()
train_ids_kept, image_features = extract_image_features(
    train_ids,
    train_img_dir,
    batch_size=256,
    cache_prefix="/kaggle/working/resnet101_train",
)

if len(train_ids_kept) != len(train_ids):
    idx = {k: i for i, k in enumerate(train_ids_kept)}
    df_train_kept = df_train[df_train["Id"].isin(idx)].copy()
    df_train_kept["_ord"] = df_train_kept["Id"].map(idx).astype(np.int32)
    df_train_kept = (
        df_train_kept.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    )
else:
    df_train_kept = df_train

features = df_train_kept[meta_cols].to_numpy(dtype=np.float32, copy=False)
responses = df_train_kept["Pawpularity"].to_numpy(dtype=np.float32, copy=False)

print("Train image_features:", image_features.shape)
print("Train meta features:", features.shape)
print("Train responses:", responses.shape)



## === cell 3
test_ids = df_test["Id"].tolist()
test_ids_kept, image_features_test = extract_image_features(
    test_ids,
    test_img_dir,
    batch_size=256,
    cache_prefix="/kaggle/working/resnet101_test",
)

if len(test_ids_kept) != len(test_ids):
    idx = {k: i for i, k in enumerate(test_ids_kept)}
    df_test_kept = df_test[df_test["Id"].isin(idx)].copy()
    df_test_kept["_ord"] = df_test_kept["Id"].map(idx).astype(np.int32)
    df_test_kept = (
        df_test_kept.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    )
else:
    df_test_kept = df_test

features_test = df_test_kept[meta_cols].to_numpy(dtype=np.float32, copy=False)

print("Test image_features:", image_features_test.shape)
print("Test meta features:", features_test.shape)



## === cell 4
X_images_train, X_images_test, X_features_train, X_features_test, y_train, y_test = (
    train_test_split(
        image_features, features, responses, test_size=0.2, random_state=42
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

batch_size = 100
epochs = 1000
early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=20,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
    start_from_epoch=3,
)

y_train_100 = y_train / 100.0
y_test_100 = y_test / 100.0

model = Model(inputs=[image_input, feature_input], outputs=output)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

X_images_train = np.ascontiguousarray(X_images_train, dtype=np.float32)
X_features_train = np.ascontiguousarray(X_features_train, dtype=np.float32)
y_train_100 = np.ascontiguousarray(y_train_100, dtype=np.float32)

X_images_test = np.ascontiguousarray(X_images_test, dtype=np.float32)
X_features_test = np.ascontiguousarray(X_features_test, dtype=np.float32)
y_test_100 = np.ascontiguousarray(y_test_100, dtype=np.float32)

n_train = X_images_train.shape[0]
val_n = int(np.floor(0.2 * n_train))
train_n = n_train - val_n

X_img_tr, X_img_val = X_images_train[:train_n], X_images_train[train_n:]
X_feat_tr, X_feat_val = X_features_train[:train_n], X_features_train[train_n:]
y_tr, y_val = y_train_100[:train_n], y_train_100[train_n:]

train_ds = tf.data.Dataset.from_tensor_slices(((X_img_tr, X_feat_tr), y_tr))
train_ds = train_ds.with_options(tf_data_options)
train_ds = train_ds.shuffle(
    buffer_size=train_n, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(((X_img_val, X_feat_val), y_val))
val_ds = val_ds.with_options(tf_data_options)
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
test_ds = test_ds.with_options(tf_data_options)
test_ds = test_ds.batch(256, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

loss, mae = model.evaluate(test_ds, verbose=0)
print(f"Test Loss (MSE on scaled target): {loss}, Test MAE (scaled): {mae}")



## === cell 5
pred_ds = tf.data.Dataset.from_tensor_slices((X_images_test, X_features_test))
pred_ds = pred_ds.with_options(tf_data_options)
pred_ds = pred_ds.batch(512, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

predictions_val = model.predict(pred_ds, verbose=0)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, predictions_val.flatten() * 100, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=2, color="red")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.grid(True)
plt.show()



## === cell 6
print("RMSE:", (np.sqrt(loss) * 100))



## === cell 7
image_features_test = np.ascontiguousarray(image_features_test, dtype=np.float32)
features_test = np.ascontiguousarray(features_test, dtype=np.float32)

test_pred_ds = tf.data.Dataset.from_tensor_slices((image_features_test, features_test))
test_pred_ds = test_pred_ds.with_options(tf_data_options)
test_pred_ds = test_pred_ds.batch(512, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

prediction = model.predict(test_pred_ds, verbose=1).astype(np.float32)
prediction = prediction.flatten() * 100.0

prediction = np.clip(prediction, 0.0, 100.0)

submission = pd.DataFrame({"Id": df_test_kept["Id"].values, "Pawpularity": prediction})

print("Submission shape:", submission.shape)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
