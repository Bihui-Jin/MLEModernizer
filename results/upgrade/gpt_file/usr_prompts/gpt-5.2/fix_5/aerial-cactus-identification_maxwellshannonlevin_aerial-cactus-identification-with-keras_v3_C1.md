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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9991726666666668

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.46298) has done: 'I fix the environment-breaking Keras import issue by switching to `tensorflow.keras`, which avoids the `MessageFactory.GetPrototype` protobuf error and restores `ImageDataGenerator`, `Sequential`, callbacks, and `load_model`. I correct the dataset paths to point at the actual `/kaggle/input/aerial-cactus-identification/train/` and `/kaggle/input/aerial-cactus-identification/test/` folders to resolve the `FileNotFoundError`. I replace deprecated `DataFrame.append` with `pd.concat` and update callback monitors from `val_acc` to `val_accuracy` so training/checkpointing works on modern TF/Keras. Finally, I fix test image loading (`target_size` must be 2-tuple), ensure deterministic file ordering, and write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


print("TF version:", tf.__version__)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.data.experimental.enable_debug_mode = getattr(
    tf.data.experimental, "enable_debug_mode", None
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
cactus_dir = "/kaggle/input/aerial-cactus-identification"
train_dir = os.path.join(cactus_dir, "train")
test_dir = os.path.join(cactus_dir, "test")

df_train_data = pd.read_csv(os.path.join(cactus_dir, "train.csv"))
df_train_data["has_cactus"] = df_train_data["has_cactus"].astype(str)

df_test = pd.read_csv(os.path.join(cactus_dir, "sample_submission.csv"))

print("train_dir exists:", os.path.isdir(train_dir), train_dir)
print("test_dir exists:", os.path.isdir(test_dir), test_dir)
print("train.csv shape:", df_train_data.shape)
print("sample_submission shape:", df_test.shape)




## === cell 3
assert set(df_train_data.columns) == {"id", "has_cactus"}
assert set(df_test.columns) == {"id", "has_cactus"}
assert df_train_data["has_cactus"].isin(["0", "1"]).all()

first_img = os.path.join(train_dir, df_train_data.loc[0, "id"])
print("Example image exists:", os.path.isfile(first_img), first_img)




## === cell 4
df_train_data.head()




## === cell 5
print("The number of training images is: {}".format(len(df_train_data)))
df_train_data["has_cactus"].value_counts()




## === cell 6
try:
    pass
except Exception as e:
    print("Visualization skipped due to:", repr(e))




## === cell 7
has_cactus = df_train_data[df_train_data["has_cactus"] == "1"]
not_cactus = df_train_data[df_train_data["has_cactus"] == "0"]

print("has_cactus:", len(has_cactus), "not_cactus:", len(not_cactus))




## === cell 8
df_train = df_train_data.sample(frac=1, random_state=SEED).reset_index(drop=True)

df_valid = (
    df_train.groupby("has_cactus", group_keys=False)
    .apply(lambda x: x.sample(frac=0.1, random_state=SEED))
    .reset_index(drop=True)
)
df_train = df_train.drop(index=df_valid.index).reset_index(drop=True)

print("Train size:", len(df_train), "Valid size:", len(df_valid))
print("Train label distribution:\n", df_train["has_cactus"].value_counts())
print("Valid label distribution:\n", df_valid["has_cactus"].value_counts())




## === cell 9
new_image_size = 64
batch_size = 128

AUTO = tf.data.AUTOTUNE

train_paths = (train_dir + "/" + df_train["id"].values).astype(str)
train_labels = df_train["has_cactus"].astype(np.float32).values

valid_paths = (train_dir + "/" + df_valid["id"].values).astype(str)
valid_labels = df_valid["has_cactus"].astype(np.float32).values


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, (new_image_size, new_image_size), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    max_dx = tf.cast(tf.round(0.2 * tf.cast(new_image_size, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.2 * tf.cast(new_image_size, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([1, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([0, 1], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    angle = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([2, 2], tf.int32),
        minval=-50.0,
        maxval=50.0,
        dtype=tf.float32,
    )
    rad = angle * (np.pi / 180.0)
    try:
        import tensorflow_addons as tfa  # may not exist

        img = tfa.image.rotate(img, rad, interpolation="BILINEAR", fill_mode="nearest")
    except Exception:
        c = (new_image_size - 1) / 2.0
        cos_a = tf.cos(rad)
        sin_a = tf.sin(rad)
        a0 = cos_a
        a1 = -sin_a
        a2 = c - cos_a * c + sin_a * c
        b0 = sin_a
        b1 = cos_a
        b2 = c - sin_a * c - cos_a * c
        transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=img[None, ...],
            transforms=transform[None, ...],
            output_shape=tf.constant([new_image_size, new_image_size], dtype=tf.int32),
            interpolation="BILINEAR",
            fill_mode="NEAREST",
            fill_value=0.0,
        )[0]
    return img


def _make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _decode_map(path, y):
        img = _decode_and_resize(path)
        return path, img, y

    ds = ds.map(_decode_map, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()  # caches (path, decoded_img, y); augmentation remains after cache

    def _aug_map(path, img, y):
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed = tf.stack([tf.cast(h, tf.int32), tf.cast(SEED, tf.int32)])
        img = _augment(img, seed)
        return img, y

    ds = ds.map(_aug_map, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


def _make_valid_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, y):
        img = _decode_and_resize(path)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()  # valid set decode+resize only once; no augmentation anyway
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


train_generator = _make_train_ds(train_paths, train_labels)
valid_generator = _make_valid_ds(valid_paths, valid_labels)

steps_per_epoch = max(1, int(np.ceil(len(df_train) / batch_size)))
validation_steps = max(1, int(np.ceil(len(df_valid) / batch_size)))




## === cell 10
model = Sequential()
model.add(
    Conv2D(
        32, (3, 3), input_shape=(new_image_size, new_image_size, 3), activation="relu"
    )
)
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(512, (3, 3), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.summary()




## === cell 11
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])




## === cell 12
checkpoint_path = "model.keras"

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.2,
    patience=10,
    min_lr=1e-9,
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=100,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)




## === cell 13
epochs = 30

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1,
)




## === cell 14
if os.path.exists(checkpoint_path):
    model = load_model(checkpoint_path)
    print(f"Loaded best model from {checkpoint_path}")
else:
    print(f"Warning: {checkpoint_path} not found; using in-memory model weights.")




## === cell 15
eval_res = model.evaluate(valid_generator, steps=validation_steps, verbose=0)
print("Validation loss, accuracy:", eval_res)




## === cell 16
test_fnames = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_paths = (test_dir + "/" + np.array(test_fnames)).astype(str)


def _make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(path):
        img = _decode_and_resize(path)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


test_ds = _make_test_ds(test_paths)
print("Test images:", len(test_fnames))




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4183008569.py in <cell line: 0>()
      1 test_fnames = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
----> 2 test_paths = (test_dir + "/" + np.array(test_fnames)).astype(str)
      3 
      4 
      5 # Speed + correctness: cache test decode+resize (used once here, but still helps stabilize I/O),

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U48'), dtype('<U36')) -> None

## === cell 17
pred = model.predict(test_ds, verbose=0).reshape(-1)
pred = np.clip(pred, 1e-7, 1 - 1e-7)  # numeric safety for probabilities
print(pred[:5])




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4018018613.py in <cell line: 0>()
----> 1 pred = model.predict(test_ds, verbose=0).reshape(-1)
      2 pred = np.clip(pred, 1e-7, 1 - 1e-7)  # numeric safety for probabilities
      3 print(pred[:5])
      4 
      5 

NameError: name 'test_ds' is not defined

## === cell 18
submission = pd.DataFrame({"id": test_fnames, "has_cactus": pred.astype(np.float32)})

if "id" in df_test.columns and len(df_test) == len(submission):
    submission = df_test[["id"]].merge(submission, on="id", how="left")
    submission["has_cactus"] = submission["has_cactus"].fillna(0.5).astype(np.float32)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/35897715.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": test_fnames, "has_cactus": pred.astype(np.float32)})
      2 
      3 if "id" in df_test.columns and len(df_test) == len(submission):
      4     submission = df_test[["id"]].merge(submission, on="id", how="left")
      5     submission["has_cactus"] = submission["has_cactus"].fillna(0.5).astype(np.float32)

NameError: name 'pred' is not defined
