# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)  # avoid surprise compile overhead in short runs
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print("TF:", tf.__version__)



## === cell 1
CANDIDATE_DIRS = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
]
DATA_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "sample_submission.csv")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate plant-pathology-2021-fgvc8 data directory. Checked: "
        + ", ".join(CANDIDATE_DIRS)
    )

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print("Using DATA_DIR:", DATA_DIR)
print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
trans_label = MultiLabelBinarizer().fit(label_split)
labels = pd.DataFrame(trans_label.transform(label_split), columns=trans_label.classes_)

for c in labels.columns:
    train[c] = labels[c].values

labels.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32

AUTOTUNE = tf.data.AUTOTUNE

y_cols = list(labels.columns)

train_df = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

n_total = len(train_df)
n_valid = int(round(0.1 * n_total))
n_train = n_total - n_valid

valid_df = train_df.iloc[:n_valid].copy()
train_df_part = train_df.iloc[n_valid:].copy()


def _build_paths_and_labels(df, img_dir, y_cols=None):
    paths = (img_dir.rstrip("/") + "/" + df["image"].astype(str)).to_numpy()
    if y_cols is None:
        return paths
    y = df[y_cols].to_numpy(dtype=np.float32, copy=False)
    return paths, y


train_paths, train_y = _build_paths_and_labels(train_df_part, TRAIN_IMG_DIR, y_cols)
valid_paths, valid_y = _build_paths_and_labels(valid_df, TRAIN_IMG_DIR, y_cols)
test_paths = _build_paths_and_labels(submissions, TEST_IMG_DIR, y_cols=None)


def _load_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")  # rgb
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _map_train(path, target):
    return _load_and_preprocess(path), target


def _map_eval(path, target):
    return _load_and_preprocess(path), target


def make_train_image_ds(paths, y, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    shuffle_buf = int(min(len(paths), 4096))
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_map_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_eval_image_ds(paths, y, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.map(_map_eval, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_image_ds(paths, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _apply_ds_options(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    try:
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    opts.autotune.enabled = True
    return ds.with_options(opts)


train_image_ds = _apply_ds_options(
    make_train_image_ds(train_paths, train_y, batch_size)
)
valid_image_ds = _apply_ds_options(make_eval_image_ds(valid_paths, valid_y, batch_size))
test_image_ds = _apply_ds_options(make_test_image_ds(test_paths, batch_size))

steps_per_epoch = int(np.ceil(len(train_paths) / batch_size))
validation_steps = int(np.ceil(len(valid_paths) / batch_size))
test_steps = int(np.ceil(len(test_paths) / batch_size))

print("train/valid/test sizes:", len(train_paths), len(valid_paths), len(test_paths))
print("steps:", steps_per_epoch, validation_steps, test_steps)



## === cell 4
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
    pooling="avg",
)
base.trainable = False

feat_dim = int(base.output_shape[-1])
head_inputs = keras.Input(shape=(feat_dim,), name="base_features")
x = keras.layers.Dropout(0.2)(head_inputs)
head_outputs = keras.layers.Dense(len(y_cols), activation="sigmoid")(x)
head_model = keras.Model(head_inputs, head_outputs, name="frozen_base_head")

head_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy"
)

head_model.summary()




## === cell 5
def _as_image_only(ds_with_labels):
    return ds_with_labels.map(
        lambda x, y: x, num_parallel_calls=AUTOTUNE, deterministic=True
    )


train_img_only = _apply_ds_options(_as_image_only(train_image_ds))
valid_img_only = _apply_ds_options(_as_image_only(valid_image_ds))
test_img_only = _apply_ds_options(test_image_ds)

train_valid_img_only = _apply_ds_options(train_img_only.concatenate(valid_img_only))
train_valid_feat = base.predict(
    train_valid_img_only,
    steps=steps_per_epoch + validation_steps,
    verbose=1,
)
train_feat = np.asarray(train_valid_feat[: len(train_paths)], dtype=np.float32)
valid_feat = np.asarray(train_valid_feat[len(train_paths) :], dtype=np.float32)

test_feat = base.predict(
    test_img_only,
    steps=test_steps,
    verbose=1,
)
test_feat = np.asarray(test_feat, dtype=np.float32)

print("Feature shapes:", train_feat.shape, valid_feat.shape, test_feat.shape)

train_feat_ds = tf.data.Dataset.from_tensor_slices((train_feat, train_y))
shuffle_buf = int(min(len(train_feat), 4096))
train_feat_ds = train_feat_ds.shuffle(
    buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
)
train_feat_ds = train_feat_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
train_feat_ds = _apply_ds_options(train_feat_ds)

valid_feat_ds = tf.data.Dataset.from_tensor_slices((valid_feat, valid_y))
valid_feat_ds = valid_feat_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
valid_feat_ds = _apply_ds_options(valid_feat_ds)

test_feat_ds = tf.data.Dataset.from_tensor_slices(test_feat)
test_feat_ds = test_feat_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
test_feat_ds = _apply_ds_options(test_feat_ds)



## === cell 6
EPOCHS = 3

history = head_model.fit(
    train_feat_ds,
    validation_data=valid_feat_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 7
preds = head_model.predict(test_feat_ds, steps=test_steps, verbose=1)
preds = np.asarray(preds, dtype=np.float32)

print("preds shape:", preds.shape)
print(preds[:2])



## === cell 8
thresh = {
    "complex": 0.25,
    "frog_eye_leaf_spot": 0.25,
    "healthy": 0.25,
    "powdery_mildew": 0.25,
    "rust": 0.25,
    "scab": 0.25,
}

missing = set(thresh.keys()) - set(y_cols)
if missing:
    raise ValueError(
        f"Thresholds defined for labels not in training classes: {missing}"
    )

label_to_index = {lab: i for i, lab in enumerate(y_cols)}
preds_np = np.asarray(preds)

n = len(submissions)
healthy_idx = label_to_index["healthy"]

argmax_idx = preds_np.argmax(axis=1)
max_vals = preds_np[np.arange(n), argmax_idx]
is_healthy_argmax = argmax_idx == healthy_idx

th_labs = list(thresh.keys())
th_idxs = np.array([label_to_index[lab] for lab in th_labs], dtype=np.int64)
th_vals = np.array([thresh[lab] for lab in th_labs], dtype=np.float32)
mask_th = preds_np[:, th_idxs] > th_vals[None, :]

out_labels = [""] * n
for i in range(n):
    if is_healthy_argmax[i]:
        out_labels[i] = "healthy"
        continue

    labs_i = [th_labs[k] for k in range(len(th_labs)) if mask_th[i, k]]
    s = " ".join(labs_i).strip()

    if (s == "") or ("healthy" in s.split()):
        best = max_vals[i]
        tie_mask = preds_np[i] >= (best - 1e-12)
        s = " ".join([y_cols[j] for j in np.flatnonzero(tie_mask)]).strip()

    if s == "":
        s = y_cols[int(argmax_idx[i])]

    out_labels[i] = s

submissions = submissions.copy()
submissions["labels"] = out_labels
submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)
print(submissions.head())
print("Wrote submission.csv with shape:", submissions.shape)



## === cell 9
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(submissions)
assert chk["labels"].isna().sum() == 0
assert (chk["labels"].astype(str).str.len() > 0).all()
print("Submission file OK:", chk.shape)
