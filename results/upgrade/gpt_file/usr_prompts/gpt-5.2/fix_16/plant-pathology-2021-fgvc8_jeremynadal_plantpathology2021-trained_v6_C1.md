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
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

print("Using tensorflow", tf.__version__)

tfa = None

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import f1_score




## === cell 1
class CFG:
    classes = [
        "complex",
        "frog_eye_leaf_spot",
        "powdery_mildew",
        "rust",
        "scab",
        "healthy",
    ]
    batch_size = 16
    test_size = 0.25
    img_size = 512
    seed = 42
    retrain = False




## === cell 2
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "../input/plant-pathology-2021-fgvc8",
]
DATA_ROOT = None
for c in DATA_ROOT_CANDIDATES:
    if os.path.exists(c):
        DATA_ROOT = c
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate plant-pathology-2021-fgvc8 dataset directory."
    )

train_dir = os.path.join(DATA_ROOT, "train_images") + "/"
train_csv = os.path.join(DATA_ROOT, "train.csv")
test_dir = os.path.join(DATA_ROOT, "test_images") + "/"

duplicates = "../input/duplicatescsv/duplicates.csv"
model_dir = "../input/models/"

png_dir = "/kaggle/working/pngs/"

print("DATA_ROOT:", DATA_ROOT)
print("train_dir exists:", os.path.exists(train_dir))
print("train_csv exists:", os.path.exists(train_csv))
print("test_dir exists :", os.path.exists(test_dir))
print("duplicates exists:", os.path.exists(duplicates))
print("png_dir exists  :", os.path.exists(png_dir))



## === cell 3
os.makedirs(png_dir, exist_ok=True)



## === cell 4
pass



## === cell 5
df_train = pd.read_csv(train_csv)

train_imgs_list = os.listdir(train_dir)
train_imgs = set(train_imgs_list)
test_imgs = os.listdir(test_dir)

missing = [im for im in df_train["image"].tolist() if im not in train_imgs]
if len(missing) > 0:
    print(
        "Warning: some train.csv images not found in train_images. Example:",
        missing[:5],
    )

print("Train rows:", df_train.shape)
print("Test images:", len(test_imgs))
print(df_train.head())



## === cell 6
pass



## === cell 7
if os.path.exists(duplicates):
    df_duplicates = pd.read_csv(duplicates, header=None)
    df_duplicates.columns = ["img1", "img2"]
else:
    df_duplicates = pd.DataFrame(columns=["img1", "img2"])

print(f"The train dataset is composed of {df_train.shape[0]} labeled images")
print(f"The test dataset is composed of {len(test_imgs)} unlabeled images")
print("\nThere are {} duplicated images.\n".format(df_duplicates.shape[0]))



## === cell 8
true_duplicates = []
false_duplicates = []

if df_duplicates.shape[0] > 0:
    train_image_set = set(df_train["image"].tolist())
    labels_by_image = dict(zip(df_train["image"].tolist(), df_train["labels"].tolist()))

    d = df_duplicates.copy()
    d["in_train_1"] = d["img1"].isin(train_image_set)
    d["in_train_2"] = d["img2"].isin(train_image_set)
    d = d[d["in_train_1"] & d["in_train_2"]]

    if len(d):
        d["lab1"] = d["img1"].map(labels_by_image)
        d["lab2"] = d["img2"].map(labels_by_image)
        m_true = d["lab1"].to_numpy() == d["lab2"].to_numpy()
        true_duplicates = d.loc[m_true, "img1"].tolist()
        false_duplicates = list(
            d.loc[~m_true, ["img1", "img2"]].itertuples(index=False, name=None)
        )

print(
    "There are {} true duplicates and {} false ones.".format(
        len(true_duplicates), len(false_duplicates)
    )
)
print("Lets display the false duplicates")



## === cell 9
pass



## === cell 10
pass



## === cell 11
labels_flat = df_train["labels"].str.split(" ")
uniques = np.unique(np.concatenate(labels_flat.to_numpy()))
assert len(uniques) == len(CFG.classes), "ERROR : labels and CFG.classes mismatch"
for unique in uniques:
    assert unique in CFG.classes, "ERROR : labels and CFG.classes mismatch"



## === cell 12
df_train = df_train.copy()
df_train["labels"] = df_train["labels"].str.split(" ")
mlb = MultiLabelBinarizer(classes=CFG.classes)
labels_bin = mlb.fit_transform(df_train["labels"].values)
labels_bin = pd.DataFrame(columns=CFG.classes, data=labels_bin, index=df_train.index)
df_train.drop("labels", axis=1, inplace=True)
df_train = pd.concat([df_train, labels_bin], axis=1)
print(df_train.head())



## === cell 13
init = df_train.shape[0]

to_drop = set(true_duplicates)
if false_duplicates:
    a, b = zip(*false_duplicates)
    to_drop.update(a)
    to_drop.update(b)

if to_drop:
    df_train = df_train[~df_train["image"].isin(to_drop)]

end = df_train.shape[0]
df_train.reset_index(drop=True, inplace=True)
print("Deleted {} files".format(init - end))



## === cell 14
pass



## === cell 15
pass



## === cell 16
y_proxy = df_train[CFG.classes].astype(np.int8).astype(str).agg("".join, axis=1)

sss = StratifiedShuffleSplit(n_splits=1, test_size=CFG.test_size, random_state=CFG.seed)
X_idx = np.arange(len(df_train))
for train_index, test_index in sss.split(X_idx, y_proxy):
    X_train, X_test = df_train.loc[train_index].reset_index(drop=True), df_train.loc[
        test_index
    ].reset_index(drop=True)

compare_X_train, compare_X_test = train_test_split(
    df_train, test_size=CFG.test_size, random_state=CFG.seed
)

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## === cell 17
pass



## === cell 18
pass



## === cell 19
pass




## === cell 20
def pred2labels(pred, thresh=0.5, labels=CFG.classes):
    assert len(pred) == len(labels), f"Predictions must have shape : ({len(labels)},)"
    pred = [labels[i] for i in range(len(labels)) if pred[i] > thresh]
    if len(pred) == 0:
        pred = ["healthy"]
    return " ".join(pred)




## === cell 21
tf.keras.utils.set_random_seed(CFG.seed)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=CFG.seed),
        tf.keras.layers.RandomRotation(0.2, seed=CFG.seed),
        tf.keras.layers.RandomContrast(0.3, seed=CFG.seed),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, seed=CFG.seed
        ),
    ]
)




## === cell 22
@tf.function
def _decode_resize_from_train(file_name):
    img = tf.io.read_file(tf.strings.join([train_dir, file_name]))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [CFG.img_size, CFG.img_size])
    return img


@tf.function
def _map_train(file_name, label):
    return _decode_resize_from_train(file_name), label


@tf.function
def _map_aug(x, y):
    return data_augmentation(x, training=True), y


def prepare_dataset(
    X, augmentation=False, repeat=False, shuffle=False, cache_name=None
):
    file_names = tf.convert_to_tensor(X["image"].values, dtype=tf.string)
    labels = tf.convert_to_tensor(
        X[CFG.classes].values.astype(np.float32), dtype=tf.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.autotune.enabled = True
    except Exception:
        pass
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_fusion = True
    except Exception:
        pass
    dataset = dataset.with_options(options)

    if shuffle:
        buf = min(int(X.shape[0]), 2048)
        dataset = dataset.shuffle(
            buffer_size=buf, seed=CFG.seed, reshuffle_each_iteration=True
        )

    dataset = dataset.map(_map_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    if cache_name is not None:
        dataset = dataset.cache(cache_name)

    dataset = dataset.batch(CFG.batch_size, drop_remainder=False)

    if augmentation:
        dataset = dataset.map(_map_aug, num_parallel_calls=AUTOTUNE, deterministic=True)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset




## === cell 23
train_cache_path = "/kaggle/working/tf_cache_train.cache"
val_cache_path = "/kaggle/working/tf_cache_val.cache"

for p in (train_cache_path, val_cache_path):
    if os.path.exists(p):
        try:
            os.remove(p)
        except Exception:
            pass

ds_train = prepare_dataset(
    X_train, augmentation=True, repeat=True, shuffle=True, cache_name=train_cache_path
)
ds_test = prepare_dataset(
    X_test, augmentation=False, repeat=False, shuffle=False, cache_name=val_cache_path
)



## === cell 24
for inputs, outputs in ds_train.take(1):
    print("Input shape is:", inputs.shape, "output shape is:", outputs.shape)
    print(
        "label of this input is",
        outputs[0].numpy(),
        "corresponding to",
        pred2labels(outputs[0].numpy()),
    )
    break



## === cell 25
pass




## === cell 26
def create_cnn(
    input_shape,
    output_length,
    nb_cnn=3,
    nb_filters=64,
    activation_cnn="relu",
    model_transfert=None,
    fine_tune=False,
    nb_FC_layer=3,
    nb_FC_neurons=512,
    reducing=False,
    activation_FC="relu",
    dropout=0.0,
    activation_output="sigmoid",
    name="my_cnn_model",
):
    """Create a CNN based model if model_transfert is None. Else, model_transfert is used for feature extraction."""
    assert input_shape[-1] == 3, "For the moment only models with rgb input is dealt"
    if reducing:
        assert (
            nb_FC_neurons % 2**nb_FC_layer == 0
        ), "If reducing, nb_FC_neurons must be multiple of 2**nb_FC_layer "

    model = tf.keras.models.Sequential(name=name)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name="Input_layer"))

    if model_transfert is None:
        for cnn in range(nb_cnn):
            model.add(
                tf.keras.layers.Conv2D(
                    filters=nb_filters,
                    kernel_size=(3, 3),
                    padding="same",
                    activation=activation_cnn,
                    name="Conv2D_" + str(cnn + 1),
                )
            )
            model.add(
                tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2), name="MaxPool_" + str(cnn + 1)
                )
            )
    else:
        if not fine_tune:
            model_transfert.trainable = False
        model.add(model_transfert)
        model.add(
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool_transfer")
        )

    model.add(tf.keras.layers.Flatten())

    if reducing:
        for FC in range(nb_FC_layer):
            model.add(
                tf.keras.layers.Dense(
                    int(nb_FC_neurons / (2**FC)),
                    activation=activation_FC,
                    name="FC_layer_" + str(FC + 1),
                )
            )
            if dropout != 0.0:
                model.add(
                    tf.keras.layers.Dropout(dropout, name="Dropout_" + str(FC + 1))
                )
    else:
        for FC in range(nb_FC_layer):
            model.add(
                tf.keras.layers.Dense(
                    nb_FC_neurons,
                    activation=activation_FC,
                    name="FC_layer_" + str(FC + 1),
                )
            )
        if dropout != 0.0:
            model.add(tf.keras.layers.Dropout(dropout, name="Dropout_" + str(FC + 1)))

    model.add(
        tf.keras.layers.Dense(
            output_length, activation=activation_output, name="Output_layer"
        )
    )
    return model


def get_callbacks(monitor="val_loss", save_name=None, patience=8):
    if save_name:
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_name,
                monitor=monitor,
                save_best_only=True,
                verbose=0,
                mode="max" if "F1" in monitor or "f1" in monitor else "min",
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                mode="max" if "F1" in monitor or "f1" in monitor else "min",
            ),
        ]
    else:
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                mode="max" if "F1" in monitor or "f1" in monitor else "min",
            )
        ]




## === cell 27
base = tf.keras.applications.Xception(
    include_top=False, weights="imagenet", input_shape=(CFG.img_size, CFG.img_size, 3)
)

model = create_cnn(
    input_shape=(CFG.img_size, CFG.img_size, 3),
    output_length=len(CFG.classes),
    model_transfert=base,
    fine_tune=False,
    nb_FC_layer=2,
    nb_FC_neurons=512,
    reducing=True,
    activation_FC="relu",
    dropout=0,
    activation_output="sigmoid",
    name="my_model",
)
optimizer = tf.keras.optimizers.Adam(learning_rate=3.5e-5)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
    jit_compile=True,
    steps_per_execution=16,
)
model.summary()



## === cell 28
steps_per_epoch = int(np.ceil(X_train.shape[0] / CFG.batch_size))
validation_steps = int(np.ceil(X_test.shape[0] / CFG.batch_size))

local_ckpt = "/kaggle/working/model.h5"
if os.path.exists(local_ckpt) and not CFG.retrain:
    print("Loading model from file:", local_ckpt)
    model = tf.keras.models.load_model(local_ckpt, compile=False)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
        jit_compile=True,
        steps_per_execution=16,
    )
    history = None
else:
    history = model.fit(
        ds_train,
        validation_data=ds_test,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=get_callbacks(monitor="val_loss", save_name=local_ckpt, patience=4),
        epochs=10,
        verbose=2,
    )



## === cell 29
val_preds = model.predict(ds_test, verbose=0)
y_true = X_test[CFG.classes].values.astype(np.int32)

threshold_grid = np.linspace(0.10, 0.90, 17)  # coarse grid, fast and stable
best_t, best_f1 = 0.50, -1.0

y_true_flat = y_true.reshape(-1)
for t in threshold_grid:
    y_hat_flat = (val_preds >= t).astype(np.int32).reshape(-1)
    f1 = f1_score(y_true_flat, y_hat_flat, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_t = float(t)

print(
    f"Best global threshold on validation: {best_t:.3f} | macro-F1 (flattened): {best_f1:.5f}"
)



## === cell 30
pass



## === cell 31
pass




## === cell 32
@tf.function
def parse_test_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [CFG.img_size, CFG.img_size])
    return img


test_paths = sorted(tf.io.gfile.glob(os.path.join(test_dir, "*.jpg")))
test_files = [os.path.basename(p) for p in test_paths]

test_ds = tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor(test_paths, dtype=tf.string)
)
options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.autotune.enabled = True
except Exception:
    pass
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_fusion = True
except Exception:
    pass
test_ds = test_ds.with_options(options)
test_ds = (
    test_ds.map(parse_test_image, num_parallel_calls=AUTOTUNE, deterministic=True)
    .apply(tf.data.experimental.ignore_errors())
    .batch(CFG.batch_size)
    .prefetch(AUTOTUNE)
)

preds = model.predict(test_ds, verbose=0)

labels_arr = np.array(CFG.classes, dtype=object)
mask = preds >= best_t
idx_list = [np.flatnonzero(row) for row in mask]
pred_labels = [
    "healthy" if idxs.size == 0 else " ".join(labels_arr[idxs].tolist())
    for idxs in idx_list
]

df_sub = pd.DataFrame({"image": test_files, "labels": pred_labels})
print(df_sub.head())
print("Submission rows:", len(df_sub), "| unique images:", df_sub["image"].nunique())

out_path = "/kaggle/working/submission.csv"
df_sub.to_csv(out_path, index=False)
print("Submission completed:", out_path)
print("File exists:", os.path.exists(out_path), "size:", os.path.getsize(out_path))
