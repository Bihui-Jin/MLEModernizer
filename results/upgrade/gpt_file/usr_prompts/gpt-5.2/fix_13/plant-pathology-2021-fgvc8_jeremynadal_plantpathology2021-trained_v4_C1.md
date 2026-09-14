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
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf

print("Using tensorflow", tf.__version__)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit


class MicroF1(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="f1_score", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        return 2.0 * precision * recall / (precision + recall + 1e-7)

    def reset_state(self):
        for v in self.variables:
            v.assign(0.0)

    def reset_states(self):
        self.reset_state()


SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except Exception:
    pass




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

    cache_train = None
    cache_val = None
    cache_test = None

    save_plots = False




## === cell 2
pass



## === cell 3
os.makedirs("/kaggle/working/pngs/", exist_ok=True)



## === cell 4
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
train_csv = "../input/plant-pathology-2021-fgvc8/train.csv"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

duplicates = "../input/duplicatescsv/duplicates.csv"

png_dir = "/kaggle/working/pngs/"
model_dir = "../input/models/"  # optional

print(os.path.exists(train_dir))
print(os.path.exists(train_csv))
print(os.path.exists(test_dir))
print(os.path.exists(duplicates))
print(os.path.exists(png_dir))
print("models dir exists:", os.path.exists(model_dir))



## === cell 5
pass



## === cell 6
df_train = pd.read_csv(train_csv)

sample_check = df_train["image"].values[:128]
missing_sample = [
    fn for fn in sample_check if not tf.io.gfile.exists(os.path.join(train_dir, fn))
]
if missing_sample:
    print(
        f"WARNING: Some images from train.csv sample are missing in train_images (showing up to 5): {missing_sample[:5]}"
    )



## === cell 7
df_duplicates = None
true_duplicates = []
false_duplicates = []

if os.path.exists(duplicates):
    df_duplicates = pd.read_csv(duplicates, header=None)
    df_duplicates.columns = ["img1", "img2"]
    print("\nThere are {} duplicated images.\n".format(df_duplicates.shape[0]))
else:
    print(
        "\nNo duplicates.csv found at {} -> skipping duplicate filtering.\n".format(
            duplicates
        )
    )

print("The train dataset is composed of {} labeled images".format(df_train.shape[0]))
print("The test dataset directory exists:", os.path.exists(test_dir))
print(df_train.head())



## === cell 8
if df_duplicates is not None:
    label_map = df_train.set_index("image")["labels"]
    df_dup = df_duplicates.copy()
    df_dup["lab1"] = df_dup["img1"].map(label_map)
    df_dup["lab2"] = df_dup["img2"].map(label_map)

    in_train = df_dup["lab1"].notna() & df_dup["lab2"].notna()
    df_dup_in = df_dup.loc[in_train]

    true_mask = df_dup_in["lab1"].to_numpy() == df_dup_in["lab2"].to_numpy()
    true_duplicates = df_dup_in.loc[true_mask, "img1"].tolist()
    false_duplicates = list(
        df_dup_in.loc[~true_mask, ["img1", "img2"]].itertuples(index=False, name=None)
    )

    print(
        "There are {} true duplicates and {} false ones.".format(
            len(true_duplicates), len(false_duplicates)
        )
    )
    print("Lets display the false duplicates")
else:
    print("No duplicates processed.")



## === cell 9
count = 0
if CFG.save_plots and len(false_duplicates) > 0:
    for img1, img2 in false_duplicates[:5]:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(plt.imread(train_dir + img1))
        axs[0].set_title(df_train[df_train["image"] == img1].reset_index()["labels"][0])
        axs[0].axis("off")
        axs[1].imshow(plt.imread(train_dir + img2))
        axs[1].set_title(df_train[df_train["image"] == img2].reset_index()["labels"][0])
        axs[1].axis("off")
        plt.savefig(png_dir + "compare_false_dup" + str(count) + ".png")
        count += 1
        plt.close(fig)
else:
    print("Skipping false-duplicate visualization (disabled or none).")



## === cell 10
pass



## === cell 11
labels_flat = df_train["labels"].str.split().explode().to_numpy()
uniques = np.unique(labels_flat)
assert len(uniques) == len(CFG.classes), "ERROR : labels and CFG.classes mismatch"
for unique in uniques:
    assert unique in CFG.classes, "ERROR : labels and CFG.classes mismatch"



## === cell 12
df_train["labels"] = [x.split(" ") for x in df_train["labels"]]
mlb = MultiLabelBinarizer(classes=CFG.classes)
labels = mlb.fit_transform(df_train["labels"].values)
labels = pd.DataFrame(columns=CFG.classes, data=labels, index=df_train.index)
df_train.drop("labels", axis=1, inplace=True)
for col in labels.columns:
    df_train[col] = labels[col]
print(df_train.head())



## === cell 13
init = df_train.shape[0]

if df_duplicates is not None:
    to_drop = set(true_duplicates)
    if false_duplicates:
        fd = np.array(false_duplicates, dtype=object)
        to_drop.update(fd[:, 0].tolist())
        to_drop.update(fd[:, 1].tolist())
    if to_drop:
        df_train = df_train[~df_train["image"].isin(to_drop)]

df_train.reset_index(drop=True, inplace=True)
end = df_train.shape[0]
print("Deleted {} files".format(init - end))



## === cell 14
pass



## === cell 15
value_counts = lambda x: pd.Series.value_counts(x, normalize=True)

vc = df_train[CFG.classes].apply(value_counts)
if 1 in vc.index:
    df_occurence = pd.DataFrame({"origin": vc.loc[1]})
else:
    df_occurence = pd.DataFrame({"origin": pd.Series(0.0, index=CFG.classes)})

if CFG.save_plots:
    bar = df_occurence.plot.barh(figsize=[15, 5], colormap="plasma")



## === cell 16
pass



## === cell 17
sss = StratifiedShuffleSplit(n_splits=1, test_size=CFG.test_size, random_state=CFG.seed)
X = df_train["image"]
y = df_train[CFG.classes].values
y_strat = np.argmax(y, axis=1)

for train_index, test_index in sss.split(X, y_strat):
    X_train = df_train.loc[train_index].reset_index(drop=True)
    X_test = df_train.loc[test_index].reset_index(drop=True)



## === cell 18
pass




## === cell 19
def pred2labels(pred, thresh=0.5, labels=CFG.classes, default_label="healthy"):
    assert len(pred) == len(labels), "Predictions must have shape : ({},)".format(
        len(labels)
    )
    picked = [labels[i] for i in range(len(labels)) if pred[i] > thresh]
    if len(picked) == 0:
        picked = [default_label]
    return " ".join(picked)




## === cell 20
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



## === cell 21
_preprocess = tf.keras.applications.xception.preprocess_input


@tf.function
def _decode_resize_bytes(img_bytes):
    def _fast():
        return tf.io.decode_and_crop_jpeg(
            img_bytes,
            crop_window=[0, 0, 0, 0],  # special case: no crop (full image)
            channels=3,
        )

    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [CFG.img_size, CFG.img_size], method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = _preprocess(img)
    img = tf.ensure_shape(img, (CFG.img_size, CFG.img_size, 3))
    return img


@tf.function
def _decode_resize_from_train(file_name):
    img_bytes = tf.io.read_file(tf.strings.join([train_dir, file_name]))
    return _decode_resize_bytes(img_bytes)


def prepare_dataset(X, augmentation=False):
    img_names = X["image"].values
    y_vals = X[CFG.classes].values

    ds = tf.data.Dataset.from_tensor_slices((img_names, y_vals))

    @tf.function
    def _decode_map(img_name, y):
        x = _decode_resize_from_train(img_name)
        return x, tf.cast(y, tf.float32)

    ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE)

    cache_path = CFG.cache_train if augmentation else CFG.cache_val
    if cache_path:
        ds = ds.cache(cache_path)
    else:
        snap_dir = (
            "/kaggle/working/tfds_snapshot_train"
            if augmentation
            else "/kaggle/working/tfds_snapshot_val"
        )
        ds = ds.snapshot(snap_dir)

    if augmentation:
        ds = ds.shuffle(
            buffer_size=min(len(X), 2048),
            seed=CFG.seed,
            reshuffle_each_iteration=True,
        )

        @tf.function
        def _aug_map(x, y):
            x = data_augmentation(x, training=True)
            return x, y

        ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(CFG.batch_size, drop_remainder=False)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    try:
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        opts.threading.private_threadpool_size = 16
    except Exception:
        pass
    ds = ds.with_options(opts)
    return ds




## === cell 22
ds_train = prepare_dataset(X_train, augmentation=True)
ds_test = prepare_dataset(X_test)



## === cell 23
for inputs, outputs in ds_train.take(1):
    print("Input shape is:", inputs.shape, "output shape is:", outputs.shape)
    out0 = outputs[0].numpy()
    print(
        "label of this input is",
        out0,
        "corresponding to",
        pred2labels(out0),
    )



## === cell 24
pass




## === cell 25
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
    """Create a CNN based model is model_transfert is None. Else, the model_transfert is used for feature extraction.
    If reducing is not False, nb_FC_neurons must be multiple of 2**nb_FC_layer
    """
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
            units = int(nb_FC_neurons // (2**FC))
            model.add(
                tf.keras.layers.Dense(
                    units,
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
                    int(nb_FC_neurons),
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
                filepath=save_name, monitor=monitor, save_best_only=True, verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, patience=patience, restore_best_weights=True
            ),
        ]
    else:
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, patience=patience, restore_best_weights=True
            )
        ]




## === cell 26
model = None
try:
    try:
        base = tf.keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            input_shape=(CFG.img_size, CFG.img_size, 3),
            classes=len(CFG.classes),
        )
        print("Loaded Xception with ImageNet weights.")
    except Exception as e:
        print(
            "WARNING: Could not load ImageNet weights (offline or blocked). Falling back to weights=None."
        )
        print("Underlying error:", repr(e))
        base = tf.keras.applications.Xception(
            include_top=False,
            weights=None,
            input_shape=(CFG.img_size, CFG.img_size, 3),
            classes=len(CFG.classes),
        )

    model = create_cnn(
        input_shape=(CFG.img_size, CFG.img_size, 3),
        output_length=len(CFG.classes),
        model_transfert=base,
        fine_tune=True,
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
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            MicroF1(name="f1_score"),
        ],
        run_eagerly=False,
    )

    model.summary()
except Exception as e:
    print("Model creation failed:", repr(e))
    raise



## === cell 27
history = None
local_ckpt = "/kaggle/working/model.h5"
external_ckpt = os.path.join(model_dir, "model.h5")
can_load_external = os.path.exists(external_ckpt)

steps_per_epoch = int(np.ceil(len(X_train) / CFG.batch_size))
validation_steps = int(np.ceil(len(X_test) / CFG.batch_size))

if can_load_external and not CFG.retrain:
    print("Loading model from file:", external_ckpt)
    model = tf.keras.models.load_model(
        external_ckpt, custom_objects={"MicroF1": MicroF1}
    )
elif os.path.exists(local_ckpt) and not CFG.retrain:
    print("Loading model from local checkpoint:", local_ckpt)
    model = tf.keras.models.load_model(local_ckpt, custom_objects={"MicroF1": MicroF1})
else:
    history = model.fit(
        ds_train,
        validation_data=ds_test,
        callbacks=get_callbacks(
            monitor="val_f1_score", save_name=local_ckpt, patience=2
        ),
        epochs=6,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
    )



## === cell 28
pass



## === cell 29
if history is not None and CFG.save_plots:
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].plot(history.history["loss"], label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss")

    axes[1].plot(history.history["acc"], label="Train accuracy")
    axes[1].plot(history.history["val_acc"], label="Validation accuracy")
    axes[1].set_title("Accuracy")

    axes[2].plot(history.history["f1_score"], label="Train micro-F1")
    axes[2].plot(history.history["val_f1_score"], label="Validation micro-F1")
    axes[2].set_title("micro-F1")

    plt.savefig(png_dir + "history_xception.png")
    plt.close(fig)
elif history is None:
    print("There is no history (model loaded from disk).")
else:
    print("History exists but plot saving disabled.")



## === cell 30
pass




## === cell 31
@tf.function
def parse_test_image(file_path):
    img_bytes = tf.io.read_file(file_path)
    return _decode_resize_bytes(img_bytes)


def predict_new(path, model):
    img = parse_test_image(path)
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)
    return pred2labels(pred[0])




## === cell 32
sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
df_sample = pd.read_csv(sample_sub_path)
test_images = df_sample["image"].tolist()

test_paths = [os.path.join(test_dir, fname) for fname in test_images]

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(parse_test_image, num_parallel_calls=AUTOTUNE)

if CFG.cache_test:
    test_ds = test_ds.cache(CFG.cache_test)
else:
    test_ds = test_ds.snapshot("/kaggle/working/tfds_snapshot_test")

test_ds = test_ds.batch(CFG.batch_size, drop_remainder=False).prefetch(AUTOTUNE)

opts = tf.data.Options()
opts.experimental_deterministic = True
opts.experimental_optimization.apply_default_optimizations = True
try:
    opts.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    opts.threading.private_threadpool_size = 16
except Exception:
    pass
test_ds = test_ds.with_options(opts)

preds = model.predict(test_ds, verbose=0)

thresh = 0.5
labels = np.array(CFG.classes, dtype=object)
default_label = "healthy"

picked_idx = preds > thresh
out_labels = []
for row in picked_idx:
    idx = np.flatnonzero(row)
    if idx.size == 0:
        out_labels.append(default_label)
    else:
        out_labels.append(" ".join(labels[idx].tolist()))

df_sub = pd.DataFrame(
    {"image": test_images, "labels": out_labels}, columns=["image", "labels"]
)
print(df_sub.head())

df_sub.to_csv("submission.csv", index=False)
print("Submission completed -> wrote submission.csv with shape", df_sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
assert (
    df_sub.shape[0] == df_sample.shape[0]
), "Row count mismatch vs sample_submission.csv"
assert list(df_sub.columns) == ["image", "labels"], "Submission columns mismatch"
