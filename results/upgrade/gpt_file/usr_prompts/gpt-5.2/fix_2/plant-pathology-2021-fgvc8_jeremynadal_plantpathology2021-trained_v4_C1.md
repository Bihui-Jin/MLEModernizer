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

# 5. Target score

0.7155493998153282

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

print("Using tensorflow", tf.__version__)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


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

    def reset_states(self):
        for v in self.variables:
            v.assign(0.0)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
pass



## === cell 3
os.makedirs("/kaggle/working/pngs/", exist_ok=True)



## === cell 4
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
train_csv = "../input/plant-pathology-2021-fgvc8/train.csv"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
duplicates = "../input/duplicatescsv/duplicates.csv"
png_dir = "/kaggle/working/pngs/"
model_dir = "../input/models/"

print(os.path.exists(train_dir))
print(os.path.exists(train_csv))
print(os.path.exists(test_dir))
print(os.path.exists(duplicates))
print(os.path.exists(png_dir))



## === cell 5
pass



## === cell 6
imgs = set(os.listdir(train_dir))
df_train = pd.read_csv(train_csv)

missing = []
for ind in range(df_train.shape[0]):
    if df_train["image"][ind] not in imgs:
        missing.append(df_train["image"][ind])
if len(missing) > 0:
    print(
        f"WARNING: {len(missing)} images from train.csv are missing in train_images (showing up to 5): {missing[:5]}"
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
print(
    "The test dataset is composed of {} unlabeled images".format(
        len(os.listdir(test_dir))
    )
)
print(df_train.head())



## === cell 8
if df_duplicates is not None:
    for ind in range(df_duplicates.shape[0]):
        if df_duplicates["img1"][ind] not in set(df_train["image"]):
            print("{} not in training dataset".format(df_duplicates["img1"][ind]))
        elif df_duplicates["img2"][ind] not in set(df_train["image"]):
            print("{} not in training dataset".format(df_duplicates["img2"][ind]))
        else:
            lab2 = df_train[
                df_train["image"] == df_duplicates["img2"][ind]
            ].reset_index()["labels"]
            lab1 = df_train[
                df_train["image"] == df_duplicates["img1"][ind]
            ].reset_index()["labels"]
            if np.all(lab2 == lab1):
                true_duplicates.append(df_duplicates["img1"][ind])
            else:
                false_duplicates.append(
                    (df_duplicates["img1"][ind], df_duplicates["img2"][ind])
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
if len(false_duplicates) > 0:
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
        plt.show()
else:
    print("No false duplicates to display (or duplicates step skipped).")



## === cell 10
pass



## === cell 11
labels_flat = [x.split(" ") for x in df_train["labels"]]
labels_flat = [l for label in labels_flat for l in label]

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
    for img1, img2 in false_duplicates:
        df_train = df_train[df_train["image"] != img1]
        df_train = df_train[df_train["image"] != img2]
    for img in true_duplicates:
        df_train = df_train[df_train["image"] != img]

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

bar = df_occurence.plot.barh(figsize=[15, 5], colormap="plasma")



## === cell 16
pass



## === cell 17
sss = StratifiedShuffleSplit(n_splits=1, test_size=CFG.test_size, random_state=CFG.seed)
X = df_train["image"]
y = df_train[CFG.classes].values
y_strat = np.argmax(y, axis=1)

for train_index, test_index in sss.split(X, y_strat):
    X_train, X_test = df_train.loc[train_index].reset_index(drop=True), df_train.loc[
        test_index
    ].reset_index(drop=True)

compare_X_train, compare_X_test = train_test_split(
    df_train, test_size=CFG.test_size, random_state=CFG.seed
)



## === cell 18
vc_all = df_train[CFG.classes].apply(value_counts)
vc_tr = X_train[CFG.classes].apply(value_counts)
vc_te = X_test[CFG.classes].apply(value_counts)
vc_ctr = compare_X_train[CFG.classes].apply(value_counts)
vc_cte = compare_X_test[CFG.classes].apply(value_counts)


def _safe_loc1(vc_):
    if 1 in vc_.index:
        return vc_.loc[1]
    return pd.Series(0.0, index=CFG.classes)


df_occurence = pd.DataFrame(
    {
        "origin": _safe_loc1(vc_all),
        "stratified_train": _safe_loc1(vc_tr),
        "stratified_test": _safe_loc1(vc_te),
        "compare_train": _safe_loc1(vc_ctr),
        "compare_test": _safe_loc1(vc_cte),
    }
)

bar = df_occurence.plot.barh(figsize=[15, 5], colormap="plasma")
plt.savefig(png_dir + "comparison_stratified.png")



## === cell 19
pass




## === cell 20
def pred2labels(pred, thresh=0.5, labels=CFG.classes):
    assert len(pred) == len(labels), "Predictions must have shape : ({},)".format(
        len(labels)
    )
    pred = [labels[i] for i in range(len(labels)) if pred[i] > thresh]
    pred = np.array(pred)
    res = ""
    for p in pred:
        if res == "":
            res += p
        else:
            res += " " + p
    return res




## === cell 21
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
def parse_image(file_path):
    img = tf.io.read_file(train_dir + file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [CFG.img_size, CFG.img_size])
    return img


def prepare_dataset(X, augmentation=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (X["image"].values, X[CFG.classes].values)
    )
    dataset = dataset.map(lambda x, y: (parse_image(x), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(CFG.batch_size)

    if augmentation:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.repeat().prefetch(buffer_size=AUTOTUNE)
    return dataset




## === cell 23
ds_train = prepare_dataset(X_train, augmentation=True)
ds_test = prepare_dataset(X_test)



## === cell 24
for inputs, outputs in ds_train.as_numpy_iterator():
    print("Input shape is:", inputs.shape, "output shape is:", outputs.shape)
    plt.imshow(inputs[0])
    plt.show()
    print(
        "label of this input is",
        outputs[0],
        "corresponding to",
        pred2labels(outputs[0]),
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
            model.add(
                tf.keras.layers.Dense(
                    nb_FC_neurons / 2**FC,
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




## === cell 27
model = None
try:
    base = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
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
    )

    model.summary()
except Exception as e:
    print("Model creation failed:", repr(e))
    print("Internet not available or weights download blocked.")
    raise



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3611381630.py in <cell line: 0>()
      9     )
     10 
---> 11     model = create_cnn(
     12         input_shape=(CFG.img_size, CFG.img_size, 3),
     13         output_length=len(CFG.classes),

/tmp/ipykernel_55/1578280436.py in create_cnn(input_shape, output_length, nb_cnn, nb_filters, activation_cnn, model_transfert, fine_tune, nb_FC_layer, nb_FC_neurons, reducing, activation_FC, dropout, activation_output, name)
     56     if reducing:
     57         for FC in range(nb_FC_layer):
---> 58             model.add(
     59                 tf.keras.layers.Dense(
     60                     nb_FC_neurons / 2**FC,

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
    120         self._layers.append(layer)
    121         if rebuild:
--> 122             self._maybe_rebuild()
    123         else:
    124             self.built = False

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in _maybe_rebuild(self)
    139         if isinstance(self._layers[0], InputLayer) and len(self._layers) > 1:
    140             input_shape = self._layers[0].batch_shape
--> 141             self.build(input_shape)
    142         elif hasattr(self._layers[0], "input_shape") and len(self._layers) > 1:
    143             # We can build the Sequential model if the first layer has the

/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py in build_wrapper(*args, **kwargs)
    226             with obj._open_name_scope():
    227                 obj._path = current_path()
--> 228                 original_build_method(*args, **kwargs)
    229             # Record build config.
    230             signature = inspect.signature(original_build_method)

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in build(self, input_shape)
    185         for layer in self._layers[1:]:
    186             try:
--> 187                 x = layer(x)
    188             except NotImplementedError:
    189                 # Can happen if shape inference is not implemented.

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in standardize_shape(shape)
    580             continue
    581         if not is_int_dtype(type(e)):
--> 582             raise ValueError(
    583                 f"Cannot convert '{shape}' to a shape. "
    584                 f"Found invalid entry '{e}' of type '{type(e)}'. "

ValueError: Cannot convert '(131072, 512.0)' to a shape. Found invalid entry '512.0' of type '<class 'float'>'. 

## === cell 28
if os.path.exists(model_dir + "model.h5") and not CFG.retrain:
    print("Loading model from file")
    model = tf.keras.models.load_model(
        model_dir + "model.h5", custom_objects={"MicroF1": MicroF1}
    )
    history = None
else:
    history = model.fit(
        ds_train,
        validation_data=ds_test,
        steps_per_epoch=(X_train.shape[0] * 0.8) // CFG.batch_size,
        validation_steps=(X_train.shape[0] * 0.2) // CFG.batch_size,
        callbacks=get_callbacks(
            monitor="val_f1_score", save_name="/kaggle/working/model.h5", patience=2
        ),
        epochs=6,
    )



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2916623548.py in <cell line: 0>()
      7     history = None
      8 else:
----> 9     history = model.fit(
     10         ds_train,
     11         validation_data=ds_test,

AttributeError: 'NoneType' object has no attribute 'fit'

## === cell 29
pass



## === cell 30
if history:
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
    plt.show()
else:
    print("There is no history")



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3484673147.py in <cell line: 0>()
----> 1 if history:
      2     fig, axes = plt.subplots(1, 3, figsize=(10, 5))
      3 
      4     axes[0].plot(history.history["loss"], label="Train loss")
      5     axes[0].plot(history.history["val_loss"], label="Validation loss")

NameError: name 'history' is not defined

## === cell 31
pass




## === cell 32
def parse_test_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [CFG.img_size, CFG.img_size])
    return img


def predict_new(path, model):
    img = parse_test_image(path)
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)
    return pred2labels(pred[0])




## === cell 33
test_images = sorted(os.listdir(test_dir))
rows = []
for fname in test_images:
    pred = predict_new(os.path.join(test_dir, fname), model)
    rows.append({"image": fname, "labels": pred})

df_sub = pd.DataFrame(rows, columns=["image", "labels"])
print(df_sub.head())

df_sub.to_csv("submission.csv", index=False)
print("Submission completed -> wrote submission.csv with shape", df_sub.shape)

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1242161680.py in <cell line: 0>()
      3 rows = []
      4 for fname in test_images:
----> 5     pred = predict_new(os.path.join(test_dir, fname), model)
      6     rows.append({"image": fname, "labels": pred})
      7 

/tmp/ipykernel_55/1845254840.py in predict_new(path, model)
     10     img = parse_test_image(path)
     11     img = tf.expand_dims(img, axis=0)
---> 12     pred = model.predict(img, verbose=0)
     13     return pred2labels(pred[0])
     14 

AttributeError: 'NoneType' object has no attribute 'predict'
