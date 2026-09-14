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
import os, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

print("Using tensorflow ", tf.__version__)




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


tf.random.set_seed(CFG.seed)
np.random.seed(CFG.seed)



## === cell 2
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
train_csv = "../input/plant-pathology-2021-fgvc8/train.csv"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
duplicates_path = "../input/duplicatescsv/duplicates.csv"  # may not exist
png_dir = "/kaggle/working/pngs/"
model_dir = "../input/models/"

os.makedirs(png_dir, exist_ok=True)

print("train_dir exists:", os.path.isdir(train_dir))
print("train_csv exists:", os.path.isfile(train_csv))
print("test_dir exists:", os.path.isdir(test_dir))
print("duplicates exists:", os.path.isfile(duplicates_path))
print("png_dir exists:", os.path.isdir(png_dir))



## === cell 3
df_train = pd.read_csv(train_csv)
df_train["labels"] = df_train["labels"].astype(str)

if os.path.isfile(duplicates_path):
    df_duplicates = pd.read_csv(duplicates_path, header=None, names=["img1", "img2"])
else:
    df_duplicates = pd.DataFrame(columns=["img1", "img2"])

print(f"Training set size: {df_train.shape[0]}")
print(f"Test set size (files): {len(os.listdir(test_dir))}")



## === cell 4
imgs = set(os.listdir(train_dir))
missing = [row for row in df_train["image"] if row not in imgs]
if missing:
    print(f"{len(missing)} images listed in CSV are missing from train_images.")
else:
    print("All images referenced in CSV are present.")



## === cell 5
mlb = MultiLabelBinarizer(classes=CFG.classes)
labels_matrix = mlb.fit_transform(df_train["labels"].str.split(" "))
labels_df = pd.DataFrame(labels_matrix, columns=CFG.classes, index=df_train.index)
df_train = df_train.drop(columns=["labels"]).join(labels_df)
print("One‑hot label columns added:")
print(df_train.head())



## === cell 6
true_duplicates = []
false_duplicates = []
for _, row in df_duplicates.iterrows():
    img1, img2 = row["img1"], row["img2"]
    if img1 not in df_train["image"].values or img2 not in df_train["image"].values:
        continue
    lbl1 = df_train.loc[df_train["image"] == img1, CFG.classes].values
    lbl2 = df_train.loc[df_train["image"] == img2, CFG.classes].values
    if np.array_equal(lbl1, lbl2):
        true_duplicates.append(img1)
    else:
        false_duplicates.append((img1, img2))

init_cnt = df_train.shape[0]
df_train = df_train[
    ~df_train["image"].isin(
        true_duplicates + [i for pair in false_duplicates for i in pair]
    )
]
df_train = df_train.reset_index(drop=True)
print(f"Removed {init_cnt - df_train.shape[0]} duplicate rows.")



## === cell 7
sss = StratifiedShuffleSplit(n_splits=1, test_size=CFG.test_size, random_state=CFG.seed)
X = df_train["image"]
y = df_train[CFG.classes]
for train_idx, test_idx in sss.split(X, y):
    df_train_split = df_train.iloc[train_idx].reset_index(drop=True)
    df_test_split = df_train.iloc[test_idx].reset_index(drop=True)

print("Train split size:", df_train_split.shape[0])
print("Validation split size:", df_test_split.shape[0])




## === cell 8
def pred2labels(pred, thresh=0.5, labels=CFG.classes):
    selected = [labels[i] for i, p in enumerate(pred) if p > thresh]
    return " ".join(selected)




## === cell 9
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=CFG.seed),
        tf.keras.layers.RandomRotation(0.2, seed=CFG.seed),
        tf.keras.layers.RandomContrast(factor=0.3, seed=CFG.seed),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, seed=CFG.seed
        ),
    ]
)

AUTOTUNE = tf.data.AUTOTUNE


def parse_image(file_path):
    img = tf.io.read_file(tf.strings.join([train_dir, file_path]))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [CFG.img_size, CFG.img_size])
    return img


def prepare_dataset(df, augment=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (df["image"].values, df[CFG.classes].values.astype(np.float32))
    )
    ds = ds.map(lambda x, y: (parse_image(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    ds = ds.batch(CFG.batch_size).prefetch(AUTOTUNE).repeat()
    return ds


ds_train = prepare_dataset(df_train_split, augment=True)
ds_val = prepare_dataset(df_test_split, augment=False)




## === cell 10
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
    assert input_shape[-1] == 3
    if reducing:
        assert nb_FC_neurons % (2**nb_FC_layer) == 0

    model = tf.keras.models.Sequential(name=name)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name="Input"))

    if model_transfert is None:
        for i in range(nb_cnn):
            model.add(
                tf.keras.layers.Conv2D(
                    nb_filters,
                    (3, 3),
                    padding="same",
                    activation=activation_cnn,
                    name=f"Conv2D_{i+1}",
                )
            )
            model.add(tf.keras.layers.MaxPooling2D((2, 2), name=f"Pool_{i+1}"))
    else:
        if not fine_tune:
            model_transfert.trainable = False
        model.add(model_transfert)
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name="Pool_transfer"))

    model.add(tf.keras.layers.Flatten())

    if reducing:
        for i in range(nb_FC_layer):
            units = nb_FC_neurons // (2**i)
            model.add(
                tf.keras.layers.Dense(units, activation=activation_FC, name=f"FC_{i+1}")
            )
            if dropout:
                model.add(tf.keras.layers.Dropout(dropout, name=f"Drop_{i+1}"))
    else:
        for i in range(nb_FC_layer):
            model.add(
                tf.keras.layers.Dense(
                    nb_FC_neurons, activation=activation_FC, name=f"FC_{i+1}"
                )
            )
            if dropout:
                model.add(tf.keras.layers.Dropout(dropout, name=f"Drop_{i+1}"))

    model.add(
        tf.keras.layers.Dense(
            output_length, activation=activation_output, name="Output"
        )
    )
    return model




## === cell 11
base = tf.keras.applications.Xception(
    include_top=False, weights="imagenet", input_shape=(CFG.img_size, CFG.img_size, 3)
)
model = create_cnn(
    input_shape=(CFG.img_size, CFG.img_size, 3),
    output_length=len(CFG.classes),
    model_transfert=base,
    fine_tune=True,
    nb_FC_layer=2,
    nb_FC_neurons=512,
    reducing=True,
    dropout=0.0,
    activation_output="sigmoid",
    name="my_model",
)

optimizer = tf.keras.optimizers.Adam(learning_rate=3.5e-5)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
)
model.summary()



## === cell 12
steps_per_epoch = max(1, len(df_train_split) // CFG.batch_size)
validation_steps = max(1, len(df_test_split) // CFG.batch_size)

history = model.fit(
    ds_train,
    validation_data=ds_val,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=6,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc", patience=2, restore_best_weights=True
        )
    ],
)



## === cell 13
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history["loss"], label="train loss")
axes[0].plot(history.history["val_loss"], label="val loss")
axes[0].set_title("Loss")
axes[0].legend()

axes[1].plot(history.history["acc"], label="train acc")
axes[1].plot(history.history["val_acc"], label="val acc")
axes[1].set_title("Accuracy")
axes[1].legend()

plt.savefig(png_dir + "history.png")
plt.close(fig)




## === cell 14
def parse_test_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [CFG.img_size, CFG.img_size])
    return img


def predict_image(path):
    img = parse_test_image(path)
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    return pred2labels(pred[0])




## === cell 15
test_files = sorted(os.listdir(test_dir))  # deterministic order
test_paths = [os.path.join(test_dir, f) for f in test_files]

ds_test = tf.data.Dataset.from_tensor_slices(test_paths)
ds_test = ds_test.map(parse_test_image, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(CFG.batch_size).prefetch(AUTOTUNE)

preds = model.predict(ds_test, verbose=0)

label_strs = [pred2labels(p) for p in preds]

submission = pd.DataFrame({"image": test_files, "labels": label_strs})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
