# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.14

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")



## === cell 1
import numpy as np
import pandas as pd
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings

warnings.filterwarnings("ignore")

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("TF_NUM_INTRAOP_THREADS", "2"))
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get("TF_NUM_INTEROP_THREADS", "2"))
    )
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("Python:", os.sys.version)
print("TF:", tf.__version__)



## === cell 2
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 3
work_base = "/kaggle/working/aerial-cactus-identification"
train_out = os.path.join(work_base, "train")
test_out = os.path.join(work_base, "test")

need_extract = not (os.path.isdir(train_out) and os.path.isdir(test_out))
if need_extract:
    os.makedirs(work_base, exist_ok=True)
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/train.zip", "r"
    ) as z:
        z.extractall(work_base)
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/test.zip", "r"
    ) as z:
        z.extractall(work_base)



## === cell 4
base_work_dir = "/kaggle/working/aerial-cactus-identification"
train_dir = os.path.join(base_work_dir, "train")
test_dir = os.path.join(base_work_dir, "test")

assert os.path.isdir(train_dir), f"Train directory not found: {train_dir}"
assert os.path.isdir(test_dir), f"Test directory not found: {test_dir}"

train_labels = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
train_labels.head()



## === cell 5
img_list1 = [os.path.join(train_dir, img_id) for img_id in train_labels["id"]]
label_list1 = list(train_labels["has_cactus"])

missing = [p for p in img_list1[:50] if not os.path.exists(p)]
print("Missing among first 50:", len(missing))



## === cell 6
df = pd.DataFrame()
df["image"] = img_list1
df["label"] = label_list1



## === cell 7
test_filenames = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_df = pd.DataFrame({"id": test_filenames})

print("Train images:", len(df), "Test images:", len(test_df))
test_df.head()



## === cell 8
assert df["image"].map(os.path.exists).all(), "Some training image paths do not exist."
assert len(test_df) > 0, "No test images found."



## === cell 9
df.head()



## === cell 10
df.shape



## === cell 11
df.info()



## === cell 12
df.isnull().sum()



## === cell 13
df["label"] = df["label"].astype(str)



## === cell 14
df.info()



## === cell 15
df["label"].value_counts()



## === cell 16
RUN_EDA_PLOTS = False

if RUN_EDA_PLOTS:
    sns.countplot(x=df["label"], palette=["salmon", "skyblue"])



## === cell 17
if RUN_EDA_PLOTS:
    fig, ax = plt.subplots(2, 5)
    fig.set_size_inches(12, 6)
    k = 0
    for i in range(2):
        for j in range(5):
            img_path = img_list1[k]
            label = label_list1[k]
            img = plt.imread(img_path)
            ax[i, j].imshow(img)
            status = "Has Cactus (1)" if label == 1 else "No Cactus(0)"
            ax[i, j].set_title(status, fontsize=10)
            ax[i, j].axis("off")
            k += 1
    plt.tight_layout()
    plt.show()



## === cell 18
test_df.head()



## === cell 19
test_df.shape



## === cell 20
if RUN_EDA_PLOTS:
    sample_test = test_df.sample(10, random_state=SEED).reset_index(drop=True)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img_name = sample_test.loc[i, "id"]
        full_path = os.path.join(test_dir, img_name)
        img = plt.imread(full_path)
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 21
IMG_SIZE = (64, 64)
BATCH_SIZE = 64



## === cell 22
y_int = df["label"].astype(int).values
cw = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=y_int
)
cw = {0: float(cw[0]), 1: float(cw[1])}

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    zoom_range=0.2,
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="image",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=SEED,
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col="image",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

inv = {v: k for k, v in train_generator.class_indices.items()}
weights_dict = {idx: cw[int(inv[idx])] for idx in inv.keys()}
print("class_indices:", train_generator.class_indices)
print("class_weight used:", weights_dict)



## === cell 23
print("Train steps:", len(train_generator), "Val steps:", len(val_generator))
assert (
    len(train_generator) > 0 and len(val_generator) > 0
), "Generator has length 0 (check paths/df)."



## === cell 24
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 25
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=weights_dict,
)



## === cell 26
history.history["accuracy"][-1]



## === cell 27
model.save("cactus.h5")



## === cell 28
pass



## === cell 29
if RUN_EDA_PLOTS:
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val_Accuracy")
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Val_Loss")
    plt.legend()



## === cell 30
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="id",
    y_col=None,
    class_mode=None,
    shuffle=False,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)



## === cell 31
predictions = model.predict(
    test_generator,
    verbose=1,
)
predictions = predictions.reshape(-1)

print("Predictions shape:", predictions.shape)
assert len(predictions) == len(test_df), "Prediction count does not match test rows."



## === cell 32
submission_df = pd.DataFrame({"id": test_df["id"].values, "has_cactus": predictions})
submission_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
