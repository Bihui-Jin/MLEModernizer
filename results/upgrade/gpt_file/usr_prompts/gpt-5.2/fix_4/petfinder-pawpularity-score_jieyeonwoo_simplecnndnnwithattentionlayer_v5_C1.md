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

3.10

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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Concatenate,
    Lambda,
)
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
)
from sklearn.model_selection import train_test_split

np.random.seed(2)
tf.random.set_seed(2)

RUN_EDA_PLOTS = False

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 1
pass



## === cell 2
physical_device = tf.config.experimental.list_physical_devices("GPU")
print(f"Device found : {physical_device}")
if len(physical_device) >= 1:
    try:
        tf.config.experimental.set_memory_growth(physical_device[0], True)
    except Exception as e:
        print("Could not set memory growth:", e)



## === cell 3
pass



## === cell 4
dir_csv = "../input/petfinder-pawpularity-score/"

train_df = pd.read_csv(dir_csv + "train.csv")
test_df = pd.read_csv(dir_csv + "test.csv")

print("Train dataset has NaN values: ", train_df.isnull().values.any())
print("Test dataset has NaN values: ", test_df.isnull().values.any())



## === cell 5
pass



## === cell 6
pass



## === cell 7
train_df.head()



## === cell 8
pass



## === cell 9
if RUN_EDA_PLOTS:
    corr_train_df = train_df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(14, 8))
    sns.set(font_scale=1)
    ax = sns.heatmap(
        corr_train_df,
        vmin=-1,
        vmax=1,
        annot=True,
        linewidths=0.5,
        xticklabels=corr_train_df.columns,
        yticklabels=corr_train_df.columns,
    )
    ax.set_ylim(len(corr_train_df.keys()), 0)



## === cell 10
pass



## === cell 11
if RUN_EDA_PLOTS:
    corr_train_df = train_df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    ax = sns.heatmap(
        corr_train_df[["Pawpularity"]],
        vmin=-1,
        vmax=1,
        annot=True,
        linewidths=0.5,
        xticklabels=["Pawpularity"],
        yticklabels=corr_train_df.index,
    )
    ax.set_ylim(len(corr_train_df.keys()), 0)



## === cell 12
pass



## === cell 13
pass



## === cell 14
print("Min value of pawpularity: ", train_df["Pawpularity"].values.min())
print("Max value of pawpularity: ", train_df["Pawpularity"].values.max())



## === cell 15
pass



## === cell 16
if RUN_EDA_PLOTS:
    train_df["Pawpularity"].plot(kind="hist", bins=100)



## === cell 17
pass



## === cell 18
pass



## === cell 19
_ = train_df["Pawpularity"].value_counts().sort_index()



## === cell 20
_ = train_df["Pawpularity"].value_counts()



## === cell 21
pass



## === cell 22
sampled_train_df = pd.DataFrame(columns=train_df.keys())



## === cell 23
max_occ = train_df["Pawpularity"].value_counts().max()
parts = []
for class_i in range(1, 101):
    idx = train_df.index[train_df["Pawpularity"] == class_i].to_numpy()
    if idx.size == 0:
        continue
    if idx.size < max_occ:
        sampled_idx = np.random.choice(idx, max_occ, replace=True)
        parts.append(train_df.loc[sampled_idx])
    else:
        parts.append(train_df.loc[idx])
sampled_train_df = pd.concat(parts, axis=0, ignore_index=True)



## === cell 24
pass



## === cell 25
if RUN_EDA_PLOTS:
    sampled_train_df["Pawpularity"].plot(kind="hist", bins=100)



## === cell 26
pass



## === cell 27
for key in sampled_train_df.keys()[1:]:
    if key != "Id":
        sampled_train_df[key] = sampled_train_df[key].astype("int64")
pass



## === cell 28
pass



## === cell 29
pass



## === cell 30
_IMAGE_CACHE = {}  # key: (split, id, h, w) -> np.float32 array (h,w,3) in [0,1]


def _load_image_fast(img_path: str, target_size):
    key = (img_path, int(target_size[0]), int(target_size[1]))
    arr = _IMAGE_CACHE.get(key)
    if arr is not None:
        return arr
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        (target_size[0], target_size[1]),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    arr = img.numpy()
    _IMAGE_CACHE[key] = arr
    return arr




## === cell 31
class CustomTrainDataGen(Sequence):

    def __init__(
        self, df, X_col, y_col, batch_size, input_size=(250, 250, 3), shuffle=True
    ):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.indexes = np.arange(len(self.df.index))
        self.shuffle = shuffle
        self.n = len(self.df)

        self._ids = self.df["Id"].to_numpy()
        self._ann = self.df[
            [
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
        ].to_numpy(dtype=np.float32, copy=True)
        self._y = self.df["Pawpularity"].to_numpy(dtype=np.float32, copy=True)

    def on_epoch_end(self):
        if self.shuffle:
            p = np.random.permutation(self.n)
            self._ids = self._ids[p]
            self._ann = self._ann[p]
            self._y = self._y[p]

    def __get_input(self, path, target_size):
        img_path = dir_csv + "train/" + str(path) + ".jpg"
        return _load_image_fast(img_path, target_size)

    def __get_data(self, batches):
        ann_batch = self._ann[batches]  # (bs, 12)
        id_batch = self._ids[batches]
        bs = len(batches)
        image_batch = np.empty(
            (bs, self.input_size[0], self.input_size[1], self.input_size[2]),
            dtype=np.float32,
        )
        for i, _id in enumerate(id_batch):
            image_batch[i] = self.__get_input(_id, self.input_size)
        image_batch = image_batch.reshape(bs, -1)

        X_batch = np.concatenate((ann_batch, image_batch), axis=1).astype(
            np.float32, copy=False
        )
        y_batch = self._y[batches]
        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))




## === cell 32
pass




## === cell 33
class CustomTestDataGen(Sequence):

    def __init__(self, df, X_col, batch_size, input_size=(250, 250, 3), shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.indexes = np.arange(len(self.df.index))
        self.shuffle = shuffle
        self.n = len(self.df)

        self._ids = self.df["Id"].to_numpy()
        self._ann = self.df[
            [
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
        ].to_numpy(dtype=np.float32, copy=True)

    def on_epoch_end(self):
        if self.shuffle:
            p = np.random.permutation(self.n)
            self._ids = self._ids[p]
            self._ann = self._ann[p]

    def __get_input(self, path, target_size):
        img_path = dir_csv + "test/" + str(path) + ".jpg"
        return _load_image_fast(img_path, target_size)

    def __get_data(self, batches):
        ann_batch = self._ann[batches]
        id_batch = self._ids[batches]
        bs = len(batches)
        image_batch = np.empty(
            (bs, self.input_size[0], self.input_size[1], self.input_size[2]),
            dtype=np.float32,
        )
        for i, _id in enumerate(id_batch):
            image_batch[i] = self.__get_input(_id, self.input_size)
        image_batch = image_batch.reshape(bs, -1)
        X_batch = np.concatenate((ann_batch, image_batch), axis=1).astype(
            np.float32, copy=False
        )
        return X_batch

    def __getitem__(self, index):
        batches = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X = self.__get_data(batches)
        return X

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))




## === cell 34
pass



## === cell 35
pass




## === cell 36
def build_model(nb_annotations, image_shape):

    input_dim = nb_annotations + image_shape[0] * image_shape[1] * image_shape[2]
    inputs = Input(shape=(input_dim,))

    annotations_input = Lambda(
        lambda t: t[:, :nb_annotations], name="annotations_slice"
    )(inputs)
    img_flat = Lambda(lambda t: t[:, nb_annotations:], name="image_slice")(inputs)

    img_input = Lambda(
        lambda t: tf.reshape(t, (-1, image_shape[0], image_shape[1], image_shape[2])),
        name="image_reshape",
    )(img_flat)

    x = Conv2D(16, 3, activation="relu")(img_input)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(32, 3, activation="relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64, 3, activation="relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(2)(x)

    x = Flatten()(x)

    x = Dense(16, activation="relu")(x)
    x = BatchNormalization(axis=-1)(x)

    x = Concatenate()([annotations_input, x])

    attention = Dense(32, activation="relu")(x)
    attention = Dense(28, activation="softmax")(attention)
    x = attention * x

    output = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        loss="mse", optimizer="adam", metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model




## === cell 37
pass



## === cell 38
pass



## === cell 39
num_epochs = 100
batch_size = 32
target_size = (250, 250, 3)
annotations = [
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



## === cell 40
pass



## === cell 41
model = build_model(len(annotations), target_size)



## === cell 42
pass



## === cell 43
model.summary()



## === cell 44
pass



## === cell 45
tr_df, val_df = train_test_split(sampled_train_df, test_size=0.1, random_state=2)
print(tr_df.shape)
print(val_df.shape)



## === cell 46
pass



## === cell 47
traingen = CustomTrainDataGen(
    tr_df,
    X_col={
        "Id": "Id",
        "Subject Focus": "Subject Focus",
        "Eyes": "Eyes",
        "Face": "Face",
        "Near": "Near",
        "Action": "Action",
        "Accessory": "Accessory",
        "Group": "Group",
        "Collage": "Collage",
        "Human": "Human",
        "Occlusion": "Occlusion",
        "Info": "Info",
        "Blur": "Blur",
    },
    y_col={"Pawpularity": "Pawpularity"},
    batch_size=batch_size,
    input_size=target_size,
)

valgen = CustomTrainDataGen(
    val_df,
    X_col={
        "Id": "Id",
        "Subject Focus": "Subject Focus",
        "Eyes": "Eyes",
        "Face": "Face",
        "Near": "Near",
        "Action": "Action",
        "Accessory": "Accessory",
        "Group": "Group",
        "Collage": "Collage",
        "Human": "Human",
        "Occlusion": "Occlusion",
        "Info": "Info",
        "Blur": "Blur",
    },
    y_col={"Pawpularity": "Pawpularity"},
    batch_size=batch_size,
    input_size=target_size,
)



## === cell 48
pass



## === cell 49
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=0.001)
early_stop = EarlyStopping(monitor="val_loss", patience=30)

dir_path_batchtr = "./Train_logs"
os.makedirs(dir_path_batchtr, exist_ok=True)

dir_weight_path_batchtr = dir_path_batchtr + "/Weights"
os.makedirs(dir_weight_path_batchtr, exist_ok=True)

checkpoint_name = dir_weight_path_batchtr + "/weights_best.weights.h5"
checkpoint = ModelCheckpoint(
    checkpoint_name,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="min",
)

dir_hist_path_batchtr = dir_path_batchtr + "/Histories"
os.makedirs(dir_hist_path_batchtr, exist_ok=True)
logger_name = dir_hist_path_batchtr + "/history_log.csv"
logger = CSVLogger(logger_name, append=True, separator=",")



## === cell 50
pass



## === cell 51
history = model.fit(
    traingen,
    validation_data=valgen,
    epochs=num_epochs,
    callbacks=[reduce_lr, early_stop, checkpoint, logger],
    verbose=1,
)



## === cell 52
if os.path.exists(checkpoint_name):
    model.load_weights(checkpoint_name)



## === cell 53
pass



## === cell 54
external_checkpoint_name = "../input/weights-final/weights_best.hdf5"
if os.path.exists(external_checkpoint_name):
    model.load_weights(external_checkpoint_name)
    print("Loaded external weights:", external_checkpoint_name)
else:
    print(
        "External weights not found; using trained/best checkpoint weights if available."
    )



## === cell 55
pass



## === cell 56
testgen = CustomTestDataGen(
    test_df,
    X_col={
        "Id": "Id",
        "Subject Focus": "Subject Focus",
        "Eyes": "Eyes",
        "Face": "Face",
        "Near": "Near",
        "Action": "Action",
        "Accessory": "Accessory",
        "Group": "Group",
        "Collage": "Collage",
        "Human": "Human",
        "Occlusion": "Occlusion",
        "Info": "Info",
        "Blur": "Blur",
    },
    batch_size=32,
    input_size=target_size,
    shuffle=False,
)



## === cell 57
pass



## === cell 58
predictions = model.predict(
    testgen,
    verbose=1,
)



## === cell 59
pass



## === cell 60
pred_df = pd.DataFrame({"Id": test_df["Id"]})
pred_df["Pawpularity"] = predictions.reshape(-1)

pred_df["Pawpularity"] = pred_df["Pawpularity"].clip(0, 100)

assert pred_df.shape[0] == test_df.shape[0], (pred_df.shape, test_df.shape)

pred_df.to_csv("./submission.csv", index=False)
print("Wrote submission to ./submission.csv with shape:", pred_df.shape)
print(pred_df.head())



## === cell 61
pass



## === cell 62
orig_tr_df, orig_val_df = train_test_split(train_df, test_size=0.1, random_state=2)
print(orig_tr_df.shape)
print(orig_val_df.shape)



## === cell 63
valgen_test = CustomTrainDataGen(
    orig_val_df,
    X_col={
        "Id": "Id",
        "Subject Focus": "Subject Focus",
        "Eyes": "Eyes",
        "Face": "Face",
        "Near": "Near",
        "Action": "Action",
        "Accessory": "Accessory",
        "Group": "Group",
        "Collage": "Collage",
        "Human": "Human",
        "Occlusion": "Occlusion",
        "Info": "Info",
        "Blur": "Blur",
    },
    y_col={"Pawpularity": "Pawpularity"},
    batch_size=32,
    input_size=target_size,
    shuffle=False,
)



## === cell 64
predictions_val = model.predict(
    valgen_test,
    verbose=0,
).reshape(-1)
print(predictions_val[:10])
print(orig_val_df.iloc[:10]["Pawpularity"].values)
rmse = np.sqrt(
    np.mean(
        (predictions_val - orig_val_df["Pawpularity"].values.astype(np.float32)) ** 2
    )
)
print("Holdout RMSE (approx):", rmse)
