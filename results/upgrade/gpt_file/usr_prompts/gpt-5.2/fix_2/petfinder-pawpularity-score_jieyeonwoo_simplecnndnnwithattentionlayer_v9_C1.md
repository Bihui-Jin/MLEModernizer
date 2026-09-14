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

# 5. Target score

22.414736117667083

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

TF_AVAILABLE = False
try:
    import tensorflow as tf  # noqa: F401

    TF_AVAILABLE = True
except Exception as e:
    TF_IMPORT_ERROR = repr(e)
    TF_AVAILABLE = False

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow import failed; falling back to metadata-only model.")
    print("TF import error:", TF_IMPORT_ERROR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if TF_AVAILABLE:
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
    )
    from tensorflow.keras.callbacks import (
        ReduceLROnPlateau,
        EarlyStopping,
        ModelCheckpoint,
        CSVLogger,
        TensorBoard,
    )
else:

    class Sequence:
        pass




## === cell 2
if TF_AVAILABLE:
    physical_device = tf.config.experimental.list_physical_devices("GPU")
    print(f"Device found : {physical_device}")
    if len(physical_device) >= 1:
        try:
            tf.config.experimental.set_memory_growth(physical_device[0], True)
        except Exception as e:
            print("Could not set memory growth:", e)
else:
    print("Skipping GPU configuration because TensorFlow is unavailable.")



## === cell 3
dir_csv = "../input/petfinder-pawpularity-score/"
if not os.path.exists(dir_csv):
    fallback = "/kaggle/input/petfinder-pawpularity-score/"
    if os.path.exists(fallback):
        dir_csv = fallback

print("Using dir_csv:", dir_csv)



## === cell 4
train_df = pd.read_csv(os.path.join(dir_csv, "train.csv"))
test_df = pd.read_csv(os.path.join(dir_csv, "test.csv"))

print("Train dataset has NaN values: ", train_df.isnull().values.any())
print("Test dataset has NaN values: ", test_df.isnull().values.any())
print("train_df shape:", train_df.shape, "test_df shape:", test_df.shape)



## === cell 5
assert "Id" in train_df.columns and "Pawpularity" in train_df.columns
assert "Id" in test_df.columns
print("Train columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())



## === cell 6
print(train_df.head(2))



## === cell 7
corr_train_df = train_df.drop(columns=["Id"]).corr(numeric_only=True)
plt.figure(figsize=(14, 8))
sns.set(font_scale=0.9)
ax = sns.heatmap(
    corr_train_df,
    vmin=-1,
    vmax=1,
    annot=False,
    linewidths=0.5,
    xticklabels=corr_train_df.columns,
    yticklabels=corr_train_df.columns,
)
ax.set_ylim(len(corr_train_df.keys()), 0)
plt.tight_layout()
plt.show()



## === cell 8
corr_train_df = train_df.drop(columns=["Id"]).corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.set(font_scale=1)
ax = sns.heatmap(
    corr_train_df[["Pawpularity"]],
    vmin=-1,
    vmax=1,
    annot=True,
    linewidths=0.5,
    xticklabels=["Pawpularity"],
    yticklabels=corr_train_df.columns,
)
ax.set_ylim(len(corr_train_df.keys()), 0)
plt.tight_layout()
plt.show()



## === cell 9
print("Min value of pawpularity: ", train_df["Pawpularity"].values.min())
print("Max value of pawpularity: ", train_df["Pawpularity"].values.max())



## === cell 10
train_df["Pawpularity"].plot(kind="hist", bins=100)
plt.title("Pawpularity distribution")
plt.tight_layout()
plt.show()



## === cell 11
print(train_df["Pawpularity"].value_counts().sort_index().head(10))
print(train_df["Pawpularity"].value_counts().head(10))



## === cell 12
tr_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2)
print(tr_df.shape)
print(val_df.shape)



## === cell 13
tr_df["Pawpularity"].plot(kind="hist", bins=100)
plt.title("Train split Pawpularity distribution")
plt.tight_layout()
plt.show()



## === cell 14
sampled_tr_df = pd.DataFrame(columns=tr_df.columns)

max_occ = tr_df["Pawpularity"].value_counts().max()
rng = np.random.default_rng(42)

for class_i in range(1, 101):
    cls = tr_df[tr_df["Pawpularity"] == class_i]
    n = len(cls)
    if n == 0:
        continue
    if n < max_occ:
        ids_class_i = cls.index.to_numpy()
        sampled_ids_class_i = rng.choice(ids_class_i, size=max_occ, replace=True)
        sampled_tr_df = pd.concat([sampled_tr_df, tr_df.loc[sampled_ids_class_i]])
    else:
        sampled_tr_df = pd.concat([sampled_tr_df, cls])

sampled_tr_df = sampled_tr_df.reset_index(drop=True)



## === cell 15
sampled_tr_df["Pawpularity"].plot(kind="hist", bins=100)
plt.title("Oversampled train split Pawpularity distribution")
plt.tight_layout()
plt.show()



## === cell 16
print("tr_df:", tr_df.info())
print("sampled_tr_df:", sampled_tr_df.info())



## === cell 17
for key in sampled_tr_df.columns:
    if key not in ["Id", "Pawpularity"]:
        sampled_tr_df[key] = sampled_tr_df[key].astype("int64")
print("sampled_tr_df:", sampled_tr_df.info())



## === cell 18
if TF_AVAILABLE:

    class CustomTrainDataGen(Sequence):
        def __init__(
            self, df, X_col, y_col, batch_size, input_size=(250, 250, 3), shuffle=True
        ):
            self.df = df.copy().reset_index(drop=True)
            self.X_col = X_col
            self.y_col = y_col
            self.batch_size = batch_size
            self.input_size = input_size
            self.shuffle = shuffle
            self.n = len(self.df)

        def on_epoch_end(self):
            if self.shuffle:
                self.df = self.df.sample(frac=1).reset_index(drop=True)

        def __get_input(self, path, target_size):
            img_path = os.path.join(dir_csv, "train", f"{path}.jpg")
            image = tf.keras.preprocessing.image.load_img(img_path)
            image_arr = tf.keras.preprocessing.image.img_to_array(image)
            image_arr = tf.image.resize(
                image_arr, (target_size[0], target_size[1])
            ).numpy()
            return image_arr / 255.0

        def __get_data(self, idxs):
            batch_df = self.df.iloc[idxs]

            ann_cols = [
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
            ann = batch_df[ann_cols].to_numpy().astype(np.float32)

            id_batch = batch_df["Id"].values
            imgs = np.asarray(
                [self.__get_input(i, self.input_size) for i in id_batch],
                dtype=np.float32,
            )
            imgs = imgs.reshape(len(idxs), -1)

            X_batch = np.concatenate([ann, imgs], axis=1)
            y_batch = batch_df["Pawpularity"].to_numpy().astype(np.float32)
            return X_batch, y_batch

        def __getitem__(self, index):
            start = index * self.batch_size
            end = min((index + 1) * self.batch_size, self.n)
            idxs = np.arange(start, end)
            return self.__get_data(idxs)

        def __len__(self):
            return int(np.ceil(self.n / self.batch_size))

    class CustomTestDataGen(Sequence):
        def __init__(
            self, df, X_col, batch_size, input_size=(250, 250, 3), shuffle=False
        ):
            self.df = df.copy().reset_index(drop=True)
            self.X_col = X_col
            self.batch_size = batch_size
            self.input_size = input_size
            self.shuffle = shuffle
            self.n = len(self.df)

        def on_epoch_end(self):
            if self.shuffle:
                self.df = self.df.sample(frac=1).reset_index(drop=True)

        def __get_input(self, path, target_size):
            img_path = os.path.join(dir_csv, "test", f"{path}.jpg")
            image = tf.keras.preprocessing.image.load_img(img_path)
            image_arr = tf.keras.preprocessing.image.img_to_array(image)
            image_arr = tf.image.resize(
                image_arr, (target_size[0], target_size[1])
            ).numpy()
            return image_arr / 255.0

        def __get_data(self, idxs):
            batch_df = self.df.iloc[idxs]
            ann_cols = [
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
            ann = batch_df[ann_cols].to_numpy().astype(np.float32)

            id_batch = batch_df["Id"].values
            imgs = np.asarray(
                [self.__get_input(i, self.input_size) for i in id_batch],
                dtype=np.float32,
            )
            imgs = imgs.reshape(len(idxs), -1)

            X_batch = np.concatenate([ann, imgs], axis=1)
            return X_batch

        def __getitem__(self, index):
            start = index * self.batch_size
            end = min((index + 1) * self.batch_size, self.n)
            idxs = np.arange(start, end)
            return self.__get_data(idxs)

        def __len__(self):
            return int(np.ceil(self.n / self.batch_size))

else:
    CustomTrainDataGen = None
    CustomTestDataGen = None



## === cell 19
if TF_AVAILABLE:

    def build_model(nb_annotations, image_shape):
        input_dim = nb_annotations + image_shape[0] * image_shape[1] * image_shape[2]
        inputs = Input(shape=(input_dim,))  # FIX: Input expects a tuple shape

        annotations_input = inputs[:, :nb_annotations]
        img_input = inputs[:, nb_annotations:]
        img_input = tf.reshape(
            img_input,
            (tf.shape(inputs)[0], image_shape[0], image_shape[1], image_shape[2]),
        )

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
            loss="mse",
            optimizer="adam",
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        return model




## === cell 20
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



## === cell 21
if TF_AVAILABLE:
    model = build_model(len(annotations), target_size)
    model.summary()

    traingen = CustomTrainDataGen(
        sampled_tr_df,
        X_col={c: c for c in ["Id"] + annotations},
        y_col={"Pawpularity": "Pawpularity"},
        batch_size=batch_size,
        input_size=target_size,
    )
    valgen = CustomTrainDataGen(
        val_df,
        X_col={c: c for c in ["Id"] + annotations},
        y_col={"Pawpularity": "Pawpularity"},
        batch_size=batch_size,
        input_size=target_size,
        shuffle=False,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=0.001
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=30)

    dir_path_batchtr = "./Train_logs"
    os.makedirs(dir_path_batchtr, exist_ok=True)

    dir_weight_path_batchtr = os.path.join(dir_path_batchtr, "Weights")
    os.makedirs(dir_weight_path_batchtr, exist_ok=True)
    checkpoint_name = os.path.join(
        dir_weight_path_batchtr, "weights_best.weights.h5"
    )  # FIX extension
    checkpoint = ModelCheckpoint(
        checkpoint_name,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )

    dir_hist_path_batchtr = os.path.join(dir_path_batchtr, "Histories")
    os.makedirs(dir_hist_path_batchtr, exist_ok=True)
    logger_name = os.path.join(dir_hist_path_batchtr, "history_log.csv")
    logger = CSVLogger(logger_name, append=True, separator=",")

    dir_tensorboard_log = "./tensorboard_logs"
    os.makedirs(dir_tensorboard_log, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=dir_tensorboard_log)

    history = model.fit(
        traingen,
        validation_data=valgen,
        epochs=num_epochs,
        callbacks=[reduce_lr, early_stop, checkpoint, logger, tensorboard_callback],
        verbose=2,
    )

    if os.path.exists(checkpoint_name):
        model.load_weights(checkpoint_name)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/304599547.py in <cell line: 0>()
      1 # TF training branch (disabled if TF not available). Checkpoint filename fixed for Keras 3.
      2 if TF_AVAILABLE:
----> 3     model = build_model(len(annotations), target_size)
      4     model.summary()
      5 

/tmp/ipykernel_55/2033176511.py in build_model(nb_annotations, image_shape)
     10         img_input = tf.reshape(
     11             img_input,
---> 12             (tf.shape(inputs)[0], image_shape[0], image_shape[1], image_shape[2]),
     13         )
     14 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 22
feature_cols = annotations
X_train = sampled_tr_df[feature_cols].astype(np.float32)
y_train = sampled_tr_df["Pawpularity"].astype(np.float32)

X_val = val_df[feature_cols].astype(np.float32)
y_val = val_df["Pawpularity"].astype(np.float32)

if not TF_AVAILABLE:
    meta_model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=10.0, random_state=0)),
        ]
    )
    meta_model.fit(X_train, y_train)

    val_pred = meta_model.predict(X_val).astype(np.float32)
    rmse = float(np.sqrt(np.mean((val_pred - y_val.to_numpy()) ** 2)))
    print("Validation RMSE (metadata-only fallback):", rmse)



## === cell 23
if TF_AVAILABLE:
    testgen = CustomTestDataGen(
        test_df,
        X_col={c: c for c in ["Id"] + annotations},
        batch_size=1,
        input_size=target_size,
        shuffle=False,
    )
    predictions = model.predict(testgen, verbose=0).reshape(-1)
else:
    X_test = test_df[feature_cols].astype(np.float32)
    predictions = meta_model.predict(X_test).astype(np.float32).reshape(-1)

predictions = np.clip(predictions, 1.0, 100.0)

sub = pd.DataFrame({"Id": test_df["Id"].values, "Pawpularity": predictions})
sub.to_csv("./submission.csv", index=False)
print("Wrote submission.csv:", sub.shape)
print(sub.head())



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3894190113.py in <cell line: 0>()
      8         shuffle=False,
      9     )
---> 10     predictions = model.predict(testgen, verbose=0).reshape(-1)
     11 else:
     12     X_test = test_df[feature_cols].astype(np.float32)

NameError: name 'model' is not defined

## === cell 24
chk = pd.read_csv("./submission.csv")
assert list(chk.columns) == ["Id", "Pawpularity"]
assert len(chk) == len(test_df)
print("Submission OK. Rows:", len(chk))

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3438464621.py in <cell line: 0>()
      1 # Final check: submission format
----> 2 chk = pd.read_csv("./submission.csv")
      3 assert list(chk.columns) == ["Id", "Pawpularity"]
      4 assert len(chk) == len(test_df)
      5 print("Submission OK. Rows:", len(chk))

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './submission.csv'
