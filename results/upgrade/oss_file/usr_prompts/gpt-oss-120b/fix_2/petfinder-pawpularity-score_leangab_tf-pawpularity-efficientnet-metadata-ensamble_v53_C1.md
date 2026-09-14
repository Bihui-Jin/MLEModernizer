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

3.9

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

18.408625212578105

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
import multiprocessing
import cv2
import matplotlib.pyplot as plt
import gc

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_SIZE = (224, 224)
folds = 5
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATA_ROOT = "./data/petfinder-pawpularity-score"
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train")
TRAIN_DS = os.path.join(DATA_ROOT, "train.csv")
TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test")
TEST_DS = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_DS = os.path.join(DATA_ROOT, "sample_submission.csv")

lazy_submit = False

CHECKPOINT_DIR = "./working"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
img_path = os.path.join(TRAIN_IMAGES_DIR, "00524dbf2637a80cbc80f70d3ff59616.jpg")
if os.path.exists(img_path):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img) / 255.0
    print("Loaded image shape:", img.shape)



## === cell 2
train_ds = pd.read_csv(TRAIN_DS)
test_ds = pd.read_csv(TEST_DS)
subm_ds = pd.read_csv(SUBMISSION_DS)
print("Train shape:", train_ds.shape, "Test shape:", test_ds.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1461502786.py in <cell line: 0>()
----> 1 train_ds = pd.read_csv(TRAIN_DS)
      2 test_ds = pd.read_csv(TEST_DS)
      3 subm_ds = pd.read_csv(SUBMISSION_DS)
      4 print("Train shape:", train_ds.shape, "Test shape:", test_ds.shape)
      5 

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

FileNotFoundError: [Errno 2] No such file or directory: './data/petfinder-pawpularity-score/train.csv'

## === cell 3
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




## === cell 4
class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        img_dir,
        batch_size,
        tab_columns,
        id,
        target,
        is_train,
        input_size=IMAGE_SIZE,
        shuffle=True,
    ):
        self.df = df.copy()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.tab_columns = tab_columns
        self.target = target
        self.id = id
        self.is_train = is_train
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        this_ds = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        images = []
        for img_id in list(this_ds[self.id].values):
            img_path = f"{self.img_dir}/{img_id}.jpg"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
            img = img.astype("float32") / 255.0
            images.append(img)
        images = np.array(images)
        if self.is_train:
            return [images, this_ds[self.tab_columns].values], this_ds[
                self.target
            ].values
        else:
            return [images, this_ds[self.tab_columns].values]




## === cell 5
NCOL = len(meta_cols)
INPUT_SHAPE = (*IMAGE_SIZE, 3)


def get_model():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomContrast(factor=0.2, seed=SEED),
            tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(factor=0.1, seed=SEED),
            tf.keras.layers.RandomTranslation(
                height_factor=0.1, width_factor=0.1, seed=SEED
            ),
        ]
    )

    conv_base = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=keras.Input(shape=INPUT_SHAPE),
    )
    conv_base.trainable = False

    inp = keras.Input(shape=INPUT_SHAPE)
    x = data_augmentation(inp)
    x = conv_base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    meta_input = keras.Input(shape=(NCOL,))
    m = layers.Dense(16, activation="relu")(meta_input)
    m = layers.Dropout(0.1)(m)
    m = layers.Dense(16, activation="relu")(m)
    m = layers.Dropout(0.1)(m)
    m = layers.Dense(16, activation="relu")(m)

    concat = layers.Concatenate(axis=1)([x, m])
    concat = layers.Dropout(0.1)(concat)
    concat = layers.Dense(16, activation="relu")(concat)
    concat = layers.Dropout(0.1)(concat)
    concat = layers.Dense(16, activation="relu")(concat)
    concat = layers.Dropout(0.1)(concat)
    out = layers.Dense(1, activation="linear")(concat)

    model = keras.Model([inp, meta_input], out)
    return model




## === cell 6
def train():
    BATCH_SIZE = 64
    EPOCHS = 5  # modest number to keep runtime reasonable
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)

    models = []
    rmses = []

    for idx, (train_idx, val_idx) in enumerate(kf.split(train_ds)):
        print(f"\nTraining fold {idx+1}/{folds}")
        params = dict(
            img_dir=TRAIN_IMAGES_DIR,
            batch_size=BATCH_SIZE,
            tab_columns=meta_cols,
            id="Id",
            target="Pawpularity",
            is_train=True,
            shuffle=True,
        )
        train_gen = CustomDataGen(df=train_ds.iloc[train_idx], **params)
        val_gen = CustomDataGen(df=train_ds.iloc[val_idx], **params)

        model = get_model()
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(1e-3),
            metrics=[keras.metrics.RootMeanSquaredError()],
        )

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"efficientnetb0_fold{idx}.h5")
        checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_root_mean_squared_error",
            save_best_only=True,
            mode="min",
            save_weights_only=True,
            verbose=0,
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_root_mean_squared_error",
            patience=2,
            factor=0.5,
            mode="min",
            verbose=1,
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error",
            patience=4,
            mode="min",
            restore_best_weights=True,
        )

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=[reduce_lr, checkpoint, early_stop],
            use_multiprocessing=True,
            workers=multiprocessing.cpu_count(),
            verbose=1,
        )

        model.load_weights(checkpoint_path)
        rmse = model.evaluate(val_gen, verbose=0)[1]
        print(f"Fold {idx+1} RMSE: {rmse:.4f}")
        rmses.append(rmse)
        models.append(model)

    avg_rmse = np.mean(rmses)
    print("\nAverage RMSE across folds:", avg_rmse)
    return models, avg_rmse




## === cell 7
if not lazy_submit:
    models, avg_rmse = train()
else:
    models = []
    avg_rmse = None



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3622751310.py in <cell line: 0>()
      1 if not lazy_submit:
----> 2     models, avg_rmse = train()
      3 else:
      4     # Placeholder for lazy evaluation path (not used here)
      5     models = []

/tmp/ipykernel_55/4241796016.py in train()
      7     rmses = []
      8 
----> 9     for idx, (train_idx, val_idx) in enumerate(kf.split(train_ds)):
     10         print(f"\nTraining fold {idx+1}/{folds}")
     11         params = dict(

NameError: name 'train_ds' is not defined

## === cell 8
BATCH_SIZE = 64
predictions = []

test_params = dict(
    img_dir=TEST_IMAGES_DIR,
    batch_size=BATCH_SIZE,
    tab_columns=meta_cols,
    id="Id",
    target="Pawpularity",
    is_train=False,
    shuffle=False,
)
test_gen = CustomDataGen(df=test_ds, **test_params)

for i, model in enumerate(models):
    print(f"Predicting with model {i+1}/{len(models)}")
    preds = model.predict(test_gen, verbose=0)
    predictions.append(preds.squeeze())

final_pred = np.mean(predictions, axis=0)

subm_ds["Pawpularity"] = final_pred
submission_path = "submission.csv"
subm_ds.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2398495938.py in <cell line: 0>()
     12     shuffle=False,
     13 )
---> 14 test_gen = CustomDataGen(df=test_ds, **test_params)
     15 
     16 for i, model in enumerate(models):

NameError: name 'test_ds' is not defined

## === cell 9
subm_ds.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2085290068.py in <cell line: 0>()
      1 # Quick look at the first few rows of the submission
----> 2 subm_ds.head()

NameError: name 'subm_ds' is not defined
