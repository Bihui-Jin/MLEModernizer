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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
lazy_submit = True  # set to False to train
SEED = 42

keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_SIZE = (224, 224)
folds = 5
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TRAIN_IMAGES_DIR = "../input/petfinder-pawpularity-score/train"
TRAIN_DS = "../input/petfinder-pawpularity-score/train.csv"
TEST_IMAGES_DIR = "../input/petfinder-pawpularity-score/test"
TEST_DS = "../input/petfinder-pawpularity-score/test.csv"
SUBMISSION_DS = "../input/petfinder-pawpularity-score/sample_submission.csv"

if lazy_submit:
    weights = "../input/weights-pawpularity"
else:
    weights = "../working/"
if not os.path.exists(weights):
    weights = "../working/"

WEIGHTS_PREFIX = "efficientnetb0_"
WEIGHTS_SUFFIX = ".weights.h5"



## === cell 2
try:
    import subprocess

    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print("nvidia-smi not available:", repr(e))



## === cell 3
img = keras.utils.load_img(
    "../input/petfinder-pawpularity-score/train/00524dbf2637a80cbc80f70d3ff59616.jpg"
)
img = keras.utils.img_to_array(img) / 255.0
print(img.shape)



## === cell 4
train_ds = pd.read_csv(TRAIN_DS)
test_ds = pd.read_csv(TEST_DS)
subm_ds = pd.read_csv(SUBMISSION_DS)
print(train_ds.shape, test_ds.shape, subm_ds.shape)



## === cell 5
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




## === cell 6
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
        input_size=(224, 224),
        shuffle=True,
    ):
        self.df = df.copy()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df)
        self.tab_columns = tab_columns
        self.target = target
        self.id = id
        self.is_train = is_train
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    def __getitem__(self, index):
        this_ds = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        images = []
        for img_id in list(this_ds[self.id].values):
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            img = keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
            img = keras.utils.img_to_array(img)
            img = np.array(img, dtype="float32")
            images.append(img)

        x_img = np.array(images, dtype="float32")
        x_tab = this_ds[self.tab_columns].values.astype("float32")

        x = (x_img, x_tab)

        if self.is_train:
            y = this_ds[self.target].values.astype("float32")
            return x, y
        else:
            return x




## === cell 7
NCOL = len(meta_cols)
INPUT_SHAPE = (*IMAGE_SIZE, 3)

_EFFNET_LOCAL = "../input/efficientnet-b0/efficientnetb0_notop.h5"


def get_model():
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomContrast(0.2, seed=SEED),
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.1, seed=SEED),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=SEED),
        ]
    )

    effnet_weights = _EFFNET_LOCAL if os.path.exists(_EFFNET_LOCAL) else "imagenet"

    conv_base = tf.keras.applications.efficientnet.EfficientNetB0(
        weights=effnet_weights,
        include_top=False,
        input_tensor=keras.Input(shape=INPUT_SHAPE),
    )
    conv_base.trainable = False

    inp = keras.Input(shape=INPUT_SHAPE)
    out = data_augmentation(inp)
    out = conv_base(out)
    out = layers.GlobalAveragePooling2D()(out)
    out = layers.BatchNormalization()(out)

    meta_input = keras.Input(shape=(NCOL,))
    out_meta = layers.Dense(16, activation="relu")(meta_input)
    out_meta = layers.Dropout(0.1)(out_meta)
    out_meta = layers.Dense(16, activation="relu")(out_meta)
    out_meta = layers.Dropout(0.1)(out_meta)
    out_meta = layers.Dense(16, activation="relu")(out_meta)

    concat = layers.Concatenate(axis=1)([out, out_meta])
    concat = layers.Dropout(0.1)(concat)
    concat = layers.Dense(16)(concat)
    concat = layers.Dropout(0.1)(concat)
    concat = layers.Dense(16)(concat)
    concat = layers.Dropout(0.1)(concat)
    concat = layers.Dense(1, activation="linear")(concat)

    model = keras.Model([inp, meta_input], concat)
    return model




## === cell 8
model = get_model()
model.summary()



## === cell 9
try:
    tf.keras.utils.plot_model(model, show_shapes=True)
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 10
BATCH_SIZE = 120
kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)


def train():
    EPOCHS = 10
    optimizer = keras.optimizers.Adam(1e-3)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_root_mean_squared_error",
        patience=2,
        factor=0.5,
        verbose=1,
        mode="min",
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_root_mean_squared_error", patience=5
    )

    def scheduler(epoch, lr):
        if epoch == 1:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1 * lr))

    sch = tf.keras.callbacks.LearningRateScheduler(scheduler)

    models = []
    evals = 0
    histories = []
    for idx, (train_idx, val_idx) in enumerate(kf.split(train_ds)):
        params = dict(
            img_dir=TRAIN_IMAGES_DIR,
            batch_size=BATCH_SIZE,
            tab_columns=meta_cols,
            id="Id",
            target="Pawpularity",
            is_train=True,
        )
        train_gen = CustomDataGen(df=train_ds.iloc[train_idx], **params)
        val_gen = CustomDataGen(df=train_ds.iloc[val_idx], **params)

        model = get_model()
        model.compile(
            loss="mse",
            optimizer=optimizer,
            metrics=[keras.metrics.RootMeanSquaredError()],
            run_eagerly=True,
        )

        checkpoint = keras.callbacks.ModelCheckpoint(
            f"../working/{WEIGHTS_PREFIX}{idx}{WEIGHTS_SUFFIX}",
            monitor="val_root_mean_squared_error",
            verbose=1,
            save_best_only=True,
            mode="min",
            save_weights_only=True,
        )

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=[reduce_lr, checkpoint, es, sch],
        )

        evals += model.evaluate(val_gen, batch_size=BATCH_SIZE, verbose=0)[1]
        models.append(model)
        histories.append(history)
        gc.collect()

    evals /= folds
    return models, evals, histories




## === cell 11
lazy_eval = False  # Set to True to evaluate with the pretrained weights

need_train_fallback = False
if lazy_submit:
    for idx in range(folds):
        wpath = os.path.join(weights, f"{WEIGHTS_PREFIX}{idx}{WEIGHTS_SUFFIX}")
        if not os.path.exists(wpath):
            wpath2 = os.path.join(
                "../working", f"{WEIGHTS_PREFIX}{idx}{WEIGHTS_SUFFIX}"
            )
            if not os.path.exists(wpath2):
                need_train_fallback = True
                break

if (not lazy_submit) or need_train_fallback:
    models, evals, histories = train()

    fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    ax = ax.flatten()
    for i in range(min(folds, len(ax))):
        ax[i].plot(histories[i].history["loss"], label="loss")
        ax[i].plot(histories[i].history["val_loss"], label="val_loss")
        ax[i].legend()
    plt.show()
    print("AVG RMSE:", evals)
else:
    models = []
    evals = 0.0
    for idx, (train_idx, val_idx) in enumerate(kf.split(train_ds)):
        params = dict(
            img_dir=TRAIN_IMAGES_DIR,
            batch_size=BATCH_SIZE,
            tab_columns=meta_cols,
            id="Id",
            target="Pawpularity",
            is_train=True,
            shuffle=False,
        )
        val_gen = CustomDataGen(df=train_ds.iloc[val_idx], **params)

        model = get_model()
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(1e-3),
            metrics=[keras.metrics.RootMeanSquaredError()],
            run_eagerly=True,
        )

        wpath = os.path.join(weights, f"{WEIGHTS_PREFIX}{idx}{WEIGHTS_SUFFIX}")
        if not os.path.exists(wpath):
            wpath = os.path.join("../working", f"{WEIGHTS_PREFIX}{idx}{WEIGHTS_SUFFIX}")
        model.load_weights(wpath)

        models.append(model)

        if lazy_eval:
            evals += model.evaluate(val_gen, verbose=0)[1] / folds
            print("AVG RMSE:", evals)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2830199056.py in <cell line: 0>()
     15 
     16 if (not lazy_submit) or need_train_fallback:
---> 17     models, evals, histories = train()
     18 
     19     fig, ax = plt.subplots(3, 2, figsize=(12, 12))

/tmp/ipykernel_55/1896287353.py in train()
     61         )
     62 
---> 63         history = model.fit(
     64             train_gen,
     65             validation_data=val_gen,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in _check_variables_are_known(self, variables)
    327         for v in variables:
    328             if self._var_key(v) not in self._trainable_variables_indices:
--> 329                 raise ValueError(
    330                     f"Unknown variable: {v}. This optimizer can only "
    331                     "be called for the variables it was originally built with. "

ValueError: Unknown variable: <Variable path=dense_12/kernel, shape=(12, 16), dtype=float32, value=[[ 0.19772065 -0.39633244  0.32008493  0.14166397  0.15210867 -0.26342177
   0.04579598  0.00681898  0.31057435 -0.17790395 -0.14068112  0.03731054
  -0.41156375  0.4158604  -0.45319405 -0.34942994]
 [-0.2667536   0.15444833  0.31970406 -0.20193842  0.15459841 -0.02588686
   0.37124074 -0.26675174  0.29553747  0.14774978 -0.2749483  -0.45362163
  -0.42229855 -0.4148401   0.36865842  0.11037982]
 [ 0.2772212  -0.38894916 -0.2960773   0.2812127   0.07818311 -0.36742473
  -0.27477413  0.35538143  0.02624688 -0.3204496  -0.19886583  0.26863158
   0.08962929  0.4520114  -0.33812708  0.27556992]
 [-0.25737303  0.0918588   0.24673247 -0.15805301  0.31228328 -0.24668503
   0.20983899 -0.32873315 -0.36238077  0.45332366  0.27302045 -0.09531128
  -0.26330522  0.1909293   0.06765038 -0.4593685 ]
 [ 0.42054826  0.01318392  0.39858234 -0.29030824 -0.26753765 -0.29764605
   0.23328722 -0.17351246 -0.00538489 -0.15950114  0.38606972 -0.45949608
  -0.23675272 -0.39866558  0.06396163  0.22070026]
 [-0.25240302 -0.40418378 -0.03981897  0.2816683   0.4515624   0.4166404
  -0.34795237 -0.33797356  0.07531446 -0.45594716 -0.15970662 -0.3671562
  -0.05034372  0.3117929  -0.44012946 -0.02264705]
 [ 0.3597123   0.04915929 -0.31030816  0.27647138  0.36477    -0.31752798
   0.42642468  0.2476142   0.14032906  0.01034176 -0.24989516 -0.24279396
   0.34631962  0.19349575  0.45499152  0.39945966]
 [ 0.02950799  0.30843544 -0.0467214   0.15485746  0.44111842 -0.44729322
  -0.26925063  0.4023217  -0.3162654  -0.39318115  0.14220738 -0.36684266
  -0.06387442 -0.19959304 -0.24103892  0.24743384]
 [ 0.07658023  0.40434515  0.18584412 -0.26113695 -0.34691688  0.12849712
  -0.225138    0.24209863 -0.34913427  0.2754764   0.11015564  0.25825894
  -0.09688357  0.45894945  0.33092505  0.36266178]
 [ 0.3098082  -0.39821714  0.3724379   0.3020264  -0.42430645 -0.2833959
  -0.23594561  0.03273776 -0.00950101  0.23534745  0.44863063  0.073304
  -0.21725199  0.35690904 -0.42964706 -0.07910553]
 [-0.38358733 -0.34245268 -0.45348212  0.25825387  0.4161688   0.3773778
  -0.42308083  0.31602162 -0.133786   -0.19325358  0.40794176 -0.34463274
  -0.38073158 -0.22060116  0.27974373 -0.39003238]
 [-0.2106685  -0.12278014 -0.29389006  0.3382296   0.3958637  -0.12420079
  -0.28987327  0.15529794  0.07384688 -0.05077747  0.44695085  0.20269191
  -0.4371914   0.03788966 -0.4494492  -0.0926286 ]]>. This optimizer can only be called for the variables it was originally built with. When working with a new set of variables, you should recreate a new optimizer instance.

## === cell 12
predictions = []
params = dict(
    img_dir=TEST_IMAGES_DIR,
    batch_size=120,
    tab_columns=meta_cols,
    id="Id",
    target="Pawpularity",
    is_train=False,
    shuffle=False,
)
test_gen = CustomDataGen(df=test_ds, **params)

for i in range(folds):
    predictions.append(models[i].predict(test_gen, verbose=0))

pred = np.mean(np.array(predictions), axis=0).reshape(-1).astype("float32")
pred = np.clip(pred, 0.0, 100.0)

submission = subm_ds.copy()
submission["Pawpularity"] = pred
submission = submission[["Id", "Pawpularity"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2561504302.py in <cell line: 0>()
     12 
     13 for i in range(folds):
---> 14     predictions.append(models[i].predict(test_gen, verbose=0))
     15 
     16 pred = np.mean(np.array(predictions), axis=0).reshape(-1).astype("float32")

NameError: name 'models' is not defined

## === cell 13
submission.head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
