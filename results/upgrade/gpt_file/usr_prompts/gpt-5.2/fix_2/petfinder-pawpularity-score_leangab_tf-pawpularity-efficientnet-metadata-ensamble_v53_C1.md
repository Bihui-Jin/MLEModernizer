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
import gc
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
lazy_submit = True  # set to False to train
SEED = 42
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

        if self.is_train:
            y = this_ds[self.target].values.astype("float32")
            return [x_img, x_tab], y
        else:
            return [x_img, x_tab]




## === cell 7
NCOL = len(meta_cols)
INPUT_SHAPE = (*IMAGE_SIZE, 3)


def get_model():
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomContrast(0.2, seed=SEED),
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.1, seed=SEED),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=SEED),
        ]
    )

    conv_base = tf.keras.applications.efficientnet.EfficientNetB0(
        weights="../input/efficientnet-b0/efficientnetb0_notop.h5",
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



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3530384866.py in <cell line: 0>()
----> 1 model = get_model()
      2 model.summary()
      3 

/tmp/ipykernel_55/207093548.py in get_model()
     15     )
     16 
---> 17     conv_base = tf.keras.applications.efficientnet.EfficientNetB0(
     18         weights="../input/efficientnet-b0/efficientnetb0_notop.h5",
     19         include_top=False,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNetB0(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)
    569     name="efficientnetb0",
    570 ):
--> 571     return EfficientNet(
    572         1.0,
    573         1.0,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNet(width_coefficient, depth_coefficient, default_size, dropout_rate, drop_connect_rate, depth_divisor, activation, blocks_args, name, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, weights_name)
    273 
    274     if not (weights in {"imagenet", None} or file_utils.exists(weights)):
--> 275         raise ValueError(
    276             "The `weights` argument should be either "
    277             "`None` (random initialization), `imagenet` "

ValueError: The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.

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
            return lr
        else:
            return lr * tf.math.exp(-0.1 * lr)

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
        )

        checkpoint = keras.callbacks.ModelCheckpoint(
            f"../working/efficientnetb0_{idx}.h5",
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
            use_multiprocessing=True,
            workers=multiprocessing.cpu_count(),
            max_queue_size=2 * BATCH_SIZE,
        )

        evals += model.evaluate(val_gen, batch_size=BATCH_SIZE)[1]
        models.append(model)
        histories.append(history)
        gc.collect()

    evals /= folds
    return models, evals, histories




## === cell 11
lazy_eval = False  # Set to True to evaluate with the pretrained weights

if not lazy_submit:
    models, evals, histories = train()

    fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    ax = ax.flatten()
    for i in range(folds):
        ax[i].plot(histories[i].history["loss"], label="loss")
        ax[i].plot(histories[i].history["val_loss"], label="val_loss")
        ax[i].legend()
    plt.show()
    print("AVG RMSE:", evals)

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
    )

    wpath = os.path.join(weights, f"efficientnetb0_{idx}.h5")
    if not os.path.exists(wpath):
        wpath = os.path.join("../working", f"efficientnetb0_{idx}.h5")
    model.load_weights(wpath)

    models.append(model)

    if lazy_eval:
        evals += (
            model.evaluate(
                val_gen,
                use_multiprocessing=True,
                workers=multiprocessing.cpu_count(),
                max_queue_size=2 * BATCH_SIZE,
            )[1]
            / folds
        )
        print("AVG RMSE:", evals)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2763561870.py in <cell line: 0>()
     27     val_gen = CustomDataGen(df=train_ds.iloc[val_idx], **params)
     28 
---> 29     model = get_model()
     30     model.compile(
     31         loss="mse",

/tmp/ipykernel_55/207093548.py in get_model()
     15     )
     16 
---> 17     conv_base = tf.keras.applications.efficientnet.EfficientNetB0(
     18         weights="../input/efficientnet-b0/efficientnetb0_notop.h5",
     19         include_top=False,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNetB0(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)
    569     name="efficientnetb0",
    570 ):
--> 571     return EfficientNet(
    572         1.0,
    573         1.0,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNet(width_coefficient, depth_coefficient, default_size, dropout_rate, drop_connect_rate, depth_divisor, activation, blocks_args, name, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, weights_name)
    273 
    274     if not (weights in {"imagenet", None} or file_utils.exists(weights)):
--> 275         raise ValueError(
    276             "The `weights` argument should be either "
    277             "`None` (random initialization), `imagenet` "

ValueError: The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.

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
subm_ds["Pawpularity"] = pred
subm_ds.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm_ds.shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/38029739.py in <cell line: 0>()
     12 
     13 for i in range(folds):
---> 14     predictions.append(models[i].predict(test_gen, verbose=0))
     15 
     16 # Bugfix: predictions are (n,1); ensure submission is 1D float array aligned to subm_ds rows.

IndexError: list index out of range

## === cell 13
subm_ds.head()
