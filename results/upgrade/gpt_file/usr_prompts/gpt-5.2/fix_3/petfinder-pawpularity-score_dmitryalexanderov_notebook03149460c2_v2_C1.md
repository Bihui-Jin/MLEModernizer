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

18.86876823169344

# 6. Current score

41.71262

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 41.71262) has done: 'The immediate blocker is the TensorFlow import crash (`MessageFactory.GetPrototype`), which prevents the script from running at all; I avoid TensorFlow entirely and keep the pipeline working by falling back to the existing tabular RandomForest path when TF is unavailable. Next, the “Invalid submission” error is because predictions include 0.0 after clipping; the competition requires values in [1, 100], so I clip to [1, 100] to produce a valid file. I also make the data root resolve robustly to the available `/kaggle/input/...` path to avoid path issues across Kaggle environments. These are minimal, execution-unblocking changes that should yield a valid `submission.csv` without changing the RandomForest core logic.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

TF_AVAILABLE = False
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # noqa: F401
    from tensorflow.keras import layers  # noqa: F401

    TF_AVAILABLE = True
except Exception as e:
    print(
        "WARNING: TensorFlow is not available in this runtime; falling back to tabular model only."
    )
    print("TensorFlow import error:", repr(e))

from sklearn.ensemble import RandomForestRegressor



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_ROOTS = [
    "../input/petfinder-pawpularity-score",
    "/kaggle/input/petfinder-pawpularity-score",
    "/kaggle/input/petfinder-pawpularity-score/petfinder-pawpularity-score",
]
DATA_ROOT = None
for p in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/petfinder-pawpularity-score"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")

image_size = 224
batch_size = 128

tabular_columns = [
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

test = pd.read_csv(TEST_CSV)
test["file_path"] = test["Id"].apply(
    lambda identifier: os.path.join(TEST_IMG_DIR, f"{identifier}.jpg")
)



## === cell 2

res_0 = None
if TF_AVAILABLE:
    from tensorflow.keras.applications import EfficientNetB0

    img_augmentation = keras.Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            layers.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )

    def rmse(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))

    def get_tabular_prediciton_model(inputs):
        width = 32
        depth = 3
        activation = "relu"
        kernel_regularizer = keras.regularizers.l2()
        x = keras.layers.Dense(
            width, activation=activation, kernel_regularizer=kernel_regularizer
        )(inputs)
        for i in range(depth):
            if i == 0:
                x = inputs
            x = keras.layers.Dense(
                width, activation=activation, kernel_regularizer=kernel_regularizer
            )(x)
            if (i + 1) % 3 == 0:
                x = keras.layers.Concatenate()([x, inputs])
        return x

    B0_NOTOP_PATH = "../input/b0-weights/efficientnetb0_notop.h5"
    TRAINED_WEIGHTS_PATH = "../input/b0-weights/pet_trained_weights_v6.h5"

    def get_model():
        image_inputs = layers.Input(shape=(image_size, image_size, 3))
        image_x = img_augmentation(image_inputs)

        if os.path.exists(B0_NOTOP_PATH):
            base_weights = B0_NOTOP_PATH
        else:
            base_weights = "imagenet"

        base_model = EfficientNetB0(
            include_top=False,
            input_tensor=image_x,
            weights=base_weights,
            input_shape=(image_size, image_size, 3),
        )
        base_model.trainable = False

        image_x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        top_dropout_rate = 0.2
        image_x = layers.Dropout(top_dropout_rate, name="top_dropout")(image_x)

        tabular_inputs = keras.Input(
            shape=(len(tabular_columns),), name="tabular_inputs"
        )
        tabular_x = get_tabular_prediciton_model(tabular_inputs)

        x = layers.Concatenate(axis=1)([image_x, tabular_x])
        outputs = layers.Dense(1)(x)

        model = keras.Model(
            inputs=[image_inputs, tabular_inputs],
            outputs=[outputs],
            name="EfficientNet",
        )
        optimizer = keras.optimizers.Adam(1e-3)
        model.compile(optimizer=optimizer, loss=rmse, metrics=["mae", "mape"])
        return model

    def preprocess_test_data(image_url, tabular):
        image_string = tf.io.read_file(image_url)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.central_crop(image, 1.0)
        image = tf.image.resize(image, (image_size, image_size))
        tabular = tf.cast(tabular, tf.float32)
        return (image, tabular), tf.constant(0.0, dtype=tf.float32)

    try:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            print("Device:", tpu.master())
            strategy = tf.distribute.TPUStrategy(tpu)
        except Exception:
            print("Not connected to a TPU runtime. Using CPU/GPU strategy")
            strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            model_up = get_model()

        if os.path.exists(TRAINED_WEIGHTS_PATH):
            model_up.load_weights(TRAINED_WEIGHTS_PATH)
        else:
            print(
                f"WARNING: trained weights not found at {TRAINED_WEIGHTS_PATH}. Using base weights only."
            )

        ds_try = (
            tf.data.Dataset.from_tensor_slices(
                (test["file_path"].values, test[tabular_columns].values)
            )
            .map(preprocess_test_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        res_0 = model_up.predict(ds_try, verbose=1).reshape(-1)
    except Exception as e:
        print(
            "WARNING: TensorFlow inference failed; falling back to tabular model only."
        )
        print("TF inference error:", repr(e))
        res_0 = None



## === cell 3
train_prev = pd.read_csv(TRAIN_CSV)
test_prev = pd.read_csv(TEST_CSV)

train_x = train_prev.drop(["Id"], axis=1)
test_x = test_prev.drop(["Id"], axis=1)

model_prev = RandomForestRegressor(
    criterion="squared_error",
    max_depth=4,
    max_features=6,
    n_estimators=1000,
    random_state=42,
    n_jobs=-1,
)
model_prev.fit(
    train_x.drop(["Pawpularity"], axis=1),
    train_x.loc[:, ["Pawpularity"]].values.ravel(),
)
res_1 = model_prev.predict(test_x)

if res_0 is None:
    predictions = res_1
else:
    w = 0.96
    predictions = w * res_0 + (1.0 - w) * res_1

predictions = np.clip(predictions, 1.0, 100.0)

submission = pd.DataFrame({"Id": test_prev["Id"].values, "Pawpularity": predictions})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Pawpularity min/max:",
    float(submission["Pawpularity"].min()),
    float(submission["Pawpularity"].max()),
)
