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

29.06754310772829

# 6. Current score

26.03422

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 35.53384) has done: 'I fix the immediate runtime crash caused by importing unnecessary heavy libraries (notably `dask`) that trigger the protobuf `MessageFactory.GetPrototype` error in this environment. Then I remove the dependency on a non-existent external dataset (`../input/paw-model/...`) by training the same CNN architecture locally on the provided `train/` images and `train.csv`, so `pre_trained_model` is always defined. Finally, I keep your existing image preprocessing and the `model_adjusted()` post-calibration logic, but compute `y_pred` from the model’s training predictions (instead of loading a missing pickle) so the output scaling is consistent and a valid `submission.csv` is always produced.'
- What this solution (achieved 35.53384) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting (it triggers the `MessageFactory.GetPrototype` issue in this environment) and keeping TF logging suppression only. Then I fix the submission validity error by ensuring final predictions are clipped to the required `[1, 100]` range (the competition disallows 0), and by calibrating with the *training* prediction distribution (the prior code mistakenly re-fit min/max using test predictions inside `model_adjusted`, which is inconsistent and can push values to 0). These changes keep your CNN, preprocessing, training loop, and calibration logic intact while making the notebook run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved 35.53384) has done: 'I fix the TensorFlow/protobuf crash happening at import time by forcing TensorFlow to use the pure-Python protobuf backend before importing `tensorflow` (this avoids the `MessageFactory.GetPrototype` issue in this Kaggle environment). Then I keep your exact CNN, preprocessing, training loop, and calibration logic, but add deterministic settings and make sure the submission columns and value range are always valid. Finally, I ensure paths resolve correctly and the script always writes `submission.csv` with the required header and `[1, 100]` clipping.'
- What this solution (achieved 26.03422) has done: 'I fix the import-time TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment override that triggers the `MessageFactory.GetPrototype` AttributeError in this Kaggle runtime. Then I keep your exact CNN, preprocessing, training loop, and calibration function, but make the calibration score-improving (and still minimal) by mapping predictions to the *training target* distribution (mean/std) rather than stretching to `[0,100]` based on the model’s min/max outputs, which is typically harmful for RMSE. Finally, I ensure the submission is always valid by clipping predictions to `[1, 100]` and writing `submission.csv` with the required columns.'
- What this solution (achieved 26.03422) has done: 'We fix the import-time TensorFlow/protobuf crash that happens before any training by pinning protobuf to the pure-Python implementation *before* importing TensorFlow (this is the minimal, known workaround for the `MessageFactory.GetPrototype` error in some Kaggle images). We keep your exact CNN, preprocessing, training loop, and calibration logic unchanged so evaluation semantics remain the same. We also add a small safety fallback to disable determinism if it triggers unexpected issues, without changing the training procedure. The script still train, predict, calibrate, clip to `[1,100]`, and write `./submission.csv` with the required columns.'
- What this solution (achieved 26.03422) has done: 'We fix the TensorFlow/protobuf import crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which is the direct cause of the `MessageFactory.GetPrototype` failure in this environment. Everything else (data loading, grayscale preprocessing, CNN architecture, training loop, and the existing mean/std calibration logic) be kept the same to preserve evaluation semantics and keep score changes minimal. We also add a small safety guard to disable op determinism if it errors, without changing training procedure otherwise, and ensure the submission is always written as `submission.csv` with the required columns and value clipping.'
- What this solution (achieved 26.03422) has done: 'We fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend *before* importing TensorFlow, which is the known workaround for the `MessageFactory.GetPrototype` AttributeError in this Kaggle image. We keep your CNN, preprocessing, training loop, and calibration logic unchanged to preserve evaluation semantics and avoid unnecessary score shifts (your current score is already better than the target, so we won’t try to improve it). We also add a small, safe fallback if op-determinism enabling triggers an exception, without changing the training procedure. The script still train end-to-end and write a valid `submission.csv` with the required columns and `[1, 100]` clipping.'
- What this solution (achieved 26.03422) has done: 'I fix the immediate runtime crash in the first cell by removing the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override that is triggering the protobuf `MessageFactory.GetPrototype` AttributeError when importing TensorFlow in this Kaggle image. I keep your CNN architecture, preprocessing, training loop, and the existing mean/std calibration exactly the same so score behavior remains essentially unchanged (and you’re already better than the target, so we won’t try to improve further). I also add a small, safe fallback to retry TensorFlow import with determinism disabled if the first import attempt still fails, without changing the training procedure otherwise. The script still run end-to-end and write `./submission.csv` with the required columns and `[1, 100]` clipping.'
- What this solution (achieved 26.03422) has done: 'I fix the TensorFlow import-time protobuf crash by ensuring we *don’t* set the problematic `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, and by adding a safe two-step fallback that re-imports TensorFlow after clearing protobuf-related env vars if the first import fails. This is the minimal change needed to make the notebook run end-to-end again, and it is score-neutral (it doesn’t alter your model, preprocessing, training loop, or calibration). I also keep the determinism call guarded so it can’t crash execution. The rest of the pipeline (training, calibration, clipping to `[1,100]`, and writing `submission.csv`) stays unchanged to preserve evaluation semantics and avoid drifting further away from your target score.'
- What this solution (achieved 26.03422) has done: 'The crash happens before training because TensorFlow’s import can still hit a protobuf `MessageFactory.GetPrototype` incompatibility in this Kaggle image, even after unsetting env vars. I make TensorFlow import robust by forcing the pure-Python protobuf backend *only if the first import fails*, and (as a second fallback) disabling C-implementation protobuf, then retrying—this is execution-only and score-neutral. I also keep your model/training/calibration intact, only adding a safe prediction fallback so submission is always produced even if the reference prediction vector is missing. The rest of the pipeline (data paths, preprocessing, CNN, epochs, calibration, clipping, and submission format) stays unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import glob
import random as rand

import numpy as np
import pandas as pd

import cv2


def _import_tensorflow_safely():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e1:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
        try:
            import tensorflow as tf  # noqa: F401

            return tf
        except Exception:
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
            import tensorflow as tf  # noqa: F401

            return tf


tf = _import_tensorflow_safely()

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten

from sklearn.model_selection import train_test_split

SEED = 42
rand.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
INPUT_ROOT = "/kaggle/input/petfinder-pawpularity-score"
if not os.path.exists(INPUT_ROOT):
    INPUT_ROOT = "../input/petfinder-pawpularity-score"

TRAIN_IMG_DIR = os.path.join(INPUT_ROOT, "train")
TEST_IMG_DIR = os.path.join(INPUT_ROOT, "test")
TRAIN_CSV = os.path.join(INPUT_ROOT, "train.csv")
TEST_CSV = os.path.join(INPUT_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(INPUT_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing test.csv at {TEST_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train/ at {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test/ at {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"



## === cell 2
train_path = os.path.join(TRAIN_IMG_DIR, "*.jpg")
test_path = os.path.join(TEST_IMG_DIR, "*.jpg")

train_images = glob.glob(train_path)
test_images = glob.glob(test_path)

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

df_test = df_test.copy()
df_test["img_path"] = df_test["Id"].apply(
    lambda x: os.path.join(TEST_IMG_DIR, f"{x}.jpg")
)

df_train.head()



## === cell 3
sq_shape = 128


def process_image(image):
    im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # reads image as greyscale
    if im is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    im = cv2.resize(im, (sq_shape, sq_shape))  # resizes for processing speed
    im = im / 255.0  # normalizes
    return im




## === cell 4
df_train = df_train.copy()
df_train["img_path"] = df_train["Id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.jpg")
)

missing = df_train.loc[~df_train["img_path"].apply(os.path.exists)]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} training images. Example: {missing.iloc[0]['img_path']}"
    )

X = np.zeros((len(df_train), sq_shape, sq_shape, 1), dtype=np.float32)
y = df_train["Pawpularity"].astype(np.float32).values

for i, p in enumerate(df_train["img_path"].values):
    im = process_image(p)
    X[i, :, :, 0] = im

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=SEED
)



## === cell 5
pre_trained_model = Sequential(
    [
        Input(shape=(sq_shape, sq_shape, 1)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.25),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(1, activation="linear"),
    ]
)

pre_trained_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)

pre_trained_model.summary()



## === cell 6
history = pre_trained_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64,
    verbose=2,
)



## === cell 7
y_pred_train_ref = pre_trained_model.predict(
    X_train, batch_size=128, verbose=0
).reshape(-1)




## === cell 8
def model_adjusted(prediction, y_pred, y_true):
    """
    Replace min-max stretch to [0,100] with an affine calibration mapping
    prediction distribution to training target distribution (mean/std).
    """
    ref = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    y_t = np.asarray(y_true, dtype=np.float32).reshape(-1)

    mu_ref = float(ref.mean())
    std_ref = float(ref.std())
    mu_y = float(y_t.mean())
    std_y = float(y_t.std())

    if std_ref <= 1e-12 or std_y <= 1e-12:
        return np.array([mu_y], dtype=np.float32)

    output = (float(prediction) - mu_ref) * (std_y / std_ref) + mu_y
    return np.array([output], dtype=np.float32)




## === cell 9
X_test_images = np.zeros((len(df_test), sq_shape, sq_shape, 1), dtype=np.float32)
Name = df_test["Id"].tolist()

missing_test = df_test.loc[~df_test["img_path"].apply(os.path.exists)]
if len(missing_test) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test images. Example: {missing_test.iloc[0]['img_path']}"
    )

for i, p in enumerate(df_test["img_path"].values):
    im = process_image(p)
    X_test_images[i, :, :, 0] = im

predictions = pre_trained_model.predict(
    X_test_images, batch_size=128, verbose=0
).reshape(-1)

if (
    "y_pred_train_ref" not in globals()
    or y_pred_train_ref is None
    or len(y_pred_train_ref) == 0
):
    y_pred_train_ref = pre_trained_model.predict(
        X_train, batch_size=128, verbose=0
    ).reshape(-1)

outputs = []
for pred in predictions:
    output = model_adjusted(pred, y_pred_train_ref, y_train)[0]
    outputs.append(float(output))

outputs = np.clip(np.asarray(outputs, dtype=np.float32), 1.0, 100.0)

df_final = pd.DataFrame({"Id": Name, "Pawpularity": outputs})
df_final.to_csv("./submission.csv", index=False)

df_final.head()
