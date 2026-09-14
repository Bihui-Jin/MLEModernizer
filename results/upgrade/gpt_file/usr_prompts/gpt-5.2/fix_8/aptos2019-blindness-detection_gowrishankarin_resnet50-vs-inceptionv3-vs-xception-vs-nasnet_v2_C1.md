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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.39099

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment/runtime issues caused by the Kaggle path assumptions and the protobuf/keras compatibility by switching to `tf.keras.preprocessing.image.ImageDataGenerator` and robustly resolving the actual `/kaggle/input/...` dataset paths. I also correct API argument names (`learning_rate` instead of `lr`), fix the `model.fit` typo (`steps_per_epoch`), and replace deprecated `predict_generator` with `predict`. Finally, I ensure the submission `id_code` is correctly aligned to `test.csv` (not derived from generator filenames) and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by pinning the Python protobuf implementation to the pure-Python fallback before TensorFlow loads, which is the minimal change that unblocks execution in this environment. I also renumber the cells to start at 1 (your current script starts at cell 0) while preserving the exact modeling/training logic and data pipeline. Finally, I keep submission generation identical but add a couple of safety asserts to ensure the submission has the right columns/row count and is aligned to `test.csv`, preventing accidental 0.0 scores from format/index mismatches.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ protobuf path before TensorFlow is imported, which directly addresses the `MessageFactory.GetPrototype` error. I also make the dataset base-path detection robust for both `/kaggle/input/...` and the provided `/kaggle/data/...` layout so the script always finds CSVs and image folders. The rest of the pipeline (ImageDataGenerator, model, training loop, prediction, and submission formatting) be kept the same to preserve core logic while producing a valid `submission.csv`. These changes should move the score from 0.0 (broken/invalid run) toward the expected non-zero baseline by ensuring correct end-to-end execution and properly aligned predictions.'
- What this solution (achieved 0.59666) has done: 'I fix the TensorFlow/protobuf crash causing the notebook to stop at cell 4 by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before any TensorFlow-related import happens. I also renumber cells to start at 1 (as required) while preserving your exact data pipeline, model, and training/prediction logic. Finally, I add a couple of small safety checks to ensure `test_df` still has the expected row count and the submission aligns to `test.csv`, which prevents accidental 0.0 scores from mismatched ordering/length. These changes are execution/validity fixes and should move the score from 0.0 (crash/invalid) toward a normal non-zero baseline without changing the core modeling approach.'
- What this solution (achieved 0.43867) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow import* and by importing protobuf early to lock in that runtime, which removes the `MessageFactory.GetPrototype` error and lets the pipeline run end-to-end. I also renumber the notebook cells to start at 1 (your current script starts at cell 0) while keeping the same data pipeline, model, training loop, and prediction logic. Since your current score (0.59666) is already better than the target (0.39099), I avoid any modeling/training changes that would deliberately push the score around; the edits are execution/stability-only. The script still produce a valid `submission.csv` with `id_code,diagnosis` aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring the pure-Python protobuf runtime is selected *before* any protobuf/TensorFlow-related import and by preventing an early `google.protobuf` import from locking in the wrong implementation. I also renumber the cells to start at 1 as required, while preserving your exact data pipeline, model, training loop, and prediction logic (so the score behavior should remain essentially unchanged). Finally, I keep the same submission alignment to `test.csv` and retain the safety asserts so the run always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by ensuring the pure-Python protobuf runtime is locked in *before* TensorFlow (or anything that triggers protobuf) is imported, and by importing `google.protobuf` right after setting the environment variables. I keep your data pipeline, model, training loop, and prediction logic the same so the score behavior remains consistent while moving you from a broken 0.0 submission to a valid non-zero submission. I also renumber cells to start at 1 (as required) without changing execution order. Finally, I keep the submission aligned to `test.csv` and retain sanity checks so you don’t accidentally get a 0.0 from formatting/index issues.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CXX_IMPLEMENTATION"] = "1"

import google.protobuf  # noqa: F401

import numpy as np
import pandas as pd

BASE_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/input",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data",
    "../input/aptos2019-blindness-detection",
    "../input",
]

BASE_DIR = None
for c in BASE_CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
        os.path.join(c, "test.csv")
    ):
        BASE_DIR = c
        break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv under expected Kaggle input/data directories."
    )

print("Using BASE_DIR:", BASE_DIR)
print("BASE_DIR listing (truncated):", sorted(os.listdir(BASE_DIR))[:20])

train_csv_path = os.path.join(BASE_DIR, "train.csv")
test_csv_path = os.path.join(BASE_DIR, "test.csv")

train_img_dir_candidates = [
    os.path.join(BASE_DIR, "train_images"),
    "/kaggle/input/aptos2019-blindness-detection/train_images",
    "/kaggle/data/aptos2019-blindness-detection/train_images",
    "/kaggle/input/train_images",
    "/kaggle/data/train_images",
    "../input/train_images",
]
test_img_dir_candidates = [
    os.path.join(BASE_DIR, "test_images"),
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
    "../input/test_images",
]

TRAIN_IMG_DIR = next((p for p in train_img_dir_candidates if os.path.isdir(p)), None)
TEST_IMG_DIR = next((p for p in test_img_dir_candidates if os.path.isdir(p)), None)
if TRAIN_IMG_DIR is None or TEST_IMG_DIR is None:
    raise FileNotFoundError(
        "Could not locate train_images/test_images directories in Kaggle input/data."
    )

print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
print("Shape of train data:", train_df.shape)
print("Shape of test data:", test_df.shape)

diagnosis_df = pd.DataFrame(
    {
        "diagnosis": [0, 1, 2, 3, 4],
        "diagnosis_label": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
    }
)
train_df = train_df.merge(diagnosis_df, how="left", on="diagnosis")

train_df["files"] = train_df["id_code"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png")
)
test_df["files"] = test_df["id_code"].apply(
    lambda x: os.path.join(TEST_IMG_DIR, f"{x}.png")
)

missing_train = (~train_df["files"].apply(os.path.exists)).sum()
missing_test = (~test_df["files"].apply(os.path.exists)).sum()
print("Missing train images:", int(missing_train), " / ", len(train_df))
print("Missing test images:", int(missing_test), " / ", len(test_df))

if missing_train > 0:
    train_df = train_df[train_df["files"].apply(os.path.exists)].reset_index(drop=True)
if missing_test > 0:
    test_df = test_df[test_df["files"].apply(os.path.exists)].reset_index(drop=True)

print("Shape of train data after path attach:", train_df.shape)
print("Shape of test data after path attach:", test_df.shape)

if len(test_df) != pd.read_csv(test_csv_path).shape[0]:
    raise RuntimeError(
        "Some test images are missing; submission would not match test.csv row count."
    )



## === cell 1
train_df.head()



## === cell 2
test_df.head()



## === cell 3
IMG_SIZE = 150
N_CLASSES = train_df.diagnosis.nunique()
CLASSES = list(map(str, range(N_CLASSES)))
BATCH_SIZE = 32
EPOCH_STEPS = 10
EPOCHS = 1



## === cell 4
import tensorflow as tf

print("TensorFlow:", tf.__version__)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_df["diagnosis"] = train_df["diagnosis"].astype(str)

train_data_gen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.3)

train_data = train_data_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col="files",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    shuffle=True,
    classes=CLASSES,
    class_mode="sparse",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="training",
)

validation_data = train_data_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col="files",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    shuffle=True,
    classes=CLASSES,
    class_mode="sparse",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="validation",
)

test_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_data = test_data_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col="files",
    y_col=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    shuffle=False,
    class_mode=None,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
    ]
)

opt = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-6)
model.compile(
    optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()



## === cell 6
history = model.fit(
    train_data,
    steps_per_epoch=EPOCH_STEPS,
    epochs=EPOCHS,
    validation_data=validation_data,
    validation_steps=10,
)



## === cell 7
classifications = model.predict(test_data, steps=len(test_df), verbose=1)
pred_labels = np.argmax(classifications, axis=1).astype(int)

results = pd.DataFrame({"id_code": test_df["id_code"].values, "diagnosis": pred_labels})
results.head()



## === cell 8
assert list(results.columns) == ["id_code", "diagnosis"]
assert len(results) == len(test_df)
assert results["id_code"].isna().sum() == 0
assert results["id_code"].is_unique
assert set(results["diagnosis"].unique()).issubset({0, 1, 2, 3, 4})

sub_path = "submission.csv"
results.to_csv(sub_path, index=False)
print("Wrote submission to:", sub_path)
print(results.shape)
print(results.head())
