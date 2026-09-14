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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
seaborn==0.12.2
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

20.49285

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_PATH = "../input/petfinder-pawpularity-score"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_CSV_PATH = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test")

train_csv = pd.read_csv(TRAIN_CSV_PATH)
test_csv = pd.read_csv(TEST_CSV_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

print(train_csv.shape, test_csv.shape, submission.shape)
print(train_csv.columns.tolist())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv = train_csv.drop_duplicates().reset_index(drop=True)
test_csv = test_csv.drop_duplicates().reset_index(drop=True)

feature_cols = [c for c in train_csv.columns if c not in ["Id", "Pawpularity"]]

assert all(
    c in test_csv.columns for c in feature_cols
), "Mismatch train/test feature columns"



## === cell 2
IMG_SIZE = (64, 64)


def load_images_from_ids(ids, img_dir, size=(64, 64)):
    images = []
    missing = 0
    for _id in ids:
        fp = os.path.join(img_dir, f"{_id}.jpg")
        img = cv2.imread(fp)
        if img is None:
            missing += 1
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        images.append(img)
    if missing:
        print(
            f"Warning: {missing} images missing in {img_dir}. They were replaced with zeros."
        )
    return np.stack(images, axis=0)


train_ids = train_csv["Id"].values
test_ids = test_csv["Id"].values

train_img = load_images_from_ids(train_ids, TRAIN_IMG_DIR, size=IMG_SIZE)
test_img = load_images_from_ids(test_ids, TEST_IMG_DIR, size=IMG_SIZE)

print("train_img:", train_img.shape, "test_img:", test_img.shape)



## === cell 3
train_csv_data = train_csv.copy()
test_csv_data = test_csv.copy()

train_csv_x = train_csv_data[feature_cols].astype(np.float32)
train_y = train_csv_data["Pawpularity"].astype(np.float32)

test_csv_x = test_csv_data[feature_cols].astype(np.float32)

print(train_csv_x.shape, train_y.shape, test_csv_x.shape)



## === cell 4
csv_input = tf.keras.Input(shape=train_csv_x.shape[1:], name="CSV_Input")
img_input = tf.keras.Input(shape=train_img.shape[1:], name="IMG_Input")

csv_hidden1 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden1"
)(csv_input)
csv_hidden2 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden2"
)(csv_hidden1)
csv_hidden3 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden3"
)(csv_hidden2)
csv_hidden4 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden4"
)(csv_hidden3)
csv_hidden5 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden5"
)(csv_hidden4)
csv_hidden6 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden6"
)(csv_hidden5)
csv_dropout = tf.keras.layers.Dropout(0.5, name="CSV_Dropout")(csv_hidden6)

img_conv1 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv1",
)(img_input)
img_conv2 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv2",
)(img_conv1)
img_pooling1 = tf.keras.layers.MaxPooling2D(4, name="IMG_Max1")(img_conv2)

img_conv3 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv3",
)(img_pooling1)
img_conv4 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv4",
)(img_conv3)
img_pooling2 = tf.keras.layers.MaxPooling2D(4, name="IMG_Max2")(img_conv4)

img_conv5 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv5",
)(img_pooling2)
img_conv6 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv6",
)(img_conv5)
img_pooling3 = tf.keras.layers.MaxPooling2D(3, name="IMG_Max3")(img_conv6)

img_dropout = tf.keras.layers.Dropout(0.5, name="IMG_Dropout")(img_pooling3)
img_conv7 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv7",
)(img_dropout)

flatten = tf.keras.layers.Flatten(name="IMG_Flatten")(img_conv7)

img_hidden1 = tf.keras.layers.Dense(
    300,
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_hidden1",
    use_bias=False,
)(flatten)
img_dropout1 = tf.keras.layers.Dropout(0.5, name="IMG_Dropout1")(img_hidden1)

img_hidden2 = tf.keras.layers.Dense(
    300,
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_hidden2",
    use_bias=False,
)(img_dropout1)
img_dropout2 = tf.keras.layers.Dropout(0.5, name="IMG_Dropout2")(img_hidden2)

csv_output = tf.keras.layers.Dense(1, name="CSV_Output")(csv_dropout)
img_output = tf.keras.layers.Dense(1, name="IMG_Output")(img_dropout2)

model = tf.keras.Model(
    inputs=[csv_input, img_input],
    outputs=[csv_output, img_output],
    name="Pythonash_model",
)
model.summary()



## === cell 5
learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002, decay_steps=10000, decay_rate=0.97
)
opt = tf.keras.optimizers.Adam(learning_rate=learning_schedule)

model.compile(
    loss=["mse", "mse"],
    loss_weights=[0.5, 0.5],
    optimizer=opt,
    metrics=tf.keras.metrics.RootMeanSquaredError(),
)

epoch_number = 5

ckpt_path = "pythonash_model.keras"
check_1 = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path, save_best_only=True, verbose=2, monitor="val_loss", mode="min"
)

history = model.fit(
    x=[train_csv_x, train_img],
    y=[train_y, train_y],
    epochs=epoch_number,
    validation_split=0.2,
    verbose=2,
    batch_size=100,
    validation_batch_size=100,
    callbacks=[check_1],
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4074140772.py in <cell line: 0>()
      5 opt = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
      6 
----> 7 model.compile(
      8     loss=["mse", "mse"],
      9     loss_weights=[0.5, 0.5],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in __init__(self, metrics, weighted_metrics, name, output_names)
    132         super().__init__(name=name)
    133         if metrics and not isinstance(metrics, (list, tuple, dict)):
--> 134             raise ValueError(
    135                 "Expected `metrics` argument to be a list, tuple, or dict. "
    136                 f"Received instead: metrics={metrics} of type {type(metrics)}"

ValueError: Expected `metrics` argument to be a list, tuple, or dict. Received instead: metrics=<RootMeanSquaredError name=root_mean_squared_error> of type <class 'keras.src.metrics.regression_metrics.RootMeanSquaredError'>

## === cell 6
best_model = tf.keras.models.load_model(ckpt_path, compile=False)

csv_result, img_result = best_model.predict(
    [test_csv_x, test_img], batch_size=100, verbose=0
)

final_pred = 0.5 * csv_result.reshape(-1) + 0.5 * img_result.reshape(-1)

final_pred = np.clip(final_pred, 0.0, 100.0)

sub = pd.DataFrame(
    {"Id": test_csv_data["Id"].values, "Pawpularity": final_pred.astype(np.float32)}
)

sub = sub[["Id", "Pawpularity"]]

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(sub.head())
print(f"Saved submission to: {out_path}  rows={len(sub)}  cols={sub.columns.tolist()}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1592161553.py in <cell line: 0>()
      1 # Inference + submission writing
----> 2 best_model = tf.keras.models.load_model(ckpt_path, compile=False)
      3 
      4 csv_result, img_result = best_model.predict(
      5     [test_csv_x, test_img], batch_size=100, verbose=0

NameError: name 'ckpt_path' is not defined
