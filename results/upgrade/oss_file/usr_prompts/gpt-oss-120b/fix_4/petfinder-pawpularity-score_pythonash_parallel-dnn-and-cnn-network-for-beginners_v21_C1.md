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

20.49703

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

import cv2, numpy as np, pandas as pd, tensorflow as tf
import matplotlib.pyplot as plt, seaborn as sns

BASE_DIR = os.path.abspath("../input/petfinder-pawpularity-score")
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.abspath("./data/petfinder-pawpularity-score")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_csv = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
submission = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))




## === cell 2
def load_images(img_dir, ids):
    """Load and resize images given a directory and list of ids."""
    imgs = []
    for img_id in ids:
        path = os.path.join(img_dir, f"{img_id}.jpg")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32) / 255.0)
    return np.stack(imgs)




## === cell 3
train_ids = train_csv["Id"].values
train_img = load_images(TRAIN_DIR, train_ids)



## === cell 4
test_ids = test_csv["Id"].values
test_img = load_images(TEST_DIR, test_ids)



## === cell 5
csv_feature_cols = [c for c in train_csv.columns if c not in ["Id", "Pawpularity"]]
train_csv_x = train_csv[csv_feature_cols].astype(np.float32).values
train_y = train_csv["Pawpularity"].astype(np.float32).values
test_csv_x = test_csv[csv_feature_cols].astype(np.float32).values



## === cell 6
csv_input = tf.keras.Input(shape=(train_csv_x.shape[1],), name="CSV_Input")
img_input = tf.keras.Input(shape=(64, 64, 3), name="IMG_Input")

csv_hidden = csv_input
for i in range(6):
    csv_hidden = tf.keras.layers.Dense(
        200, activation="elu", kernel_initializer="he_normal", name=f"CSV_Hidden{i+1}"
    )(csv_hidden)
csv_dropout = tf.keras.layers.Dropout(0.5, name="CSV_Dropout")(csv_hidden)

x = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv1",
)(img_input)
x = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv2",
)(x)
x = tf.keras.layers.MaxPooling2D(4, name="IMG_Max1")(x)

x = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv3",
)(x)
x = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv4",
)(x)
x = tf.keras.layers.MaxPooling2D(4, name="IMG_Max2")(x)

x = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv5",
)(x)
x = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv6",
)(x)
x = tf.keras.layers.MaxPooling2D(3, name="IMG_Max3")(x)

x = tf.keras.layers.Dropout(0.5, name="IMG_Dropout")(x)
x = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv7",
)(x)

x = tf.keras.layers.Dense(
    300, activation="elu", kernel_initializer="he_normal", name="IMG_hidden1"
)(x)
x = tf.keras.layers.Dropout(0.5, name="IMG_Dropout1")(x)
x = tf.keras.layers.Dense(
    600, activation="elu", kernel_initializer="he_normal", name="IMG_hidden2"
)(x)
x = tf.keras.layers.GlobalAvgPool2D(name="IMG_Gpool")(x)
img_dropout2 = tf.keras.layers.Dropout(0.5, name="IMG_Dropout2")(x)

csv_output = tf.keras.layers.Dense(1, name="CSV_Output")(csv_dropout)
img_output = tf.keras.layers.Dense(1, name="IMG_Output")(img_dropout2)

model = tf.keras.Model(
    inputs=[csv_input, img_input],
    outputs=[csv_output, img_output],
    name="Pythonash_model",
)



## === cell 7
learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002, decay_steps=10000, decay_rate=0.99
)
opt = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
model.compile(
    loss=["mse", "mse"],
    loss_weights=[0.5, 0.5],
    optimizer=opt,
    metrics=[
        tf.keras.metrics.RootMeanSquaredError(),
        tf.keras.metrics.RootMeanSquaredError(),
    ],
)



## === cell 8
epochs = 20
checkpoint_path = "pythonash_model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)

check_1 = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, save_best_only=True, verbose=2
)

model.fit(
    x=[train_csv_x, train_img],
    y=[train_y, train_y],
    epochs=epochs,
    validation_split=0.2,
    verbose=2,
    batch_size=100,
    callbacks=[check_1],
)



## === cell 9
if os.path.exists(checkpoint_path):
    try:
        best_model = tf.keras.models.load_model(checkpoint_path, compile=False)
    except Exception:
        best_model = model
else:
    best_model = model

csv_pred, img_pred = best_model.predict([test_csv_x, test_img])
final_pred = 0.5 * csv_pred + 0.5 * img_pred
final_df = pd.DataFrame({"Id": test_ids, "Pawpularity": final_pred.ravel()})



## === cell 10
output_path = "./working/submission.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
