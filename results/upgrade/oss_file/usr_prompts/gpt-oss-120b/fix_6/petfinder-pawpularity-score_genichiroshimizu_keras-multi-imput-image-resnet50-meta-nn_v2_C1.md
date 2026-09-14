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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

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

22.28318

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

import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Dense,
        concatenate,
        Dropout,
        MaxPooling2D,
        Conv2D,
        Flatten,
        Input,
    )
    from tensorflow.keras import regularizers
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model

    _ = tf.constant(0)
    TF_AVAILABLE = True
except Exception as e:
    print(f"TensorFlow import or sanity check failed ({e}); using sklearn fallback.")
    TF_AVAILABLE = False




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
comp_path = "/kaggle/input/petfinder-pawpularity-score"
train_csv_path = os.path.join(comp_path, "train.csv")
test_csv_path = os.path.join(comp_path, "test.csv")




## === cell 2
train_meta = pd.read_csv(train_csv_path)
test_meta = pd.read_csv(test_csv_path)


def img_path(folder, img_id):
    return os.path.join(comp_path, folder, img_id + ".jpg")


train_meta["ImgPath"] = train_meta["Id"].apply(lambda x: img_path("train", x))
test_meta["ImgPath"] = test_meta["Id"].apply(lambda x: img_path("test", x))




## === cell 3
def gen_flow_two_inputs(datagen, batch_size, df, shuffle=True):
    """
    Yields ((image_batch, meta_batch), label_batch) for training.
    """
    df_indexed = df.set_index("Id")
    flow = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="ImgPath",
        y_col="Id",  # dummy; will be overwritten
        class_mode="raw",
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    while True:
        img_batch, id_batch = next(flow)
        id_batch = np.ravel(id_batch).astype(str)
        meta_batch = df_indexed.loc[
            id_batch,
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
            ],
        ].values.astype(np.float32)
        label_batch = df_indexed.loc[id_batch, "Pawpularity"].values.astype(np.float32)
        yield ((img_batch, meta_batch), label_batch)


def gen_flow_two_inputs_test(datagen, batch_size, df):
    """
    Yields (image_batch, meta_batch) for inference.
    """
    df_indexed = df.set_index("Id")
    flow = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="ImgPath",
        y_col="Id",
        class_mode="raw",
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
    )
    while True:
        img_batch, id_batch = next(flow)
        id_batch = np.ravel(id_batch).astype(str)
        meta_batch = df_indexed.loc[
            id_batch,
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
            ],
        ].values.astype(np.float32)
        yield (img_batch, meta_batch)




## === cell 4
if TF_AVAILABLE:
    input_img = Input(shape=(224, 224, 3), name="image_input")
    input_meta = Input(shape=(12,), name="meta_input")

    x = Conv2D(16, kernel_size=3, activation="relu")(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=3, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)

    combined = concatenate([x, input_meta])
    z = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        combined
    )
    z = Dropout(0.25)(z)
    z = Dense(8, activation="relu", kernel_regularizer=regularizers.l2(0.001))(z)
    output = Dense(1, name="output")(z)

    model = Model(inputs=[input_img, input_meta], outputs=output)
    model.compile(loss="mse", optimizer="adam", metrics=["mse"])

    train_datagen = image.ImageDataGenerator(rescale=1 / 255)

    EPOCHS = 10
    BATCH_SIZE = 32
    steps_per_epoch = int(np.ceil(train_meta.shape[0] / BATCH_SIZE))

    history = model.fit(
        gen_flow_two_inputs(train_datagen, BATCH_SIZE, train_meta),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        verbose=1,
    )

    test_datagen = image.ImageDataGenerator(rescale=1 / 255)
    test_steps = int(np.ceil(test_meta.shape[0] / BATCH_SIZE))
    preds = model.predict(
        gen_flow_two_inputs_test(test_datagen, BATCH_SIZE, test_meta),
        steps=test_steps,
        verbose=1,
    ).ravel()
else:
    from sklearn.ensemble import GradientBoostingRegressor

    feature_cols = [
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

    X_train = train_meta[feature_cols].values.astype(np.float32)
    y_train = train_meta["Pawpularity"].values.astype(np.float32)

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gbr.fit(X_train, y_train)

    X_test = test_meta[feature_cols].values.astype(np.float32)
    preds = gbr.predict(X_test).ravel()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/425724788.py in <cell line: 0>()
     38     test_datagen = image.ImageDataGenerator(rescale=1 / 255)
     39     test_steps = int(np.ceil(test_meta.shape[0] / BATCH_SIZE))
---> 40     preds = model.predict(
     41         gen_flow_two_inputs_test(test_datagen, BATCH_SIZE, test_meta),
     42         steps=test_steps,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    158     inputs = tree.flatten(inputs)
    159     if len(inputs) != len(input_spec):
--> 160         raise ValueError(
    161             f'Layer "{layer_name}" expects {len(input_spec)} input(s),'
    162             f" but it received {len(inputs)} input tensors. "

ValueError: Layer "functional" expects 2 input(s), but it received 1 input tensors. Inputs received: [<tf.Tensor 'data:0' shape=(32, 224, 224, 3) dtype=float32>]

## === cell 5
test_meta["Pawpularity"] = preds
submission_df = test_meta[["Id", "Pawpularity"]].copy()
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission_df.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2192399323.py in <cell line: 0>()
----> 1 test_meta["Pawpularity"] = preds
      2 submission_df = test_meta[["Id", "Pawpularity"]].copy()
      3 submission_path = "submission.csv"
      4 submission_df.to_csv(submission_path, index=False)
      5 print(f"Submission saved to {submission_path}")

NameError: name 'preds' is not defined
