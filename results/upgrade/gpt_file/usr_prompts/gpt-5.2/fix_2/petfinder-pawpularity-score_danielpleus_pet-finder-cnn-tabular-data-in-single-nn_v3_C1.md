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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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

21.43546

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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, utils
import pandas as pd
import numpy as np
from tensorflow import keras
from PIL import Image
import math

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
batch_size = 32
validation_split = 0.2
random_seed = 42
img_size = (256, 256)



## === cell 2
training_dir = "../input/petfinder-pawpularity-score/train"
test_dir = "../input/petfinder-pawpularity-score/test"



## === cell 3
train_all = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
training_df = train_all.iloc[0:8000].reset_index(drop=True)
validation_df = train_all.iloc[8000:].reset_index(drop=True)
test_df = pd.read_csv("../input/petfinder-pawpularity-score/test.csv").reset_index(
    drop=True
)

print(training_df.shape, validation_df.shape, test_df.shape)



## === cell 4
target = "Pawpularity"
features = training_df.drop(columns={"Id", target}).columns.values
print("n_features:", len(features), features)




## === cell 5
def get_image(image_id, image_folder, img_size, resized_folder):
    resized_path = os.path.join(resized_folder, f"{image_id}.jpg")

    if os.path.isfile(resized_path):
        img = Image.open(resized_path)
        return img

    img_path = os.path.join(image_folder, f"{image_id}.jpg")
    img = Image.open(img_path).convert("RGB")
    img = img.resize(img_size)
    img.save(resized_path)
    return img


class DataGenerator(utils.Sequence):
    """
    Fixes:
    - Must return tuples (not lists) for multi-input Keras with TF 2.18.
    - Correct test/no-target handling: `target is None` is the reliable check.
    - Ensure images are float32 tensors with consistent shape.
    """

    def __init__(
        self,
        data_type,
        data,
        target,
        img_folder,
        img_size,
        batch_size,
        features,
        shuffle=False,
        seed=42,
    ):
        self.data = data.reset_index(drop=True)
        self.target = target  # can be None
        self.batch_size = batch_size
        self.img_folder = img_folder
        self.img_size = img_size
        self.data_type = data_type
        self.features = features
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.resized_folder = os.path.join("./resized", data_type)
        os.makedirs(self.resized_folder, exist_ok=True)

        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.data))
        batch_idx = self.indexes[start:end]

        batch = self.data.iloc[batch_idx]
        ids = batch["Id"].values

        images = np.stack(
            [
                np.asarray(
                    get_image(i, self.img_folder, self.img_size, self.resized_folder),
                    dtype=np.uint8,
                )
                for i in ids
            ],
            axis=0,
        )
        meta = batch[self.features].to_numpy(dtype=np.float32)

        images = tf.cast(images, tf.float32) / 255.0

        x = (images, meta)

        if self.target is None:
            return x

        y = (
            batch[self.target.name].to_numpy(dtype=np.float32)
            if isinstance(self.target, pd.Series)
            else np.asarray(self.target)[batch_idx].astype(np.float32)
        )
        return x, y




## === cell 6
training_ds = DataGenerator(
    "training",
    training_df,
    training_df[target],
    training_dir,
    img_size,
    batch_size,
    features,
    shuffle=True,
    seed=random_seed,
)
validation_ds = DataGenerator(
    "validation",
    validation_df,
    validation_df[target],
    training_dir,
    img_size,
    batch_size,
    features,
    shuffle=False,
    seed=random_seed,
)
test_ds = DataGenerator(
    "test",
    test_df,
    None,
    test_dir,
    img_size,
    batch_size,
    features,
    shuffle=False,
    seed=random_seed,
)



## === cell 7
len(features)



## === cell 8
inputs = keras.Input(shape=(12,))
img_inputs = keras.Input(shape=(256, 256, 3))

img1 = layers.Conv2D(4, (4, 4), activation="relu")(img_inputs)
img2 = layers.MaxPooling2D((4, 4))(img1)
img3 = layers.Conv2D(8, (4, 4), activation="relu")(img2)
img4 = layers.MaxPooling2D((4, 4))(img3)
img5 = layers.Conv2D(8, (4, 4), activation="relu")(img4)
img6 = layers.Flatten()(img5)
img7 = layers.Dense(
    64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)
)(img6)
img8 = layers.Dropout(0.3)(img7)
img9 = layers.Dense(8)(img8)
img = keras.Model(inputs=img_inputs, outputs=img9)

tab1 = layers.Dense(8, input_shape=(12,))(inputs)
tab = keras.Model(inputs=inputs, outputs=tab1)

x1 = layers.Concatenate(axis=1)([tab.output, img.output])
x2 = layers.Dense(32)(x1)
x3 = layers.Dense(16)(x2)
outputs = layers.Dense(1)(x3)

model = keras.Model(inputs=[img.input, tab.input], outputs=outputs, name="model")



## === cell 9
model.summary()



## === cell 10
model.compile(optimizer="adam", loss=tf.keras.losses.MSE, metrics=["mse"])



## === cell 11
history = model.fit(training_ds, epochs=10, validation_data=validation_ds)



## === cell 12
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend(loc="lower right")
plt.show()



## === cell 13
prediction = model.predict(test_ds, verbose=1).reshape(-1)

prediction = np.clip(prediction, 0.0, 100.0)

submission = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")
submission["Pawpularity"] = prediction.astype(np.float32)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2419044113.py in <cell line: 0>()
----> 1 prediction = model.predict(test_ds, verbose=1).reshape(-1)
      2 
      3 # Pawpularity is typically in [0, 100]; clipping avoids extreme outputs that can hurt RMSE.
      4 prediction = np.clip(prediction, 0.0, 100.0)
      5 

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

ValueError: Layer "model" expects 2 input(s), but it received 1 input tensors. Inputs received: [<tf.Tensor 'data:0' shape=(32, 256, 256, 3) dtype=float32>]
