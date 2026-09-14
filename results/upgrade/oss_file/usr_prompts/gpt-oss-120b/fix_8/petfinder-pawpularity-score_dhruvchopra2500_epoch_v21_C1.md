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

3.12

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

20.88166

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




## === cell 1
def pythonic_loader_train():
    IMAGES_PATH = "/kaggle/input/petfinder-pawpularity-score/train/train/"
    CSV_PATH = "/kaggle/input/petfinder-pawpularity-score/train.csv"

    relevant_columns = [
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
    target = "Pawpularity"

    image_names = [
        f for f in sorted(os.listdir(IMAGES_PATH)) if f.lower().endswith(".jpg")
    ]
    image_names = image_names[: int(len(image_names) * 0.9)]
    np.random.shuffle(image_names)
    metadata_csv = pd.read_csv(CSV_PATH)

    for i in image_names:
        img = tf.keras.utils.load_img(
            path=os.path.join(IMAGES_PATH, i), color_mode="rgb", target_size=(256, 256)
        )
        img = np.array(img).astype(np.float32) / 255.0  # scale to [0,1]

        metadata = metadata_csv[metadata_csv["Id"] == i[:-4]]
        features = metadata[relevant_columns].values[0].astype(np.float32)
        y = metadata[target].values[0].astype(np.float32)

        yield ({"Image": img, "Feature": features}, y)




## === cell 2
def pythonic_loader_val():
    IMAGES_PATH = "/kaggle/input/petfinder-pawpularity-score/train/train/"
    CSV_PATH = "/kaggle/input/petfinder-pawpularity-score/train.csv"

    relevant_columns = [
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
    target = "Pawpularity"

    image_names = [
        f for f in sorted(os.listdir(IMAGES_PATH)) if f.lower().endswith(".jpg")
    ]
    image_names = image_names[int(len(image_names) * 0.9) :]  # last 10 %
    np.random.shuffle(image_names)
    metadata_csv = pd.read_csv(CSV_PATH)

    for i in image_names:
        img = tf.keras.utils.load_img(
            path=os.path.join(IMAGES_PATH, i), color_mode="rgb", target_size=(256, 256)
        )
        img = np.array(img).astype(np.float32) / 255.0

        metadata = metadata_csv[metadata_csv["Id"] == i[:-4]]
        features = metadata[relevant_columns].values[0].astype(np.float32)
        y = metadata[target].values[0].astype(np.float32)

        yield ({"Image": img, "Feature": features}, y)




## === cell 3
train_loader = tf.data.Dataset.from_generator(
    pythonic_loader_train,
    output_signature=(
        {
            "Image": tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            "Feature": tf.TensorSpec(shape=(12,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ),
)

val_loader = tf.data.Dataset.from_generator(
    pythonic_loader_val,
    output_signature=(
        {
            "Image": tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            "Feature": tf.TensorSpec(shape=(12,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ),
)




## === cell 4
input_image = tf.keras.Input(shape=(256, 256, 3), name="Image")
input_meta = tf.keras.Input(shape=(12,), name="Feature")

l2 = tf.keras.layers.MaxPool2D((2, 2))(input_image)
l3 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu")(l2)
l4 = tf.keras.layers.MaxPool2D((2, 2))(l3)
l5 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(l4)
l6 = tf.keras.layers.MaxPool2D((2, 2))(l5)
l7 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(l6)
l_mid1 = tf.keras.layers.MaxPool2D((2, 2))(l7)
l_mid2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(l_mid1)
l8 = tf.keras.layers.Flatten()(l_mid2)
l10 = tf.keras.layers.Dense(1024, activation="gelu")(l8)

combined = tf.keras.layers.concatenate([l10, input_meta])
l12 = tf.keras.layers.Dense(512, activation="gelu")(combined)
bn1 = tf.keras.layers.BatchNormalization()(l12)
le1 = tf.keras.layers.Dense(512, activation="gelu")(bn1)
bn2 = tf.keras.layers.BatchNormalization()(le1)
le2 = tf.keras.layers.Dense(512, activation="gelu")(bn2)
bn3 = tf.keras.layers.BatchNormalization()(le2)
le3 = tf.keras.layers.Dense(512, activation="gelu")(bn3)
bn4 = tf.keras.layers.BatchNormalization()(le3)
le4 = tf.keras.layers.Dense(256, activation="gelu")(bn4)
bn_5 = tf.keras.layers.BatchNormalization()(le4)
le5 = tf.keras.layers.Dense(256, activation="gelu")(bn_5)
bn_6 = tf.keras.layers.BatchNormalization()(le5)

layer1 = tf.keras.layers.Dense(256, activation="gelu")(bn_6)
layer2 = tf.keras.layers.BatchNormalization()(layer1)
layer3 = tf.keras.layers.Dense(256, activation="gelu")(layer2)
layer4 = tf.keras.layers.BatchNormalization()(layer3)
layer5 = tf.keras.layers.Dense(256, activation="gelu")(layer4)
layer6 = tf.keras.layers.BatchNormalization()(layer5)

le6 = tf.keras.layers.Dense(128, activation="gelu")(layer6)
bn5 = tf.keras.layers.BatchNormalization()(le6)
le9 = tf.keras.layers.Dense(64, activation="gelu")(bn5)
bn6 = tf.keras.layers.BatchNormalization()(le9)
l13 = tf.keras.layers.Dense(16, activation="gelu")(bn6)
bn7 = tf.keras.layers.BatchNormalization()(l13)
l14 = tf.keras.layers.Dense(1, activation="sigmoid")(bn7)
output = l14 * tf.constant([100], dtype=tf.float32)

model = tf.keras.Model(
    inputs={"Image": input_image, "Feature": input_meta}, outputs=output
)




## === cell 5
model.summary()




## === cell 6
initial_learning_rate = 10 ** (-3)
first_decay_steps = 200
lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate, decay_steps=first_decay_steps
)




## === cell 7
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_warmup_decayed_fn),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)




## === cell 8
gen_train = train_loader.shuffle(1024).batch(64).prefetch(tf.data.AUTOTUNE).repeat()
gen_val = val_loader.batch(64).prefetch(tf.data.AUTOTUNE)




## === cell 9
print("Logical GPUs:", tf.config.list_logical_devices("GPU"))




## === cell 10
train_image_count = int(
    len(
        [
            f
            for f in os.listdir(
                "/kaggle/input/petfinder-pawpularity-score/train/train/"
            )
            if f.lower().endswith(".jpg")
        ]
    )
    * 0.9
)
steps_per_epoch = max(1, train_image_count // 64)

model.fit(
    gen_train,
    steps_per_epoch=steps_per_epoch,
    epochs=2,
    validation_data=gen_val,
    validation_steps=1,
)




## === cell 11
def pythonic_loader_predict():
    IMAGES_PATH = "/kaggle/input/petfinder-pawpularity-score/test/test/"
    CSV_PATH = "/kaggle/input/petfinder-pawpularity-score/test.csv"

    relevant_columns = [
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

    image_names = [
        f for f in sorted(os.listdir(IMAGES_PATH)) if f.lower().endswith(".jpg")
    ]  # deterministic order
    metadata_csv = pd.read_csv(CSV_PATH)

    for i in image_names:
        img = tf.keras.utils.load_img(
            path=os.path.join(IMAGES_PATH, i), color_mode="rgb", target_size=(256, 256)
        )
        img = np.array(img).astype(np.float32) / 255.0

        metadata = metadata_csv[metadata_csv["Id"] == i[:-4]]
        features = metadata[relevant_columns].values[0].astype(np.float32)

        yield {"Image": img, "Feature": features}




## === cell 12
predict_loader = tf.data.Dataset.from_generator(
    pythonic_loader_predict,
    output_signature={
        "Image": tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        "Feature": tf.TensorSpec(shape=(12,), dtype=tf.float32),
    },
)

TEST_PATH = "/kaggle/input/petfinder-pawpularity-score/test/test/"
num_test_samples = sum(1 for f in os.listdir(TEST_PATH) if f.lower().endswith(".jpg"))

preds = model.predict(predict_loader.batch(1), verbose=0, steps=num_test_samples)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
/tmp/ipykernel_55/1179847811.py in <cell line: 0>()
     11 
     12 # Explicitly provide the number of steps so Keras does not rely on dataset cardinality.
---> 13 preds = model.predict(predict_loader.batch(1), verbose=0, steps=num_test_samples)
     14 
     15 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py in predict(self, x, batch_size, verbose, steps, callbacks)
    567         callbacks.on_predict_end()
    568         outputs = tree.map_structure_up_to(
--> 569             batch_outputs, potentially_ragged_concat, outputs
    570         )
    571         return tree.map_structure(convert_to_np_if_not_ragged, outputs)

UnboundLocalError: cannot access local variable 'batch_outputs' where it is not associated with a value

## === cell 13
ser = pd.Series(preds.squeeze(), name="Pawpularity")




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1391551905.py in <cell line: 0>()
----> 1 ser = pd.Series(preds.squeeze(), name="Pawpularity")
      2 
      3 

NameError: name 'preds' is not defined

## === cell 14
test_ids = sorted(
    [
        i[:-4]
        for i in os.listdir("/kaggle/input/petfinder-pawpularity-score/test/test/")
        if i.lower().endswith(".jpg")
    ]
)
id_ = pd.Series(test_ids, name="Id")




## === cell 15
ans = pd.DataFrame({"Id": id_, "Pawpularity": ser})




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3743986363.py in <cell line: 0>()
----> 1 ans = pd.DataFrame({"Id": id_, "Pawpularity": ser})
      2 
      3 

NameError: name 'ser' is not defined

## === cell 16
ans.to_csv("submission.csv", index=False)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/899919599.py in <cell line: 0>()
----> 1 ans.to_csv("submission.csv", index=False)

NameError: name 'ans' is not defined
