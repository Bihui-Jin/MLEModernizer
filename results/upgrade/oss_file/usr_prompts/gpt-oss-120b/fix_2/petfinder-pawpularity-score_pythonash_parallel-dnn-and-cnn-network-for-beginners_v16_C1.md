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

20.45483

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
import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

base_path = "../input/petfinder-pawpularity-score"
train_csv_path = os.path.join(base_path, "train.csv")
test_csv_path = os.path.join(base_path, "test.csv")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")

train_csv = pd.read_csv(train_csv_path)
test_csv = pd.read_csv(test_csv_path)
submission = pd.read_csv(sample_sub_path)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_images(img_dir):
    """Load JPG images, resize to 64x64, normalize to [0,1]. Returns list of arrays and list of ids."""
    imgs = []
    ids = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        img = img.astype("float32") / 255.0
        imgs.append(img)
        ids.append(os.path.splitext(fname)[0])
    return np.array(imgs), ids


train_img_dir = os.path.join(base_path, "train")
test_img_dir = os.path.join(base_path, "test")

train_images, train_ids = load_images(train_img_dir)
test_images, test_ids = load_images(test_img_dir)



## === cell 2
train_df = train_csv.set_index("Id").loc[train_ids].reset_index()
test_df = test_csv.set_index("Id").loc[test_ids].reset_index()

train_csv_x = train_df.drop(["Id", "Pawpularity"], axis=1)
train_y = train_df["Pawpularity"]
test_csv_x = test_df.drop(["Id"], axis=1)



## === cell 3
X_csv_train, X_csv_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
    train_csv_x.values, train_images, train_y.values, test_size=0.2, random_state=42
)



## === cell 4
csv_input = tf.keras.Input(shape=(X_csv_train.shape[1],), name="CSV_Input")
img_input = tf.keras.Input(shape=(64, 64, 3), name="IMG_Input")

csv_hidden1 = tf.keras.layers.Dense(200, activation="relu")(csv_input)
csv_hidden2 = tf.keras.layers.Dense(200, activation="relu")(csv_hidden1)
csv_hidden3 = tf.keras.layers.Dense(200, activation="relu")(csv_hidden2)
csv_dropout = tf.keras.layers.Dropout(0.5)(csv_hidden3)
csv_hidden4 = tf.keras.layers.Dense(200, activation="relu")(csv_dropout)
csv_hidden5 = tf.keras.layers.Dense(200, activation="relu")(csv_hidden4)

img_conv1 = tf.keras.layers.Conv2D(120, 5, padding="same", activation="relu")(img_input)
img_pool1 = tf.keras.layers.MaxPool2D(3)(img_conv1)
img_conv2 = tf.keras.layers.Conv2D(120, 4, padding="same", activation="relu")(img_pool1)
img_conv3 = tf.keras.layers.Conv2D(120, 4, padding="same", activation="relu")(img_conv2)
img_pool2 = tf.keras.layers.MaxPool2D(3)(img_conv3)
img_conv4 = tf.keras.layers.Conv2D(120, 3, padding="same", activation="relu")(img_pool2)
img_pool3 = tf.keras.layers.MaxPool2D(3)(img_conv4)
img_conv5 = tf.keras.layers.Conv2D(120, 3, padding="same", activation="relu")(img_pool3)
img_conv6 = tf.keras.layers.Conv2D(120, 3, padding="same", activation="relu")(img_conv5)
img_pool4 = tf.keras.layers.MaxPool2D(2)(img_conv6)
img_flatten = tf.keras.layers.Flatten()(img_pool4)
img_dense1 = tf.keras.layers.Dense(300, activation="relu")(img_flatten)
img_dropout1 = tf.keras.layers.Dropout(0.5)(img_dense1)
img_dense2 = tf.keras.layers.Dense(300, activation="relu")(img_dropout1)
img_dropout2 = tf.keras.layers.Dropout(0.5)(img_dense2)

csv_output = tf.keras.layers.Dense(1, name="CSV_Output")(csv_hidden5)
img_output = tf.keras.layers.Dense(1, name="IMG_Output")(img_dropout2)

model = tf.keras.Model(
    inputs=[csv_input, img_input],
    outputs=[csv_output, img_output],
    name="PawpularityModel",
)



## === cell 5
learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002, decay_steps=10000, decay_rate=0.9
)
opt = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
model.compile(
    loss=["mse", "mse"],
    loss_weights=[0.5, 0.5],
    optimizer=opt,
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)



## === cell 6
epochs = 20
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "pythonash_model.h5", save_best_only=True, verbose=0
)

model.fit(
    x=[X_csv_train, X_img_train],
    y=[y_train, y_train],
    validation_data=([X_csv_val, X_img_val], [y_val, y_val]),
    epochs=epochs,
    batch_size=64,
    callbacks=[checkpoint],
    verbose=2,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/166683672.py in <cell line: 0>()
      5 )
      6 
----> 7 model.fit(
      8     x=[X_csv_train, X_img_train],
      9     y=[y_train, y_train],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in _build_metrics_set(self, metrics, num_outputs, output_names, y_true, y_pred, argument_name)
    252             if isinstance(metrics, (list, tuple)):
    253                 if len(metrics) != len(y_pred):
--> 254                     raise ValueError(
    255                         "For a model with multiple outputs, "
    256                         f"when providing the `{argument_name}` argument as a "

ValueError: For a model with multiple outputs, when providing the `metrics` argument as a list, it should have as many entries as the model has outputs. Received:
metrics=[<RootMeanSquaredError name=root_mean_squared_error>]
of length 1 whereas the model has 2 outputs.

## === cell 7
best_model = tf.keras.models.load_model("pythonash_model.h5")
csv_pred, img_pred = best_model.predict([test_csv_x.values, test_images], verbose=0)

final_pred = 0.5 * csv_pred.squeeze() + 0.5 * img_pred.squeeze()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1037917716.py in <cell line: 0>()
      1 # Load best weights and predict on test set
----> 2 best_model = tf.keras.models.load_model("pythonash_model.h5")
      3 csv_pred, img_pred = best_model.predict([test_csv_x.values, test_images], verbose=0)
      4 
      5 # Ensemble the two predictions equally

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'pythonash_model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 8
submission = pd.read_csv(sample_sub_path)
submission["Pawpularity"] = final_pred
submission = submission[["Id", "Pawpularity"]]

out_dir = "./working"
os.makedirs(out_dir, exist_ok=True)
submission_path = os.path.join(out_dir, "submission.csv")
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3255177352.py in <cell line: 0>()
      1 # Prepare submission
      2 submission = pd.read_csv(sample_sub_path)
----> 3 submission["Pawpularity"] = final_pred
      4 # Ensure ordering matches the required Id column
      5 submission = submission[["Id", "Pawpularity"]]

NameError: name 'final_pred' is not defined
