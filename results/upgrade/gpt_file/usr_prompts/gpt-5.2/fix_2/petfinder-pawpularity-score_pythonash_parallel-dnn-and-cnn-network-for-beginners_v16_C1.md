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
import os, sys, subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as pb_ver

    major = int(pb_ver.split(".")[0])
    if major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib
        import google.protobuf as gp

        importlib.reload(gp)
except Exception as e:
    print("protobuf compatibility step warning:", repr(e))

import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATA_DIR = "/kaggle/input/petfinder-pawpularity-score"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
TEST_CSV_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train"
TEST_IMG_DIR = f"{DATA_DIR}/test"

train_csv = pd.read_csv(TRAIN_CSV_PATH)
test_csv = pd.read_csv(TEST_CSV_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

print(train_csv.shape, test_csv.shape, submission.shape)



## === cell 1
train_csv.head()



## === cell 2
train_csv.isnull().sum()



## === cell 3
train_csv = train_csv.drop_duplicates()
train_csv.shape



## === cell 4
for col in train_csv.drop(["Id", "Pawpularity"], axis=1).columns:
    sns.countplot(data=train_csv, x=col)
    plt.show()



## === cell 5
sns.histplot(train_csv["Pawpularity"], kde=True)
plt.show()



## === cell 6
test_csv.head()



## === cell 7
test_csv.isnull().sum()



## === cell 8
submission.head()



## === cell 9
jpg_files = [f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith(".jpg")]
size_rows = []
for f in jpg_files[
    :200
]:  # limit to keep this diagnostic fast; does not affect modeling core logic
    img = cv2.imread(os.path.join(TRAIN_IMG_DIR, f))
    if img is None:
        continue
    h, w, c = img.shape
    size_rows.append({"h": h, "w": w, "c": c, "pixels": img.size / 3.0})
size_data = pd.DataFrame(size_rows)
size_data.head(), size_data.describe(include="all")



## === cell 10
if len(size_data) > 0:
    size_data[size_data["pixels"] == size_data["pixels"].min()].head()
else:
    size_data



## === cell 11
if len(size_data) > 0:
    size_data["pixels"].value_counts().head()
else:
    size_data



## === cell 12
if len(size_data) > 0:
    size_data[size_data["pixels"] == 691200].head()
else:
    size_data



## === cell 13
IMG_SIZE = (64, 64)

train_img = []
train_img_name = []

for fname in sorted(os.listdir(TRAIN_IMG_DIR)):
    if not fname.lower().endswith(".jpg"):
        continue
    fpath = os.path.join(TRAIN_IMG_DIR, fname)
    img = cv2.imread(fpath)
    if img is None:
        continue
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    train_img.append(img.astype(np.float32) / 255.0)
    train_img_name.append(fname)

len(train_img), train_img_name[:5]



## === cell 14
bad = [n for n in train_img_name if not n.lower().endswith(".jpg")]
bad[:10], len(bad)



## === cell 15
train_id_to_row = train_csv.set_index("Id")
train_ids = [n[:-4] for n in train_img_name if n.lower().endswith(".jpg")]

missing = [i for i in train_ids if i not in train_id_to_row.index]
print("Missing train ids in CSV:", len(missing))

train_ids_kept = [i for i in train_ids if i in train_id_to_row.index]
if len(train_ids_kept) != len(train_ids):
    keep_mask = [i in set(train_ids_kept) for i in train_ids]
    train_img = [im for im, m in zip(train_img, keep_mask) if m]
    train_img_name = [nm for nm, m in zip(train_img_name, keep_mask) if m]

train_csv_data = train_id_to_row.loc[train_ids_kept].reset_index()
train_csv_data.head()



## === cell 16
train_csv_data.shape



## === cell 17
image_1 = cv2.imread(os.path.join(TRAIN_IMG_DIR, train_csv_data["Id"].iloc[0] + ".jpg"))
plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 18
plt.imshow(train_img[0])
plt.axis("off")
plt.show()



## === cell 19
image_2 = cv2.imread(os.path.join(TRAIN_IMG_DIR, train_csv_data["Id"].iloc[1] + ".jpg"))
plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 20
plt.imshow(train_img[1])
plt.axis("off")
plt.show()



## === cell 21
test_img = []
test_img_name = []

for fname in sorted(os.listdir(TEST_IMG_DIR)):
    if not fname.lower().endswith(".jpg"):
        continue
    fpath = os.path.join(TEST_IMG_DIR, fname)
    img = cv2.imread(fpath)
    if img is None:
        continue
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    test_img.append(img.astype(np.float32) / 255.0)
    test_img_name.append(fname)

len(test_img), test_img_name[:5]



## === cell 22
test_id_to_row = test_csv.set_index("Id")
test_ids = [n[:-4] for n in test_img_name]

missing_t = [i for i in test_ids if i not in test_id_to_row.index]
print("Missing test ids in CSV:", len(missing_t))

test_ids_kept = [i for i in test_ids if i in test_id_to_row.index]
if len(test_ids_kept) != len(test_ids):
    keep_mask = [i in set(test_ids_kept) for i in test_ids]
    test_img = [im for im, m in zip(test_img, keep_mask) if m]
    test_img_name = [nm for nm, m in zip(test_img_name, keep_mask) if m]

test_csv_data = test_id_to_row.loc[test_ids_kept].reset_index()
test_csv_data.head()



## === cell 23
test_1 = cv2.imread(os.path.join(TEST_IMG_DIR, test_csv_data["Id"].iloc[0] + ".jpg"))
plt.imshow(cv2.cvtColor(test_1, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 24
plt.imshow(test_img[0])
plt.axis("off")
plt.show()



## === cell 25
train_csv_x = train_csv_data.drop(["Id", "Pawpularity"], axis=1)
train_y = train_csv_data["Pawpularity"].astype(np.float32)

test_csv_x = test_csv_data.drop(["Id"], axis=1)

train_csv_x = train_csv_x.astype(np.float32)
test_csv_x = test_csv_x.astype(np.float32)

X_img_train = np.asarray(train_img, dtype=np.float32)
X_img_test = np.asarray(test_img, dtype=np.float32)

train_csv_x.shape, X_img_train.shape, test_csv_x.shape, X_img_test.shape



## === cell 26
csv_input = tf.keras.Input(shape=train_csv_x.shape[1:], name="CSV_Input")
img_input = tf.keras.Input(shape=X_img_train.shape[1:], name="IMG_Input")

csv_hidden1 = tf.keras.layers.Dense(200, activation="relu", name="CSV_Hidden1")(
    csv_input
)
csv_hidden2 = tf.keras.layers.Dense(200, activation="relu", name="CSV_Hidden2")(
    csv_hidden1
)
csv_hidden3 = tf.keras.layers.Dense(200, activation="relu", name="CSV_Hidden3")(
    csv_hidden2
)
csv_dropout = tf.keras.layers.Dropout(0.5, name="CSV_Dropout")(csv_hidden3)
csv_hidden4 = tf.keras.layers.Dense(200, activation="relu", name="CSV_Hidden4")(
    csv_dropout
)
csv_hidden5 = tf.keras.layers.Dense(200, activation="relu", name="CSV_Hidden5")(
    csv_hidden4
)

img_conv1 = tf.keras.layers.Conv2D(
    filters=120,
    kernel_size=5,
    strides=1,
    padding="same",
    activation="relu",
    name="IMG_Conv1",
)(img_input)
img_pool1 = tf.keras.layers.MaxPool2D(3, name="IMG_Pool1")(img_conv1)
img_conv2 = tf.keras.layers.Conv2D(
    filters=120,
    kernel_size=4,
    strides=1,
    padding="same",
    activation="relu",
    name="IMG_Conv2",
)(img_pool1)
img_conv3 = tf.keras.layers.Conv2D(
    filters=120,
    kernel_size=4,
    strides=1,
    padding="same",
    activation="relu",
    name="IMG_Conv3",
)(img_conv2)
img_pool2 = tf.keras.layers.MaxPool2D(3, name="IMG_Pool2")(img_conv3)
img_conv4 = tf.keras.layers.Conv2D(
    filters=120,
    kernel_size=3,
    strides=1,
    padding="same",
    activation="relu",
    name="IMG_Conv4",
)(img_pool2)
img_pool3 = tf.keras.layers.MaxPool2D(3, name="IMG_Pool3")(img_conv4)
img_conv5 = tf.keras.layers.Conv2D(
    filters=120,
    kernel_size=3,
    strides=1,
    padding="same",
    activation="relu",
    name="IMG_Conv5",
)(img_pool3)
img_conv6 = tf.keras.layers.Conv2D(
    filters=120,
    kernel_size=3,
    strides=1,
    padding="same",
    activation="relu",
    name="IMG_Conv6",
)(img_conv5)
img_pool4 = tf.keras.layers.MaxPool2D(2, name="IMG_Pool4")(img_conv6)
img_flatten = tf.keras.layers.Flatten(name="IMG_Flatten")(img_pool4)
img_dense1 = tf.keras.layers.Dense(300, activation="relu", name="IMG_Dense1")(
    img_flatten
)
img_dropout1 = tf.keras.layers.Dropout(0.5, name="IMG_Dropout1")(img_dense1)
img_dense2 = tf.keras.layers.Dense(300, activation="relu", name="IMG_Dense2")(
    img_dropout1
)
img_dropout2 = tf.keras.layers.Dropout(0.5, name="IMG_Dropout2")(img_dense2)

csv_output = tf.keras.layers.Dense(1, name="CSV_Output")(csv_hidden5)
img_output = tf.keras.layers.Dense(1, name="IMG_Output")(img_dropout2)

model = tf.keras.Model(
    inputs=[csv_input, img_input],
    outputs=[csv_output, img_output],
    name="Pythonash_model",
)

model.summary()



## === cell 27
learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002, decay_steps=10000, decay_rate=0.9
)
opt = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
model.compile(
    loss=["mse", "mse"],
    loss_weights=[0.5, 0.5],
    optimizer=opt,
    metrics=tf.keras.metrics.RootMeanSquaredError(),
)

epoch_number = 100

check_1 = tf.keras.callbacks.ModelCheckpoint(
    "pythonash_model.keras", save_best_only=True, verbose=2
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_12/1479103130.py in <cell line: 0>()
      4 )
      5 opt = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
----> 6 model.compile(
      7     loss=["mse", "mse"],
      8     loss_weights=[0.5, 0.5],

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

## === cell 28
history = model.fit(
    x=[train_csv_x, X_img_train],
    y=[train_y, train_y],
    epochs=epoch_number,
    validation_split=0.2,
    verbose=2,
    batch_size=100,
    validation_batch_size=100,
    callbacks=[check_1],
)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/4217937224.py in <cell line: 0>()
      3     x=[train_csv_x, X_img_train],
      4     y=[train_y, train_y],
----> 5     epochs=epoch_number,
      6     validation_split=0.2,
      7     verbose=2,

NameError: name 'epoch_number' is not defined

## === cell 29
best_model = tf.keras.models.load_model("pythonash_model.keras", compile=False)
csv_result, img_result = best_model.predict([test_csv_x, X_img_test], verbose=0)

final_pred = 0.5 * csv_result + 0.5 * img_result
final_pred = final_pred.reshape(-1)

final_pred = np.clip(final_pred, 0.0, 100.0)

final_result = pd.DataFrame({"Pawpularity": final_pred.astype(np.float32)})
final_result.head(), final_result.shape



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_12/1539440760.py in <cell line: 0>()
      1 # Predict using best saved model
----> 2 best_model = tf.keras.models.load_model("pythonash_model.keras", compile=False)
      3 csv_result, img_result = best_model.predict([test_csv_x, X_img_test], verbose=0)
      4 
      5 final_pred = 0.5 * csv_result + 0.5 * img_result

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=pythonash_model.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 30
sub = submission.copy()
pred_df = pd.DataFrame(
    {
        "Id": test_csv_data["Id"].values,
        "Pawpularity": final_result["Pawpularity"].values,
    }
)

sub = sub.drop(columns=["Pawpularity"]).merge(pred_df, on="Id", how="left")

if sub["Pawpularity"].isna().any():
    sub["Pawpularity"] = sub["Pawpularity"].fillna(sub["Pawpularity"].mean())

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
sub.head()

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3484867460.py in <cell line: 0>()
      4     {
      5         "Id": test_csv_data["Id"].values,
----> 6         "Pawpularity": final_result["Pawpularity"].values,
      7     }
      8 )

NameError: name 'final_result' is not defined
