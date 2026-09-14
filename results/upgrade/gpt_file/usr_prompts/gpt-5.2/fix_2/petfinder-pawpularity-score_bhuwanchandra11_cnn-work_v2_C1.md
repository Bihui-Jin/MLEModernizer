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

20.49045

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
import cv2



## === cell 2
INPUT_DIR = "/kaggle/input/petfinder-pawpularity-score"
TRAIN_CSV_PATH = f"{INPUT_DIR}/train.csv"
TEST_CSV_PATH = f"{INPUT_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{INPUT_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{INPUT_DIR}/train"
TEST_IMG_DIR = f"{INPUT_DIR}/test"
WORKING_DIR = "/kaggle/working"

df = pd.read_csv(TRAIN_CSV_PATH)
test_csv = pd.read_csv(TEST_CSV_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

df.shape, test_csv.shape, submission.shape



## === cell 3
df.head()



## === cell 4
df.info()



## === cell 5
sns.histplot(df["Pawpularity"], kde=True)
plt.show()




## === cell 6
def list_jpg_files(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    files.sort()
    return files


def load_and_resize_images(folder, img_ids, size=(64, 64)):
    """
    img_ids: iterable of Id strings (without .jpg)
    Returns: float32 array [N, H, W, 3] scaled to [0,1]
    """
    imgs = []
    missing = 0
    for _id in img_ids:
        path = os.path.join(folder, f"{_id}.jpg")
        img = cv2.imread(path)
        if img is None:
            missing += 1
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32) / 255.0)
    if missing:
        print(
            f"Warning: {missing} images could not be read and were replaced with zeros."
        )
    return np.stack(imgs, axis=0)




## === cell 7
train_files = list_jpg_files(TRAIN_IMG_DIR)[
    :200
]  # limit for quick inspection only (score-neutral)
rows = []
for f in train_files:
    img = cv2.imread(os.path.join(TRAIN_IMG_DIR, f))
    if img is None:
        continue
    h, w, c = img.shape
    rows.append((h, w, c, img.size / 3))
size_data = pd.DataFrame(rows, columns=["h", "w", "c", "pixels"])
size_data.head()



## === cell 8
size_data["pixels"].value_counts().head()



## === cell 9
train_ids = df["Id"].tolist()
train_img = load_and_resize_images(TRAIN_IMG_DIR, train_ids, size=(64, 64))
train_img.shape



## === cell 10
train_img_name = [f"{_id}.jpg" for _id in train_ids]
train_img_name[:5]



## === cell 11
for name in train_img_name[:5]:
    if name[:-4] == ".jpg":
        print(name)



## === cell 12
import tensorflow as tf

tf.__version__



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 13
train_csv_data = df.copy()
train_csv_data.head()



## === cell 14
train_csv_data = train_csv_data.reset_index(drop=True)
train_csv_data.head()



## === cell 15
image_1 = cv2.imread(os.path.join(TRAIN_IMG_DIR, train_csv_data.loc[0, "Id"] + ".jpg"))
if image_1 is not None:
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()



## === cell 16
plt.imshow(cv2.cvtColor((train_img[0] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 17
test_files = list_jpg_files(TEST_IMG_DIR)[:5]
for f in test_files:
    img = cv2.imread(os.path.join(TEST_IMG_DIR, f))
    print(f, None if img is None else img.shape)



## === cell 18
test_ids = test_csv["Id"].tolist()
test_img = load_and_resize_images(TEST_IMG_DIR, test_ids, size=(64, 64))
test_img.shape



## === cell 19
test_img_name = [f"{_id}.jpg" for _id in test_ids]
test_img_name[:1]



## === cell 20
test_csv_data = test_csv.copy().reset_index(drop=True)
test_csv_data.head(1)



## === cell 21
plt.imshow(cv2.cvtColor((test_img[0] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 22
train_csv_x = train_csv_data.drop(["Id", "Pawpularity"], axis=1)
train_y = train_csv_data["Pawpularity"].astype(np.float32)

test_csv_x = test_csv_data.drop(["Id"], axis=1)

train_csv_x.shape, test_csv_x.shape, train_y.shape



## === cell 23
train_csv_x.shape



## === cell 24
csv_input = tf.keras.Input(shape=train_csv_x.shape[1:], name="CSV_Input")
img_input = tf.keras.Input(shape=train_img.shape[1:], name="IMG_input")

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

img_hidden1 = tf.keras.layers.Dense(
    300,
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_hidden1",
    use_bias=False,
)(img_conv7)
img_dropout1 = tf.keras.layers.Dropout(0.5, name="IMG_Dropout1")(img_hidden1)
img_hidden2 = tf.keras.layers.Dense(
    300,
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_hidden2",
    use_bias=False,
)(img_dropout1)

img_gpool = tf.keras.layers.GlobalAvgPool2D(name="IMG_Gpool")(img_hidden2)
img_dropout2 = tf.keras.layers.Dropout(0.5, name="IMG_Dropout2")(img_gpool)

csv_output = tf.keras.layers.Dense(1, name="CSV_Output")(csv_dropout)
img_output = tf.keras.layers.Dense(1, name="IMG_Output")(img_dropout2)

model = tf.keras.Model(
    inputs=[csv_input, img_input], outputs=[csv_output, img_output], name="MY_WORK"
)



## === cell 25
model.summary()



## === cell 26
try:
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(WORKING_DIR, "model.png"),
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
    )
    print("Saved model diagram to /kaggle/working/model.png")
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 27
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

epoch_number = 20

check_1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(WORKING_DIR, "MY_WORK.keras"), save_best_only=True, verbose=2
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/182307449.py in <cell line: 0>()
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

## === cell 28
history = model.fit(
    x=[train_csv_x.to_numpy(dtype=np.float32), train_img],
    y=[train_y.to_numpy(), train_y.to_numpy()],
    epochs=epoch_number,
    validation_split=0.2,
    verbose=2,
    batch_size=100,
    callbacks=[check_1],
)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2030012309.py in <cell line: 0>()
      2     x=[train_csv_x.to_numpy(dtype=np.float32), train_img],
      3     y=[train_y.to_numpy(), train_y.to_numpy()],
----> 4     epochs=epoch_number,
      5     validation_split=0.2,
      6     verbose=2,

NameError: name 'epoch_number' is not defined

## === cell 29
best_model = tf.keras.models.load_model(os.path.join(WORKING_DIR, "MY_WORK.keras"))
csv_result, img_result = best_model.predict(
    [test_csv_x.to_numpy(dtype=np.float32), test_img], batch_size=100, verbose=1
)

final_pred = 0.5 * csv_result.reshape(-1) + 0.5 * img_result.reshape(-1)

final_pred = np.clip(final_pred, 0.0, 100.0)

final_result = pd.DataFrame({"Pawpularity": final_pred})
final_result.head()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2196747664.py in <cell line: 0>()
----> 1 best_model = tf.keras.models.load_model(os.path.join(WORKING_DIR, "MY_WORK.keras"))
      2 csv_result, img_result = best_model.predict(
      3     [test_csv_x.to_numpy(dtype=np.float32), test_img], batch_size=100, verbose=1
      4 )
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/kaggle/working/MY_WORK.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 30
out = pd.DataFrame(
    {
        "Id": test_csv_data["Id"].values,
        "Pawpularity": final_result["Pawpularity"].values,
    }
)
out_path = os.path.join(WORKING_DIR, "submission.csv")
out.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(out.head())
print("Shape:", out.shape)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3152480934.py in <cell line: 0>()
      3     {
      4         "Id": test_csv_data["Id"].values,
----> 5         "Pawpularity": final_result["Pawpularity"].values,
      6     }
      7 )

NameError: name 'final_result' is not defined
