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

20.48685

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATA_DIR = "../input/petfinder-pawpularity-score"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_csv = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

tf.keras.utils.set_random_seed(42)
np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv



## === cell 2
train_csv.isnull().sum()



## === cell 3
train_csv = train_csv.drop_duplicates()
train_csv.shape



## === cell 4
for col in train_csv.drop(["Id", "Pawpularity"], axis=1).columns:
    sns.countplot(x=train_csv[col])
    plt.show()



## === cell 5
sns.histplot(train_csv["Pawpularity"], kde=True)
plt.show()



## === cell 6
test_csv



## === cell 7
test_csv.isnull().sum()



## === cell 8
submission



## === cell 9
rows = []
files = sorted([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")])
for file in files[:200]:
    path = os.path.join(TRAIN_DIR, file)
    imgg = cv2.imread(path)
    if imgg is None:
        continue
    h, w, c = imgg.shape
    rows.append([w, h, c, imgg.size / 3.0])

size_data = pd.DataFrame(rows, columns=["w", "h", "c", "pixels"])
size_data.head()



## === cell 10
if len(size_data) > 0:
    size_data[size_data["pixels"] == size_data["pixels"].min()]
else:
    size_data



## === cell 11
if len(size_data) > 0:
    size_data["pixels"].value_counts().head(10)
else:
    pd.Series(dtype=int)



## === cell 12
if len(size_data) > 0:
    size_data[size_data["pixels"] == 691200]
else:
    size_data




## === cell 13
def load_images_from_dir(img_dir, target_size=(64, 64)):
    imgs = []
    names = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        fpath = os.path.join(img_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32) / 255.0)
        names.append(fname)
    return np.array(imgs, dtype=np.float32), names


train_img, train_img_name = load_images_from_dir(TRAIN_DIR, target_size=(64, 64))
train_img.shape, len(train_img_name)



## === cell 14
train_img_name[:5]



## === cell 15
bad = [n for n in train_img_name if not n.lower().endswith(".jpg")]
bad[:5], len(bad)



## === cell 16
train_ids = [n[:-4] for n in train_img_name]
train_csv_indexed = train_csv.set_index("Id")
train_csv_data = train_csv_indexed.loc[train_ids].reset_index()
train_csv_data.head()



## === cell 17
img0 = cv2.imread(os.path.join(TRAIN_DIR, train_csv_data["Id"].iloc[0] + ".jpg"))
plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 18
plt.imshow(train_img[0][..., ::-1])
plt.axis("off")
plt.show()



## === cell 19
img1 = cv2.imread(os.path.join(TRAIN_DIR, train_csv_data["Id"].iloc[1] + ".jpg"))
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 20
plt.imshow(train_img[1][..., ::-1])
plt.axis("off")
plt.show()



## === cell 21
test_img, test_img_name = load_images_from_dir(TEST_DIR, target_size=(64, 64))
test_img.shape, len(test_img_name)



## === cell 22
test_img[:1].shape



## === cell 23
test_img_name[:5]



## === cell 24
test_ids = [n[:-4] for n in test_img_name]
test_csv_indexed = test_csv.set_index("Id")
test_csv_data = test_csv_indexed.loc[test_ids].reset_index()
test_csv_data.head()



## === cell 25
test0 = cv2.imread(os.path.join(TEST_DIR, test_csv_data["Id"].iloc[0] + ".jpg"))
plt.imshow(cv2.cvtColor(test0, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 26
plt.imshow(test_img[0][..., ::-1])
plt.axis("off")
plt.show()



## === cell 27
train_csv_x = train_csv_data.drop(["Id", "Pawpularity"], axis=1).astype(np.float32)
train_y = train_csv_data["Pawpularity"].astype(np.float32)

test_csv_x = test_csv_data.drop(["Id"], axis=1).astype(np.float32)

train_csv_x.shape, train_y.shape, test_csv_x.shape



## === cell 28
assert len(train_csv_x) == len(train_img), "Train CSV rows and images misaligned"
assert len(test_csv_x) == len(test_img), "Test CSV rows and images misaligned"
(len(train_img), len(test_img))



## === cell 29
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
    inputs=[csv_input, img_input],
    outputs=[csv_output, img_output],
    name="Pythonash_model",
)



## === cell 30
model.summary()



## === cell 31
try:
    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
    )
    print("Saved model plot to model.png")
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 32
learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002, decay_steps=10000, decay_rate=0.97
)

opt = tf.keras.optimizers.Adam(learning_rate=learning_schedule)

rmse = tf.keras.metrics.RootMeanSquaredError()
model.compile(
    loss=["mse", "mse"],
    loss_weights=[1, 1],
    optimizer=opt,
    metrics=[[rmse], [tf.keras.metrics.RootMeanSquaredError()]],
)

epoch_number = 20

check_1 = tf.keras.callbacks.ModelCheckpoint(
    "pythonash_model.h5", save_best_only=True, verbose=2
)



## === cell 33
history = model.fit(
    x=[train_csv_x, train_img],
    y=[train_y, train_y],
    epochs=epoch_number,
    validation_split=0.2,
    verbose=2,
    batch_size=100,
    callbacks=[check_1],
)



## === cell 34
if os.path.exists("pythonash_model.h5"):
    best_model = tf.keras.models.load_model("pythonash_model.h5")
else:
    best_model = model

csv_result, img_result = best_model.predict([test_csv_x, test_img], verbose=0)
final_pred = (0.5 * csv_result + 0.5 * img_result).reshape(-1)

final_pred = np.clip(final_pred, 1.0, 100.0)

final_result = pd.DataFrame({"Pawpularity": final_pred})
final_result.head()



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/291375609.py in <cell line: 0>()
      1 if os.path.exists("pythonash_model.h5"):
----> 2     best_model = tf.keras.models.load_model("pythonash_model.h5")
      3 else:
      4     best_model = model
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    153             # Compile model.
    154             model.compile(
--> 155                 **saving_utils.compile_args_from_training_config(
    156                     training_config, custom_objects
    157                 )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/saving_utils.py in compile_args_from_training_config(training_config, custom_objects)
    141         loss_config = training_config.get("loss", None)
    142         if loss_config is not None:
--> 143             loss = _deserialize_nested_config(losses.deserialize, loss_config)
    144             # Ensure backwards compatibility for losses in legacy H5 files
    145             loss = _resolve_compile_arguments_compat(loss, loss_config, losses)

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/saving_utils.py in _deserialize_nested_config(deserialize_fn, config)
    207         }
    208     elif isinstance(config, (tuple, list)):
--> 209         return [
    210             _deserialize_nested_config(deserialize_fn, obj) for obj in config
    211         ]

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/saving_utils.py in <listcomp>(.0)
    208     elif isinstance(config, (tuple, list)):
    209         return [
--> 210             _deserialize_nested_config(deserialize_fn, obj) for obj in config
    211         ]
    212 

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/saving_utils.py in _deserialize_nested_config(deserialize_fn, config)
    200         return None
    201     if _is_single_object(config):
--> 202         return deserialize_fn(config)
    203     elif isinstance(config, dict):
    204         return {

/usr/local/lib/python3.11/dist-packages/keras/src/losses/__init__.py in deserialize(name, custom_objects)
    153         A Keras `Loss` instance or a loss function.
    154     """
--> 155     return serialization_lib.deserialize_keras_object(
    156         name,
    157         module_objects=ALL_OBJECTS_DICT,

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    573                 return config
    574             if isinstance(module_objects[config], types.FunctionType):
--> 575                 return deserialize_keras_object(
    576                     serialize_with_public_fn(
    577                         module_objects[config], config, fn_module_name

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    676     if class_name == "function":
    677         fn_name = inner_config
--> 678         return _retrieve_class_or_fn(
    679             fn_name,
    680             registered_name,

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in _retrieve_class_or_fn(name, registered_name, module, obj_type, full_config, custom_objects)
    801             return obj
    802 
--> 803     raise TypeError(
    804         f"Could not locate {obj_type} '{name}'. "
    805         "Make sure custom classes are decorated with "

TypeError: Could not locate function 'mse'. Make sure custom classes are decorated with `@keras.saving.register_keras_serializable()`. Full object config: {'module': 'keras.metrics', 'class_name': 'function', 'config': 'mse', 'registered_name': 'mse'}

## === cell 35
sub_out = pd.DataFrame(
    {
        "Id": test_csv_data["Id"].values,
        "Pawpularity": final_result["Pawpularity"].values,
    }
)

assert list(sub_out.columns) == ["Id", "Pawpularity"]
assert len(sub_out) == len(
    submission
), "Submission row count mismatch vs sample_submission"
assert (
    sub_out["Pawpularity"].between(1.0, 100.0).all()
), "Pawpularity must be between 1 and 100"

sub_path = "submission.csv"
sub_out.to_csv(sub_path, index=False)
sub_out.head(), sub_path

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3391068795.py in <cell line: 0>()
      2     {
      3         "Id": test_csv_data["Id"].values,
----> 4         "Pawpularity": final_result["Pawpularity"].values,
      5     }
      6 )

NameError: name 'final_result' is not defined
