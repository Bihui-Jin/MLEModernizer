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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.5005

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd

print(os.listdir("../input"))



## === cell 1
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalMaxPool2D,
    Dropout,
    Dense,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
output_dir = "../input/aerial-cactus-identification/model_output/CNN"
seed = 7
np.random.seed(seed)
tf.random.set_seed(seed)



## === cell 3
train_df = pd.read_csv("../input/train.csv")
train_df.head()



## === cell 4
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
print(class_weights)



## === cell 5
train_image = []
for idx in tqdm(range(len(train_df)), desc="Loading train images"):
    img_path = os.path.join("../input/train/train", train_df["id"][idx])
    img = load_img(img_path, target_size=(32, 32))
    img = img_to_array(img) / 255.0
    train_image.append(img)
X = np.array(train_image)



## === cell 6
X.shape



## === cell 7
plt.imshow(X[1])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3712634548.py in <cell line: 0>()
----> 1 plt.imshow(X[1])
      2 

NameError: name 'plt' is not defined

## === cell 8
y = np.array(train_df.drop(["id"], axis=1))
y.shape



## === cell 9
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



## === cell 10
X_train.shape, X_test.shape, y_train.shape, y_test.shape



## === cell 11
img_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    rotation_range=40,
    brightness_range=(0.5, 1.0),
    height_shift_range=0.2,
    width_shift_range=0.2,
)

test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow(X_test, y_test, batch_size=32)



## === cell 12
model = Sequential(
    [
        Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation="relu"),
        BatchNormalization(),
        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ]
)
model.summary()



## === cell 13
callbacks = [
    ModelCheckpoint(
        filepath="weights.best.hdf5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss", mode="auto", patience=20, restore_best_weights=True
    ),
    ReduceLROnPlateau(monitor="val_loss", mode="auto", patience=3, min_lr=1e-4),
]



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2298481924.py in <cell line: 0>()
      1 callbacks = [
----> 2     ModelCheckpoint(
      3         filepath="weights.best.hdf5",
      4         monitor="val_accuracy",
      5         save_best_only=True,

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=weights.best.hdf5

## === cell 14
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(
    img_gen.flow(X_train, y_train, batch_size=32),
    epochs=80,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight={0: class_weights[0], 1: class_weights[1]},
    verbose=2,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4111056326.py in <cell line: 0>()
      4     epochs=80,
      5     validation_data=validation_generator,
----> 6     callbacks=callbacks,
      7     class_weight={0: class_weights[0], 1: class_weights[1]},
      8     verbose=2,

NameError: name 'callbacks' is not defined

## === cell 15
model.load_weights("weights.best.hdf5")



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2061856298.py in <cell line: 0>()
----> 1 model.load_weights("weights.best.hdf5")
      2 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'weights.best.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 16
y_hat = model.predict(X_test).ravel()
get_auc = roc_auc_score(y_test, y_hat) * 100.0
print("AUC:", get_auc)



## === cell 17
pred = {}


def predictions(imagepath, imagename):
    img = load_img(imagepath, target_size=(32, 32, 3))
    img = img_to_array(img) / 255.0
    proba = model.predict(img.reshape(1, 32, 32, 3))[0][0]
    pred[imagename] = float(proba)




## === cell 18
test_dir = "../input/aerial-cactus-identification/test"
if not os.path.isdir(test_dir):
    test_dir = "../input/test"
files = os.listdir(test_dir)
for file in tqdm(files, desc="Predicting test set"):
    predictions(os.path.join(test_dir, file), file)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_55/636571447.py in <cell line: 0>()
      5 files = os.listdir(test_dir)
      6 for file in tqdm(files, desc="Predicting test set"):
----> 7     predictions(os.path.join(test_dir, file), file)
      8 

/tmp/ipykernel_55/3885386759.py in predictions(imagepath, imagename)
      3 
      4 def predictions(imagepath, imagename):
----> 5     img = load_img(imagepath, target_size=(32, 32, 3))
      6     img = img_to_array(img) / 255.0
      7     proba = model.predict(img.reshape(1, 32, 32, 3))[0][0]

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

IsADirectoryError: [Errno 21] Is a directory: '../input/aerial-cactus-identification/test/test'

## === cell 19
pred_df = pd.DataFrame(list(pred.items()), columns=["id", "has_cactus"])
print("Submission shape:", pred_df.shape)
pred_df.head()



## === cell 20
pred_df.to_csv("Submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
