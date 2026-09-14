# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.8

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

0.9986

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!unzip /kaggle/input/aerial-cactus-identification/train.zip
!unzip /kaggle/input/aerial-cactus-identification/test.zip


## === cell 1
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.image as pimg
import seaborn as sns
import math
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from importlib.metadata import version as _pkg_version

    _pb_ver = _pkg_version("protobuf")
except Exception:
    _pb_ver = None


def _major(ver):
    try:
        return int(str(ver).split(".")[0])
    except Exception:
        return None


if _pb_ver is None or (_major(_pb_ver) is not None and _major(_pb_ver) >= 6):
    get_ipython().system("pip -q install 'protobuf==5.28.3'")

    import importlib

    for _m in list(sys.modules):
        if _m.startswith("google.protobuf") or _m == "protobuf":
            sys.modules.pop(_m, None)
    importlib.invalidate_caches()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Dropout,
    GlobalMaxPooling2D,
)
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler


## === cell 2
print(sys.version)
print('tensorflow -> ', tf.__version__)


## === cell 3

np.random.seed(12)
tf.random.set_seed(12)


## === cell 4
main_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
sub_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')

train_dir = '/kaggle/working/train/'
test_dir = '/kaggle/working/test/'


## === cell 5
main_df.head()


## === cell 6
print('shape: ', main_df.shape)
print('===================================')
print(main_df['has_cactus'].value_counts())


## === cell 7
plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('Pastel2')

x = ['has cactus', 'hasn\'t cactus']
y = main_df.groupby('has_cactus').size()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.pie(y, labels=x, autopct="%1.1f%%")

plt.show()


## === cell 8
_train_dir_candidates = [
    train_dir,  # as set in cell 4
    "/kaggle/working/aerial-cactus-identification/train/",
    "/kaggle/working/train/",
    "/kaggle/input/aerial-cactus-identification/train/",
]
for _cand in _train_dir_candidates:
    if os.path.isdir(_cand):
        _resolved_train_dir = _cand
        break
else:
    raise FileNotFoundError(
        f"Could not find a valid training image directory. Tried: {_train_dir_candidates}"
    )

fig, ax = plt.subplots(2, 5, figsize=(12, 6))

for i, idx in enumerate(main_df[main_df["has_cactus"] == 1]["id"][-5:]):
    path = os.path.join(_resolved_train_dir, idx)
    img = load_img(path)
    ax[0, i].axis("off")
    ax[0, i].set_title("has cactus")
    ax[0, i].imshow(img)

for i, idx in enumerate(main_df[main_df["has_cactus"] == 0]["id"][-5:]):
    path = os.path.join(_resolved_train_dir, idx)
    img = load_img(path)
    ax[1, i].axis("off")
    ax[1, i].set_title("hasn't cactus")
    ax[1, i].imshow(img)


## === cell 9
train_df, val_df = train_test_split(main_df, test_size=0.25, stratify=main_df['has_cactus'], shuffle=True, random_state=12)

train_df = train_df.reset_index()
val_df = val_df.reset_index()

total_train = train_df.shape[0]
total_val = val_df.shape[0]

print('total_train: {}, total_val: {}'.format(total_train, total_val))


## === cell 10
img_width, img_height = 32, 32
target_size = (img_width, img_height)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_df['has_cactus'] = train_df['has_cactus'].astype(str)
val_df['has_cactus'] = val_df['has_cactus'].astype(str)


## === cell 11
batch_size = 32
x_col, y_col = 'id', 'has_cactus'
class_mode = 'binary'


train_gen = train_datagen.flow_from_dataframe(train_df,
                                            train_dir,
                                            x_col=x_col,
                                            y_col=y_col,
                                            class_mode=class_mode,
                                            target_size=target_size,
                                            batch_size=batch_size,
                                            )

val_gen = val_datagen.flow_from_dataframe(val_df,
                                        train_dir,
                                        x_col=x_col,
                                        y_col=y_col,
                                        class_mode=class_mode,
                                        target_size=target_size,
                                        batch_size=batch_size,
                                        )


## === cell 12
input_shape = (img_width, img_height, 3)

optimizer = optimizers.Adam(learning_rate=1e-3)


## === cell 13
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(GlobalMaxPooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=optimizer)
model.summary()


## === cell 14
def step_decay(epoch):
    initial_rate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_rate * math.pow(drop, math.floor((epoch) / epochs_drop))
    
    return lrate


## === cell 15
lrate = LearningRateScheduler(step_decay)
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

callbacks = [lrate, es]


## === cell 16

_train_dir_candidates = [
    train_dir,  # as set in cell 4
    "/kaggle/working/aerial-cactus-identification/train/",
    "/kaggle/working/train/",
    "/kaggle/input/aerial-cactus-identification/train/",
]
for _cand in _train_dir_candidates:
    if os.path.isdir(_cand):
        _resolved_train_dir = _cand
        break
else:
    raise FileNotFoundError(
        f"Could not find a valid training image directory. Tried: {_train_dir_candidates}"
    )

if (
    (not hasattr(train_gen, "directory"))
    or (not os.path.isdir(getattr(train_gen, "directory", "")))
    or len(train_gen) == 0
):
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        _resolved_train_dir,
        x_col=x_col,
        y_col=y_col,
        class_mode=class_mode,
        target_size=target_size,
        batch_size=batch_size,
    )

if (
    (not hasattr(val_gen, "directory"))
    or (not os.path.isdir(getattr(val_gen, "directory", "")))
    or len(val_gen) == 0
):
    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        _resolved_train_dir,
        x_col=x_col,
        y_col=y_col,
        class_mode=class_mode,
        target_size=target_size,
        batch_size=batch_size,
    )

epochs = 30

steps_per_epoch = max(1, len(train_gen))
validation_steps = max(1, len(val_gen))

history = model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
)


## === cell 17
sns.set_palette('Dark2')
fig,ax = plt.subplots(2, 1)

plot_acc = pd.DataFrame({'acc': history.history['acc'],
                         'val_acc': history.history['val_acc']})

plot_loss = pd.DataFrame({'loss': history.history['loss'],
                          'val_loss': history.history['val_loss']})

plot_acc.plot(ax=ax[0])
plot_loss.plot(ax=ax[1])


## === cell 18
def predict(model, sub_df):
    pred = np.empty((sub_df.shape[0],))
    
    for n in tqdm(range(sub_df.shape[0])):
        image = np.array(Image.open(test_dir + sub_df.id[n]))
        pred[n] = model.predict(image.reshape((1, 32, 32, 3))/255.0)[0]
    
    sub_df['has_cactus'] = pred
    return sub_df


## === cell 19
def predict(model, sub_df):
    _test_dir_candidates = [
        test_dir,  # as set in cell 4
        "/kaggle/working/aerial-cactus-identification/test/",
        "/kaggle/working/test/",
        "/kaggle/input/aerial-cactus-identification/test/",
    ]
    for _cand in _test_dir_candidates:
        if os.path.isdir(_cand):
            _resolved_test_dir = _cand
            break
    else:
        raise FileNotFoundError(
            f"Could not find a valid test image directory. Tried: {_test_dir_candidates}"
        )

    pred = np.empty((sub_df.shape[0],))

    for n in tqdm(range(sub_df.shape[0])):
        img_path = os.path.join(_resolved_test_dir, sub_df.id[n])
        image = np.array(Image.open(img_path))
        pred[n] = model.predict(image.reshape((1, 32, 32, 3)) / 255.0, verbose=0)[0]

    sub_df["has_cactus"] = pred
    return sub_df


## === cell 20

!rm -r *


## === cell 21
pass
