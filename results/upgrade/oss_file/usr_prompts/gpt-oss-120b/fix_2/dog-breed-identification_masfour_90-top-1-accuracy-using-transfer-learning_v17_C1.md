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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.28106

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, shutil
import numpy as np, pandas as pd
import cv2, matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "../input/dog-breed-identification"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
SAMPLE_SUBMIT = os.path.join(BASE_DIR, "sample_submission.csv")
LABELS_CSV = os.path.join(BASE_DIR, "labels.csv")



## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels["filename"] = labels["id"] + ".jpg"
classes = labels["breed"].unique()
num_classes = len(classes)
print("Num classes:", num_classes)



## === cell 3
train_df, val_df = train_test_split(
    labels, test_size=0.2, stratify=labels["breed"], random_state=42
)



## === cell 4
TMP_DIR = "/root/tmp_dog_data"
TMP_TRAIN = os.path.join(TMP_DIR, "train")
TMP_VAL = os.path.join(TMP_DIR, "val")
TMP_TEST = os.path.join(TMP_DIR, "test")
for d in [TMP_TRAIN, TMP_VAL, TMP_TEST]:
    os.makedirs(d, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(d, cls), exist_ok=True)

for _, row in train_df.iterrows():
    src = os.path.join(TRAIN_DIR, row["filename"])
    dst = os.path.join(TMP_TRAIN, row["breed"], row["filename"])
    shutil.copyfile(src, dst)

for _, row in val_df.iterrows():
    src = os.path.join(TRAIN_DIR, row["filename"])
    dst = os.path.join(TMP_VAL, row["breed"], row["filename"])
    shutil.copyfile(src, dst)

TEST_SUBDIR = os.path.join(TMP_TEST, "test")
os.makedirs(TEST_SUBDIR, exist_ok=True)
test_filenames = sorted(os.listdir(TEST_DIR))
for fname in test_filenames:
    shutil.copyfile(os.path.join(TEST_DIR, fname), os.path.join(TEST_SUBDIR, fname))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_55/1016511183.py in <cell line: 0>()
     26 test_filenames = sorted(os.listdir(TEST_DIR))
     27 for fname in test_filenames:
---> 28     shutil.copyfile(os.path.join(TEST_DIR, fname), os.path.join(TEST_SUBDIR, fname))
     29 

/usr/lib/python3.11/shutil.py in copyfile(src, dst, follow_symlinks)
    254         os.symlink(os.readlink(src), dst)
    255     else:
--> 256         with open(src, 'rb') as fsrc:
    257             try:
    258                 with open(dst, 'wb') as fdst:

IsADirectoryError: [Errno 21] Is a directory: '../input/dog-breed-identification/test/test'

## === cell 5
img_height, img_width = 128, 128
batch_size = 32
norm_factor = 1 / 255.0

train_datagen = ImageDataGenerator(
    rescale=norm_factor,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=norm_factor)

train_gen = train_datagen.flow_from_directory(
    TMP_TRAIN,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    TMP_VAL,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)




## === cell 6
class Plotter(Callback):
    def on_train_begin(self, logs=None):
        self.losses, self.val_losses, self.acc, self.val_acc, self.epochs = (
            [],
            [],
            [],
            [],
            [],
        )

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))
        self.epochs.append(epoch)
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(self.epochs, self.losses, label="train loss")
        ax1.plot(self.epochs, self.val_losses, label="val loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.legend()
        ax2.plot(self.epochs, self.acc, label="train acc")
        ax2.plot(self.epochs, self.val_acc, label="val acc")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("accuracy")
        ax2.legend()
        plt.show()


plotter = Plotter()
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)




## === cell 7
def build_model(input_shape, n_classes):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=output)
    return model


model = build_model((img_height, img_width, 3), num_classes)
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 8
epochs = 8
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[plotter, reduce_lr, early_stop],
    verbose=0,
)



## === cell 9
test_datagen = ImageDataGenerator(rescale=norm_factor)
test_gen = test_datagen.flow_from_directory(
    TMP_TEST,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode=None,
    shuffle=False,
)

preds = model.predict(test_gen, steps=len(test_filenames), verbose=0)
pred_df = pd.DataFrame(preds, columns=sorted(classes))
pred_df.insert(0, "id", [fname[:-4] for fname in test_filenames])



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3826982290.py in <cell line: 0>()
     12 # ensure probabilities sum to 1 (softmax already does)
     13 pred_df = pd.DataFrame(preds, columns=sorted(classes))
---> 14 pred_df.insert(0, "id", [fname[:-4] for fname in test_filenames])
     15 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in insert(self, loc, column, value, allow_duplicates)
   5169             value = value.iloc[:, 0]
   5170 
-> 5171         value, refs = self._sanitize_column(value)
   5172         self._mgr.insert(loc, column, value, refs=refs)
   5173 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (1024) does not match length of index (1023)

## === cell 10
sample_sub = pd.read_csv(SAMPLE_SUBMIT, nrows=1)
ordered_cols = ["id"] + [c for c in sample_sub.columns if c != "id"]
submission = pred_df[ordered_cols]
submission_path = "/root/submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission saved to", submission_path)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1598157069.py in <cell line: 0>()
      2 sample_sub = pd.read_csv(SAMPLE_SUBMIT, nrows=1)
      3 ordered_cols = ["id"] + [c for c in sample_sub.columns if c != "id"]
----> 4 submission = pred_df[ordered_cols]
      5 submission_path = "/root/submission.csv"
      6 submission.to_csv(submission_path, index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['id'] not in index"
