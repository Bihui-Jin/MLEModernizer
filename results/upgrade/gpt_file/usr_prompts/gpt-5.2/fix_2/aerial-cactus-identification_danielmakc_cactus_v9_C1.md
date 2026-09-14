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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
joblib==1.5.2
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

0.9914

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

image_size = (32, 32)

BASE = Path("/kaggle/input/aerial-cactus-identification")
if not BASE.exists():
    BASE = Path("/kaggle/data/aerial-cactus-identification")
if not BASE.exists():
    BASE = Path("../input/aerial-cactus-identification")

TRAIN_DIR = BASE / "train"
TEST_DIR = BASE / "test"
TRAIN_CSV = BASE / "train.csv"
SAMPLE_SUB = BASE / "sample_submission.csv"

assert TRAIN_DIR.exists(), f"Train dir not found: {TRAIN_DIR}"
assert TEST_DIR.exists(), f"Test dir not found: {TEST_DIR}"
assert TRAIN_CSV.exists(), f"Train CSV not found: {TRAIN_CSV}"
assert SAMPLE_SUB.exists(), f"Sample submission not found: {SAMPLE_SUB}"

print("Using paths:")
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR :", TEST_DIR)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)



## === cell 1

print(
    "Skipping interactive image preview cell to avoid environment display/protobuf error."
)



## === cell 2
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from numpy import array
from tqdm import tqdm


def extract_features(label_path, set_path):
    images = []
    labels = []

    model = VGG19(include_top=False, input_shape=(image_size[0], image_size[1], 3))

    train_labels = pd.read_csv(label_path)
    label_map = dict(zip(train_labels["id"].values, train_labels["has_cactus"].values))

    file_names = [f for f in os.listdir(set_path) if (Path(set_path) / f).is_file()]

    for image_name in tqdm(
        file_names, desc=f"Extracting VGG19 features from {Path(set_path).name}"
    ):
        image = load_img(Path(set_path) / image_name, target_size=image_size)
        images.append(img_to_array(image))
        labels.append(label_map[image_name])

    training_images = preprocess_input(array(images))
    training_labels = array(labels)

    features = model.predict(training_images, verbose=0)

    return features, training_labels




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
from joblib import dump
from os import listdir

features, training_labels = extract_features(str(TRAIN_CSV), str(TRAIN_DIR))

dump(features, "features.dat")
dump(training_labels, "labels.dat")

print("Saved:", [p for p in listdir(".") if p.endswith(".dat")])



## === cell 4
from joblib import load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from matplotlib.pyplot import legend, plot, show, title, xlabel, ylabel

x_train = load("features.dat")
y_train = load("labels.dat")

model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
fit_model = model.fit(
    x_train, y_train, epochs=10, shuffle=True, validation_split=0.1, verbose=2
)

plot(fit_model.history.get("accuracy", []))
plot(fit_model.history.get("val_accuracy", []))
title("model accuracy")
ylabel("accuracy")
xlabel("epoch")
legend(["train", "val"], loc="upper left")
show()

plot(fit_model.history["loss"])
plot(fit_model.history["val_loss"])
title("model loss")
ylabel("loss")
xlabel("epoch")
legend(["train", "val"], loc="upper left")
show()

Path("model_structure.json").write_text(model.to_json())
model.save_weights("model_weights.h5")

print(
    "Saved model files:",
    [p for p in listdir(".") if p.endswith(".json") or p.endswith(".h5")],
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2494181877.py in <cell line: 0>()
     37 
     38 Path("model_structure.json").write_text(model.to_json())
---> 39 model.save_weights("model_weights.h5")
     40 
     41 print(

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in save_weights(model, filepath, overwrite, **kwargs)
    222 def save_weights(model, filepath, overwrite=True, **kwargs):
    223     if not str(filepath).endswith(".weights.h5"):
--> 224         raise ValueError(
    225             "The filename must end in `.weights.h5`. "
    226             f"Received: filepath={filepath}"

ValueError: The filename must end in `.weights.h5`. Received: filepath=model_weights.h5

## === cell 5
from csv import writer
from keras.models import model_from_json

model_structure = Path("model_structure.json").read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")

sub = pd.read_csv(SAMPLE_SUB)
test_ids = sub["id"].tolist()

images = []
for image_name in tqdm(test_ids, desc="Loading test images"):
    image_path = TEST_DIR / image_name
    image = load_img(image_path, target_size=image_size)
    images.append(img_to_array(image))

images_to_predict = preprocess_input(np.array(images))

feature_extractor = VGG19(
    include_top=False, input_shape=(image_size[0], image_size[1], 3)
)
features = feature_extractor.predict(images_to_predict, verbose=0)

predictions = model.predict(features, verbose=0).reshape(-1)

submission_path = Path("submission.csv")
with submission_path.open("w", newline="") as submissionCsvFile:
    csvWriter = writer(submissionCsvFile, lineterminator="\n")
    csvWriter.writerow(["id", "has_cactus"])
    for image_name, pred in zip(test_ids, predictions):
        csvWriter.writerow([image_name, float(pred)])

print("Wrote:", submission_path.resolve())
print(pd.read_csv(submission_path).head())
print("Rows:", len(pd.read_csv(submission_path)))
assert len(pd.read_csv(submission_path)) == len(
    sub
), "Submission row count mismatch vs sample_submission.csv"
assert list(pd.read_csv(submission_path).columns) == [
    "id",
    "has_cactus",
], "Submission columns mismatch"

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/641101777.py in <cell line: 0>()
      5 model_structure = Path("model_structure.json").read_text()
      6 model = model_from_json(model_structure)
----> 7 model.load_weights("model_weights.h5")
      8 
      9 # Load test IDs in the required submission order

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model_weights.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
