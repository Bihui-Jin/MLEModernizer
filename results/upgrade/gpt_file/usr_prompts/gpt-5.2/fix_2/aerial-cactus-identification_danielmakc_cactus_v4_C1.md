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

0.9788

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path

image_size = (32, 32)

CANDIDATE_ROOTS = [
    Path("/kaggle/input/aerial-cactus-identification"),
    Path("/kaggle/data/aerial-cactus-identification"),
    Path("../input/aerial-cactus-identification"),
    Path("../input"),
    Path("/kaggle/input"),
]

DATA_ROOT = None
for p in CANDIDATE_ROOTS:
    if (p / "train.csv").exists() and (p / "train").exists() and (p / "test").exists():
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate dataset root. Looked for train.csv/train/test under: "
        + ", ".join(str(p) for p in CANDIDATE_ROOTS)
    )

TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"


def list_image_files(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [
        p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts
    ]
    files.sort()
    return files


print("Using DATA_ROOT:", DATA_ROOT)
print("Train images:", len(list_image_files(TRAIN_DIR)))
print("Test images:", len(list_image_files(TEST_DIR)))
print(
    "Train CSV exists:",
    TRAIN_CSV.exists(),
    "| Sample submission exists:",
    SAMPLE_SUB.exists(),
)



## === cell 1

import numpy as np
import pandas as pd
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.utils import load_img, img_to_array
from tqdm import tqdm


def extract_features(label_path, set_path):
    images = []
    labels = []

    model = VGG19(include_top=False, input_shape=(image_size[0], image_size[1], 3))

    train_labels = pd.read_csv(label_path)
    label_map = dict(
        zip(train_labels["id"].astype(str), train_labels["has_cactus"].astype(int))
    )

    image_names = list_image_files(Path(set_path))
    for image_name in tqdm(image_names, desc=f"Loading+labeling from {set_path}"):
        image = load_img(str(Path(set_path) / image_name), target_size=image_size)
        images.append(img_to_array(image))
        labels.append(label_map[image_name])

    training_images = preprocess_input(np.array(images, dtype=np.float32))
    training_labels = np.array(labels, dtype=np.float32)

    features = model.predict(training_images, verbose=0)

    return features, training_labels




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
from joblib import dump

features, training_labels = extract_features(TRAIN_CSV, TRAIN_DIR)

dump(features, "features.dat")
dump(training_labels, "labels.dat")

print("Saved:", Path("features.dat").resolve(), Path("labels.dat").resolve())
print("features shape:", features.shape, "labels shape:", training_labels.shape)



## === cell 3
from joblib import load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path

x_train = load("features.dat")
y_train = load("labels.dat")

model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=100, shuffle=True, validation_split=0.2, verbose=2)

Path("model_structure.json").write_text(model.to_json())
model.save_weights("model_weights.h5")

print("Saved model_structure.json and model_weights.h5")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/592594734.py in <cell line: 0>()
     17 
     18 Path("model_structure.json").write_text(model.to_json())
---> 19 model.save_weights("model_weights.h5")
     20 
     21 print("Saved model_structure.json and model_weights.h5")

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

## === cell 4
from csv import writer
from keras.models import model_from_json

model_structure = Path("model_structure.json").read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")

test_image_names = list_image_files(TEST_DIR)

images = []
for image_name in tqdm(test_image_names, desc="Loading test images"):
    image = load_img(str(TEST_DIR / image_name), target_size=image_size)
    images.append(img_to_array(image))

images_to_predict = preprocess_input(np.array(images, dtype=np.float32))

feature_extractor = VGG19(
    include_top=False, input_shape=(image_size[0], image_size[1], 3)
)
features = feature_extractor.predict(images_to_predict, verbose=0)
predictions = model.predict(features, verbose=0).reshape(-1)

with open("submission.csv", "w", newline="") as submissionCsvFile:
    csvWriter = writer(submissionCsvFile, lineterminator="\n")
    csvWriter.writerow(["id", "has_cactus"])
    for image_name, pred in zip(test_image_names, predictions):
        csvWriter.writerow([image_name, float(pred)])

print("Wrote submission.csv with rows:", len(test_image_names))
print(pd.read_csv("submission.csv").head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4150457380.py in <cell line: 0>()
      5 model_structure = Path("model_structure.json").read_text()
      6 model = model_from_json(model_structure)
----> 7 model.load_weights("model_weights.h5")
      8 
      9 # Load test images deterministically (sorted), so IDs match predictions order.

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
