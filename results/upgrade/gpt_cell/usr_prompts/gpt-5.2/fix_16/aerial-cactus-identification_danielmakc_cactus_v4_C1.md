# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
from IPython.display import display

image_size = (32, 32)


## === cell 1
from PIL import Image
from os import listdir
from os.path import join
from pandas import read_csv

train_labels = read_csv("../input/train.csv")
for image_name in listdir("../input/train/train")[:10]:
    image = (
        Image.open(join("../input/train/train", image_name))
        .convert("RGB")
        .resize(image_size)
    )
    label = train_labels[train_labels["id"] == image_name]["has_cactus"].item()
    print(label)
    print(image)


## === cell 2
import os

os.environ.setdefault("KERAS_BACKEND", "numpy")

from keras.applications.vgg19 import VGG19, preprocess_input
from keras.utils import img_to_array, load_img
from numpy import array
from os import listdir
from os.path import join
from pandas import read_csv
from tqdm import tqdm_notebook


def extract_features(label_path, set_path):
    images = []
    labels = []

    model = VGG19(include_top=False, input_shape=(image_size[0], image_size[1], 3))

    train_labels = read_csv(label_path)
    for image_name in tqdm_notebook(listdir(set_path)):
        image = load_img(join(set_path, image_name), target_size=image_size)
        images.append(img_to_array(image))
        label = train_labels[train_labels["id"] == image_name]["has_cactus"].item()
        labels.append(label)

    training_images = preprocess_input(array(images))
    training_labels = array(labels)

    features = model.predict(training_images)

    return features, training_labels


## === cell 3
from joblib import dump
from os import listdir
from os.path import isfile, join

_train_dir = "../input/train/train"
_image_exts = (".jpg", ".jpeg", ".png", ".bmp")

_filtered_files = [
    fn
    for fn in listdir(_train_dir)
    if isfile(join(_train_dir, fn)) and fn.lower().endswith(_image_exts)
]

_original_listdir = listdir


def _safe_listdir(path):
    if path == _train_dir:
        return _filtered_files
    return _original_listdir(path)


globals()["listdir"] = _safe_listdir
try:
    features, training_labels = extract_features("../input/train.csv", _train_dir)
finally:
    globals()["listdir"] = _original_listdir

dump(features, "features.dat")
dump(training_labels, "labels.dat")

display(listdir("."))


## === cell 4
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ["KERAS_BACKEND"] = "tensorflow"
for _m in list(sys.modules):
    if _m == "keras" or _m.startswith("keras."):
        del sys.modules[_m]

from joblib import load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
from os import listdir

x_train = load("features.dat")
y_train = load("labels.dat")

model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=100, shuffle=True, validation_split=0.2)

Path("model_structure.json").write_text(model.to_json())
model.save_weights("model_weights.h5")

display(listdir("."))


## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
from csv import writer
from keras.applications.vgg19 import preprocess_input
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from numpy import array
from pathlib import Path
from tqdm import tqdm_notebook 

model_structure = Path('model_structure.json').read_text()
model = model_from_json(model_structure)
model.load_weights('model_weights.h5')

images = []
for image_name in tqdm_notebook(listdir('../input/test/test')):
    image = load_img(join('../input/test/test', image_name), target_size=image_size)
    images.append(img_to_array(image))
    
images_to_predict = preprocess_input(array(images))

feature_extractor = VGG19(include_top=False, input_shape=(image_size[0], image_size[1], 3))
features = feature_extractor.predict(images_to_predict)
predictions = model.predict(features)

display(predictions)

with open('submission.csv', 'w+') as submissionCsvFile:
    csvWriter = writer(submissionCsvFile, lineterminator='\n')
    csvWriter.writerow(['id', 'has_cactus'])
    
    for index, image_name in enumerate(listdir('../input/test/test')):        
        csvWriter.writerow([image_name, predictions[index][0]])
