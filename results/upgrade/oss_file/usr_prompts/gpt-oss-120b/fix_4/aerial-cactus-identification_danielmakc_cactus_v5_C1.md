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

0.9844

# 6. Current score

0.8681

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.8681) has done: 'The fixes address the three runtime errors and remove the dependency on TensorFlow/VGG which caused protobuf import failures.  
1. **Cell 0** now uses Pillow (`Image`) instead of TensorFlow for loading images.  
2. **Cell 1** rewrites `extract_features` to load raw image arrays directly, scale them, and skip VGG‑19, eliminating the problematic import.  
3. **Cell 3** saves weights with the required “`.weights.h5`” suffix and updates the loading path.  
4. **Cell 4** loads the correct weight file, selects the proper test directory, and extracts test features using the same simple preprocessing as the training step.  
These minimal changes restore end‑to‑end execution and produce a valid `submission.csv` while keeping the overall model‑training logic intact.'

# 9. Code solution

## === cell 0
image_size = (32, 32)

from os import listdir
from os.path import join
from pandas import read_csv
from PIL import Image

train_labels = read_csv("../input/train.csv")
for image_name in listdir("../input/train/train")[:5]:
    _ = Image.open(join("../input/train/train", image_name)).resize(image_size)
    _ = train_labels[train_labels["id"] == image_name]["has_cactus"].item()




## === cell 1
from keras.utils import load_img
from keras.preprocessing.image import img_to_array
from numpy import array
from os import listdir
from os.path import join, isfile
from pandas import read_csv
from tqdm import tqdm
import joblib


def extract_features(label_path, set_path):
    images = []
    labels = []

    train_labels = read_csv(label_path)
    for image_name in tqdm(listdir(set_path), desc="Extracting features"):
        if not image_name.lower().endswith(".jpg"):
            continue
        img_path = join(set_path, image_name)
        if not isfile(img_path):
            continue
        image = load_img(img_path, target_size=image_size)
        images.append(img_to_array(image))

        label = train_labels[train_labels["id"] == image_name]["has_cactus"].item()
        labels.append(label)

    training_images = array(images, dtype="float32") / 255.0
    training_labels = array(labels, dtype="float32")
    return training_images, training_labels




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
features, training_labels = extract_features(
    "../input/train.csv", "../input/train/train"
)

joblib.dump(features, "features.dat")
joblib.dump(training_labels, "labels.dat")
print("Saved features and labels:", listdir("."))




## === cell 3
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import joblib

x_train = joblib.load("features.dat")
y_train = joblib.load("labels.dat")

model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.2, verbose=2)

Path("model_structure.json").write_text(model.to_json())
model.save_weights("model_weights.weights.h5")
print("Model saved:", listdir("."))




## === cell 4
from csv import writer
from keras.utils import load_img
from keras.preprocessing.image import img_to_array
from numpy import array
from pathlib import Path
from tqdm import tqdm
from os import listdir
from os.path import join, isfile

model_structure = Path("model_structure.json").read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.weights.h5")

possible_test_dirs = [
    "../input/aerial-cactus-identification/test",
    "../input/test/test",
    "../input/test",
    "../input/aerial-cactus-identification/test",
]
test_dir = None
for d in possible_test_dirs:
    if isfile(
        join(d, listdir(d)[0])
    ):  # crude check that directory exists and contains files
        test_dir = d
        break
if test_dir is None:
    raise FileNotFoundError("Test directory not found.")

image_names = [
    fn
    for fn in listdir(test_dir)
    if fn.lower().endswith(".jpg") and isfile(join(test_dir, fn))
]

images = []
for image_name in tqdm(image_names, desc="Loading test images"):
    img = load_img(join(test_dir, image_name), target_size=image_size)
    images.append(img_to_array(img))

images_to_predict = array(images, dtype="float32") / 255.0

predictions = model.predict(images_to_predict, verbose=0)

with open("submission.csv", "w", newline="") as f:
    csv_writer = writer(f)
    csv_writer.writerow(["id", "has_cactus"])
    for idx, image_name in enumerate(image_names):
        csv_writer.writerow([image_name, float(predictions[idx][0])])

print("Submission file created:", "submission.csv" in listdir("."))
