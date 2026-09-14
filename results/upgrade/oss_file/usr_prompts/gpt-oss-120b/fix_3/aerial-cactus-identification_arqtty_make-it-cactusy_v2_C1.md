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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.9185

# 6. Current score

0.45166

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.45166) has done: 'I set the protobuf implementation flag before importing TensorFlow to avoid the `MessageFactory` error, and I filter out directory entries when building the lists of image file paths for both training and test sets. These fixes unblock the data loading, allow the model to train, and ensure a proper CSV submission is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split

keras = tf.keras

print("Input directory contents:", os.listdir("../input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_img_dir = "../input/aerial-cactus-identification/train/"
train_img_pathes = [
    os.path.join(train_img_dir, f)
    for f in sorted(os.listdir(train_img_dir))
    if os.path.isfile(os.path.join(train_img_dir, f))
]

df = pd.read_csv("../input/aerial-cactus-identification/train.csv")
print("Number of training images:", len(train_img_pathes))
print(df.head())




## === cell 2
lst = sorted(
    [
        f
        for f in os.listdir(train_img_dir)
        if os.path.isfile(os.path.join(train_img_dir, f))
    ]
)
err = False

for i, idx in enumerate(df["id"]):
    if idx != lst[i]:
        print("mismatch after %d iterations" % i)
        err = True
        break

if not err:
    print("1:1 corresponding between train_img_pathes and df labels")




## === cell 3
img_size = 32


def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):
    img_load_batch_size = 900
    output = None

    for i in range(0, len(img_paths), img_load_batch_size):
        print("process batch %d" % i)
        batch_paths = img_paths[i : i + img_load_batch_size]
        tmp_imgs = [
            load_img(p, target_size=(img_height, img_width))
            for p in batch_paths
            if os.path.isfile(p)
        ]
        tmp_img_array = np.array([img_to_array(img) for img in tmp_imgs])

        if not isinstance(output, np.ndarray):
            output = preprocess_input(tmp_img_array)
        else:
            output = np.vstack((output, preprocess_input(tmp_img_array)))

    return output


train_imgs = read_and_prep_images(train_img_pathes)




## === cell 4
num_classes = 2
out_y = to_categorical(df["has_cactus"], num_classes)

print("Shape of a single training image:", train_imgs[0].shape)
print("Shape of training data:", train_imgs.shape)




## === cell 5
model = Sequential()
model.add(
    Conv2D(filters=50, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu")
)
model.add(Dropout(0.5))
model.add(Conv2D(30, kernel_size=(3, 3), activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(54, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"]
)




## === cell 6
model.fit(
    train_imgs,
    out_y,
    batch_size=int(len(train_imgs) * 0.8 / 100),
    epochs=4,
    validation_split=0.2,
    verbose=2,
)




## === cell 7
test_img_dir = "../input/aerial-cactus-identification/test/"
test_img_pathes = [
    os.path.join(test_img_dir, f)
    for f in sorted(os.listdir(test_img_dir))
    if os.path.isfile(os.path.join(test_img_dir, f))
]
test_imgs = read_and_prep_images(test_img_pathes)
print("Number of test images:", len(test_imgs))




## === cell 8
prediction = model.predict(test_imgs)[:, 1]

answer = pd.DataFrame(columns=("id", "has_cactus"))
getFilename = lambda s: s.split("/")[-1]

for i, path in enumerate(test_img_pathes):
    answer.loc[i] = (getFilename(path), float(prediction[i]))

print("Submission preview:")
print(answer.head())




## === cell 9
answer.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
