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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.3266

# 6. Current score

0.37234

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.37234) has done: 'I fixed the import errors, updated the optimizer arguments, replaced deprecated `fit_generator` and `DataFrame.append`, corrected image resizing, ensured the prediction uses the probability of the cactus class, and rewrote the shell commands with pure Python so the notebook runs end‑to‑end and writes a proper `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os, shutil
import numpy as np
import pandas as pd

print(os.listdir("../input/"))



## === cell 1
os.makedirs("has_cactus", exist_ok=True)
os.makedirs("has_no_cactus", exist_ok=True)



## === cell 2
df = pd.read_csv("../input/train.csv")



## === cell 3
for i in df[df["has_cactus"] == 1]["id"]:
    src = os.path.join("../input/train/train/", i)
    dst = os.path.join("has_cactus", i)
    shutil.copy(src, dst)

for i in df[df["has_cactus"] == 0]["id"]:
    src = os.path.join("../input/train/train/", i)
    dst = os.path.join("has_no_cactus", i)
    shutil.copy(src, dst)



## === cell 4
print("Has Cactus: {}".format(df[df["has_cactus"] == 1]["id"].count()))
print("Has No Cactus: {}".format(df[df["has_cactus"] == 0]["id"].count()))




## === cell 5
def augument_data(directory, number_of_images_to_add):
    """Simple flip augmentation to balance the dataset."""
    print("Images to add: {}".format(number_of_images_to_add))
    import cv2
    from glob import glob

    img_paths = glob(os.path.join(directory, "*.jpg"))
    for image_path in img_paths:
        if number_of_images_to_add <= 0:
            break
        img = cv2.imread(image_path)
        h_img = cv2.flip(img, 0)
        v_img = cv2.flip(img, 1)
        cv2.imwrite(
            os.path.join(directory, f"h_img_{number_of_images_to_add}.jpg"), h_img
        )
        number_of_images_to_add -= 1
        if number_of_images_to_add <= 0:
            break
        cv2.imwrite(
            os.path.join(directory, f"v_img_{number_of_images_to_add}.jpg"), v_img
        )
        number_of_images_to_add -= 1




## === cell 6
imbalance = (
    df[df["has_cactus"] == 1]["id"].count() - df[df["has_cactus"] == 0]["id"].count()
)
if imbalance > 0:
    augument_data("./has_no_cactus/", imbalance)



## === cell 7
os.makedirs("curated_data/train_data", exist_ok=True)
os.makedirs("curated_data/validation_data/has_cactus", exist_ok=True)
os.makedirs("curated_data/validation_data/has_no_cactus", exist_ok=True)

shutil.move("has_cactus", "curated_data/train_data")
shutil.move("has_no_cactus", "curated_data/train_data")



## === cell 8
from glob import glob

train_cactus = glob("curated_data/train_data/has_cactus/*.jpg")
train_no_cactus = glob("curated_data/train_data/has_no_cactus/*.jpg")

for img_path in train_cactus[:300]:
    shutil.move(img_path, "curated_data/validation_data/has_cactus")
for img_path in train_no_cactus[:300]:
    shutil.move(img_path, "curated_data/validation_data/has_no_cactus")



## === cell 9
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
datagen = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
)

train_data = datagen.flow_from_directory(
    "curated_data/train_data/",
    target_size=(256, 256),
    batch_size=32,
    class_mode="categorical",
)

validation_data = datagen.flow_from_directory(
    "curated_data/validation_data/",
    target_size=(256, 256),
    batch_size=32,
    class_mode="categorical",
)



## === cell 11
inputs = Input(shape=(256, 256, 3))
x = Conv2D(32, (3, 3), activation="relu")(inputs)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)



## === cell 12
model.compile(
    loss="binary_crossentropy",
    optimizer=SGD(learning_rate=0.0001, momentum=0.9),
    metrics=["accuracy"],
)



## === cell 13
model.fit(train_data, epochs=4, validation_data=validation_data, verbose=2)



## === cell 14
import cv2

test_images = glob("../input/test/test/*.jpg")
pred_ids = []
pred_probs = []
for img_path in test_images:
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (256, 256))
    img_array = img_resized.astype("float32") / 255.0
    prob = model.predict(img_array[np.newaxis, ...], verbose=0)[0][
        1
    ]  # probability of class "cactus"
    pred_ids.append(os.path.basename(img_path))
    pred_probs.append(prob)

submission = pd.DataFrame({"id": pred_ids, "has_cactus": pred_probs})



## === cell 15
shutil.rmtree("curated_data", ignore_errors=True)



## === cell 16
submission.to_csv("submission.csv", index=False)
