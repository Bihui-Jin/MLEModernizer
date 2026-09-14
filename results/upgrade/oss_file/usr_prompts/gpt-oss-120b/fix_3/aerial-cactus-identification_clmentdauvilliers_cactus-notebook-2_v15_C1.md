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

3.9

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
protobuf==6.33.0
scikit-image==0.25.2
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.8573

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from zipfile import ZipFile

path = "/kaggle/input/aerial-cactus-identification/"
files_dataframe = pd.read_csv(path + "train.csv", dtype=str)
files_dataframe.head()




## === cell 1
with ZipFile(path + "train.zip", "r") as zipper:
    zipper.extractall("./training/")
with ZipFile(path + "test.zip", "r") as zipper:
    zipper.extractall("./test/")




## === cell 2
class_reparts = files_dataframe["has_cactus"].value_counts()
ax = class_reparts.plot.bar()
plt.title("Class distribution")
plt.show()

total_samples = files_dataframe["has_cactus"].size
has_cactus_weight = total_samples / (2 * class_reparts.get("1", 1))
no_cactus_weight = total_samples / (2 * class_reparts.get("0", 1))
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights:", class_weights)




## === cell 3
plt.figure(figsize=(36, 12))
training_files = "train/" + files_dataframe["id"]
for i, k in enumerate(np.random.randint(0, len(files_dataframe), size=20)):
    plt.subplot(4, 5, i + 1)
    img_path = os.path.join("./training", training_files.iloc[k])
    plt.imshow(imread(img_path))
    plt.title("Label: " + str(files_dataframe["has_cactus"].iloc[k]))
plt.show()




## === cell 4
import skimage.exposure as exposure


def preprocess(img):
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale


plt.figure(figsize=(12, 6))
img_example = imread(os.path.join("./training", training_files.iloc[0]))
plt.subplot(1, 2, 1)
plt.imshow(img_example)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(preprocess(img_example))
plt.title("Preprocessed")
plt.show()




## === cell 5
from tensorflow.keras.preprocessing.image import ImageDataGenerator


generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=45,
    validation_split=0.1,
    preprocessing_function=preprocess,
)

noAugmentationGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    preprocessing_function=preprocess,
)




## === cell 6
training_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
)

validation_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=False,
)




## === cell 7
try:
    plt.figure(figsize=(12, 6))
    imgs, labels = next(validation_generator)
    for idx in range(min(12, len(imgs))):
        plt.subplot(3, 4, idx + 1)
        plt.imshow(imgs[idx])
        lbl = np.argmax(labels[idx])
        plt.title(f"Label: {lbl}")
    plt.show()
except Exception as e:
    print("Visualization skipped:", e)




## === cell 8
from tensorflow.keras import layers, models




## === cell 9
model = models.Sequential()
model.add(
    layers.Conv2D(
        32, (3, 3), padding="valid", activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(64, (5, 5), padding="same", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()




## === cell 10
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

save_best_model = ModelCheckpoint(
    "/tmp/checkpoint.keras", monitor="val_accuracy", mode="max", save_best_only=True
)




## === cell 11
history = model.fit(
    training_generator,
    validation_data=validation_generator,
    steps_per_epoch=training_generator.n // training_generator.batch_size,
    epochs=5,
    class_weight=class_weights,
    callbacks=[reduce_lr, save_best_model],
    verbose=1,
)




## === cell 12
test_filenames = sorted(os.listdir("./test/test/"))
test_df = pd.DataFrame({"id": test_filenames})
test_generator = noAugmentationGenerator.flow_from_dataframe(
    dataframe=test_df,
    directory="./test/test/",
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=(32, 32),
    batch_size=32,
    shuffle=False,
)




## === cell 13
from keras.models import load_model

model = load_model("/tmp/checkpoint.keras")




## === cell 14
preds = model.predict(test_generator, verbose=0)
prob_cactus = preds[:, 1]  # probability of class 1
output = pd.DataFrame({"id": test_generator.filenames, "has_cactus": prob_cactus})
output["id"] = output["id"].apply(lambda s: os.path.basename(s))
output.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")




## === cell 15
import shutil

for folder in ["test", "training"]:
    try:
        shutil.rmtree(folder)
    except OSError:
        pass
