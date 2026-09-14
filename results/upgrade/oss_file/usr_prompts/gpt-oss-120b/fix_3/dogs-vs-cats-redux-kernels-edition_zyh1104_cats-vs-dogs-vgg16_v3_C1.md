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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

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
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        input/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
            test/
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
            train/
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        working/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

# 5. Target score

2.99801

# 6. Current score

0.20747

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.20747) has done: 'I added a small monkey‑patch for the protobuf `MessageFactory` before importing TensorFlow to stop the `GetPrototype` attribute error, and rewrote the test‑image loader to search recursively for JPG files so that it actually finds images inside the nested test folders. The rest of the pipeline (data preparation, model definition, training, and CSV creation) is unchanged, ensuring a valid `submission.csv` is written without altering the core modelling logic.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, *args, **kwargs):
            return None

        setattr(message_factory.MessageFactory, "GetPrototype", _get_prototype)
except Exception:
    pass  # If protobuf is not available, TensorFlow will raise later

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import zipfile
import shutil
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16




## === cell 1
base_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"

train_zip_path = os.path.join(base_dir, "train.zip")
test_zip_path = os.path.join(base_dir, "test.zip")

extract_root = "/kaggle/working"
train_dir = os.path.join(extract_root, "dogs-vs-cats-redux-kernels-edition", "train")
test_dir = os.path.join(extract_root, "dogs-vs-cats-redux-kernels-edition", "test")

if not os.path.isdir(train_dir):
    with zipfile.ZipFile(train_zip_path, "r") as z:
        z.extractall(extract_root)

if not os.path.isdir(test_dir):
    with zipfile.ZipFile(test_zip_path, "r") as z:
        z.extractall(extract_root)




## === cell 2
assert os.path.isdir(os.path.join(train_dir, "cat")), "cat folder missing"
assert os.path.isdir(os.path.join(train_dir, "dog")), "dog folder missing"




## === cell 3
batch_size = 32
img_size = (150, 150)

datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True,
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False,
)




## === cell 4
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

model = models.Sequential(
    [
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]
)




## === cell 5
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)




## === cell 6
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    verbose=2,
)




## === cell 7
model.save("cats_vs_dogs_vgg16_model.h5")




## === cell 8
def plot_history(hist):
    acc = hist.history.get("accuracy", [])
    val_acc = hist.history.get("val_accuracy", [])
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.figure()
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


plot_history(history)




## === cell 9
def prepare_test_images(test_dir_path, img_size=(150, 150)):
    jpg_files = []
    for root, _, files in os.walk(test_dir_path):
        for f in files:
            if f.lower().endswith(".jpg"):
                jpg_files.append(os.path.join(root, f))
    jpg_files.sort()  # ensure deterministic order
    filenames = [os.path.basename(p) for p in jpg_files]

    images = []
    for img_path in jpg_files:
        img = load_img(img_path, target_size=img_size)
        img_arr = img_to_array(img) / 255.0
        images.append(img_arr)
    if not images:
        raise ValueError("No test images found in the provided directory.")
    return filenames, np.stack(images)


test_filenames, test_images = prepare_test_images(test_dir, img_size=(150, 150))




## === cell 10
predictions = model.predict(test_images, batch_size=32, verbose=0).flatten()




## === cell 11
test_ids = [int(os.path.splitext(fname)[0]) for fname in test_filenames]

submission_df = pd.DataFrame(
    {"id": test_ids, "label": predictions}  # probability of being a dog
)
submission_df["id"] = submission_df["id"].astype("Int32")
submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
