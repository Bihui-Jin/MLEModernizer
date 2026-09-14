# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception as e:
    tf = None
    keras = None
    ImageDataGenerator = None
    print("TensorFlow import failed, falling back to dummy predictions:", e)

random.seed(42)
np.random.seed(42)
if tf is not None:
    tf.random.set_seed(42)




## === cell 1
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")




## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()




## === cell 3
label_lists = train_df["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
multihot = mlb.fit_transform(label_lists)

train_df[mlb.classes_] = multihot
num_classes = len(mlb.classes_)




## === cell 4
IMG_SIZE = (224, 224)
BATCH_SIZE = 128

if ImageDataGenerator is not None:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    workers = max(1, os.cpu_count() // 2)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="../input/plant-pathology-2021-fgvc8/train_images",
        x_col="image",
        y_col=mlb.classes_.tolist(),
        target_size=IMG_SIZE,
        color_mode="rgb",
        class_mode="raw",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        workers=workers,
        max_queue_size=8,
        use_multiprocessing=True,
    )
else:
    train_generator = None




## === cell 5
if tf is not None:
    base_model = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(num_classes, activation="sigmoid")(x)

    model = keras.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
else:
    model = None




## === cell 6
EPOCHS = 3
if tf is not None and train_generator is not None:
    model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
        verbose=1,
    )
else:
    print("Skipping model training due to missing TensorFlow.")




## === cell 7
if ImageDataGenerator is not None:
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    workers = max(1, os.cpu_count() // 2)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=submissions,
        directory="../input/plant-pathology-2021-fgvc8/test_images",
        x_col="image",
        y_col=None,
        target_size=IMG_SIZE,
        color_mode="rgb",
        class_mode=None,
        shuffle=False,
        batch_size=128,
        workers=workers,
        max_queue_size=8,
        use_multiprocessing=True,
    )
else:
    test_generator = None




## === cell 8
if tf is not None and model is not None and test_generator is not None:
    preds = model.predict(test_generator, verbose=1)
else:
    preds = np.zeros((test_generator.samples, num_classes))




## === cell 9
thresh = 0.5
pred_labels = []
for prob in preds:
    idxs = np.where(prob >= thresh)[0]
    if len(idxs) == 0:
        idxs = [np.argmax(prob)]
    pred_labels.append(" ".join(mlb.classes_[idxs]))

submissions["labels"] = pred_labels
submissions.to_csv("submission.csv", index=False)




## === cell 10
submissions.head()
