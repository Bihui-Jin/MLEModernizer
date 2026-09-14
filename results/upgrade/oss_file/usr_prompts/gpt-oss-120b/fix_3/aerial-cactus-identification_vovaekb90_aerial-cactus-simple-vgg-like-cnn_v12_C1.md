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

3.7

# 3. Installed packages

geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
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

0.9966

# 6. Current score

0.23404

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.23404) has done: 'The fix replaces the direct `keras` imports with `tensorflow.keras` to avoid the protobuf error, restores missing imports (e.g., `ImageDataGenerator`, `train_test_split`, `plt`), renumbers the cells starting at 1, and ensures the data pipeline, model, training, and submission steps run sequentially. No core‑logic changes are made, only the necessary imports and ordering so the notebook executes end‑to‑end and produces a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import gc
import glob
import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

print("Input root contents:", os.listdir("../input/"))




## === cell 1
def load_images(glob_path):
    images = []
    names = []
    for img_path in glob.glob(glob_path):
        names.append(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        images.append(img)  # already 32x32
    return images, names


train_images, train_names = load_images("../input/train/*.jpg")
print(f"Loaded {len(train_images)} training images")
plt.figure(figsize=(4, 2))
cols = 4
for i in range(min(8, len(train_images))):
    plt.subplot(2, cols, i + 1)
    plt.imshow(cv2.cvtColor(train_images[i], cv2.COLOR_BGR2RGB))
    plt.axis("off")
plt.show()




## === cell 2
train_meta = pd.read_csv("../input/train.csv")
print("Metadata shape:", train_meta.shape)
print(train_meta["has_cactus"].value_counts())
lookupY = dict(zip(train_meta["id"], train_meta["has_cactus"]))

data_list = []
for img, name in zip(train_images, train_names):
    label = lookupY.get(name, 0)
    data_list.append({"label": label, "data": img})
random.shuffle(data_list)
train_df = pd.DataFrame(data_list)
gc.collect()
print("Dataframe shape:", train_df.shape)
print(train_df["label"].value_counts())




## === cell 3
data_stack = np.stack(train_df["data"].values)
data_floats = data_stack.astype(np.float32)
all_x = data_floats / 255.0
all_y = np.array(train_df["label"]).astype(np.float32)
print("X shape:", all_x.shape, "Y shape:", all_y.shape)




## === cell 4
train_x, test_x, train_y, test_y = train_test_split(
    all_x, all_y, test_size=0.2, random_state=7, stratify=all_y
)
print("Train/X shape:", train_x.shape, "Test/X shape:", test_x.shape)




## === cell 5
datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=60,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)
datagen.fit(train_x)




## === cell 6
input_shape = train_x.shape[1:]  # (32, 32, 3)
output_shape = 1

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(output_shape, activation="sigmoid"))

model.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()




## === cell 7
batch_size = 32
history = model.fit(
    datagen.flow(train_x, train_y, batch_size=batch_size),
    steps_per_epoch=len(train_x) // batch_size,
    epochs=20,
    validation_data=(test_x, test_y),
    workers=4,
    verbose=2,
)




## === cell 8
test_images, test_names = load_images("../input/test/*.jpg")
print(f"Loaded {len(test_images)} test images")
test_stack = np.stack(test_images)
test_floats = test_stack.astype(np.float32)
unknown_x = test_floats / 255.0

predicted = np.ravel(model.predict(unknown_x, batch_size=64))
submission_df = pd.DataFrame({"id": test_names, "has_cactus": predicted})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, rows:", len(submission_df))
