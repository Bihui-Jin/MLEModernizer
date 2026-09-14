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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
tqdm==4.67.1

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

0.7965

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.98565) has done: 'I fix the import error caused by the protobuf‑TensorFlow mismatch, guard image loading so that only valid images are used, correct the custom noisyand layer (use input_shape[-1] instead of the outdated .value attribute), adjust the model compilation and history key names, and finally ensure the script writes a proper `sample_submission.csv` with the required columns. These changes remove the runtime crashes, allow the model to train, and produce a valid submission file, moving the solution toward the target AUC score.'

# 9. Code solution

## === cell 0
import os, glob, cv2, numpy as np, pandas as pd
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dense,
    Layer,
)
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

BASE_DIR = "/kaggle/input/aerial-cactus-identification"
print("Base directory contents:", os.listdir(BASE_DIR))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def find_image_dir(base_dir, name_hint):
    """
    Return a directory containing images that matches the hint.
    If the expected folder (base_dir/name_hint) does not exist, search recursively.
    """
    candidate = os.path.join(base_dir, name_hint)
    if os.path.isdir(candidate):
        return candidate
    for root, dirs, files in os.walk(base_dir):
        if any(f.lower().endswith(".jpg") for f in files):
            return root
    raise FileNotFoundError(
        f"No image directory found for hint '{name_hint}' under {base_dir}"
    )


def load_imgs(path):
    """
    Load all 32x32 jpg images from a directory,
    keep only those that are read successfully and have the correct shape.
    """
    imgs = {}
    for f in os.listdir(path):
        if not f.lower().endswith(".jpg"):
            continue
        fp = os.path.join(path, f)
        img = cv2.imread(fp)
        if img is not None and img.shape == (32, 32, 3):
            imgs[f] = img.astype(np.float32) / 255.0  # normalise
    return imgs




## === cell 2
train_img_dir = find_image_dir(BASE_DIR, "train")
test_img_dir = find_image_dir(BASE_DIR, "test")

print("Train image dir:", train_img_dir)
print("Test image dir :", test_img_dir)

img_train = load_imgs(train_img_dir)
img_test = load_imgs(test_img_dir)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3545610930.py in <cell line: 0>()
      1 # Resolve possible missing subfolders
----> 2 train_img_dir = find_image_dir(BASE_DIR, "train")
      3 test_img_dir = find_image_dir(BASE_DIR, "test")
      4 
      5 print("Train image dir:", train_img_dir)

/tmp/ipykernel_11/2031333916.py in find_image_dir(base_dir, name_hint)
     11         if any(f.lower().endswith(".jpg") for f in files):
     12             return root
---> 13     raise FileNotFoundError(
     14         f"No image directory found for hint '{name_hint}' under {base_dir}"
     15     )

FileNotFoundError: No image directory found for hint 'train' under /kaggle/input/aerial-cactus-identification

## === cell 3
train_csv = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
X_train = []
Y_train = []

for _, row in train_csv.iterrows():
    img_id = row["id"]
    if img_id in img_train:
        X_train.append(img_train[img_id])
        Y_train.append(int(row["has_cactus"]))

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print("Training data shape:", X_train.shape, "=>", Y_train.shape)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2147415661.py in <cell line: 0>()
      5 for _, row in train_csv.iterrows():
      6     img_id = row["id"]
----> 7     if img_id in img_train:
      8         X_train.append(img_train[img_id])
      9         Y_train.append(int(row["has_cactus"]))

NameError: name 'img_train' is not defined

## === cell 4
from scipy.ndimage import gaussian_filter


def img_sharpen(img):
    blurred = gaussian_filter(img, 2)
    blurred2 = gaussian_filter(blurred, 2)
    alpha = 15
    return blurred + alpha * (blurred - blurred2)




## === cell 5
if X_train.shape[0] > 0:
    sharp_img_xtrain = [img_sharpen(im) for im in X_train]
    sharp_xtrain = np.array(sharp_img_xtrain)
else:
    sharp_xtrain = np.empty((0, 32, 32, 3), dtype=np.float32)

if sharp_xtrain.shape[0] == 0:
    raise RuntimeError("No training images were loaded – cannot continue.")

x_train, x_val, y_train, y_val = train_test_split(
    sharp_xtrain, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2817888064.py in <cell line: 0>()
      1 # Apply sharpening only if we have images
----> 2 if X_train.shape[0] > 0:
      3     sharp_img_xtrain = [img_sharpen(im) for im in X_train]
      4     sharp_xtrain = np.array(sharp_img_xtrain)
      5 else:

AttributeError: 'list' object has no attribute 'shape'

## === cell 6
class noisyand(Layer):
    def __init__(self, num_classes, a=20, **kwargs):
        super(noisyand, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.a = max(1, a)

    def build(self, input_shape):
        channel_dim = int(input_shape[-1])
        self.b = self.add_weight(
            name="b", shape=(1, channel_dim), initializer="uniform", trainable=True
        )
        super(noisyand, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])  # (batch, channels)
        b = self.b
        a = self.a
        top = tf.nn.sigmoid(a * (mean - b)) - tf.nn.sigmoid(-a * b)
        bot = tf.nn.sigmoid(a * (1 - b)) - tf.nn.sigmoid(-a * b)
        return top / bot

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])




## === cell 7
def define_model(input_shape=(32, 32, 3), num_classes=1):
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (1, 1), activation="relu"))

    model.add(noisyand(num_classes + 1))
    model.add(Dense(num_classes, activation="sigmoid"))
    return model




## === cell 8
model = define_model()
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=RMSprop(),
    metrics=["accuracy"],
)
model.summary()




## === cell 9
epochs = 30
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=1,
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3978666708.py in <cell line: 0>()
      1 epochs = 30
      2 history = model.fit(
----> 3     x_train,
      4     y_train,
      5     batch_size=32,

NameError: name 'x_train' is not defined

## === cell 10
submission = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
preds = np.empty(len(submission), dtype=np.float32)

for idx in tqdm(range(len(submission)), desc="Predicting"):
    img_id = submission.loc[idx, "id"]
    if img_id in img_test:
        img = img_test[img_id]
    else:
        img_path = os.path.join(test_img_dir, img_id)
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((32, 32, 3), dtype=np.float32)
        else:
            img = img.astype(np.float32) / 255.0
    img = img_sharpen(img)
    prob = model.predict(img.reshape((1, 32, 32, 3)), verbose=0)[0][0]
    preds[idx] = prob

submission["has_cactus"] = preds
submission.to_csv("sample_submission.csv", index=False)
print("Submission saved to sample_submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/723069815.py in <cell line: 0>()
      4 for idx in tqdm(range(len(submission)), desc="Predicting"):
      5     img_id = submission.loc[idx, "id"]
----> 6     if img_id in img_test:
      7         img = img_test[img_id]
      8     else:

NameError: name 'img_test' is not defined
