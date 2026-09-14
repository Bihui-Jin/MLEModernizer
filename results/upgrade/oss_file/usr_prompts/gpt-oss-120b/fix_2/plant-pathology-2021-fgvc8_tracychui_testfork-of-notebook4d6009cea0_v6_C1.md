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
import os, glob, gc
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import ResNet50, resnet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
import warnings, random

warnings.filterwarnings("ignore")
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
K.set_image_data_format("channels_last")
print("TF:", tf.__version__, "Keras:", keras.__version__)



## === cell 1
IMG_WIDTH, IMG_HEIGHT, NR_CHANNELS = 300, 300, 3
DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SUBMISSION_PATH = "./submission.csv"



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
print("Train rows:", len(train_df))
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
train_targets = mlb.fit_transform(train_df["label_list"])
tag_names = mlb.classes_
num_tags = len(tag_names)
print("Num tags:", num_tags)




## === cell 3
def preprocess_path(img_path):
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet50.preprocess_input(x)
    return x[0]


train_images = []
for img_id in train_df["image"].values[
    :2000
]:  # limit to 2000 samples to keep runtime low
    img_path = os.path.join(TRAIN_IMG_DIR, img_id)
    if os.path.exists(img_path):
        train_images.append(preprocess_path(img_path))
train_X = np.stack(train_images, axis=0).astype("float32")
train_y = train_targets[: len(train_X)]
print("Training data shape:", train_X.shape, train_y.shape)



## === cell 4
base = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
    pooling="avg",
)
base.trainable = False  # freeze backbone for quick training
inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
x = base(inputs, training=False)
outputs = layers.Dense(num_tags, activation="sigmoid")(x)
model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.summary()



## === cell 5
model.fit(train_X, train_y, epochs=2, batch_size=32, verbose=1)



## === cell 6
test_img_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
print("Test images:", len(test_img_paths))
test_imgs = []
valid_paths = []
for p in test_img_paths:
    try:
        test_imgs.append(preprocess_path(p))
        valid_paths.append(os.path.basename(p))
    except Exception:
        continue
test_X = np.stack(test_imgs, axis=0).astype("float32")
print("Test data shape:", test_X.shape)



## === cell 7
test_probs = model.predict(test_X, batch_size=32, verbose=0)



## === cell 8
THRESH = 0.5
test_binary = (test_probs > THRESH).astype(int)
pred_tags = []
for row in test_binary:
    tags = [tag_names[i] for i, val in enumerate(row) if val]
    pred_tags.append(" ".join(tags) if tags else "healthy")  # fallback to 'healthy'



## === cell 9
submission_df = pd.DataFrame({"image": valid_paths, "labels": pred_tags})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print("Submission written to", SUBMISSION_PATH)
