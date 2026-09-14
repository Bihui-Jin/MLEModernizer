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

# 5. Target score

0.2924284395198524

# 6. Current score

0.21955

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.21955) has done: 'I fix the import crash by removing the problematic `from sklearn import *` (it’s unnecessary here and triggers the protobuf `MessageFactory` error in Kaggle). Then I fix the test image glob/path joining so it reliably finds all `test_images/*.jpg`, and replace the missing external model (`../input/resnet50-customized1-model/abc.h5`) with an on-the-fly ResNet50 backbone + Dense head that preserves the “ResNet feature extractor → predict probabilities → threshold → space-delimited labels” core flow. Finally, I ensure the submission uses the exact `sample_submission.csv` image order and writes a valid `submission.csv` with columns `image,labels` (and avoid attaching dataframes to the pandas module). This run end-to-end and produce a valid submission file.'

# 9. Code solution

## === cell 0
import os
import glob
import gc
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

import tensorflow.keras.applications.resnet50 as resnet

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3
TOTAL_INPUTS = NR_CHANNELS * IMG_HEIGHT * IMG_WIDTH



## === cell 2
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"
imglist_test = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
print("Found test images:", len(imglist_test))
print("First test image:", imglist_test[0] if imglist_test else None)



## === cell 3
Xim_test = np.zeros(
    (len(imglist_test), IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS), dtype=np.float32
)

for i, img_path in enumerate(imglist_test):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet.preprocess_input(x)
    Xim_test[i, :] = x[0]

print("Xim_test shape:", Xim_test.shape)



## === cell 4
if len(Xim_test) > 0:
    print(Xim_test[0])



## === cell 5
training_csv = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")

training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
tagnames = np.unique(training_class)
n_classes = len(tagnames)
print("Number of classes:", n_classes)
print("Classes:", tagnames)

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
base = resnet.ResNet50(
    include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
)
x = keras.layers.Dense(n_classes, activation="sigmoid")(base.output)
model_f = keras.Model(inputs=inputs, outputs=x)

X_test = model_f.predict(Xim_test, batch_size=16, verbose=1)
print("X_test shape:", X_test.shape)



## === cell 6
print(X_test)



## === cell 7
training_csv = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
tagnames = np.unique(training_class)




## === cell 8
def class2tags(classes, tagnames):
    tags = []
    for n in range(classes.shape[0]):
        tmp = []
        for i in range(classes.shape[1]):
            if classes[n, i]:
                tmp.append(tagnames[i])
        if len(tmp) == 0 and "healthy" in tagnames:
            tmp = ["healthy"]
        tags.append(" ".join(tmp))
    return tags




## === cell 9
test_predclass = X_test > 0.5
test_predtags = class2tags(test_predclass, tagnames)
print("Example predicted tags:", test_predtags[0] if len(test_predtags) else None)



## === cell 10
del test_predclass
gc.collect()



## === cell 11
sample_sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
print(sample_sub.head())

pred_map = {os.path.basename(p): tag for p, tag in zip(imglist_test, test_predtags)}



## === cell 12
sub = sample_sub.copy()
sub["labels"] = sub["image"].map(pred_map)

default_label = "healthy" if "healthy" in tagnames else ""
sub["labels"] = sub["labels"].fillna(default_label)

print(sub.head())
print("Submission rows:", len(sub), "null labels:", sub["labels"].isna().sum())



## === cell 13
sub[["labels"]].head()



## === cell 14
sub.head()



## === cell 15
sub.to_csv("./submission.csv", index=False)
print("Wrote ./submission.csv with shape:", sub.shape)
