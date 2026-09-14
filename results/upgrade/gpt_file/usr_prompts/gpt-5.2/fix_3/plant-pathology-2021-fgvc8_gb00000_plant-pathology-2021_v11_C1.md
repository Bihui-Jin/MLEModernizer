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

0.7410156971375809

# 6. Current score

0.28148

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.28148) has done: 'I fix the TensorFlow import crash by removing the forced protobuf Python implementation setting that breaks TF’s protobuf bindings in this environment, while keeping the same TF/Keras workflow. Then I fix the missing model path issue by loading an available EfficientNet family model from `tf.keras.applications` (same inference-style pipeline) instead of a non-existent `/kaggle/input/effnet3/...` SavedModel, so the notebook can run end-to-end. I also correct the label handling: `flow_from_dataframe(..., class_mode="categorical")` cannot directly consume the space-delimited multi-label strings, so I build a proper multi-label one-hot matrix and use `class_mode="raw"` with a custom dataframe. Finally, I generate submission labels by thresholding sigmoid outputs into a space-delimited list (falling back to the top-1 label if none exceed the threshold) and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        pass



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import cv2
import re
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import os



## === cell 3
sam_sub = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
sam_sub.head()



## === cell 4
train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"



## === cell 5
train = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 6
test_filepaths = []
test_ids = []

for dirname, _, filenames in os.walk(test_dir):
    for filename in filenames:
        test_ids.append(filename)
        test_filepaths.append(os.path.join(dirname, filename))

test_df = pd.DataFrame(test_ids, columns=["image"])
test_df.head()



## === cell 7
all_labels = sorted(
    {lab for s in train["labels"].fillna("").values for lab in s.split(" ") if lab}
)
label2idx = {lab: i for i, lab in enumerate(all_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}
num_classes = len(all_labels)


def encode_multi_hot(label_str: str) -> np.ndarray:
    vec = np.zeros(num_classes, dtype=np.float32)
    if isinstance(label_str, str) and label_str.strip():
        for lab in label_str.split(" "):
            if lab in label2idx:
                vec[label2idx[lab]] = 1.0
    return vec


y = np.stack([encode_multi_hot(s) for s in train["labels"].values], axis=0)
y_cols = [f"y_{lab}" for lab in all_labels]

train_ml = train[["image"]].copy()
for j, col in enumerate(y_cols):
    train_ml[col] = y[:, j]

train_ml.head()



## === cell 8
train_datagen_sub = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

train_generator_sub = train_datagen_sub.flow_from_dataframe(
    train_ml,
    directory=train_dir,
    x_col="image",
    y_col=y_cols,
    target_size=(224, 336),
    batch_size=16,
    class_mode="raw",
    shuffle=False,
)



## === cell 9
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="image",
    target_size=(224, 336),
    batch_size=16,
    class_mode=None,
    shuffle=False,
)



## === cell 10
base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 336, 3),
    pooling="avg",
)
inp = keras.Input(shape=(224, 336, 3), name="image")
x = keras.applications.efficientnet.preprocess_input(inp)
x = base(x, training=False)
out = keras.layers.Dense(num_classes, activation="sigmoid")(x)
trained_model_sub = keras.Model(inp, out)
trained_model_sub.trainable = False

trained_model_sub.summary()



## === cell 11
y_pred = trained_model_sub.predict(test_generator, verbose=1)
y_pred = np.asarray(y_pred)
print("y_pred shape:", y_pred.shape)



## === cell 12
threshold = 0.5

pred_label_strs = []
for row in y_pred:
    inds = np.where(row >= threshold)[0].tolist()
    if len(inds) == 0:
        inds = [int(np.argmax(row))]
    labs = [idx2label[i] for i in inds]
    pred_label_strs.append(" ".join(labs))

gen_files = list(test_generator.filenames)
gen_images = [os.path.basename(f) for f in gen_files]

sub = pd.DataFrame({"image": gen_images, "labels": pred_label_strs})
sub.head()



## === cell 13
assert list(sub.columns) == ["image", "labels"]
assert sub["image"].nunique() == len(sub)
assert sub["labels"].isna().sum() == 0

sub = sub.set_index("image").reindex(sam_sub["image"]).reset_index()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
