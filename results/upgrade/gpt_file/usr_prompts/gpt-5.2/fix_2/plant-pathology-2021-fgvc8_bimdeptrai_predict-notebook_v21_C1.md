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

0.807700831024933

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(label_split)
class_names = list(mlb.classes_)

labels_df = pd.DataFrame(mlb.transform(label_split), columns=class_names)
print("Num classes:", len(class_names))
print("Classes:", class_names)



## === cell 3
h_target = 384
w_target = 384
batch_size = 32

test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_generator = test_data_generator.flow_from_dataframe(
    submissions,
    directory=TEST_DIR,
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode=None,
    shuffle=False,  # critical: keep order aligned with submissions.image
    batch_size=batch_size,
)



## === cell 4


def find_model_path(preferred_path: str):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    roots = ["/kaggle/input", "../input"]
    candidates = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                lfn = fn.lower()
                if lfn.endswith(".h5") or lfn.endswith(".hdf5"):
                    candidates.append(os.path.join(dirpath, fn))
            if "saved_model.pb" in filenames:
                candidates.append(dirpath)

    keywords = ["eff", "efficientnet", "plant", "pathology", "fgvc"]
    scored = []
    for p in candidates:
        lp = p.lower()
        score = sum(k in lp for k in keywords)
        scored.append((score, p))
    scored.sort(reverse=True)

    if scored:
        return scored[0][1]
    return None


preferred = "../input/effnetb0-512-to-384/EffNetB0_512to384.h5"
model_path = find_model_path(preferred)

if model_path is None:
    raise FileNotFoundError(
        "Could not find any .h5/.hdf5 model or SavedModel directory under /kaggle/input. "
        "Please add the trained model as a Kaggle Dataset input."
    )

print("Loading model from:", model_path)

model = keras.models.load_model(model_path, compile=False)
model.summary()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2817672172.py in <cell line: 0>()
     42 
     43 if model_path is None:
---> 44     raise FileNotFoundError(
     45         "Could not find any .h5/.hdf5 model or SavedModel directory under /kaggle/input. "
     46         "Please add the trained model as a Kaggle Dataset input."

FileNotFoundError: Could not find any .h5/.hdf5 model or SavedModel directory under /kaggle/input. Please add the trained model as a Kaggle Dataset input.

## === cell 5
preds = model.predict(test_generator, verbose=1)

preds = np.asarray(preds)
print("Preds shape:", preds.shape)

if preds.ndim == 1:
    preds = preds.reshape(-1, 1)

n_test = len(submissions)
if preds.shape[0] != n_test:
    preds = preds[:n_test]

print("Preds aligned shape:", preds.shape)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1697630813.py in <cell line: 0>()
      1 # Predict
----> 2 preds = model.predict(test_generator, verbose=1)
      3 
      4 preds = np.asarray(preds)
      5 print("Preds shape:", preds.shape)

NameError: name 'model' is not defined

## === cell 6

thresh = 0.3

healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

out_labels = []
for i in range(n_test):
    p = preds[i]

    c = min(len(class_names), p.shape[-1])
    p = p[:c]

    if healthy_idx is not None and healthy_idx < c and p[healthy_idx] == np.max(p):
        lab = "healthy"
    else:
        chosen = [class_names[j] for j in range(c) if p[j] >= thresh]

        if (len(chosen) == 0) or ("healthy" in chosen):
            j = int(np.argmax(p))
            lab = class_names[j]
        else:
            lab = " ".join(chosen)

    out_labels.append(lab)

submissions = submissions.copy()
submissions["labels"] = out_labels

submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)

print(submissions.head())
print("Wrote submission.csv with shape:", submissions.shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1429879002.py in <cell line: 0>()
     12 
     13 out_labels = []
---> 14 for i in range(n_test):
     15     p = preds[i]
     16 

NameError: name 'n_test' is not defined

## === cell 7
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(submissions)
chk.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/1200893835.py in <cell line: 0>()
      1 # Sanity checks for Kaggle format
----> 2 assert os.path.exists("submission.csv")
      3 chk = pd.read_csv("submission.csv")
      4 assert list(chk.columns) == ["image", "labels"]
      5 assert len(chk) == len(submissions)

AssertionError:
