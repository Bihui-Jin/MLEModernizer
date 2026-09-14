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

0.7676084949215155

# 6. Current score

0.28656

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the problematic TensorFlow‑Addons import, handle the missing pretrained model by falling back to a simple dummy predictor that always gives the highest score to the “healthy” class, and correct the submission‑generation logic (use assignment instead of comparison and properly map class probabilities to labels). These fixes ensure the notebook runs end‑to‑end, creates a valid `submission.csv`, and keeps the original workflow otherwise unchanged.'
- What this solution (achieved 0.28656) has done: 'The fix removes the TensorFlow import that crashes in the environment, replaces the broken model loading with a lightweight frequency‑based predictor built from the training label distribution, and keeps the original submission‑generation logic while ensuring a proper `submission.csv` is written. This makes the notebook run end‑to‑end and improves the mean F1 score by predicting the most common disease classes instead of always “healthy”.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

try:
    import tensorflow as tf
    import tensorflow.keras as keras
except Exception:
    tf = None
    keras = None



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")



## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 3
h_target = 512
w_target = 512
batch_size = 32



## === cell 4
n_samples = len(submissions)



## === cell 5
if keras is not None:
    try:
        model = keras.models.load_model("../input/mobilenetv2-512/mobilenetv2_512.h5")
    except Exception as e:
        print(f"Model load failed ({e}); using frequency‑based predictor.")
        model = None
else:
    model = None



## === cell 6
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_split)
labels_bin = pd.DataFrame(mlb.transform(label_split), columns=mlb.classes_)

class_freq = labels_bin.mean(axis=0)  # proportion of images containing each class
class_names = np.array(mlb.classes_)



## === cell 7
if model is not None:
    test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255
    )
    test_generator = test_data_generator.flow_from_dataframe(
        submissions,
        directory="../input/plant-pathology-2021-fgvc8/test_images",
        x_col="image",
        y_col=None,
        target_size=(h_target, w_target),
        color_mode="rgb",
        class_mode=None,
        shuffle=False,
        batch_size=batch_size,
    )
    preds = model.predict(test_generator, verbose=0)
else:
    preds = np.tile(class_freq.values, (n_samples, 1)).astype(np.float32)



## === cell 8
thresh = 0.4  # probability threshold for multi‑label prediction
for i in range(n_samples):
    prob_vec = preds[i]
    if "healthy" in class_names:
        healthy_idx = np.where(class_names == "healthy")[0][0]
        if prob_vec[healthy_idx] == prob_vec.max():
            submissions.at[i, "labels"] = "healthy"
            continue
    pred_idxs = np.where(prob_vec >= thresh)[0]
    if len(pred_idxs) == 0 or "healthy" in class_names[pred_idxs]:
        pred_idxs = np.where(prob_vec == prob_vec.max())[0]
    submissions.at[i, "labels"] = " ".join(class_names[pred_idxs])

submissions.to_csv("submission.csv", index=False)



## === cell 9
submissions.head()
