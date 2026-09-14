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

0.1578947368421052

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'The fixes remove the unused and incompatible `tensorflow_addons` import, add a safe fallback when the pretrained model file is missing (creating dummy predictions that mark every image as “healthy”), and correct the submission‑generation loop (use proper assignment, handle thresholds correctly, and ensure a label is always chosen). These changes allow the notebook to run end‑to‑end and produce a valid `submission.csv` file while keeping the original workflow intact.'
- What this solution (achieved 0.24507) has done: 'The fix adds a protobuf environment setting before importing TensorFlow to avoid the `MessageFactory` AttributeError, and wraps the TensorFlow import in a safe try/except while keeping the rest of the pipeline unchanged. This ensures the notebook runs end‑to‑end and produces a valid `submission.csv` without altering the core modeling logic or score‑affecting behavior.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np

try:
    import tensorflow as tf
    import tensorflow.keras as keras
except Exception as e:
    print(f"TensorFlow import failed ({e}); proceeding with dummy pipeline.")
    tf = None
    keras = None

import PIL
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")




## === cell 2
h_target = 512
w_target = 512




## === cell 3
if tf is not None:
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
        classes=None,
        class_mode=None,
        shuffle=False,
        batch_size=1,
    )
else:
    test_generator = None  # placeholder; not used when TF is unavailable




## === cell 4
model_path = "../input/mobilenetv2-512/mobilenetv2_512.h5"
try:
    if keras is None:
        raise RuntimeError("Keras not available")
    model = keras.models.load_model(model_path)
    if test_generator is not None:
        preds = model.predict(test_generator)
    else:
        raise RuntimeError("Test generator unavailable")
except Exception as e:
    print(f"Model load failed ({e}), using dummy predictions.")
    label_split = train.labels.apply(lambda x: x.split())
    trans_label = MultiLabelBinarizer().fit(label_split)
    n_classes = len(trans_label.classes_)
    preds = np.zeros((len(submissions), n_classes), dtype=np.float32)
    if "healthy" in trans_label.classes_:
        healthy_idx = list(trans_label.classes_).index("healthy")
        preds[:, healthy_idx] = 1.0




## === cell 5
label_split = train.labels.apply(lambda x: x.split())
trans_label = MultiLabelBinarizer().fit(label_split)
labels = pd.DataFrame(trans_label.transform(label_split), columns=trans_label.classes_)




## === cell 6
thresh = 0.4
for i in range(len(submissions)):
    idxs = np.where(preds[i] >= thresh)[0]
    if len(idxs) == 0:
        idxs = [int(np.argmax(preds[i]))]
    chosen_labels = [trans_label.classes_[j] for j in idxs]
    submissions.at[i, "labels"] = (
        " ".join(chosen_labels) if chosen_labels else "healthy"
    )

submissions.to_csv("submission.csv", index=False)




## === cell 7
submissions.head()
