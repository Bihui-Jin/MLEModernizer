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

0.7977100646352737

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14534) has done: 'I remove the import that triggers the protobuf/TensorFlow Addons `GetPrototype` crash, and I add a safe model-loading fallback so the notebook can still run when the external `.h5` model file isn’t available in your `/kaggle/input` (the current error). I also fix the submission-generation logic bugs (`==` instead of `=`, chained indexing, and the always-true `if submissions['labels'][i] == ''or 'healthy' in ...`) and make it robust to any class ordering by using the MultiLabelBinarizer classes. Finally, I ensure the script always writes a valid `submission.csv` with the required `image,labels` columns and correct row alignment with the test generator.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf crash by removing the unnecessary TensorFlow import at the top and using the built-in `tf.keras` APIs only (this is the runtime blocker). Then I remove the untrained “fallback model” path (it produces near-random predictions and explains the very low 0.145 score) and instead load an existing Keras model from any available `.h5`/SavedModel path under `/kaggle/input`, failing fast with a clear error if none is found. Finally, I keep your label-thresholding logic but make it deterministic and aligned to the sample submission ordering, then always write a valid `submission.csv` with the required `image,labels` columns.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32



## === cell 4
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(SEED)

test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_generator = test_data_generator.flow_from_dataframe(
    submissions,
    directory="../input/plant-pathology-2021-fgvc8/test_images",
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode=None,
    shuffle=False,  # must be False for correct row alignment
    batch_size=batch_size,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(label_split)
class_names = list(mlb.classes_)
print("Num classes:", len(class_names))
print("Classes:", class_names)




## === cell 6
def find_model_candidates(root="/kaggle/input"):
    h5_paths = []
    savedmodel_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".h5") or fn.lower().endswith(".keras"):
                h5_paths.append(os.path.join(dirpath, fn))
        if "saved_model.pb" in filenames:
            savedmodel_dirs.append(dirpath)
    preferred = [
        p
        for p in h5_paths
        if os.path.basename(p).lower() in ("effnetb4.h5", "effnetb4.keras")
    ]
    if preferred:
        return preferred + [p for p in h5_paths if p not in preferred] + savedmodel_dirs
    return h5_paths + savedmodel_dirs


candidates = find_model_candidates("/kaggle/input")
print(f"Found {len(candidates)} model candidate(s) under /kaggle/input")

model = None
loaded_path = None
last_err = None

for p in candidates:
    try:
        model = keras.models.load_model(p, compile=False)
        loaded_path = p
        break
    except Exception as e:
        last_err = e
        continue

if model is None:
    raise FileNotFoundError(
        "No loadable Keras model (.h5/.keras or SavedModel) was found under /kaggle/input. "
        "Please add the pretrained model dataset to the notebook inputs. "
        f"Last load error: {repr(last_err)}"
    )

print("Loaded model from:", loaded_path)
preds = model.predict(test_generator, verbose=1)
preds = np.asarray(preds)
print("preds shape:", preds.shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/460263495.py in <cell line: 0>()
     42 if model is None:
     43     # Fail fast instead of producing a misleading low-score submission.
---> 44     raise FileNotFoundError(
     45         "No loadable Keras model (.h5/.keras or SavedModel) was found under /kaggle/input. "
     46         "Please add the pretrained model dataset to the notebook inputs. "

FileNotFoundError: No loadable Keras model (.h5/.keras or SavedModel) was found under /kaggle/input. Please add the pretrained model dataset to the notebook inputs. Last load error: None

## === cell 7
if len(submissions) != preds.shape[0]:
    raise RuntimeError(
        f"Prediction rows ({preds.shape[0]}) do not match submission rows ({len(submissions)}). "
        "Check generator ordering/shuffle settings."
    )



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4168448296.py in <cell line: 0>()
----> 1 if len(submissions) != preds.shape[0]:
      2     raise RuntimeError(
      3         f"Prediction rows ({preds.shape[0]}) do not match submission rows ({len(submissions)}). "
      4         "Check generator ordering/shuffle settings."
      5     )

NameError: name 'preds' is not defined

## === cell 8
thresh = 0.25

healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

out_labels = []
for i in range(preds.shape[0]):
    row = preds[i]
    row_argmax = int(np.argmax(row))

    if healthy_idx is not None and row_argmax == healthy_idx:
        out_labels.append("healthy")
        continue

    chosen = [
        class_names[j] for j in range(len(class_names)) if float(row[j]) >= thresh
    ]

    if (len(chosen) == 0) or ("healthy" in chosen):
        chosen = [class_names[row_argmax]]

    out_labels.append(" ".join(chosen))

submissions = submissions.copy()
submissions["labels"] = out_labels



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1889216907.py in <cell line: 0>()
      5 
      6 out_labels = []
----> 7 for i in range(preds.shape[0]):
      8     row = preds[i]
      9     row_argmax = int(np.argmax(row))

NameError: name 'preds' is not defined

## === cell 9
submissions[["image", "labels"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
submissions.head()



## === cell 10
submissions
