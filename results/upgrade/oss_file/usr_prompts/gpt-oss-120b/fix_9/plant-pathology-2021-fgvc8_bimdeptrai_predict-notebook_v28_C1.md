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
os.environ["PROTOCOL_BUFFERS_PYTHON_PREFER_GOOGLE"] = "false"

import random
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    import tensorflow.keras as keras
except Exception as e:
    print(f"TensorFlow import failed ({e}); predictions will use fallback model.")
    tf = None
    keras = None

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if tf is not None:
    tf.random.set_seed(SEED)




## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")




## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_split)
labels = pd.DataFrame(mlb.transform(label_split), columns=mlb.classes_)




## === cell 3
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import colorsys

train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"


def extract_features(path):
    """
    Return a 9‑dimensional feature vector:
    mean R,G,B ; std R,G,B ; mean H,S,V (HSV space)
    """
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((32, 32))  # cheap resize
        arr = np.array(img) / 255.0  # shape (32,32,3)

        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))

        hsv = np.apply_along_axis(
            lambda rgb: colorsys.rgb_to_hsv(*rgb), 2, arr
        )  # shape (32,32,3)
        mean_hsv = hsv.mean(axis=(0, 1))

        return np.concatenate([mean_rgb, std_rgb, mean_hsv]).astype(np.float32)
    except Exception:
        return np.zeros(9, dtype=np.float32)


def _process_train_image(img_name):
    img_path = os.path.join(train_img_dir, img_name)
    return extract_features(img_path)


with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    X_train_list = list(executor.map(_process_train_image, train["image"]))
X_train = np.vstack(X_train_list)  # shape (n_samples, 9)




## === cell 4
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, labels.values, test_size=0.1, random_state=SEED, stratify=labels.values
)

clf = OneVsRestClassifier(
    LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
        random_state=SEED,
        n_jobs=-1,
    )
)

clf.fit(X_tr, y_tr)

val_probs = clf.predict_proba(X_val)

thresholds_vec = []
for i, cls in enumerate(mlb.classes_):
    y_true = y_val[:, i]
    y_prob = val_probs[:, i]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(y_true, y_prob >= t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    thresholds_vec.append(best_t)

thresholds_vec = np.array(thresholds_vec)




## === cell 5
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")




## === cell 6
h_target = 256
w_target = 256
batch_size = 32

model_path = "../input/effnetb4-512-to-256/EffNetB4_512to256.h5"
num_classes = len(mlb.classes_)

if keras is not None:
    try:
        model = keras.models.load_model(model_path)
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255
        )
        test_generator = test_datagen.flow_from_dataframe(
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
    except Exception as e:
        print(f"Model load failed ({e}); using logistic‑regression predictions.")
        test_img_dir = "../input/plant-pathology-2021-fgvc8/test_images"

        def _process_test_image(img_name):
            img_path = os.path.join(test_img_dir, img_name)
            return extract_features(img_path)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            X_test_list = list(executor.map(_process_test_image, submissions["image"]))
        X_test = np.vstack(X_test_list)
        preds = clf.predict_proba(X_test)
else:
    test_img_dir = "../input/plant-pathology-2021-fgvc8/test_images"

    def _process_test_image(img_name):
        img_path = os.path.join(test_img_dir, img_name)
        return extract_features(img_path)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        X_test_list = list(executor.map(_process_test_image, submissions["image"]))
    X_test = np.vstack(X_test_list)
    preds = clf.predict_proba(X_test)




## === cell 7
class_list = mlb.classes_
mask = preds >= thresholds_vec

pred_labels = []
for row_mask, prob_row in zip(mask, preds):
    labels_selected = [cls for cls, keep in zip(class_list, row_mask) if keep]
    if not labels_selected:
        top_cls = class_list[np.argmax(prob_row)]
        labels_selected = [top_cls]
    pred_labels.append(" ".join(labels_selected))

submissions["labels"] = pred_labels




## === cell 8
submissions.to_csv("submission.csv", index=False)




## === cell 9
print(submissions.head())
