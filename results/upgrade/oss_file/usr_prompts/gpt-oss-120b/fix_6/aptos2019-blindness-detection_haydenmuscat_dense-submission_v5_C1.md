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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8360317782738592

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from PIL import Image

IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"
TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images/")



## === cell 1
test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["id_code"] = test_df["id_code"].astype(str) + ".png"

train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
class_counts = train_df["diagnosis"].value_counts().sort_index()
class_probs = class_counts.values / class_counts.values.sum()  # fallback probabilities


def simple_load_and_resize(img_path):
    """
    Load an image with Pillow, resize to IMG_DIM×IMG_DIM, and return a NumPy array.
    This avoids a hard dependency on cv2 and keeps the original processing logic untouched.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMG_DIM, IMG_DIM))
        return np.asarray(img, dtype=np.float32)


def extract_features(img_array):
    """
    Return a 2‑dimensional feature vector:
    0 – overall mean brightness,
    1 – mean of the green channel (often informative for retinal images).
    """
    overall_mean = img_array.mean()
    green_mean = img_array[..., 1].mean()  # channel index 1 is Green
    return np.array([overall_mean, green_mean], dtype=np.float32)


train_features = []
train_labels = []

for idx, row in train_df.iterrows():
    img_path = os.path.join(INPUT_FOLDER, "train_images", f"{row['id_code']}.png")
    if not os.path.exists(img_path):
        img_array = np.full((IMG_DIM, IMG_DIM, CHANNEL_SIZE), 128.0, dtype=np.float32)
    else:
        img_array = simple_load_and_resize(img_path)
    train_features.append(extract_features(img_array))
    train_labels.append(row["diagnosis"])

train_features = np.stack(train_features)  # shape (n_samples, 2)
train_labels = np.array(train_labels)

class_centroids = np.zeros((NUM_CLASSES, 2), dtype=np.float32)
for cls in range(NUM_CLASSES):
    cls_feats = train_features[train_labels == cls]
    if cls_feats.shape[0] == 0:
        class_centroids[cls] = train_features.mean(axis=0)
    else:
        class_centroids[cls] = cls_feats.mean(axis=0)

X_tr, X_val, y_tr, y_val = train_test_split(
    train_features, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

model = LogisticRegression(
    multi_class="multinomial",
    max_iter=500,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
)

model.fit(X_tr, y_tr)
val_pred = model.predict(X_val)
val_kappa = cohen_kappa_score(y_val, val_pred, weights="quadratic")
print(f"Validation Quadratic Weighted Kappa: {val_kappa:.5f}")

model.fit(train_features, train_labels)


def predict_by_features(img_array):
    """
    Use the trained LogisticRegression model to predict the class.
    Falls back to centroid nearest‑neighbor if the model fails for any reason.
    """
    try:
        feats = extract_features(img_array).reshape(1, -1)
        return int(model.predict(feats)[0])
    except Exception:
        feats = extract_features(img_array)
        distances = np.linalg.norm(class_centroids - feats, axis=1)
        return int(np.argmin(distances))




## === cell 2
block_size = 500
total = test_df.shape[0]
y_pred_list = np.zeros(total, dtype=int)

for start in range(0, total, block_size):
    gc.collect()
    end = min(start + block_size, total)

    img_batch = np.empty(
        (end - start, IMG_DIM, IMG_DIM, CHANNEL_SIZE), dtype=np.float32
    )
    for i, fname in enumerate(test_df.iloc[start:end]["id_code"]):
        img_path = os.path.join(TEST_IMAGES_DIR, fname)
        if not os.path.exists(img_path):
            img_batch[i] = 128.0  # fallback gray image
        else:
            img_batch[i] = simple_load_and_resize(img_path)

    for i in range(img_batch.shape[0]):
        y_pred_list[start + i] = predict_by_features(img_batch[i])

    print(f"{start} - {end} finished")

output_path = "submission.csv"
test_df["diagnosis"] = y_pred_list
test_df[["id_code", "diagnosis"]].to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
