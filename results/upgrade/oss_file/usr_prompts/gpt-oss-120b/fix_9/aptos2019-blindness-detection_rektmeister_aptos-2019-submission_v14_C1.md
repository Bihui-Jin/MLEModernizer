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

0.8348565227449207

# 6. Current score

0.74233

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix wraps TensorFlow‑related imports in a safe try/except, provides a fallback that skips all model‑training code when TensorFlow isn’t available, and generates predictions using a simple “most common class” heuristic. This removes the protobuf‑related crash, ensures a valid `submission.csv` is written, and keeps the original pipeline structure intact for environments where TensorFlow does work.'
- What this solution (achieved 0.0) has done: 'I add a missing sklearn import and replace the simple “most‑common” fallback with a lightweight RandomForest that uses basic image statistics (mean ± std per channel). This keeps the original pipeline when TensorFlow works, but when it does not (the case here) it trains a quick model on the training images and generates realistic predictions, which should raise the Quadratic Weighted Kappa from 0.0 toward the target score.'
- What this solution (achieved 0.0) has done: 'I adjust the fallback image‑feature extraction to use the correct absolute input paths and give the RandomForest a stronger configuration (more trees). This fixes the path‑related bug that caused all images to be read as empty arrays, and the richer forest should lift the Quadratic Weighted Kappa score toward the target while keeping the original pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'The fix updates the fallback image‑loading paths so the script can actually read the train and test images when TensorFlow is unavailable. By using the correct relative input directory (`../input/aptos2019-blindness-detection`) we ensure feature extraction works, allowing the RandomForest to train on real image statistics and produce a meaningful submission instead of a zero‑score result.'
- What this solution (achieved 0.0) has done: 'I improve the fallback feature extraction to use richer statistics (mean, std, median per color channel) and make the RandomForest classifier a bit stronger with more trees and balanced class weighting. These changes keep the overall pipeline unchanged, fix the zero‑score issue by giving the model better data, and still produce a valid `submission.csv`.'
- What this solution (achieved 0.67259) has done: 'Implemented a robust fallback that avoids TensorFlow imports, correctly loads image files, enriches image statistics (mean, std, median, min, max), trains a stronger RandomForest, and evaluates a validation split with Quadratic Weighted Kappa to ensure the model is learning. The script now always produces a valid `submission.csv` and includes diagnostic output of the internal validation score. This fixes the import crash and improves predictive performance toward the target metric.'
- What this solution (achieved 0.73503) has done: 'Implemented a modest yet effective tweak to the fallback RandomForest pipeline: increased the number of trees to capture more patterns and switched prediction to a probability‑based expected value rounded to the nearest class. This change aligns the model’s output more closely with the quadratic weighted kappa metric, raising validation performance toward the target without altering the overall architecture or training flow. The script now writes a proper `submission.csv` after these adjustments.'
- What this solution (achieved 0.74233) has done: 'Implemented a modest feature‑enhancement and a slight increase in forest size. The new `bright_ratio` (fraction of bright pixels) is added to the image statistics, giving the model a richer signal while keeping the original RandomForest pipeline unchanged. The number of trees is raised from 1500 to 2000 to capture more patterns, which together are expected to lift the Quadratic Weighted Kappa toward the target score.'

# 9. Code solution

## === cell 0
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score




## === cell 1
TRAINING = False




## === cell 2
train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])




## === cell 3
train["id_code"] = train["id_code"].apply(lambda x: str(x) + ".png")
test["id_code"] = test["id_code"].apply(lambda x: str(x) + ".png")
train["diagnosis"] = train["diagnosis"].astype(str)

label_cols = ["lbl_0", "lbl_1", "lbl_2", "lbl_3", "lbl_4"]
label_mat = np.zeros((train.shape[0], len(label_cols)), dtype=np.int32)

for i in range(train.shape[0]):
    for j in range(int(train["diagnosis"][i]) + 1):
        label_mat[i, j] = 1

train = pd.concat([train, pd.DataFrame(label_mat, columns=label_cols)], axis=1)

print(train.head(10))




## === cell 4
IMG_DATA_GEN_AVAILABLE = False




## === cell 5
import cv2

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5  # 0, 1, 2, 3, 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1


def crop_image(img, tol=10):
    def crop_image_1(img):
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    if img.ndim == 2:
        return crop_image_1(img)
    elif img.ndim == 3:
        try:
            img_cpy = img.copy()
            h, w, _ = img.shape
            img1 = cv2.resize(crop_image_1(img[:, :, 0]), (w, h))
            img2 = cv2.resize(crop_image_1(img[:, :, 1]), (w, h))
            img3 = cv2.resize(crop_image_1(img[:, :, 2]), (w, h))
            img[:, :, 0] = img1
            img[:, :, 1] = img2
            img[:, :, 2] = img3
        except:
            return img_cpy
        return img


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE / 10), -4, 128)
    return img




## === cell 6
TF_AVAILABLE = False




## === cell 7
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)




## === cell 8
def get_resnet50(input_shape, nb_out):
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.resnet50 import ResNet50

    inputs = Input(shape=input_shape)
    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(nb_out, activation="softmax", name="final_output")(x)
    return Model(inputs, output)


def get_densenet121(input_shape, nb_out):
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.densenet import DenseNet121

    inputs = Input(shape=input_shape)
    base_model = DenseNet121(weights=None, include_top=False, input_tensor=inputs)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    output = Dense(nb_out, activation="sigmoid")(x)
    return Model(inputs, output)




## === cell 9
def get_model(name, input_shape, nb_out):
    if not TF_AVAILABLE:
        print("TensorFlow unavailable – returning None for model.")
        return None
    models = {
        "resnet50": get_resnet50,
        "conv1": get_conv1,
        "conv2": get_conv2,
        "densenet121": get_densenet121,
    }
    if name not in models:
        print(f"No model named '{name}'")
        return None
    model = models[name](input_shape, nb_out)
    weights_path = os.path.join(
        "../input/aptos-2019-densenet121-weights/", f"{name}_weights.hdf5"
    )
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
        print(f"loaded model weights from {weights_path}")
    return model




## === cell 10
def train_model(name, input_shape, nb_out, train_generator, val_generator):
    model = get_model(name, input_shape, nb_out)
    if model is None:
        print("Model not created – skipping training.")
        return




## === cell 11
if TRAINING:
    train_model(
        "densenet121",
        (IMG_SIZE, IMG_SIZE, NB_CHANNELS),
        NB_CLASSES,
        None,
        None,
    )




## === cell 12
if TF_AVAILABLE and False:
    pass
else:
    base_input = os.path.abspath("../input/aptos2019-blindness-detection")
    train_dir = os.path.join(base_input, "train_images")
    test_dir = os.path.join(base_input, "test_images")

    def extract_features(df, img_dir):
        """Extract richer statistical features from each image, including a bright‑pixel ratio."""
        feats = []
        for fname in df["id_code"]:
            path = os.path.join(img_dir, fname)
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            mean = img.mean(axis=(0, 1))
            std = img.std(axis=(0, 1))
            median = np.median(img, axis=(0, 1))
            img_min = img.min(axis=(0, 1))
            img_max = img.max(axis=(0, 1))
            bright_mask = img > 200
            bright_ratio = bright_mask.mean()  # scalar in [0,1]
            feats.append(
                np.concatenate([mean, std, median, img_min, img_max, [bright_ratio]])
            )
        return np.array(feats, dtype=np.float32)

    X_all = extract_features(train, train_dir)
    y_all = train["diagnosis"].astype(int).values
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    clf = RandomForestClassifier(
        n_estimators=2000,  # increased from 1500 for a bit more capacity
        max_depth=None,
        max_features="sqrt",
        n_jobs=4,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)

    val_proba = clf.predict_proba(X_val)
    val_exp = np.rint(np.dot(val_proba, np.arange(NB_CLASSES))).astype(int)
    val_exp = np.clip(val_exp, 0, NB_CLASSES - 1)
    qwk = cohen_kappa_score(y_val, val_exp, weights="quadratic")
    print(f"Local validation Quadratic Weighted Kappa: {qwk:.5f}")

    clf.fit(X_all, y_all)

    X_test = extract_features(test, test_dir)
    test_proba = clf.predict_proba(X_test)
    test_exp = np.rint(np.dot(test_proba, np.arange(NB_CLASSES))).astype(int)
    test_exp = np.clip(test_exp, 0, NB_CLASSES - 1)

    ids = test["id_code"].str.replace(".png", "", regex=False)
    results = pd.DataFrame({"id_code": ids, "diagnosis": test_exp})

results.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written with", results.shape[0], "rows.")
