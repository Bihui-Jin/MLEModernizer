# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
shap==0.44.1
shapely==2.1.2
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

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

# 5. Code solution

## === cell 0
import os
import cv2
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sns.set(rc={"figure.figsize": (11.7, 8.27)})

_CANDIDATE_BASES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
]
BASE_DIR = next((p for p in _CANDIDATE_BASES if os.path.exists(p)), None)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate dataset base directory in known locations."
    )

TRAIN_CSV = (
    os.path.join(BASE_DIR, "train.csv")
    if os.path.exists(os.path.join(BASE_DIR, "train.csv"))
    else os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train.csv")
)
TEST_CSV = (
    os.path.join(BASE_DIR, "test.csv")
    if os.path.exists(os.path.join(BASE_DIR, "test.csv"))
    else os.path.join(BASE_DIR, "aptos2019-blindness-detection", "test.csv")
)


def _resolve_dir(*parts):
    p = os.path.join(*parts)
    return p if os.path.isdir(p) else None


TRAIN_IMG_DIR = (
    _resolve_dir(BASE_DIR, "train_images")
    or _resolve_dir(BASE_DIR, "aptos2019-blindness-detection", "train_images")
    or _resolve_dir("/kaggle/input/aptos2019-blindness-detection", "train_images")
    or _resolve_dir("/kaggle/data/aptos2019-blindness-detection", "train_images")
)
TEST_IMG_DIR = (
    _resolve_dir(BASE_DIR, "test_images")
    or _resolve_dir(BASE_DIR, "aptos2019-blindness-detection", "test_images")
    or _resolve_dir("/kaggle/input/aptos2019-blindness-detection", "test_images")
    or _resolve_dir("/kaggle/data/aptos2019-blindness-detection", "test_images")
)

if TRAIN_IMG_DIR is None or TEST_IMG_DIR is None:
    raise FileNotFoundError("Could not locate train_images/test_images directories.")


def _read_image_png(img_dir, id_code):
    path = os.path.join(img_dir, f"{id_code}.png")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def _collapse_predictions(pred_labels: np.ndarray) -> np.ndarray:
    pred_labels = pred_labels.astype(int).copy()
    pred_labels[pred_labels >= 3] = 2  # {3,4} -> 2
    return pred_labels




## === cell 1
train = pd.read_csv(TRAIN_CSV)
train_results = []

for _, row in tqdm(train.iterrows(), total=train.shape[0]):
    img = _read_image_png(TRAIN_IMG_DIR, row["id_code"])
    if img is None:
        continue

    height, width, channels = img.shape
    ratio = width / height
    pixel_count = width * height
    gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_cnt = pixel_count - cv2.countNonZero(gray_scaled)
    black_pct = black_cnt / pixel_count
    mean0, mean1, mean2, _ = cv2.mean(img)

    observation = np.array(
        (
            row["diagnosis"],
            height,
            width,
            ratio,
            pixel_count,
            black_cnt,
            black_pct,
            mean0,
            mean1,
            mean2,
        ),
        dtype=np.float32,
    )
    train_results.append(observation)

train_results_df = pd.DataFrame(train_results)
train_results_df.columns = [
    "diagnosis",
    "height",
    "width",
    "ratio",
    "pixel_count",
    "black_cnt",
    "black_pct",
    "mean_c0",
    "mean_c1",
    "mean_c2",
]

train_results_df["diagnosis"] = train_results_df["diagnosis"].astype(int)




## === cell 2
test = pd.read_csv(TEST_CSV)
test_results = []

for _, row in tqdm(test.iterrows(), total=test.shape[0]):
    img = _read_image_png(TEST_IMG_DIR, row["id_code"])
    if img is None:
        continue

    height, width, channels = img.shape
    ratio = width / height
    pixel_count = width * height
    gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_cnt = pixel_count - cv2.countNonZero(gray_scaled)
    black_pct = black_cnt / pixel_count
    mean0, mean1, mean2, _ = cv2.mean(img)

    observation = np.array(
        (
            np.nan,
            height,
            width,
            ratio,
            pixel_count,
            black_cnt,
            black_pct,
            mean0,
            mean1,
            mean2,
        ),
        dtype=np.float32,
    )
    test_results.append(observation)

test_results_df = pd.DataFrame(test_results)
test_results_df.columns = [
    "diagnosis",
    "height",
    "width",
    "ratio",
    "pixel_count",
    "black_cnt",
    "black_pct",
    "mean_c0",
    "mean_c1",
    "mean_c2",
]




## === cell 3
params = {
    "booster": "gbtree",
    "objective": "multi:softprob",
    "eval_metric": "merror",
    "eta": 0.005,
    "max_depth": 10,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "num_class": 5,
    "seed": 1337,
}




## === cell 4
X = train_results_df.drop(columns=["diagnosis"])
y = train_results_df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1337, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

watchlist = [(dtrain, "train"), (dvalid, "valid")]

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100000,
    evals=watchlist,
    early_stopping_rounds=250,
    verbose_eval=100,
)




## === cell 5
valid_raw = np.argmax(bst.predict(dvalid), axis=1)
valid_pred = _collapse_predictions(valid_raw)

pred = pd.DataFrame(valid_pred)
results = pd.concat([y_test.reset_index(drop=True), pred], axis=1)
score = cohen_kappa_score(results.iloc[:, 0], results.iloc[:, 1], weights="quadratic")

print("Validation Kappa (with collapsed preds):", score)




## === cell 6
cm = confusion_matrix(
    y_true=results.iloc[:, 0], y_pred=results.iloc[:, 1], labels=[0, 1, 2, 3, 4]
)
cm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, ax=ax)
ax.set(ylabel="True label", xlabel="Predicted label")




## === cell 7
train["diagnosis"].value_counts() / train.shape[0]




## === cell 8
fig, ax = plt.subplots(figsize=(12, 18))
xgb.plot_importance(bst, importance_type="gain", height=0.8, ax=ax)




## === cell 9
train_results_df.groupby(["ratio", "diagnosis"])["pixel_count"].count()




## === cell 10
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))

img0a = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.000000
                ].index[0],
                0,
            ]
        ),
    )
)
ax[0, 0].imshow(cv2.cvtColor(img0a, cv2.COLOR_BGR2RGB))
ax[0, 0].axis("off")

img0b = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.000000
                ].index[1],
                0,
            ]
        ),
    )
)
ax[1, 0].imshow(cv2.cvtColor(img0b, cv2.COLOR_BGR2RGB))
ax[1, 0].axis("off")

img0c = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.000000
                ].index[2],
                0,
            ]
        ),
    )
)
ax[2, 0].imshow(cv2.cvtColor(img0c, cv2.COLOR_BGR2RGB))
ax[2, 0].axis("off")

img1a = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.333333
                ].index[0],
                0,
            ]
        ),
    )
)
ax[0, 1].imshow(cv2.cvtColor(img1a, cv2.COLOR_BGR2RGB))
ax[0, 1].axis("off")

img1b = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.333333
                ].index[1],
                0,
            ]
        ),
    )
)
ax[1, 1].imshow(cv2.cvtColor(img1b, cv2.COLOR_BGR2RGB))
ax[1, 1].axis("off")

img1c = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.333333
                ].index[2],
                0,
            ]
        ),
    )
)
ax[2, 1].imshow(cv2.cvtColor(img1c, cv2.COLOR_BGR2RGB))
ax[2, 1].axis("off")

img2a = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.505618
                ].index[0],
                0,
            ]
        ),
    )
)
ax[0, 2].imshow(cv2.cvtColor(img2a, cv2.COLOR_BGR2RGB))
ax[0, 2].axis("off")

img2b = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.505618
                ].index[1],
                0,
            ]
        ),
    )
)
ax[1, 2].imshow(cv2.cvtColor(img2b, cv2.COLOR_BGR2RGB))
ax[1, 2].axis("off")

img2c = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.505618
                ].index[2],
                0,
            ]
        ),
    )
)
ax[2, 2].imshow(cv2.cvtColor(img2c, cv2.COLOR_BGR2RGB))
ax[2, 2].axis("off")




## === cell 11
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))

img0a = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.000000
                ].index[0],
                0,
            ]
        ),
    )
)
img0a = cv2.resize(img0a, (224, 224))
ax[0, 0].imshow(cv2.cvtColor(img0a, cv2.COLOR_BGR2RGB))
ax[0, 0].axis("off")

img0b = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.000000
                ].index[1],
                0,
            ]
        ),
    )
)
img0b = cv2.resize(img0b, (224, 224))
ax[1, 0].imshow(cv2.cvtColor(img0b, cv2.COLOR_BGR2RGB))
ax[1, 0].axis("off")

img0c = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.000000
                ].index[2],
                0,
            ]
        ),
    )
)
img0c = cv2.resize(img0c, (224, 224))
ax[2, 0].imshow(cv2.cvtColor(img0c, cv2.COLOR_BGR2RGB))
ax[2, 0].axis("off")

img1a = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.333333
                ].index[0],
                0,
            ]
        ),
    )
)
img1a = cv2.resize(img1a, (224, 224))
ax[0, 1].imshow(cv2.cvtColor(img1a, cv2.COLOR_BGR2RGB))
ax[0, 1].axis("off")

img1b = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.333333
                ].index[1],
                0,
            ]
        ),
    )
)
img1b = cv2.resize(img1b, (224, 224))
ax[1, 1].imshow(cv2.cvtColor(img1b, cv2.COLOR_BGR2RGB))
ax[1, 1].axis("off")

img1c = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.333333
                ].index[2],
                0,
            ]
        ),
    )
)
img1c = cv2.resize(img1c, (224, 224))
ax[2, 1].imshow(cv2.cvtColor(img1c, cv2.COLOR_BGR2RGB))
ax[2, 1].axis("off")

img2a = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.505618
                ].index[0],
                0,
            ]
        ),
    )
)
img2a = cv2.resize(img2a, (224, 224))
ax[0, 2].imshow(cv2.cvtColor(img2a, cv2.COLOR_BGR2RGB))
ax[0, 2].axis("off")

img2b = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.505618
                ].index[1],
                0,
            ]
        ),
    )
)
img2b = cv2.resize(img2b, (224, 224))
ax[1, 2].imshow(cv2.cvtColor(img2b, cv2.COLOR_BGR2RGB))
ax[1, 2].axis("off")

img2c = cv2.imread(
    os.path.join(
        TRAIN_IMG_DIR,
        "{}.png".format(
            train.iloc[
                train_results_df[
                    np.round(train_results_df["ratio"], 6) == 1.505618
                ].index[2],
                0,
            ]
        ),
    )
)
img2c = cv2.resize(img2c, (224, 224))
ax[2, 2].imshow(cv2.cvtColor(img2c, cv2.COLOR_BGR2RGB))
ax[2, 2].axis("off")




## === cell 12
dtest = xgb.DMatrix(test_results_df.drop(columns=["diagnosis"]))

test_raw = np.argmax(bst.predict(dtest), axis=1).astype(int)
test_pred = _collapse_predictions(test_raw)

submission = pd.DataFrame(
    {
        "id_code": test["id_code"].reset_index(drop=True),
        "diagnosis": test_pred.astype(int),
    }
)
submission.to_csv("submission.csv", index=False)
submission.head()
