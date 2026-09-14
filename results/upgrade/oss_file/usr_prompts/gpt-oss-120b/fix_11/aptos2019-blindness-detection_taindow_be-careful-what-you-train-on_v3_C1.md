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

# 5. Target score

0.463379

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.64979) has done: 'I added a lightweight custom evaluation metric that directly computes the quadratic weighted kappa on the validation set and tells XGBoost to maximize it. This aligns early‑stopping with the competition metric, which should move the validation score toward the target without altering the core modeling pipeline. The change is limited to the training cell and keeps all other logic unchanged.'
- What this solution (achieved 0.65377) has done: 'I slightly reduce the model’s training length so the validation kappa drops closer to the target (since a higher score is already better than needed). Specifically, I change the boost round count from 5000 to 500 and shorten early‑stopping from 250 to 50. This keeps the core modeling pipeline unchanged while modestly lowering performance toward the desired range.'

# 9. Code solution

## === cell 0
import cv2
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm  # use CPU tqdm compatible import
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import os
import concurrent.futures

sns.set(rc={"figure.figsize": (11.7, 8.27)})




## === cell 1
train = pd.read_csv("../input/train.csv")
feature_path = "train_features.csv"

if os.path.exists(feature_path):
    train_results_df = pd.read_csv(feature_path)
else:

    def process_train(row_tuple):
        idx, row = row_tuple  # row is a pandas Series
        img_path = f'../input/train_images/{row["id_code"]}.png'
        img = cv2.imread(img_path)
        if img is None:
            return None
        height, width, _ = img.shape
        ratio = width / height
        pixel_count = width * height
        gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black_cnt = pixel_count - cv2.countNonZero(gray_scaled)
        black_pct = black_cnt / pixel_count
        mean0, mean1, mean2, _ = cv2.mean(img)
        return np.array(
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
            )
        )

    observations = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for obs in tqdm(
            executor.map(process_train, train.iterrows()), total=len(train)
        ):
            if obs is not None:
                observations.append(obs)

    train_results_df = pd.DataFrame(
        observations,
        columns=[
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
        ],
    )
    train_results_df.to_csv(feature_path, index=False)




## === cell 2
test = pd.read_csv("../input/test.csv")
test_feature_path = "test_features.csv"

if os.path.exists(test_feature_path):
    test_results_df = pd.read_csv(test_feature_path)
else:

    def process_test(row_tuple):
        idx, row = row_tuple
        img_path = f'../input/test_images/{row["id_code"]}.png'
        img = cv2.imread(img_path)
        if img is None:
            return None
        height, width, _ = img.shape
        ratio = width / height
        pixel_count = width * height
        gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black_cnt = pixel_count - cv2.countNonZero(gray_scaled)
        black_pct = black_cnt / pixel_count
        mean0, mean1, mean2, _ = cv2.mean(img)
        return np.array(
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
            )
        )

    observations_test = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for obs in tqdm(executor.map(process_test, test.iterrows()), total=len(test)):
            if obs is not None:
                observations_test.append(obs)

    test_results_df = pd.DataFrame(
        observations_test,
        columns=[
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
        ],
    )
    test_results_df.to_csv(test_feature_path, index=False)




## === cell 3
params = {
    "booster": "gbtree",
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "eta": 0.005,
    "max_depth": 4,
    "min_child_weight": 20,  # increased (harder splits)
    "lambda": 5.0,  # stronger L2 regularisation
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "num_class": 5,
    "seed": 1337,
}




## === cell 4
X = train_results_df.drop(columns=["diagnosis"])
y = train_results_df["diagnosis"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=1337, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

watchlist = [(dtrain, "train"), (dvalid, "valid")]


def kappa_eval(preds, dmatrix):
    labels = dmatrix.get_label()
    preds = np.argmax(preds, axis=1)
    kappa = cohen_kappa_score(labels, preds, weights="quadratic")
    return "kappa", kappa


bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=10,
    evals=watchlist,
    feval=kappa_eval,
    maximize=True,
    early_stopping_rounds=3,  # earlier stop to avoid over‑fitting
    verbose_eval=100,
)




## === cell 5
pred = pd.DataFrame(np.argmax(bst.predict(dvalid), axis=1), columns=["pred"])
results = pd.concat([y_valid.reset_index(drop=True), pred], axis=1)
score = cohen_kappa_score(results["diagnosis"], results["pred"], weights="quadratic")
print("Validation Kappa:", score)




## === cell 6
cm = confusion_matrix(y_true=results["diagnosis"], y_pred=results["pred"])
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set(ylabel="True label", xlabel="Predicted label")
plt.show()




## === cell 7
train["diagnosis"].value_counts(normalize=True)




## === cell 8
fig, ax = plt.subplots(figsize=(12, 18))
xgb.plot_importance(bst, importance_type="gain", height=0.8, ax=ax)
plt.show()




## === cell 9
train_results_df.groupby(["ratio", "diagnosis"])["pixel_count"].count()




## === cell 10
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))


def load_resized(idx):
    img_path = f'../input/train_images/{train.iloc[idx]["id_code"]}.png'
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.zeros((224, 224, 3), dtype=np.uint8)


ratios = [1.0, 1.333333, 1.505618]
for col, r in enumerate(ratios):
    ids = train_results_df[np.round(train_results_df["ratio"], 6) == r].index[:3]
    for row, idx in enumerate(ids):
        ax[row, col].imshow(load_resized(idx))
        ax[row, col].axis("off")
plt.show()




## === cell 11
dtest = xgb.DMatrix(test_results_df.drop(columns=["diagnosis"]))
test_pred = pd.DataFrame(np.argmax(bst.predict(dtest), axis=1), columns=["diagnosis"])
submission = pd.concat([test["id_code"], test_pred], axis=1)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
submission.head()
