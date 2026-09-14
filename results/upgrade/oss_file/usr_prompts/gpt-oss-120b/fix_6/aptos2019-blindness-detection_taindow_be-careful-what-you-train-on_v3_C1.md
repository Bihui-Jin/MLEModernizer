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

0.64919

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.71887) has done: 'I change the XGBoost parameters to use the CPU‑based “hist” tree method (the original GPU setting crashes on systems without a GPU) and replace the problematic image‑visualisation cells with harmless placeholders so the notebook runs from start to finish and writes a proper `submission.csv`. This fixes the runtime errors while keeping the original feature engineering and modeling logic unchanged, allowing a valid submission and a score that can be evaluated toward the target.'
- What this solution (achieved 0.65057) has done: 'I slightly weaken the XGBoost model (reduce tree depth) and use a different random seed for the train/validation split. This keeps the overall pipeline unchanged but should lower the validation quadratic weighted kappa, moving the score from the current 0.71887 toward the target range around 0.46.'
- What this solution (achieved 0.66056) has done: 'I decrease the model capacity so the validation quadratic weighted kappa moves down toward the target (≈0.46). Specifically, I lower `max_depth` from 4 to 2 and add modest regularisation by reducing `subsample` and `colsample_bytree` to 0.8. These minimal tweaks keep the overall pipeline unchanged while making the model slightly less expressive, which should lower the score into the desired range.'
- What this solution (achieved 0.63807) has done: 'I slightly increase regularisation and reduce model complexity by setting `max_depth` to 1 and lowering both `subsample` and `colsample_bytree` to 0.6 (and add an L2 term). These minimal tweaks keep the overall pipeline unchanged while making the XGBoost model less expressive, which should lower the validation quadratic weighted kappa from 0.66 toward the target 0.463  without drastic changes.'
- What this solution (achieved 0.64919) has done: 'I increase regularisation and make the train/validation split slightly harder so the validation quadratic weighted kappa drops closer to the target 0.463 while keeping the overall pipeline unchanged. Specifically, I lower `subsample` and `colsample_bytree` to 0.4, raise the L2 term `lambda` to 2.0, and change the split’s `random_state` to 7. These minimal tweaks reduce model capacity and make validation more challenging, which should move the score down toward the desired range.'

# 9. Code solution

## === cell 0
import cv2
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sns.set(rc={"figure.figsize": (11.7, 8.27)})




## === cell 1
train = pd.read_csv("../input/train.csv")
train_results = []

for index, row in tqdm(train.iterrows(), total=train.shape[0]):
    img = cv2.imread("../input//train_images/{}.png".format(row["id_code"]))

    height, width, channels = img.shape
    ratio = width / height
    pixel_count = width * height
    gray_scaled = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        )
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




## === cell 2
test = pd.read_csv("../input/test.csv")
test_results = []

for index, row in tqdm(test.iterrows(), total=test.shape[0]):
    img = cv2.imread("../input//test_images/{}.png".format(row["id_code"]))

    height, width, channels = img.shape
    ratio = width / height
    pixel_count = width * height
    gray_scaled = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        )
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
    "eval_metric": "mlogloss",
    "eta": 0.005,
    "max_depth": 1,
    "subsample": 0.4,  # stronger row subsampling
    "colsample_bytree": 0.4,  # stronger column subsampling
    "lambda": 2.0,  # increased L2 regularisation
    "tree_method": "hist",
    "num_class": 5,
}




## === cell 4
X = train_results_df.drop(columns=["diagnosis"])
y = train_results_df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

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
pred = pd.DataFrame(np.argmax(bst.predict(dvalid), axis=1))
results = pd.concat([y_test.reset_index(drop=True), pred], axis=1)
score = cohen_kappa_score(results.iloc[:, 0], results.iloc[:, 1], weights="quadratic")
print("Validation Kappa:", score)




## === cell 6
cm = confusion_matrix(y_true=results.iloc[:, 0], y_pred=results.iloc[:, 1])
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True)
ax.set(ylabel="True label", xlabel="Predicted label")




## === cell 7
train["diagnosis"].value_counts() / train.shape[0]




## === cell 8
fig, ax = plt.subplots(figsize=(12, 18))
xgb.plot_importance(bst, importance_type="gain", height=0.8, ax=ax)




## === cell 9
pass




## === cell 10
pass




## === cell 11
dtest = xgb.DMatrix(test_results_df.drop(columns=["diagnosis"]))
test_pred = pd.DataFrame(np.argmax(bst.predict(dtest), axis=1))
submission = pd.concat([test["id_code"], test_pred], axis=1)
submission.columns = ["id_code", "diagnosis"]
submission.to_csv("submission.csv", index=False)
submission.head()




## === cell 12
import os

print("Submission saved to:", os.path.abspath("submission.csv"))
