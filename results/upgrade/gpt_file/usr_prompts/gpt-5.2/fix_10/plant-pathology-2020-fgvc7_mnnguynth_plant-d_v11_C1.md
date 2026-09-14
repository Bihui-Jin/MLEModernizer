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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.10

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
sklearn-pandas==2.2.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.7531873774606723

# 6. Current score

0.84421

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.92345) has done: 'I first fix the pipeline so it reliably produces a valid submission with the correct filename/columns and aligned row order (your current code can silently misalign labels due to index-based access and also outputs hard one-hot labels, which is suboptimal for ROC AUC). Next, I keep your exact feature extraction logic and still use XGBoost, but switch from `XGBRegressor` with `multi:softmax` (class labels) to the matching classification objective that outputs class probabilities (`multi:softprob`), so the submission contains probabilities as required by the metric. I also correct `return_label()`/`y_array()` to avoid using `train_table` as a global with label-based indexing (can be wrong if indices are not 0..n-1), which can otherwise hurt performance and reproducibility. These are minimal, metric-aligned changes that should move the score upward toward your target by producing probability outputs rather than hard classes.'
- What this solution (achieved 0.89541) has done: 'Your current score (0.92345) is much higher than the target (0.75319), so to move *toward* the target we should slightly reduce model performance while keeping the same feature extraction and the same XGBoost multiclass-probability setup. The smallest, safest way is to add a bit more regularization (shallower trees, higher `min_child_weight`, nonzero `reg_lambda`, small `gamma`, and mild subsampling/colsampling), which tends to reduce overfitting and lower AUC without changing the pipeline’s semantics. I also fix the metric check in-cell (you’re printing an “Accuracy” using `r2_score`, which is misleading) without affecting training or submission. The submission generation stays identical (same columns, same row order from `sample_submission.csv`, probabilities clipped to [0,1], and a `submit.csv` written).'
- What this solution (achieved 0.91803) has done: 'I (1) fix your notebook’s cell numbering so it runs cleanly end-to-end in Kaggle (your current script starts at cell 0, but you asked for cell 1..N), and (2) add a single, metric-aligned calibration step that keeps your exact feature extraction and XGBoost multiclass-probability training intact while producing probabilities that typically score better on mean column-wise ROC AUC. Specifically, after fitting the same `XGBClassifier`, I fit a lightweight one-vs-rest Platt scaling (logistic regression) on the model’s predicted probabilities to better calibrate per-class scores for ROC AUC without changing the underlying model or training loop. The submission continue to be built from `sample_submission.csv` to guarantee correct row order/columns, and it write `./submit.csv` as before.'
- What this solution (achieved 0.92174) has done: 'Your current score (0.91803) is substantially higher than the target (0.75319), so we should *intentionally* move performance downward toward the target band with the smallest safe change. I keep your exact feature extraction + XGBoost multiclass-probability approach and simply remove the Platt-scaling calibration step, which is a known AUC improver in your pipeline and should reduce the score while keeping valid probability outputs. I also keep submission construction from `sample_submission.csv` to preserve correct row order/columns and still write `./submit.csv`. No changes are made to the model type, features, or training loop beyond skipping calibration.'
- What this solution (achieved 0.9078) has done: 'Your current score (0.92174) is well above the target (0.75319), so we should intentionally reduce performance slightly to move closer to the target band while keeping your exact feature extraction and XGBoost multiclass-probability pipeline intact. The smallest, safest lever is to increase regularization a bit and add stronger row/column subsampling (same model type/objective, same training call), which typically reduces AUC without breaking semantics. I keep submission construction from `sample_submission.csv` so the row order/columns remain correct, and still write `./submit.csv`. I also add `verbosity=0` for cleaner logs and keep determinism via `random_state` and `tree_method="hist"`.'
- What this solution (achieved 0.84421) has done: 'You currently don’t get a Kaggle score because the notebook is in a cell format starting at `cell 0`, which can break your run order in the required 1..N format and prevent a submission from being produced reliably. I keep your exact feature extraction (HSV histogram + Hu moments) and your XGBoost multiclass probability setup, but I only make execution-stability fixes: renumber cells to start at 1, add a small deterministic seed block, and keep the submission alignment checks so the CSV is always valid. These changes are minimal and should let you finally generate `./submit.csv`, enabling Kaggle scoring; once you have a current score we can then make tiny parameter nudges to move toward the target band.'

# 9. Code solution

## === cell 0
from math import e
import os
import pickle
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb



## === cell 1
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

BASE1 = "../input/plant-pathology-2020-fgvc7"
BASE2 = "/kaggle/input/plant-pathology-2020-fgvc7"
BASE = BASE1 if os.path.exists(BASE1) else BASE2

test_table = pd.read_csv(f"{BASE}/test.csv")
train_table = pd.read_csv(f"{BASE}/train.csv")

LABELS = ["healthy", "multiple_diseases", "rust", "scab"]



## === cell 2
RUN_EDA = False

if RUN_EDA:
    display(train_table.head())



## === cell 3
if RUN_EDA:
    print(train_table.shape)



## === cell 4
if RUN_EDA:
    print(train_table.dtypes)



## === cell 5
if RUN_EDA:
    print(train_table.index.duplicated().sum())



## === cell 6
if RUN_EDA:
    _, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(10, 3))
    for ax, column in zip(axes, LABELS):
        train_table[column].value_counts().plot.bar(title=column, ax=ax)
    plt.show()



## === cell 7
if RUN_EDA:
    plt.title("Label dist")
    train_table[LABELS].idxmax(axis=1).value_counts().plot.bar()
    plt.show()




## === cell 8
def getSampleToShow(keyColumn, sample):
    list_sample_image = []
    sample_train = train_table[train_table[keyColumn] == 1].sample(
        n=sample, random_state=0
    )
    for image_id in sample_train["image_id"]:
        name = f"{BASE}/images/" + image_id + ".jpg"
        image = cv.imread(name)
        im_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        list_sample_image.append(im_rgb)
    return np.array(list_sample_image)


def showImages(images):
    plt.figure(figsize=(15, 15))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
    plt.show()


def getImageToTest():
    return cv.imread(f"{BASE}/images/Train_382.jpg")




## === cell 9
if RUN_EDA:
    rust_iamges = getSampleToShow("rust", 9)
    showImages(rust_iamges)



## === cell 10
if RUN_EDA:
    scab_images = getSampleToShow("scab", 9)
    showImages(scab_images)



## === cell 11
if RUN_EDA:
    image = getImageToTest()
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show()




## === cell 12
def applyCannyThreshold(frame, val):
    ratio = 1.2
    kernel_size = 3
    low_threshold = val
    img_blur = cv.GaussianBlur(frame, (3, 3), 0)
    detected_edges = cv.Canny(
        img_blur, low_threshold, low_threshold * ratio, kernel_size
    )
    mask = detected_edges != 0
    dst = frame * (mask[:, :].astype(frame.dtype))
    return dst




## === cell 13
if RUN_EDA:
    image = getImageToTest()
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    drivative_image = applyCannyThreshold(gray_image, 12)
    plt.imshow(drivative_image)
    plt.show()



## === cell 14
if RUN_EDA:
    print(drivative_image.shape)




## === cell 15
def zipImage(src, zip_x, zip_y, ratio):
    rs, cs = src.shape
    zip_rs = int(rs / zip_y)
    zip_cs = int(cs / zip_x)

    for idx in range(0, zip_rs * zip_y, zip_y):
        for jdx in range(0, zip_cs * zip_x, zip_x):
            block_img = src[idx : idx + zip_y, jdx : jdx + zip_x]
            num_pixel = np.sum(block_img > 0)
            if num_pixel >= zip_x * zip_y * ratio:
                block_img[:, :] = 1
            else:
                block_img[:, :] = 0

    return src




## === cell 16
if RUN_EDA:
    image = getImageToTest()
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    drivative_image = applyCannyThreshold(gray_image, 12)

    mask_image = zipImage(drivative_image.copy(), 8, 8, 0.12)
    mask_image = zipImage(mask_image, 16, 16, 0.2)

    plt.imshow(mask_image * 255)
    plt.show()




## === cell 17
def joinNeiboorPixel(src, zip_x, zip_y, mask_size, ratio):
    rs, cs = src.shape
    zip_rs = int(rs / zip_y)
    zip_cs = int(cs / zip_x)
    half = int(mask_size / 2)
    dst = src.copy()
    for idx in range(half, zip_rs - half):
        for jdx in range(half, zip_cs - half):
            start_row_mask = (idx - half) * zip_y
            end_row_mask = (idx + half + 1) * zip_y
            start_col_mask = (jdx - half) * zip_x
            end_col_mask = (jdx + half + 1) * zip_x
            mask_block = src[start_row_mask:end_row_mask, start_col_mask:end_col_mask]
            block_img = dst[
                idx * zip_y : (idx + 1) * zip_y, jdx * zip_x : (jdx + 1) * zip_x
            ]
            num_pixel = np.sum(mask_block > 0)
            if num_pixel >= mask_size * mask_size * zip_x * zip_y * ratio:
                block_img[:, :] = 1
    return dst




## === cell 18
if RUN_EDA:
    image = getImageToTest()
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    drivative_image = applyCannyThreshold(gray_image, 12)

    mask_image = zipImage(drivative_image.copy(), 8, 8, 0.12)
    mask_image = zipImage(mask_image, 16, 16, 0.2)
    mask_image = joinNeiboorPixel(mask_image, 8, 8, 3, 0.2)
    mask_image = joinNeiboorPixel(mask_image, 16, 16, 3, 1 / 3)

    plt.imshow(mask_image * 255)
    plt.show()



## === cell 19
if RUN_EDA:
    image = getImageToTest()
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    drivative_image = applyCannyThreshold(gray_image, 12)

    mask_image = zipImage(drivative_image.copy(), 8, 8, 0.12)
    mask_image = zipImage(mask_image, 16, 16, 0.2)
    mask_image = joinNeiboorPixel(mask_image, 8, 8, 3, 0.2)
    mask_image = joinNeiboorPixel(mask_image, 16, 16, 3, 1 / 3)
    for chanel in range(0, 3):
        image[:, :, chanel] = image[:, :, chanel] * mask_image

    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show()




## === cell 20
def fd_histogram(image, mask=None):
    bins = 8
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist(
        [image], [0, 1, 2], None, [bins, bins, bins], [1, 256, 1, 256, 1, 256]
    )
    cv.normalize(hist, hist)
    return hist.flatten()


def fd_hu_moments(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    feature = cv.HuMoments(cv.moments(image)).flatten()
    return feature




## === cell 21
def getPathImageById(image_id):
    return f"{BASE}/images/" + image_id + ".jpg"


def getFigureForImage(path):
    img = cv.imread(path)
    hist_figure = fd_histogram(img).astype(np.float64)
    hu_monents = fd_hu_moments(img)
    fig = np.concatenate((hist_figure, hu_monents))
    return fig




## === cell 22
def createColsTrainName():
    cols = []
    for i in range(0, 512):
        cols.append("color_hist_" + str(i))

    for i in range(0, 7):
        cols.append("hu_moents_" + str(i))

    return cols


def createTrainData(table_):
    series = table_["image_id"]
    figs = None
    for id in series:
        name = getPathImageById(id)
        fig = getFigureForImage(name)
        if figs is None:
            figs = fig
        else:
            figs = np.vstack((figs, fig))

    cols = createColsTrainName()
    data = pd.DataFrame(figs, columns=cols)
    data["image_id"] = table_["image_id"].values
    return data




## === cell 23
train_data = createTrainData(train_table)
train_data.to_csv("./train_data2.csv", index=False)



## === cell 24
test_data = createTrainData(test_table)
test_data.to_csv("./test_data2.csv", index=False)



## === cell 25
X_train = train_data.drop(columns=["image_id"]).values
X_test = test_data.drop(columns=["image_id"]).values




## === cell 26
def y_array(train_table_):
    return (
        train_table_[["healthy", "multiple_diseases", "rust", "scab"]]
        .values.argmax(axis=1)
        .astype(int)
    )


Y_train = y_array(train_table)
print("Y_train sample:", Y_train[:10])



## === cell 27
tuned_params = {"learning_rate": 0.18, "max_depth": 1, "n_estimators": 35}

xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    n_estimators=tuned_params["n_estimators"],
    max_depth=tuned_params["max_depth"],
    learning_rate=tuned_params["learning_rate"],
    n_jobs=1,
    eval_metric="mlogloss",
    random_state=0,
    subsample=0.40,
    colsample_bytree=0.40,
    min_child_weight=35.0,
    reg_lambda=35.0,
    reg_alpha=5.0,
    gamma=5.0,
    tree_method="hist",
    verbosity=0,
)

xgb_model.fit(X_train, Y_train)
pickle.dump(xgb_model, open("xgbModel.pkl", "wb"))
print("Saved model to: xgbModel.pkl")



## === cell 28
pred_proba_train = xgb_model.predict_proba(X_train)



## === cell 29
from sklearn.metrics import accuracy_score

pred_y_train = pred_proba_train.argmax(axis=1)
print("Train accuracy (sanity check only):", accuracy_score(Y_train, pred_y_train))



## === cell 30
pred_proba_test = xgb_model.predict_proba(X_test)
print("pred_proba_test shape:", pred_proba_test.shape)



## === cell 31
submit = pd.read_csv(f"{BASE}/sample_submission.csv")
print("sample_submission shape:", submit.shape)
print("sample_submission columns:", list(submit.columns))



## === cell 32
if not np.array_equal(test_data["image_id"].values, test_table["image_id"].values):
    raise RuntimeError(
        "Internal mismatch: test_data image_id order differs from test_table."
    )

pred_df = pd.DataFrame(pred_proba_test, columns=LABELS)
pred_df["image_id"] = test_table["image_id"].values

df = submit[["image_id"]].merge(
    pred_df, on="image_id", how="left", validate="one_to_one"
)

if df[LABELS].isna().any().any():
    missing = df.loc[df[LABELS].isna().any(axis=1), "image_id"].tolist()[:10]
    raise RuntimeError(
        f"Missing predictions for some image_id (showing up to 10): {missing}"
    )

df[LABELS] = df[LABELS].astype(float).clip(0.0, 1.0)
print(df.head(2))



## === cell 33
df.to_csv("./submit.csv", index=False)
print("Wrote submission to ./submit.csv with shape:", df.shape)
print("Columns:", list(df.columns))
