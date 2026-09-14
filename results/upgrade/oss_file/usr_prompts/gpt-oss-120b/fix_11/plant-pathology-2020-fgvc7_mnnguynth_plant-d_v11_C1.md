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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.91288) has done: 'I keep the original data‑loading and feature‑engineering unchanged, but replace the single‑label classification with a proper multi‑label approach. Four independent binary XGBoost classifiers (one per disease) be trained on the same features, and their probability outputs be used directly for the submission. This aligns the training objective with the mean column‑wise ROC‑AUC metric and should move the score toward the target while still producing a valid `submit.csv`.'
- What this solution (achieved 0.91835) has done: 'I increase the XGBoost hyperparameters modestly so each binary model can learn richer patterns, which should raise the mean ROC‑AUC toward the target 0.753 while keeping the original pipeline unchanged. The only modification is in **cell 24** where `tuned_params` now uses a smaller learning rate, deeper trees, and more estimators.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
from xgboost import (
    XGBClassifier,
)  # switched to classifier for proper class probabilities

train_table = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
test_table = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")



## === cell 1
train_table.head()



## === cell 2
train_table.shape



## === cell 3
train_table.dtypes



## === cell 4
LABELS = ["healthy", "multiple_diseases", "rust", "scab"]
_, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(10, 3))
for ax, column in zip(axes, LABELS):
    train_table[column].value_counts().plot.bar(title=column, ax=ax)
plt.show()



## === cell 5
plt.title("Label distribution")
train_table[LABELS].idxmax(axis=1).value_counts().plot.bar()
plt.show()




## === cell 6
def getSampleToShow(keyColumn, sample):
    list_sample_image = []
    sample_train = train_table[train_table[keyColumn] == 1].sample(
        n=sample, random_state=42
    )
    for image_id in sample_train["image_id"]:
        name = f"../input/plant-pathology-2020-fgvc7/images/{image_id}.jpg"
        image = cv.imread(name)
        im_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        list_sample_image.append(im_rgb)
    return np.array(list_sample_image)


def showImages(images):
    plt.figure(figsize=(15, 15))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.axis("off")


def getImageToTest():
    return cv.imread("../input/plant-pathology-2020-fgvc7/images/Train_382.jpg")




## === cell 7
rust_images = getSampleToShow("rust", 9)
showImages(rust_images)



## === cell 8
scab_images = getSampleToShow("scab", 9)
showImages(scab_images)



## === cell 9
image = getImageToTest()
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()




## === cell 10
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


gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
drivative_image = applyCannyThreshold(gray_image, 12)
plt.imshow(drivative_image, cmap="gray")
plt.axis("off")
plt.show()



## === cell 11
drivative_image.shape




## === cell 12
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


mask_image = zipImage(drivative_image, 8, 8, 0.12)
mask_image = zipImage(mask_image, 16, 16, 0.2)
plt.imshow(mask_image * 255, cmap="gray")
plt.axis("off")
plt.show()




## === cell 13
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


mask_image = joinNeiboorPixel(mask_image, 8, 8, 3, 0.2)
mask_image = joinNeiboorPixel(mask_image, 16, 16, 3, 1 / 3)
plt.imshow(mask_image * 255, cmap="gray")
plt.axis("off")
plt.show()



## === cell 14
mask_image = zipImage(drivative_image, 8, 8, 0.12)
mask_image = zipImage(mask_image, 16, 16, 0.2)
mask_image = joinNeiboorPixel(mask_image, 8, 8, 3, 0.2)
mask_image = joinNeiboorPixel(mask_image, 16, 16, 3, 1 / 3)
for channel in range(3):
    image[:, :, channel] = image[:, :, channel] * mask_image
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()




## === cell 15
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




## === cell 16
def getPathImageById(image_id):
    return f"../input/plant-pathology-2020-fgvc7/images/{image_id}.jpg"


def getFigureForImage(path):
    img = cv.imread(path)
    hist_figure = fd_histogram(img).astype(np.float64)
    hu_moments = fd_hu_moments(img)
    fig = np.concatenate((hist_figure, hu_moments))
    return fig




## === cell 17
def createColsTrainName():
    cols = [f"color_hist_{i}" for i in range(512)]
    cols += [f"hu_moments_{i}" for i in range(7)]
    return cols


def createTrainData(df):
    """
    Build a feature dataframe for the given image table (train or test).
    """
    figs = None
    for img_id in df["image_id"]:
        name = getPathImageById(img_id)
        fig = getFigureForImage(name)
        if figs is None:
            figs = fig.reshape(1, -1)
        else:
            figs = np.vstack((figs, fig))
    cols = createColsTrainName()
    train_data = pd.DataFrame(figs, columns=cols)
    train_data["image_id"] = df["image_id"].reset_index(drop=True)
    return train_data




## === cell 18
train_data = createTrainData(train_table)
train_data.to_csv("./train_data2.csv", index=False)



## === cell 19
test_data = createTrainData(test_table)
test_data.to_csv("./test_data2.csv", index=False)



## === cell 20
X_train = train_data.drop(columns=["image_id"]).values
X_test = test_data.drop(columns=["image_id"]).values



## === cell 21
y_multi = train_table[LABELS]
y_multi.head(2)



## === cell 22
label_to_int = {label: i for i, label in enumerate(LABELS)}



## === cell 23
print(
    "Y_train shape:",
    y_multi.shape,
    "unique classes per column:",
    y_multi.nunique().to_dict(),
)

tuned_params = {"learning_rate": 0.10, "max_depth": 2, "n_estimators": 20}
models = {}
train_auc = {}

for label in LABELS:
    y = y_multi[label].values
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",  # aligns with competition metric
        n_estimators=tuned_params["n_estimators"],
        max_depth=tuned_params["max_depth"],
        learning_rate=tuned_params["learning_rate"],
        nthread=1,
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X_train, y)
    models[label] = model

    prob = model.predict_proba(X_train)[:, 1]
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y, prob)
    train_auc[label] = auc
    print(f"Training ROC‑AUC for {label}: {auc:.4f}")

print(f"Mean training ROC‑AUC: {np.mean(list(train_auc.values())):.4f}")



## === cell 24
test_probs = {}
for label in LABELS:
    model = models[label]
    prob = model.predict_proba(X_test)[:, 1]
    test_probs[label] = prob

test_probs_df = pd.DataFrame(test_probs)



## === cell 25
submission = pd.concat(
    [test_table["image_id"].reset_index(drop=True), test_probs_df[LABELS]], axis=1
)
submission.to_csv("./submit.csv", index=False)
print("Submission file written to ./submit.csv with shape", submission.shape)



## === cell 26
pass



## === cell 27
pass



## === cell 28
pass



## === cell 29
pass



## === cell 30
pass
