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

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
test_table = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
train_table = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3199283155.py in <cell line: 0>()
----> 1 test_table = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
      2 train_table = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
      3 

NameError: name 'pd' is not defined

## === cell 1
train_table.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1418230925.py in <cell line: 0>()
----> 1 train_table.head()
      2 

NameError: name 'train_table' is not defined

## === cell 2
train_table.shape



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/321006560.py in <cell line: 0>()
----> 1 train_table.shape
      2 

NameError: name 'train_table' is not defined

## === cell 3
train_table.dtypes



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2128360668.py in <cell line: 0>()
----> 1 train_table.dtypes
      2 

NameError: name 'train_table' is not defined

## === cell 4
train_table.index.duplicated().sum()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3767144029.py in <cell line: 0>()
----> 1 train_table.index.duplicated().sum()
      2 

NameError: name 'train_table' is not defined

## === cell 5
LABELS = ["healthy", "multiple_diseases", "rust", "scab"]
_, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(10, 3))
for ax, column in zip(axes, LABELS):
    train_table[column].value_counts().plot.bar(title=column, ax=ax)
plt.show()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3073644634.py in <cell line: 0>()
      1 LABELS = ["healthy", "multiple_diseases", "rust", "scab"]
----> 2 _, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(10, 3))
      3 for ax, column in zip(axes, LABELS):
      4     train_table[column].value_counts().plot.bar(title=column, ax=ax)
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 6
plt.title("Label dist")
train_table[LABELS].idxmax(axis=1).value_counts().plot.bar()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/507606733.py in <cell line: 0>()
----> 1 plt.title("Label dist")
      2 train_table[LABELS].idxmax(axis=1).value_counts().plot.bar()
      3 
      4 

NameError: name 'plt' is not defined

## === cell 7
def getSampleToShow(keyColumn, sample):
    list_sample_image = []
    sample_train = train_table[train_table[keyColumn] == 1].sample(n=sample)
    for image_id in sample_train["image_id"]:
        name = "../input/plant-pathology-2020-fgvc7/images/" + image_id + ".jpg"
        image = cv.imread(name)
        im_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        list_sample_image.append(im_rgb)
    return np.array(list_sample_image)


def showImages(images):
    plt.figure(figsize=(15, 15))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])


def getImageToTest():
    return cv.imread("../input/plant-pathology-2020-fgvc7/images/Train_382.jpg")




## === cell 8
rust_iamges = getSampleToShow("rust", 9)
showImages(rust_iamges)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3814345942.py in <cell line: 0>()
----> 1 rust_iamges = getSampleToShow("rust", 9)
      2 showImages(rust_iamges)
      3 

/tmp/ipykernel_55/1321551268.py in getSampleToShow(keyColumn, sample)
      1 def getSampleToShow(keyColumn, sample):
      2     list_sample_image = []
----> 3     sample_train = train_table[train_table[keyColumn] == 1].sample(n=sample)
      4     for image_id in sample_train["image_id"]:
      5         name = "../input/plant-pathology-2020-fgvc7/images/" + image_id + ".jpg"

NameError: name 'train_table' is not defined

## === cell 9
rust_iamges = getSampleToShow("scab", 9)
showImages(rust_iamges)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2168589682.py in <cell line: 0>()
----> 1 rust_iamges = getSampleToShow("scab", 9)
      2 showImages(rust_iamges)
      3 

/tmp/ipykernel_55/1321551268.py in getSampleToShow(keyColumn, sample)
      1 def getSampleToShow(keyColumn, sample):
      2     list_sample_image = []
----> 3     sample_train = train_table[train_table[keyColumn] == 1].sample(n=sample)
      4     for image_id in sample_train["image_id"]:
      5         name = "../input/plant-pathology-2020-fgvc7/images/" + image_id + ".jpg"

NameError: name 'train_table' is not defined

## === cell 10
image = getImageToTest()
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3693673556.py in <cell line: 0>()
----> 1 image = getImageToTest()
      2 plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
      3 
      4 

/tmp/ipykernel_55/1321551268.py in getImageToTest()
     18 
     19 def getImageToTest():
---> 20     return cv.imread("../input/plant-pathology-2020-fgvc7/images/Train_382.jpg")
     21 
     22 

NameError: name 'cv' is not defined

## === cell 11
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
plt.imshow(drivative_image)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/527001731.py in <cell line: 0>()
     12 
     13 
---> 14 gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
     15 drivative_image = applyCannyThreshold(gray_image, 12)
     16 plt.imshow(drivative_image)

NameError: name 'cv' is not defined

## === cell 12
drivative_image.shape




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3850574570.py in <cell line: 0>()
----> 1 drivative_image.shape
      2 
      3 

NameError: name 'drivative_image' is not defined

## === cell 13
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

plt.imshow(mask_image * 255)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2936276855.py in <cell line: 0>()
     16 
     17 
---> 18 mask_image = zipImage(drivative_image, 8, 8, 0.12)
     19 mask_image = zipImage(mask_image, 16, 16, 0.2)
     20 

NameError: name 'drivative_image' is not defined

## === cell 14
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

plt.imshow(mask_image * 255)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3543590219.py in <cell line: 0>()
     21 
     22 
---> 23 mask_image = joinNeiboorPixel(mask_image, 8, 8, 3, 0.2)
     24 mask_image = joinNeiboorPixel(mask_image, 16, 16, 3, 1 / 3)
     25 

NameError: name 'mask_image' is not defined

## === cell 15
mask_image = zipImage(drivative_image, 8, 8, 0.12)
mask_image = zipImage(mask_image, 16, 16, 0.2)
mask_image = joinNeiboorPixel(mask_image, 8, 8, 3, 0.2)
mask_image = joinNeiboorPixel(mask_image, 16, 16, 3, 1 / 3)
for chanel in range(0, 3):
    image[:, :, chanel] = image[:, :, chanel] * mask_image

plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2221100198.py in <cell line: 0>()
----> 1 mask_image = zipImage(drivative_image, 8, 8, 0.12)
      2 mask_image = zipImage(mask_image, 16, 16, 0.2)
      3 mask_image = joinNeiboorPixel(mask_image, 8, 8, 3, 0.2)
      4 mask_image = joinNeiboorPixel(mask_image, 16, 16, 3, 1 / 3)
      5 for chanel in range(0, 3):

NameError: name 'drivative_image' is not defined

## === cell 16
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




## === cell 17
def getPathImageById(image_id):
    return "../input/plant-pathology-2020-fgvc7/images/" + image_id + ".jpg"


def getFigureForImage(path):
    img = cv.imread(path)
    hist_figure = fd_histogram(img).astype(np.float64)
    hu_monents = fd_hu_moments(img)
    fig = np.concatenate((hist_figure, hu_monents))
    return fig




## === cell 18
def createColsTrainName():
    cols = []
    for i in range(0, 512):
        cols.append("color_hist_" + str(i))

    for i in range(0, 7):
        cols.append("hu_moents_" + str(i))

    return cols


def createTrainData(df):
    """
    Build a feature dataframe for the given image table (train or test).
    The function now strictly uses the passed dataframe `df` and does not rely
    on the global `train_table`, preventing mismatched row counts.
    """
    series = df["image_id"]
    figs = None
    for img_id in series:
        name = getPathImageById(img_id)
        fig = getFigureForImage(name)
        if figs is None:
            figs = fig
        else:
            figs = np.vstack((figs, fig))

    cols = createColsTrainName()
    train_data = pd.DataFrame(figs, columns=cols)
    train_data["image_id"] = df["image_id"].reset_index(drop=True)
    return train_data




## === cell 19
train_data = createTrainData(train_table)
train_data.to_csv("./train_data2.csv", index=False)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3133665308.py in <cell line: 0>()
----> 1 train_data = createTrainData(train_table)
      2 train_data.to_csv("./train_data2.csv", index=False)
      3 

NameError: name 'train_table' is not defined

## === cell 20
test_data = createTrainData(test_table)
test_data.to_csv("./test_data2.csv", index=False)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4257638259.py in <cell line: 0>()
----> 1 test_data = createTrainData(test_table)
      2 test_data.to_csv("./test_data2.csv", index=False)
      3 

NameError: name 'test_table' is not defined

## === cell 21
X_train = train_data.drop(columns=["image_id"]).values
X_test = test_data.drop(columns=["image_id"]).values



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2474561431.py in <cell line: 0>()
----> 1 X_train = train_data.drop(columns=["image_id"]).values
      2 X_test = test_data.drop(columns=["image_id"]).values
      3 

NameError: name 'train_data' is not defined

## === cell 22
y = train_table[["healthy", "multiple_diseases", "rust", "scab"]]
y.head(2)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1347265710.py in <cell line: 0>()
----> 1 y = train_table[["healthy", "multiple_diseases", "rust", "scab"]]
      2 y.head(2)
      3 
      4 

NameError: name 'train_table' is not defined

## === cell 23
def return_label(_id):
    d = ["healthy", "multiple_diseases", "rust", "scab"]
    for i in range(4):
        if train_table[d[i]][_id]:
            return i


def y_array(train_table_):
    Y_train_ = train_table_[["healthy", "multiple_diseases", "rust", "scab"]]
    Y_train_ = pd.Series(Y_train_.index.values).apply(return_label)
    return Y_train_.values




## === cell 24
Y_train = y_array(train_table)
print(type(Y_train))
Y_train



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2959411715.py in <cell line: 0>()
----> 1 Y_train = y_array(train_table)
      2 print(type(Y_train))
      3 Y_train
      4 

NameError: name 'train_table' is not defined

## === cell 25
tuned_params = {"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}
xgb_model = XGBRegressor(
    objective="multi:softprob",  # produce class probabilities
    num_class=4,
    n_estimators=tuned_params["n_estimators"],
    max_depth=tuned_params["max_depth"],
    learning_rate=tuned_params["learning_rate"],
    nthread=1,
    eval_metric="mlogloss",  # stable evaluation metric
    random_state=42,
)
xgb_model.fit(X_train, Y_train)
pickle.dump(xgb_model, open("xgbModel.pkl", "wb"))
print("Saved model to: xgbModel.pkl")



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2692144181.py in <cell line: 0>()
      1 # Slightly increase model capacity to improve ROC‑AUC while keeping the same architecture.
      2 tuned_params = {"learning_rate": 0.05, "max_depth": 6, "n_estimators": 200}
----> 3 xgb_model = XGBRegressor(
      4     objective="multi:softprob",  # produce class probabilities
      5     num_class=4,

NameError: name 'XGBRegressor' is not defined

## === cell 26
pred_y = xgb_model.predict(X_train)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1520650909.py in <cell line: 0>()
----> 1 pred_y = xgb_model.predict(X_train)
      2 

NameError: name 'xgb_model' is not defined

## === cell 27
from sklearn.metrics import accuracy_score, roc_auc_score

train_pred_labels = pred_y.argmax(axis=1)
print("Training accuracy:", accuracy_score(Y_train, train_pred_labels))

roc_auc_vals = []
for i, label in enumerate(LABELS):
    true_binary = (Y_train == i).astype(int)
    prob = pred_y[:, i]
    roc_auc = roc_auc_score(true_binary, prob)
    roc_auc_vals.append(roc_auc)
    print(f"ROC‑AUC for {label}: {roc_auc:.4f}")
print(f"Mean ROC‑AUC (train): {np.mean(roc_auc_vals):.4f}")



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/902027639.py in <cell line: 0>()
      1 from sklearn.metrics import accuracy_score, roc_auc_score
      2 
----> 3 train_pred_labels = pred_y.argmax(axis=1)
      4 print("Training accuracy:", accuracy_score(Y_train, train_pred_labels))
      5 

NameError: name 'pred_y' is not defined

## === cell 28
pred_y_test = xgb_model.predict(X_test)  # shape (n_test, 4)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2360014519.py in <cell line: 0>()
----> 1 pred_y_test = xgb_model.predict(X_test)  # shape (n_test, 4)
      2 

NameError: name 'xgb_model' is not defined

## === cell 29
df = pd.concat(
    [
        test_table["image_id"].reset_index(drop=True),
        pd.DataFrame(pred_y_test, columns=LABELS),
    ],
    axis=1,
)
df.to_csv("./submit.csv", index=False)
print("Submission file written to ./submit.csv with shape", df.shape)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3635797322.py in <cell line: 0>()
----> 1 df = pd.concat(
      2     [
      3         test_table["image_id"].reset_index(drop=True),
      4         pd.DataFrame(pred_y_test, columns=LABELS),
      5     ],

NameError: name 'pd' is not defined

## === cell 30
pass



## === cell 31
pass



## === cell 32
pass
