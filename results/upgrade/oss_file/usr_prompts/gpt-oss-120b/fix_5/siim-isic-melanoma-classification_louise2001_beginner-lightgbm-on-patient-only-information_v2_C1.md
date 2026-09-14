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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.6573

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
submission = pd.read_csv(
    "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv"
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/211164207.py in <cell line: 0>()
----> 1 train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
      2 test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
      3 submission = pd.read_csv(
      4     "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv"
      5 )

NameError: name 'pd' is not defined

## === cell 1
train.columns



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2223934433.py in <cell line: 0>()
----> 1 train.columns
      2 

NameError: name 'train' is not defined

## === cell 2
test.columns



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2688281865.py in <cell line: 0>()
----> 1 test.columns
      2 

NameError: name 'test' is not defined

## === cell 3
submission.columns



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2740939156.py in <cell line: 0>()
----> 1 submission.columns
      2 

NameError: name 'submission' is not defined

## === cell 4
patient_only_cols = ["patient_id", "sex", "age_approx", "anatom_site_general_challenge"]
patient_only_train = train[patient_only_cols + ["target"]].drop_duplicates(
    inplace=False
)
patient_only_test = test[patient_only_cols].drop_duplicates(inplace=False)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3507805009.py in <cell line: 0>()
      1 patient_only_cols = ["patient_id", "sex", "age_approx", "anatom_site_general_challenge"]
----> 2 patient_only_train = train[patient_only_cols + ["target"]].drop_duplicates(
      3     inplace=False
      4 )
      5 patient_only_test = test[patient_only_cols].drop_duplicates(inplace=False)

NameError: name 'train' is not defined

## === cell 5
categoricals = ["sex", "anatom_site_general_challenge"]



## === cell 6
patient_only_train.describe()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3631336113.py in <cell line: 0>()
----> 1 patient_only_train.describe()
      2 

NameError: name 'patient_only_train' is not defined

## === cell 7
set(patient_only_train.sex.values.tolist())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3273038869.py in <cell line: 0>()
----> 1 set(patient_only_train.sex.values.tolist())
      2 

NameError: name 'patient_only_train' is not defined

## === cell 8
matching_sex = {"female": 1, "male": 0}



## === cell 9
set(patient_only_train.anatom_site_general_challenge.values.tolist())



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/108010430.py in <cell line: 0>()
----> 1 set(patient_only_train.anatom_site_general_challenge.values.tolist())
      2 

NameError: name 'patient_only_train' is not defined

## === cell 10
matching_anatom = {
    "head/neck": 0,
    "lower extremity": 1,
    "oral/genital": 2,
    "palms/soles": 3,
    "torso": 4,
    "upper extremity": 5,
}



## === cell 11
patient_only_train.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1177367433.py in <cell line: 0>()
----> 1 patient_only_train.head()
      2 

NameError: name 'patient_only_train' is not defined

## === cell 12
patient_only_test.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/579293578.py in <cell line: 0>()
----> 1 patient_only_test.head()
      2 

NameError: name 'patient_only_test' is not defined

## === cell 13
patient_only_train.replace(
    to_replace={"anatom_site_general_challenge": matching_anatom, "sex": matching_sex},
    inplace=True,
)
patient_only_test.replace(
    to_replace={"anatom_site_general_challenge": matching_anatom, "sex": matching_sex},
    inplace=True,
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1323117250.py in <cell line: 0>()
----> 1 patient_only_train.replace(
      2     to_replace={"anatom_site_general_challenge": matching_anatom, "sex": matching_sex},
      3     inplace=True,
      4 )
      5 patient_only_test.replace(

NameError: name 'patient_only_train' is not defined

## === cell 14
patient_only_train.head()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1177367433.py in <cell line: 0>()
----> 1 patient_only_train.head()
      2 

NameError: name 'patient_only_train' is not defined

## === cell 15
Cols = ["sex", "age_approx", "anatom_site_general_challenge"]
patient_only_train[Cols] = patient_only_train[Cols].astype("int32", errors="ignore")
patient_only_test[Cols] = patient_only_test[Cols].astype("int32", errors="ignore")
patient_only_train.head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2817774488.py in <cell line: 0>()
      1 Cols = ["sex", "age_approx", "anatom_site_general_challenge"]
----> 2 patient_only_train[Cols] = patient_only_train[Cols].astype("int32", errors="ignore")
      3 patient_only_test[Cols] = patient_only_test[Cols].astype("int32", errors="ignore")
      4 patient_only_train.head()
      5 

NameError: name 'patient_only_train' is not defined

## === cell 16
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold


def derivative(func, x, n=1, dx=1e-6):
    if n == 1:
        return (func(x + dx) - func(x - dx)) / (2 * dx)
    elif n == 2:
        return func(x + dx) - 2 * func(x) + func(x - dx) / (dx**2)
    else:
        raise NotImplementedError("Higher order derivatives not implemented.")




## === cell 17
def MeanAveragePrecision(y_pred, y_true):
    y_true = y_true.get_label()
    df = pd.DataFrame({"true": y_true, "pred_probas": y_pred})
    n = df.shape[0]
    df.sort_values(by="pred_probas", ascending=False, inplace=True)
    df["loss"] = df["true"].cumsum() / list(range(1, n + 1))
    df = df.loc[df["true"] == 1, "loss"]
    return "MeanAveragePrecision", max(0, df.mean(axis=0)), True


def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label

    def fl(x, t):
        p = 1 / (1 + np.exp(-x))
        return (
            -(a * t + (1 - a) * (1 - t))
            * ((1 - (t * p + (1 - t) * (1 - p))) ** g)
            * (t * np.log(p) + (1 - t) * np.log(1 - p))
        )

    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label
    p = 1 / (1 + np.exp(-y_pred))
    loss = (
        -(a * y_true + (1 - a) * (1 - y_true))
        * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g)
        * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    )
    return "focal_loss", np.mean(loss), False


def focal_loss_lgb_f1_score(preds, lgbDataset):
    preds = 1 / (1 + np.exp(-preds))
    binary_preds = [int(p > 0.5) for p in preds]
    y_true = lgbDataset.get_label()
    return "f1", f1_score(y_true, binary_preds), True


focal_loss = lambda x, y: focal_loss_lgb(x, y, alpha=0.45, gamma=2.0)
focal_loss_eval = lambda x, y: focal_loss_lgb_eval_error(x, y, alpha=0.45, gamma=2.0)


def DataSetLgbm(Data, trn_idx, val_idx, target, features, categorical_features=""):
    trn_data = lgb.Dataset(
        Data.iloc[trn_idx][features],
        label=Data[target].iloc[trn_idx],
        categorical_feature=categorical_features,
    )
    val_data = lgb.Dataset(
        Data.iloc[val_idx][features],
        label=Data[target].iloc[val_idx],
        categorical_feature=categorical_features,
    )
    return trn_data, val_data


def TrainSimpleLgbm(
    Params, DataTrain, trn_idx, val_idx, target, features, categorical_features=""
):
    trn_data, val_data = DataSetLgbm(
        DataTrain, trn_idx, val_idx, target, features, categorical_features
    )
    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=500, verbose=False),
    ]
    clf = lgb.train(
        Params,
        trn_data,
        num_boost_round=30000,
        valid_sets=[trn_data, val_data],
        feval=MeanAveragePrecision,
        callbacks=callbacks,
    )
    return clf


def mean_average_p(y_true, y_pred_p):
    df = pd.DataFrame({"true": y_true, "pred_probas": y_pred_p})
    n = df.shape[0]
    df.sort_values(by="pred_probas", ascending=False, inplace=True)
    df["loss"] = df["true"].cumsum() / list(range(1, n + 1))
    df = df.loc[df["true"] == 1, "loss"]
    return max(0, df.mean(axis=0))




## === cell 18
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20481998)

ParN1 = {
    "bagging_freq": 1,
    "bagging_fraction": 0.95,
    "boost_from_average": "true",
    "boost": "gbdt",
    "feature_fraction": 0.5,
    "learning_rate": 0.04,
    "max_depth": -1,
    "metric": "auc",
    "is_unbalance": "true",
    "min_data_in_leaf": 80,
    "lambda_l1": 1,
    "lambda_l2": 1,
    "num_leaves": 2000,
    "colsample_bytree": 0.9,
    "tree_learner": "serial",
    "objective": "binary",
    "verbosity": 1,
}
ParN2 = {
    "bagging_freq": 20,
    "bagging_fraction": 0.9,
    "boost_from_average": "true",
    "boost": "gbdt",
    "feature_fraction": 0.9,
    "learning_rate": 0.04,
    "max_depth": -1,
    "metric": "auc",
    "is_unbalance": "true",
    "lambda_l1": 10,
    "lambda_l2": 10,
    "num_leaves": 7,
    "colsample_bytree": 0.7,
    "tree_learner": "serial",
    "objective": "binary",
    "verbosity": 1,
}




## === cell 19
test_preds = []
last_clf = None  # will hold the most recent model for feature importance
for fold, (trn_idx, val_idx) in enumerate(
    folds.split(patient_only_train[Cols], patient_only_train["target"])
):
    print(f"********************* Fitting on Fold {fold + 1} ... ******************")
    clf1 = TrainSimpleLgbm(
        ParN1, patient_only_train, trn_idx, val_idx, "target", Cols, categoricals
    )
    clf2 = TrainSimpleLgbm(
        ParN2, patient_only_train, trn_idx, val_idx, "target", Cols, categoricals
    )

    pred_oof1 = clf1.predict(
        patient_only_train.iloc[val_idx][Cols], num_iteration=clf1.best_iteration
    )
    pred_test1 = clf1.predict(
        patient_only_test[Cols], num_iteration=clf1.best_iteration
    )
    test_preds.append(pred_test1)

    pred_oof2 = clf2.predict(
        patient_only_train.iloc[val_idx][Cols], num_iteration=clf2.best_iteration
    )
    pred_test2 = clf2.predict(
        patient_only_test[Cols], num_iteration=clf2.best_iteration
    )
    test_preds.append(pred_test2)

    mean_pred_oof = 0.5 * pred_oof1 + 0.5 * pred_oof2

    m1 = mean_average_p(patient_only_train["target"].iloc[val_idx], pred_oof1)
    m2 = mean_average_p(patient_only_train["target"].iloc[val_idx], pred_oof2)
    m3 = mean_average_p(patient_only_train["target"].iloc[val_idx], mean_pred_oof)

    print(f" Mean Average M1 : {m1}  , M2 : {m2}   M3 : {m3}")

    pred_oof1_bin = (pred_oof1 >= 0.5).astype(int)
    pred_oof2_bin = (pred_oof2 >= 0.5).astype(int)
    mean_pred_oof_bin = (mean_pred_oof >= 0.5).astype(int)

    print("*************  CR Param 1 *************************")
    print(
        classification_report(patient_only_train["target"].iloc[val_idx], pred_oof1_bin)
    )

    print("*************  CR Param 2 *************************")
    print(
        classification_report(patient_only_train["target"].iloc[val_idx], pred_oof2_bin)
    )

    print("*************  CR Mean *************************")
    print(
        classification_report(
            patient_only_train["target"].iloc[val_idx], mean_pred_oof_bin
        )
    )
    last_clf = clf1




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/315923075.py in <cell line: 0>()
      2 last_clf = None  # will hold the most recent model for feature importance
      3 for fold, (trn_idx, val_idx) in enumerate(
----> 4     folds.split(patient_only_train[Cols], patient_only_train["target"])
      5 ):
      6     print(f"********************* Fitting on Fold {fold + 1} ... ******************")

NameError: name 'patient_only_train' is not defined

## === cell 20
if last_clf is not None:
    feature_importance = pd.DataFrame(
        {"Value": last_clf.feature_importance(), "Feature": Cols}
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_importance.sort_values(by="Value", ascending=False),
    )
    plt.title("Features Importance")
    plt.tight_layout()
    plt.show()
else:
    print("No model was trained; feature importance cannot be displayed.")




## === cell 21
submission.head()




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/960005582.py in <cell line: 0>()
----> 1 submission.head()
      2 
      3 

NameError: name 'submission' is not defined

## === cell 22
test.head()




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1260514712.py in <cell line: 0>()
----> 1 test.head()
      2 
      3 

NameError: name 'test' is not defined

## === cell 23
patient_only_test.head()




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3752540357.py in <cell line: 0>()
----> 1 patient_only_test.head()
      2 
      3 

NameError: name 'patient_only_test' is not defined

## === cell 24
if len(test_preds) > 0:
    patient_only_test["target"] = np.stack(test_preds, axis=1).mean(axis=1)
else:
    patient_only_test["target"] = patient_only_train["target"].mean()
patient_only_test.head()




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3550286470.py in <cell line: 0>()
      2     patient_only_test["target"] = np.stack(test_preds, axis=1).mean(axis=1)
      3 else:
----> 4     patient_only_test["target"] = patient_only_train["target"].mean()
      5 patient_only_test.head()
      6 

NameError: name 'patient_only_train' is not defined

## === cell 25
sample_submission = test[["image_name", "patient_id"]].copy()
sample_submission = sample_submission.merge(
    patient_only_test[["patient_id", "target"]],
    on="patient_id",
    how="left",  # keep all test rows
)
sample_submission = sample_submission[["image_name", "target"]]

overall_mean = patient_only_train["target"].mean()
sample_submission["target"].fillna(overall_mean, inplace=True)

assert sample_submission.shape[0] == test.shape[0], "Row count mismatch!"

sample_submission.head()




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2864632608.py in <cell line: 0>()
      1 # Build submission ensuring every test row is present by using a left join
----> 2 sample_submission = test[["image_name", "patient_id"]].copy()
      3 sample_submission = sample_submission.merge(
      4     patient_only_test[["patient_id", "target"]],
      5     on="patient_id",

NameError: name 'test' is not defined

## === cell 26
sample_submission.isna().describe()




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3626330593.py in <cell line: 0>()
----> 1 sample_submission.isna().describe()
      2 
      3 

NameError: name 'sample_submission' is not defined

## === cell 27
sample_submission.to_csv("sample_submission.csv", index=False)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/664803168.py in <cell line: 0>()
----> 1 sample_submission.to_csv("sample_submission.csv", index=False)

NameError: name 'sample_submission' is not defined
