# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<4"])

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    ZeroPadding2D,
    UpSampling2D,
    ThresholdedReLU,
    Conv2DTranspose,
    Cropping2D,
    DepthwiseConv2D,
    add,
    Activation,
    LeakyReLU,
    Dense,
    Conv2D,
    GlobalMaxPooling2D,
    MaxPooling2D,
    Flatten,
    Concatenate,
    Input,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    SeparableConv2D,
    AveragePooling2D,
)
import cv2
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.metrics import (
    TruePositives,
    FalsePositives,
    TrueNegatives,
    FalseNegatives,
    AUC,
    BinaryAccuracy,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import softsign
from keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import random
import keras.backend as K
import gc
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)
random.seed(42)



## === cell 1
TRAIN = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
TRAIN22 = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
TEST = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
SUB = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
TRAIN_DIR = "../input/osic-pulmonary-fibrosis-progression/train/"
TEST_DIR = "../input/osic-pulmonary-fibrosis-progression/test/"
TRAIN.reset_index(drop=True, inplace=True)
TEST.reset_index(drop=True, inplace=True)

BATCH = 15
SHAPE_RESIZE = 256
CUT = 10
COUNT_MODEL = 4
TRAIN["dir"] = TRAIN_DIR
TEST["dir"] = TEST_DIR

TRAIN = pd.concat([TRAIN, TEST], ignore_index=True).copy()
TRAIN.drop_duplicates(subset=["Patient", "Weeks"], keep=False, inplace=True)



## === cell 2
PACIENT = TEST["Patient"].unique()
TRAIN_NEW = pd.DataFrame()
for ID in tqdm(PACIENT):
    data = TEST[TEST.Patient == ID].copy()
    data.reset_index(inplace=True, drop=True)
    data = data[:1]
    r = range(-12, 134)
    count_week = len(r)
    data = data.loc[data.index.repeat(count_week)].reset_index(drop=True)
    week_predict = [i for i in r]
    data["week_predict"] = week_predict
    TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)
TEST = TRAIN_NEW




## === cell 3
def counsruct(dataframe, test=False):
    PACIENT = dataframe["Patient"].unique()
    TRAIN_NEW = pd.DataFrame()
    for ID in tqdm(PACIENT):
        data = dataframe[dataframe.Patient == ID].copy()
        data.reset_index(inplace=True, drop=True)
        week_start, FVC_start, Percent_kt, d = data.loc[
            0, ["Weeks", "FVC", "Percent", "dir"]
        ].values
        DIR = d
        DIR_DCM = DIR + ID + "/"
        dlist = sorted(os.listdir(DIR_DCM), key=lambda v: int(v.split(".")[0]))
        count_slice = len(dlist)
        dmap = {i + 1: dcm for i, dcm in enumerate(dlist)}
        center = len(dmap) // 2
        c = center - (center * 40 // 100)
        arr_slice = [c]
        count_repeat = len(arr_slice)

        dd = np.array(
            [[dmap[i], int(i)] for j in range(data.shape[0]) for i in arr_slice]
        )
        dd = pd.DataFrame(dd, columns=["dcm", "num_slice"])
        dd["num_slice"] = dd["num_slice"].astype("int")
        data = data.loc[data.index.repeat(count_repeat)].reset_index(drop=True)
        data[["dcm", "num_slice"]] = dd[["dcm", "num_slice"]]
        data["count_slice"] = count_slice
        data["week_kt"] = week_start
        data["FVC_kt"] = FVC_start
        data["Percent_kt"] = Percent_kt
        TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)
    return TRAIN_NEW


TRAIN_C = counsruct(TRAIN)
TEST_C = counsruct(TEST, True)
TEST_C["Count_weks"] = TEST_C["week_predict"] - TEST_C["week_kt"]
TRAIN_C["Count_weks"] = TRAIN_C["Weeks"] - TRAIN_C["week_kt"]



## === cell 4
pass



## === cell 5
r = 1
e = 70


def custom_data(dataframe):
    dataframe["FVC_n"] = dataframe["FVC_kt"] * 100 / dataframe["Percent_kt"]
    for i in range(r, e):
        name = "FVC_mean" + str(i)
        name2 = "FVC_custom" + str(i)
        dataframe[name] = (dataframe["FVC_n"] - dataframe["FVC_kt"]) / (30 * i)
        dataframe[name2] = (
            dataframe["FVC_kt"] - (dataframe["Count_weks"]) * dataframe[name]
        ) - ((dataframe["Count_weks"] + 90))
    return dataframe


TRAIN_C = custom_data(TRAIN_C)
TEST_C = custom_data(TEST_C)

name = ["FVC_custom" + str(i) for i in range(r, e)]
TEST_C["FVC_PRE"] = TEST_C[name[10:]].mean(axis=1)
TRAIN_C["FVC_PRE"] = TRAIN_C[name[10:]].mean(axis=1)

TEST_C["FVC_PRE2"] = TEST_C["FVC_PRE"] ** 2
TRAIN_C["FVC_PRE2"] = TRAIN_C["FVC_PRE"] ** 2
TEST_C["FVC_n2"] = TEST_C["FVC_n"] ** 2
TRAIN_C["FVC_n2"] = TRAIN_C["FVC_n"] ** 2

TEST_C["r1"] = TEST_C["FVC_n"] - TEST_C["FVC_PRE"]
TEST_C["r1mean"] = TEST_C[["FVC_n", "FVC_PRE"]].mean(axis=1)
TEST_C["r2"] = TEST_C[["FVC_n", "FVC_PRE"]].std(axis=1)

TRAIN_C["r1"] = TRAIN_C["FVC_n"] - TRAIN_C["FVC_PRE"]
TRAIN_C["r1mean"] = TRAIN_C[["FVC_n", "FVC_PRE"]].mean(axis=1)
TRAIN_C["r2"] = TRAIN_C[["FVC_n", "FVC_PRE"]].std(axis=1)

TRAIN_C[["Female", "Male", "Currently smokes", "Ex-smoker", "Never smoked"]] = 0
TEST_C[["Female", "Male", "Currently smokes", "Ex-smoker", "Never smoked"]] = 0


def calculate_height(row):
    if row["Sex"] == "Male":
        return row["FVC_kt"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FVC_kt"] / (21.78 - 0.101 * row["Age"])


def calculate_all(row):
    if row["Sex"] == "Male":
        row["Male"] = 1
    else:
        row["Female"] = 1

    if row["SmokingStatus"] == "Currently smokes":
        row["Currently smokes"] = 1
    if row["SmokingStatus"] == "Ex-smoker":
        row["Ex-smoker"] = 1
    if row["SmokingStatus"] == "Never smoked":
        row["Never smoked"] = 1
    return row


TRAIN_C["Height"] = TRAIN_C.apply(calculate_height, axis=1)
TEST_C["Height"] = TEST_C.apply(calculate_height, axis=1)
TRAIN_C = TRAIN_C.apply(calculate_all, axis=1)
TEST_C = TEST_C.apply(calculate_all, axis=1)
TRAIN_C = TRAIN_C.drop(columns=["Sex", "SmokingStatus"])
TEST_C = TEST_C.drop(columns=["Sex", "SmokingStatus"])



## === cell 6
Height = TRAIN_C["Height"].unique().tolist()
Height.extend(TEST_C["Height"].unique().tolist())
all_Height = np.unique(Height)
bins = np.linspace(all_Height.min(), all_Height.max(), 5)
witch_bin = np.digitize(TRAIN_C.Height, bins)
witch_bin2 = np.digitize(TEST_C.Height, bins)

encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder.fit(witch_bin.reshape(-1, 1))
Height_binned = encoder.transform(witch_bin.reshape(-1, 1))
Height_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))
name_height_binned = ["Height_binned" + str(i) for i in range(Height_binned.shape[1])]
TRAIN_C[name_height_binned] = Height_binned
TEST_C[name_height_binned] = Height_binned2



## === cell 7
FVC_kt_train = TRAIN_C["FVC_kt"].unique().tolist()
FVC_kt_train.extend(TEST_C["FVC_kt"].unique().tolist())
all_FVC_kt = np.unique(FVC_kt_train)
bins = np.linspace(all_FVC_kt.min(), all_FVC_kt.max(), 11)
witch_bin = np.digitize(TRAIN_C.FVC_kt, bins)
witch_bin2 = np.digitize(TEST_C.FVC_kt, bins)

encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder.fit(witch_bin.reshape(-1, 1))
FVC_binned = encoder.transform(witch_bin.reshape(-1, 1))
FVC_binned2 = encoder.transform(witch_bin2.reshape(-1, 1))
name_bin_fvckt = ["FVC_KT_bin" + str(i) for i in range(FVC_binned.shape[1])]
TRAIN_C[name_bin_fvckt] = FVC_binned
TEST_C[name_bin_fvckt] = FVC_binned2



## === cell 8
FVC_PRE_train = TRAIN_C["FVC_PRE"].unique().tolist()
FVC_PRE_train.extend(TEST_C["FVC_PRE"].unique().tolist())
all_FVC_PRE_kt = np.unique(FVC_PRE_train)
bins2 = np.linspace(all_FVC_PRE_kt.min(), all_FVC_PRE_kt.max(), 5)
witch_bin_pre_tr = np.digitize(TRAIN_C.FVC_PRE, bins2)
witch_bin_pre_te = np.digitize(TEST_C.FVC_PRE, bins2)

encoder_pre = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoder_pre.fit(witch_bin_pre_tr.reshape(-1, 1))
FVC_PRE_binned = encoder_pre.transform(witch_bin_pre_tr.reshape(-1, 1))
FVC_PRE_binned2 = encoder_pre.transform(witch_bin_pre_te.reshape(-1, 1))

name_bin_pre = ["FVC_PRE_bin" + str(i) for i in range(FVC_PRE_binned.shape[1])]
TRAIN_C[name_bin_pre] = FVC_PRE_binned
TEST_C[name_bin_pre] = FVC_PRE_binned2




## === cell 9
def calculate_FVC(row):
    if row["Weeks"] == row["week_kt"]:
        row["FVC_PRE"] = row["FVC"]
    return row


TRAIN_C = TRAIN_C.apply(calculate_FVC, axis=1)



## === cell 10
Patient = LabelEncoder()
train_pac = TRAIN_C["Patient"].unique().tolist()
train_pac.extend(TEST_C["Patient"].unique().tolist())
all_pacient = np.unique(train_pac)
Patient.fit(all_pacient)


def LE(dataframe, val=False, dense=False):
    dataframe["patiet_id"] = Patient.transform(dataframe["Patient"])
    col = ["Patient", "dcm"]
    if not val:
        col.extend(["FVC", "Weeks"])
    if val:
        col.extend(["week_predict"])
    col.extend(name_bin_pre)
    col.extend(name_bin_fvckt)
    col.extend(name_height_binned)
    col.extend(
        [
            "FVC_PRE",
            "FVC_PRE2",
            "Age",
            "count_slice",
            "week_kt",
            "Count_weks",
            "Height",
            "FVC_kt",
            "Currently smokes",
            "Ex-smoker",
            "Never smoked",
            "Female",
            "Male",
            "FVC_n",
            "r1",
            "r2",
            "r1mean",
            "Percent_kt",
        ]
    )
    dataframe = dataframe[col]
    dataframe["dcm"] = dataframe.agg("{0[Patient]}/{0[dcm]}".format, axis=1)
    return dataframe


TRAIN2 = LE(TRAIN_C.copy())
TEST2 = LE(TEST_C.copy(), True)



## === cell 11
train = 0
validation = 0


def Fold(dataframe):
    train = []
    val = []
    PACIENT = dataframe["Patient"].unique()

    for ID in PACIENT:
        d = dataframe[dataframe.Patient == ID].copy()
        d.reset_index(inplace=True, drop=True)
        weeks = sorted(d["Weeks"].unique().tolist())

        if len(weeks) == 1:
            week_val = [weeks[0]]
            week_train = [weeks[0]]
        else:
            week_val = [weeks[-1]]  # hold out last observed week
            week_train = weeks[:-1]  # train on earlier weeks

        train_index = dataframe[
            (dataframe.Patient == ID) & (dataframe.Weeks.isin(week_train))
        ].index.tolist()
        val_index = dataframe[
            (dataframe.Patient == ID) & (dataframe.Weeks.isin(week_val))
        ].index.tolist()

        train.extend(train_index)
        val.extend(val_index)

    return train, val


train_index, val_index = Fold(TRAIN2)
train = TRAIN2.loc[train_index]
validation = TRAIN2.loc[val_index]
train.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)




## === cell 12
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    else:
        return np.mean(metric)




## === cell 13
train_split = train.copy()
test_split = validation.copy()

X_train = train_split[train_split.columns.tolist()[3:]].copy()
X_val = test_split[test_split.columns.tolist()[3:]].copy()
X_val2 = test_split[test_split.columns.tolist()[3:]].copy()

Y_train_raw = train_split["FVC"].copy()
Y_train2 = (Y_train_raw - train_split["FVC_PRE"]).abs()
Y_train = Y_train2.copy()

Y_val = test_split["FVC"].copy()
Y_val2 = (Y_val - test_split["FVC_PRE"]).abs()

X_test = TEST2[TEST2.columns.tolist()[2:]]
X_train_kneigboards = train_split[train_split.columns.tolist()[3:]].copy()
X_val_kneigboards = test_split[test_split.columns.tolist()[3:]].copy()
X_test_kneigboards = TEST2[TEST2.columns.tolist()[2:]]

alpha = 0.9
tree1 = GradientBoostingRegressor(
    loss="quantile",
    alpha=alpha,
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=49,
    min_samples_split=49,
)
tree1.fit(X_train_kneigboards, Y_train)
y_upper = tree1.predict(X_val_kneigboards)

tree1.set_params(alpha=0.1)
tree1.fit(X_train_kneigboards, Y_train)
y_lower = tree1.predict(X_val_kneigboards)

tree1.set_params(loss="squared_error")
tree1.fit(X_train_kneigboards, Y_train)
y_pred = tree1.predict(X_val_kneigboards)

tree3 = KNeighborsRegressor(n_neighbors=252)
tree3.fit(X_train_kneigboards, Y_train)
pred_k = tree3.predict(X_val_kneigboards)

tree2 = RandomForestRegressor(
    n_estimators=30, min_samples_leaf=2, min_samples_split=2, random_state=42
)
tree2.fit(X_train, Y_train)
pred_r = tree2.predict(X_val)

tree4 = LinearRegression()
tree4.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:-11]], Y_train)
pred_lr = tree4.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:-11]])

tree5 = Ridge(alpha=0.03, random_state=42)
tree5.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:-11]], Y_train)
pred_ridge = tree5.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:-11]])

lf = BayesianRidge()
lf.fit(X_train_kneigboards[X_train_kneigboards.columns.tolist()[:]], Y_train)
pred_baes = lf.predict(X_val_kneigboards[X_val_kneigboards.columns.tolist()[:]])

X_val2["y_upper"] = y_upper
X_val2["y_lower"] = y_lower
X_val2["y_pred"] = y_pred
X_val2["pred_k"] = pred_k
X_val2["pred_r"] = pred_r
X_val2["pred_lr"] = pred_lr
X_val2["pred_ridge"] = pred_ridge
X_val2["baes"] = pred_baes

trend_tbl = (
    train_split.groupby("Patient")[["Weeks", "FVC"]]
    .apply(
        lambda g: (
            np.polyfit(
                g["Weeks"].values.astype(np.float64),
                g["FVC"].values.astype(np.float64),
                1,
            )[0]
            if g.shape[0] >= 2
            else 0.0
        )
    )
    .reset_index(name="slope_fvc_per_week")
)
X_val2 = X_val2.merge(trend_tbl, on="Patient", how="left")
X_val2["slope_fvc_per_week"] = X_val2["slope_fvc_per_week"].fillna(0.0)
X_val2["sign_trend"] = np.where(X_val2["slope_fvc_per_week"] >= 0.0, 1.0, -1.0)

X_val2["delta_blend"] = X_val2[["baes", "pred_r"]].mean(axis=1)
X_val2["delta_signed"] = X_val2["delta_blend"] * X_val2["sign_trend"]
X_val2["FVC_pred_final"] = X_val2["FVC_PRE"] + X_val2["delta_signed"]

val_abs_err = np.abs(Y_val.values - X_val2["FVC_pred_final"].values)
val_abs_weeks = np.abs(X_val2["Count_weks"].values.astype(np.float64))

X_design = np.vstack([np.ones_like(val_abs_weeks), val_abs_weeks]).T
coef, _, _, _ = np.linalg.lstsq(X_design, val_abs_err, rcond=None)
a_abs, b_abs = float(coef[0]), float(coef[1])

a_sigma = max(70.0, np.sqrt(2) * max(a_abs, 0.0))
b_sigma = max(0.0, np.sqrt(2) * b_abs)

X_val2["Conf_final"] = np.maximum(70.0, a_sigma + b_sigma * val_abs_weeks)

print(
    "val LaplaceLL (baseline):",
    laplace_log_likelihood(Y_val, X_val2["FVC_PRE"], X_val2["Conf_final"]),
)
print(
    "val LaplaceLL (blend+signed):",
    laplace_log_likelihood(Y_val, X_val2["FVC_pred_final"], X_val2["Conf_final"]),
)
print("Chosen confidence calibrator: a_sigma=%.3f, b_sigma=%.6f" % (a_sigma, b_sigma))

if ("week_predict" in X_test_kneigboards.columns) and (
    "Weeks" not in X_test_kneigboards.columns
):
    X_test_kneigboards = X_test_kneigboards.rename(columns={"week_predict": "Weeks"})
if "week_predict" in X_test.columns and "Weeks" not in X_test.columns:
    X_test = X_test.rename(columns={"week_predict": "Weeks"})

X_test_kneigboards = X_test_kneigboards[X_train_kneigboards.columns.tolist()]
X_test = X_test[X_train.columns.tolist()]

tree1.set_params(loss="squared_error")
y_pred2 = tree1.predict(X_test_kneigboards)
pred_k2 = tree3.predict(X_test_kneigboards)
pred_r2 = tree2.predict(X_test)
pred_lr2 = tree4.predict(X_test_kneigboards[X_test_kneigboards.columns.tolist()[:-11]])
pred_ridge2 = tree5.predict(
    X_test_kneigboards[X_test_kneigboards.columns.tolist()[:-11]]
)
pred_baes2 = lf.predict(X_test_kneigboards[X_test_kneigboards.columns.tolist()[:]])

TEST2["y_pred"] = y_pred2
TEST2["pred_k"] = pred_k2
TEST2["pred_r"] = pred_r2
TEST2["pred_lr"] = pred_lr2
TEST2["pred_ridge"] = pred_ridge2
TEST2["baes"] = pred_baes2

TEST2["delta_blend"] = TEST2[["baes", "pred_r"]].mean(axis=1)

TEST2 = TEST2.merge(trend_tbl, on="Patient", how="left")
TEST2["slope_fvc_per_week"] = TEST2["slope_fvc_per_week"].fillna(0.0)
TEST2["sign_trend"] = np.where(TEST2["slope_fvc_per_week"] >= 0.0, 1.0, -1.0)
TEST2["delta_signed"] = TEST2["delta_blend"] * TEST2["sign_trend"]
TEST2["FVC_pred_final"] = TEST2["FVC_PRE"] + TEST2["delta_signed"]

test_abs_weeks = np.abs(TEST2["Count_weks"].values.astype(np.float64))
TEST2["Confidence_final"] = np.maximum(70.0, a_sigma + b_sigma * test_abs_weeks)

TEST2["Patient_Week"] = TEST2.agg("{0[Patient]}_{0[week_predict]}".format, axis=1)
pred_out = TEST2[["Patient_Week", "FVC_pred_final", "Confidence_final"]].copy()
pred_out.columns = ["Patient_Week", "FVC", "Confidence"]

SUB_aligned = SUB[["Patient_Week"]].merge(pred_out, on="Patient_Week", how="left")
SUB_aligned["FVC"] = SUB_aligned["FVC"].fillna(
    TEST2.groupby("Patient")["FVC_PRE"].first().mean()
)
SUB_aligned["Confidence"] = SUB_aligned["Confidence"].fillna(70.0)

SUB_aligned.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", SUB_aligned.shape)
print(SUB_aligned.head())

## --- ERROR in cell 13, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3303776532.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     87[0m     [0;34m.[0m[0mreset_index[0m[0;34m([0m[0mname[0m[0;34m=[0m[0;34m"slope_fvc_per_week"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     88[0m )
[0;32m---> 89[0;31m [0mX_val2[0m [0;34m=[0m [0mX_val2[0m[0;34m.[0m[0mmerge[0m[0;34m([0m[0mtrend_tbl[0m[0;34m,[0m [0mon[0m[0;34m=[0m[0;34m"Patient"[0m[0;34m,[0m [0mhow[0m[0;34m=[0m[0;34m"left"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     90[0m [0mX_val2[0m[0;34m[[0m[0;34m"slope_fvc_per_week"[0m[0;34m][0m [0;34m=[0m [0mX_val2[0m[0;34m[[0m[0;34m"slope_fvc_per_week"[0m[0;34m][0m[0;34m.[0m[0mfillna[0m[0;34m([0m[0;36m0.0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     91[0m [0mX_val2[0m[0;34m[[0m[0;34m"sign_trend"[0m[0;34m][0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mwhere[0m[0;34m([0m[0mX_val2[0m[0;34m[[0m[0;34m"slope_fvc_per_week"[0m[0;34m][0m [0;34m>=[0m [0;36m0.0[0m[0;34m,[0m [0;36m1.0[0m[0;34m,[0m [0;34m-[0m[0;36m1.0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mmerge[0;34m(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)[0m
[1;32m  10830[0m         [0;32mfrom[0m [0mpandas[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mreshape[0m[0;34m.[0m[0mmerge[0m [0;32mimport[0m [0mmerge[0m[0;34m[0m[0;34m[0m[0m
[1;32m  10831[0m [0;34m[0m[0m
[0;32m> 10832[0;31m         return merge(
[0m[1;32m  10833[0m             [0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m  10834[0m             [0mright[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36mmerge[0;34m(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)[0m
[1;32m    168[0m         )
[1;32m    169[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 170[0;31m         op = _MergeOperation(
[0m[1;32m    171[0m             [0mleft_df[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    172[0m             [0mright_df[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36m__init__[0;34m(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)[0m
[1;32m    792[0m             [0mleft_drop[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    793[0m             [0mright_drop[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 794[0;31m         ) = self._get_merge_keys()
[0m[1;32m    795[0m [0;34m[0m[0m
[1;32m    796[0m         [0;32mif[0m [0mleft_drop[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36m_get_merge_keys[0;34m(self)[0m
[1;32m   1308[0m                         [0;31m#  the latter of which will raise[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1309[0m                         [0mlk[0m [0;34m=[0m [0mcast[0m[0;34m([0m[0mHashable[0m[0;34m,[0m [0mlk[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1310[0;31m                         [0mleft_keys[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mleft[0m[0;34m.[0m[0m_get_label_or_level_values[0m[0;34m([0m[0mlk[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1311[0m                         [0mjoin_names[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mlk[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1312[0m                     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m_get_label_or_level_values[0;34m(self, key, axis)[0m
[1;32m   1909[0m             [0mvalues[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0maxes[0m[0;34m[[0m[0maxis[0m[0;34m][0m[0;34m.[0m[0mget_level_values[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m.[0m[0m_values[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1910[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1911[0;31m             [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1912[0m [0;34m[0m[0m
[1;32m   1913[0m         [0;31m# Check for duplicates[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: 'Patient'
