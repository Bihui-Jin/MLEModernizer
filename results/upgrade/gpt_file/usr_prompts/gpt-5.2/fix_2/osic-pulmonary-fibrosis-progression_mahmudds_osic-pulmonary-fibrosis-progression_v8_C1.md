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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

category_encoders==2.7.0
geopandas==0.14.4
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
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

# 4. Data file paths

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

# 5. Target score

-6.9498

# 6. Current score

-8.76216

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -8.76216) has done: 'I fix the notebook so it runs end-to-end under modern pandas/seaborn/TensorFlow without changing the modeling approach. Specifically: (1) fix the correlation heatmap crash by restricting to numeric columns; (2) replace deprecated `DataFrame.append` with `pd.concat`; (3) fix the TensorFlow import crash caused by an incompatible protobuf version by forcing the pure-Python protobuf implementation; and (4) fix the Keras input shape error by ensuring the feature list `FE` is built before training and by correcting the model input shape to `len(FE)` (instead of a hardcoded 9). These are runtime/logic fixes and should also improve score versus the current run because the deep model block actually train and generate predictions instead of failing.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)



## === cell 2
df_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
df_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



## === cell 3
print(
    f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(f"Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f'Test Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB")



## === cell 4
training_sample_counts = (
    df_train.rename(columns={"Weeks": "Samples"})
    .groupby("Patient")
    .agg("count")["Samples"]
    .value_counts()
)
print(
    f'Training Set FVC Measurements Per Patient \n{("-") * 41}\n{training_sample_counts}'
)



## === cell 5
df_test.head(2)



## === cell 6
df_submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
df_submission.head()



## === cell 7
print(f'FVC Statistical Summary\n{"-" * 23}')

print(
    f'Mean: {df_train["FVC"].mean():.6}  -  Median: {df_train["FVC"].median():.6}  -  Std: {df_train["FVC"].std():.6}'
)
print(
    f'Min: {df_train["FVC"].min()}  -  25%: {df_train["FVC"].quantile(0.25)}  -  50%: {df_train["FVC"].quantile(0.5)}  -  75%: {df_train["FVC"].quantile(0.75)}  -  Max: {df_train["FVC"].max()}'
)
print(
    f'Skew: {df_train["FVC"].skew():.6}  -  Kurtosis: {df_train["FVC"].kurtosis():.6}'
)
missing_values_count = df_train[df_train["FVC"].isnull()].shape[0]
training_samples_count = df_train.shape[0]
print(
    f"Missing Values: {missing_values_count}/{training_samples_count} ({missing_values_count * 100 / training_samples_count:.4}%)"
)

fig, axes = plt.subplots(ncols=2, figsize=(18, 6), dpi=150)

sns.distplot(df_train["FVC"], label="FVC", ax=axes[0])
stats.probplot(df_train["FVC"], plot=axes[1])

for i in range(2):
    axes[i].tick_params(axis="x", labelsize=12)
    axes[i].tick_params(axis="y", labelsize=12)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

axes[0].set_title("FVC Distribution in Training Set", size=15, pad=15)
axes[1].set_title("FVC Probability Plot", size=15, pad=15)

plt.show()




## === cell 8
def plot_fvc(df, patient):
    df[["Weeks", "FVC"]].set_index("Weeks").plot(figsize=(30, 6), label="_nolegend_")

    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis="y", labelsize=20)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(
        f'Patient: {patient} - {df["Age"].tolist()[0]} - {df["Sex"].tolist()[0]} - {df["SmokingStatus"].tolist()[0]} '
        f'({len(df)} Measurements in {(df["Weeks"].max() - df["Weeks"].min())} Weeks Period)',
        size=25,
        pad=25,
    )
    plt.legend().set_visible(False)
    plt.show()


for patient, df in list(df_train.groupby("Patient"))[:3]:
    df = df.copy()
    df["FVC_diff-1"] = np.abs(df["FVC"].diff(-1))

    print(f'Patient: {patient} FVC Statistical Summary\n{"-" * 58}')
    print(
        f'Mean: {df["FVC"].mean():.6}  -  Median: {df["FVC"].median():.6}  -  Std: {df["FVC"].std():.6}'
    )
    print(f'Min: {df["FVC"].min()} -  Max: {df["FVC"].max()}')
    print(f'Skew: {df["FVC"].skew():.6}  -  Kurtosis: {df["FVC"].kurtosis():.6}')
    print(
        f'Change Mean: {df["FVC_diff-1"].mean():.6}  - Change Median: {df["FVC_diff-1"].median():.6}  - Change Std: {df["FVC_diff-1"].std():.6}'
    )
    print(
        f'Change Min: {df["FVC_diff-1"].min()} -  Change Max: {df["FVC_diff-1"].max()}'
    )
    print(
        f'Change Skew: {df["FVC_diff-1"].skew():.6} -  Change Kurtosis: {df["FVC_diff-1"].kurtosis():.6}'
    )

    plot_fvc(df, patient)



## === cell 9
try:
    g = sns.pairplot(
        df_train[["FVC", "Weeks", "Percent", "Age"]],
        aspect=1.4,
        height=3,
        diag_kind="kde",
        kind="reg",
    )
    g.fig.suptitle(
        "Tabular Data Feature Distributions and Interactions", fontsize=16, y=1.02
    )
    plt.show()
except Exception as e:
    print("Skipping pairplot due to:", repr(e))



## === cell 10
try:
    g = sns.pairplot(
        df_train[["FVC", "Weeks", "Percent", "Age", "Sex"]],
        hue="Sex",
        aspect=1.4,
        height=3,
        diag_kind="kde",
        kind="reg",
    )
    g.fig.suptitle(
        "Tabular Data Feature Distributions and Interactions Between Sex Groups",
        fontsize=16,
        y=1.02,
    )
    plt.show()
except Exception as e:
    print("Skipping pairplot (Sex) due to:", repr(e))



## === cell 11
try:
    g = sns.pairplot(
        df_train[["FVC", "Weeks", "Percent", "Age", "SmokingStatus"]],
        hue="SmokingStatus",
        aspect=1.4,
        height=3,
        diag_kind="kde",
        kind="reg",
    )
    g.fig.suptitle(
        "Tabular Data Feature Distributions and Interactions Between SmokingStatus Groups",
        fontsize=16,
        y=1.02,
    )
    plt.show()
except Exception as e:
    print("Skipping pairplot (SmokingStatus) due to:", repr(e))



## === cell 12
fig = plt.figure(figsize=(10, 10), dpi=100)
corr = df_train.select_dtypes(include=[np.number]).corr()
sns.heatmap(
    corr, annot=True, square=True, cmap="coolwarm", annot_kws={"size": 10}, fmt=".2f"
)
plt.tick_params(axis="x", labelsize=12, rotation=75)
plt.tick_params(axis="y", labelsize=12, rotation=0)
plt.title("Tabular Data Feature Correlations", size=16, pad=20)
plt.show()



## === cell 13
import random
import math

from tqdm.notebook import tqdm

from sklearn.model_selection import GroupKFold, KFold
import category_encoders as ce

from sklearn.linear_model import ElasticNet
from functools import partial
import scipy as sp

import warnings

warnings.filterwarnings("ignore")




## === cell 14
def seed_everything(seed=777):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)




## === cell 15
OUTPUT_DICT = "./"

ID = "Patient_Week"
TARGET = "FVC"
SEED = 777
seed_everything(seed=SEED)

N_FOLD = 7



## === cell 16
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



## === cell 17
train = pd.concat([train, otest], ignore_index=True)
output = pd.DataFrame()
gb = train.groupby("Patient")
tk0 = tqdm(gb, total=len(gb))
for _, usr_df in tk0:
    usr_output = pd.DataFrame()
    for week, tmp in usr_df.groupby("Weeks"):
        rename_cols = {"Weeks": "base_Week", "FVC": "base_FVC", "Age": "base_Age"}
        tmp = tmp.rename(columns=rename_cols)
        drop_cols = ["Age", "Sex", "SmokingStatus", "Percent"]
        _usr_output = (
            usr_df.drop(columns=drop_cols)
            .rename(columns={"Weeks": "predict_Week"})
            .merge(tmp, on="Patient")
        )
        _usr_output["Week_passed"] = (
            _usr_output["predict_Week"] - _usr_output["base_Week"]
        )
        usr_output = pd.concat([usr_output, _usr_output], ignore_index=True)
    output = pd.concat([output, usr_output], ignore_index=True)

train = output[output["Week_passed"] != 0].reset_index(drop=True)



## === cell 18
test = otest.rename(
    columns={"Weeks": "base_Week", "FVC": "base_FVC", "Age": "base_Age"}
)
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["predict_Week"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
test = submission.drop(columns=["FVC", "Confidence"]).merge(test, on="Patient")
test["Week_passed"] = test["predict_Week"] - test["base_Week"]
test.set_index("Patient_Week", inplace=True)



## === cell 19
folds = train[["Patient", TARGET]].copy()
Fold = GroupKFold(n_splits=N_FOLD)
groups = folds["Patient"].values
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):
    folds.loc[val_index, "fold"] = int(n)
folds["fold"] = folds["fold"].astype(int)




## === cell 20
def run_single_model(clf, train_df, test_df, folds, features, target, fold_num=0):
    trn_idx = folds[folds.fold != fold_num].index
    val_idx = folds[folds.fold == fold_num].index

    y_tr = target.iloc[trn_idx].values
    X_tr = train_df.iloc[trn_idx][features].values
    y_val = target.iloc[val_idx].values
    X_val = train_df.iloc[val_idx][features].values

    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))
    clf.fit(X_tr, y_tr)

    oof[val_idx] = clf.predict(X_val)
    predictions += clf.predict(test_df[features].values)
    return oof, predictions


def run_kfold_model(clf, train, test, folds, features, target, n_fold=7):
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    for fold_ in range(n_fold):
        _oof, _predictions = run_single_model(
            clf, train, test, folds, features, target, fold_num=fold_
        )
        oof += _oof
        predictions += _predictions / n_fold

    return oof, predictions




## === cell 21
target = train[TARGET]
test[TARGET] = np.nan

cat_features = ["Sex", "SmokingStatus"]
num_features = [
    c for c in test.columns if (test.dtypes[c] != "object") and (c not in cat_features)
]
features = num_features + cat_features
drop_features = [TARGET, "predict_Week", "Percent", "base_Week"]
features = [c for c in features if c not in drop_features]

if cat_features:
    ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown="impute")
    ce_oe.fit(train)
    train = ce_oe.transform(train)
    test = ce_oe.transform(test)



## === cell 22
""" ... """



## === cell 23
for alpha1 in [0.3]:
    for l1s in [0.8]:
        print(" For alpha:", alpha1, "& l1_ratio:", l1s)
        clf = ElasticNet(alpha=alpha1, l1_ratio=l1s)
        oof, predictions = run_kfold_model(
            clf, train, test, folds, features, target, n_fold=N_FOLD
        )

        train["FVC_pred"] = oof
        test["FVC_pred"] = predictions

        train["Confidence"] = 100
        train["sigma_clipped"] = train["Confidence"].apply(lambda x: max(x, 70))
        train["diff"] = abs(train["FVC"] - train["FVC_pred"])
        train["delta"] = train["diff"].apply(lambda x: min(x, 1000))
        train["score"] = -math.sqrt(2) * train["delta"] / train[
            "sigma_clipped"
        ] - np.log(math.sqrt(2) * train["sigma_clipped"])
        score = train["score"].mean()
        print("Baseline (conf=100) train-score:", score)

        def loss_func(weight, row):
            confidence = float(weight)
            sigma_clipped = max(confidence, 70)
            diff = abs(row["FVC"] - row["FVC_pred"])
            delta = min(diff, 1000)
            score = -math.sqrt(2) * delta / sigma_clipped - np.log(
                math.sqrt(2) * sigma_clipped
            )
            return -score

        results = []
        tk0 = tqdm(train.iterrows(), total=len(train))
        for _, row in tk0:
            loss_partial = partial(loss_func, row=row)
            weight0 = [100.0]
            result = sp.optimize.minimize(loss_partial, weight0, method="SLSQP")
            x = result["x"]
            results.append(float(x[0]))

        train["Confidence"] = results
        train["sigma_clipped"] = train["Confidence"].apply(lambda x: max(x, 70))
        train["diff"] = abs(train["FVC"] - train["FVC_pred"])
        train["delta"] = train["diff"].apply(lambda x: min(x, 1000))
        train["score"] = -math.sqrt(2) * train["delta"] / train[
            "sigma_clipped"
        ] - np.log(math.sqrt(2) * train["sigma_clipped"])
        score = train["score"].mean()
        print("Optimized (per-row conf) train-score:", score)



## === cell 24
TARGET = "Confidence"

target = train[TARGET]
test[TARGET] = np.nan

cat_features = ["Sex", "SmokingStatus"]
num_features = [
    c for c in test.columns if (test.dtypes[c] != "object") and (c not in cat_features)
]
features = num_features + cat_features
drop_features = [ID, TARGET, "predict_Week", "base_Week", "FVC", "FVC_pred"]
features = [c for c in features if c not in drop_features]

oof, predictions = run_kfold_model(
    clf, train, test, folds, features, target, n_fold=N_FOLD
)



## === cell 25
train["Confidence"] = oof
train["sigma_clipped"] = train["Confidence"].apply(lambda x: max(x, 70))
train["diff"] = abs(train["FVC"] - train["FVC_pred"])
train["delta"] = train["diff"].apply(lambda x: min(x, 1000))
train["score"] = -math.sqrt(2) * train["delta"] / train["sigma_clipped"] - np.log(
    math.sqrt(2) * train["sigma_clipped"]
)
score = train["score"].mean()
print("OOF score (conf model):", score)



## === cell 26
test["Confidence"] = predictions
test = test.reset_index()



## === cell 27
sub = submission[["Patient_Week"]].merge(
    test[["Patient_Week", "FVC_pred", "Confidence"]], on="Patient_Week"
)
sub = sub.rename(columns={"FVC_pred": "FVC"})

for i in range(len(otest)):
    sub.loc[
        sub["Patient_Week"] == otest.Patient[i] + "_" + str(otest.Weeks[i]), "FVC"
    ] = otest.FVC[i]
    sub.loc[
        sub["Patient_Week"] == otest.Patient[i] + "_" + str(otest.Weeks[i]),
        "Confidence",
    ] = 0.1

sub.to_csv("submission_2.csv", index=False, float_format="%.1f")



## === cell 28
import pydicom
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error



## === cell 29
ROOT = "../input/osic-pulmonary-fibrosis-progression"
DESIRED_SIZE = 128



## === cell 30
tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient")



## === cell 31
tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"
data = pd.concat([tr, chunk, sub], ignore_index=True)



## === cell 32
print(tr.shape, chunk.shape, sub.shape, data.shape)
print(
    tr.Patient.nunique(),
    chunk.Patient.nunique(),
    sub.Patient.nunique(),
    data.Patient.nunique(),
)



## === cell 33
data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")



## === cell 34
base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)



## === cell 35
data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base



## === cell 36
COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in data[col].dropna().unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)



## === cell 37
data["age"] = (data["Age"] - data["Age"].min()) / (
    data["Age"].max() - data["Age"].min()
)
data["BASE"] = (data["min_FVC"] - data["min_FVC"].min()) / (
    data["min_FVC"].max() - data["min_FVC"].min()
)
data["week"] = (data["base_week"] - data["base_week"].min()) / (
    data["base_week"].max() - data["base_week"].min()
)
data["percent"] = (data["Percent"] - data["Percent"].min()) / (
    data["Percent"].max() - data["Percent"].min()
)
FE += ["age", "percent", "week", "BASE"]



## === cell 38
tr = data.loc[data.WHERE == "train"].copy()
chunk = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data



## === cell 39
tr.shape, chunk.shape, sub.shape




## === cell 40
def get_images(df, how="train"):
    xo = []
    p = []
    w = []
    for i in tqdm(range(df.shape[0])):
        patient = df.iloc[i, 0]
        week = df.iloc[i, 1]
        try:
            img_path = f"{ROOT}/{how}/{patient}/{week}.dcm"
            ds = pydicom.dcmread(img_path)
            im = Image.fromarray(ds.pixel_array)
            im = im.resize((DESIRED_SIZE, DESIRED_SIZE))
            im = np.array(im)
            xo.append(im[np.newaxis, :, :])
            p.append(patient)
            w.append(week)
        except Exception:
            pass
    data = pd.DataFrame({"Patient": p, "Weeks": w})
    return np.concatenate(xo, axis=0), data




## === cell 41
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 42
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model(n_features):
    z = L.Input((n_features,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    model = M.Model(z, preds, name="CNN")
    model.compile(loss=mloss(0.8), optimizer="adam", metrics=[score])
    return model




## === cell 43
net = make_model(len(FE))
print(net.summary())
print(net.count_params())



## === cell 44
y = tr["FVC"].values.astype(np.float32)
z = tr[FE].values.astype(np.float32)
ze = sub[FE].values.astype(np.float32)
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)



## === cell 45
NFOLD = 9
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)



## === cell 46
cnt = 0
for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(len(FE))
    net.fit(
        z[tr_idx],
        y[tr_idx],
        batch_size=200,
        epochs=500,
        validation_data=(z[val_idx], y[val_idx]),
        verbose=0,
    )
    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=500))
    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=500))
    pred[val_idx] = net.predict(z[val_idx], batch_size=500, verbose=0)
    pe += net.predict(ze, batch_size=500, verbose=0) / NFOLD



## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1087794604.py in <cell line: 0>()
      6     print(f"FOLD {cnt}")
      7     net = make_model(len(FE))
----> 8     net.fit(
      9         z[tr_idx],
     10         y[tr_idx],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/358391139.py in loss(y_true, y_pred)
     28 def mloss(_lambda):
     29     def loss(y_true, y_pred):
---> 30         return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
     31 
     32     return loss

/tmp/ipykernel_11/358391139.py in score(y_true, y_pred)
      9 
     10     sigma_clip = tf.maximum(sigma, C1)
---> 11     delta = tf.abs(y_true[:, 0] - fvc_pred)
     12     delta = tf.minimum(delta, C2)
     13     sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))

ValueError: Index out of range using input dim 1; input has only 1 dims for '{{node compile_loss/loss/strided_slice_3}} = StridedSlice[Index=DT_INT32, T=DT_FLOAT, begin_mask=1, ellipsis_mask=0, end_mask=1, new_axis_mask=0, shrink_axis_mask=2](data_1, compile_loss/loss/strided_slice_3/stack, compile_loss/loss/strided_slice_3/stack_1, compile_loss/loss/strided_slice_3/stack_2)' with input shapes: [?], [2], [2], [2] and with computed input tensors: input[3] = <1 1>.

## === cell 47
sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = np.mean(unc)
print("sigma_opt, sigma_mean:", sigma_opt, sigma_mean)



## === cell 48
idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()



## === cell 49
print(
    "unc min/mean/max, nonneg rate:",
    float(unc.min()),
    float(unc.mean()),
    float(unc.max()),
    float((unc >= 0).mean()),
)



## === cell 50
plt.hist(unc, bins=50)
plt.title("uncertainty in prediction")
plt.show()



## === cell 51
sub.head()



## === cell 52
sub["FVC1"] = pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]



## === cell 53
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()



## === cell 54
subm.loc[~subm.FVC1.isnull()].head(10)



## === cell 55
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]
if sigma_mean < 70:
    subm["Confidence"] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]



## === cell 56
subm.head()



## === cell 57
subm.describe().T



## === cell 58
subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    subm[["Patient_Week", "FVC", "Confidence"]].shape,
)
