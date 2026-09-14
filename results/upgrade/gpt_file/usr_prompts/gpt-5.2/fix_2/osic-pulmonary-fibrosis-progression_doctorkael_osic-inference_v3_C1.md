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

3.9

# 3. Installed packages

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

-6.847914207559203

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

plt.style.use("dark_background")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
main_dir = "../input/osic-pulmonary-fibrosis-progression"

print("Listing:", main_dir)
print("\n".join(sorted(os.listdir(main_dir))[:50]))



## === cell 2
train_files = glob.glob(main_dir + "/train/*/*")
test_files = glob.glob(main_dir + "/test/*/*")

sample_sub = pd.read_csv(main_dir + "/sample_submission.csv")
train = pd.read_csv(main_dir + "/train.csv")
test = pd.read_csv(main_dir + "/test.csv")

print(
    "Number of train patients: {}\nNumber of test patients: {:4}".format(
        train.Patient.nunique(), test.Patient.nunique()
    )
)
print(
    "\nTotal number of Train patient records: {}\nTotal number of Test patient records: {:6}".format(
        len(train_files), len(test_files)
    )
)

print(train.shape, test.shape, sample_sub.shape)
sample_sub.head()




## === cell 3
def laplace_log_likelihood(y_true, y_pred, sigma=70):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)

    sigma_clipped = np.maximum(sigma, 70.0).astype(np.float32)
    delta = np.abs(y_true - y_pred)
    delta_clipped = np.minimum(delta, 1000.0).astype(np.float32)

    score = -np.sqrt(2.0) * delta_clipped / sigma_clipped - np.log(
        np.sqrt(2.0) * sigma_clipped
    )
    return float(np.mean(score))




## === cell 4
def l1(s):
    def scorer_func(x, y, sigma=s):
        return laplace_log_likelihood(x, y, sigma=s)

    return make_scorer(scorer_func, greater_is_better=False)




## === cell 5
def base_shift(data, q=50):
    x = data.copy()

    temp = (
        x.groupby("Patient")
        .apply(
            lambda g: g.loc[
                int(np.percentile(g["Weeks"].index, q=q)), ["Weeks", "FVC", "Percent"]
            ]
        )
        .reset_index(level=0)
    )

    temp = temp.rename(
        columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
    )

    x = x.merge(temp, on="Patient", how="left")
    x["Week_Offset"] = x["Weeks"] - x["Base_Week"]
    return x




## === cell 6
def multi_baseweek_frame(data, display=True):
    """Creates multiple base week frames to help model learn past/future dynamics."""
    op = data.merge(
        data[["Patient", "Weeks", "FVC", "Percent"]].rename(
            {"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}, axis=1
        ),
        on="Patient",
    )

    op["Week_Offset"] = op["Weeks"] - op["Base_Week"]
    op = op[op["Week_Offset"] != 0]

    if display:
        print(
            "Number of Samples:{:5} -> {:5}\nNumber of Columns:{:5} -> {:5}".format(
                data.shape[0], op.shape[0], data.shape[1], op.shape[1]
            )
        )

    return op.sort_values(by=["Patient", "Base_Week"]).reset_index(drop=True)




## === cell 7
def get_model_data(
    data,
    cat_cols,
    num_cols,
    to_drop,
    cat_method="1h",
    transform_stats=None,
    age_bins=None,
    train=True,
    display_stats=True,
    math=None,
    factor=False,
):
    X = data.copy().reset_index(drop=True)

    if age_bins:
        X["binned_age"] = (
            pd.cut(X["Age"], bins=range(0, 101, 100 // (age_bins - 1))).cat.codes
            / age_bins
        )
        to_drop = to_drop + ["Age"]

    if math:
        prod = np.ones(X.shape[0])
        if "sin" in math:
            X["Sin_week"] = X.groupby("Patient")["Weeks"].transform(np.sin)
            prod = prod * X["Sin_week"].values
        if "cos" in math:
            X["Cos_week"] = X.groupby("Patient")["Weeks"].transform(np.cos)
            prod = prod * X["Cos_week"].values
        if "tan" in math:
            X["Tan_week"] = X.groupby("Patient")["Weeks"].transform(np.tan)
            prod = prod * X["Cos_week"].values

        if len(math) > 1:
            X["Math_Prod"] = prod

    if factor:
        X["factor"] = X["Base_FVC"] / X["Base_Percent"]

    if cat_cols != []:
        if cat_method == "ord":
            global ordenc
            if train:
                ordenc = OrdinalEncoder()
                X = X.merge(
                    pd.DataFrame(
                        ordenc.fit_transform(X[cat_cols]).astype(int),
                        columns=[c + "_ord" for c in cat_cols],
                    ),
                    left_index=True,
                    right_index=True,
                )
            else:
                X = X.merge(
                    pd.DataFrame(
                        ordenc.transform(X[cat_cols]).astype(int),
                        columns=[c + "_ord" for c in cat_cols],
                    ),
                    left_index=True,
                    right_index=True,
                )

        elif cat_method == "1h":
            global onehenc
            if train:
                onehenc = OneHotEncoder(handle_unknown="ignore")
                X_oh = onehenc.fit_transform(X[cat_cols]).toarray()
                X = X.merge(
                    pd.DataFrame(X_oh, columns=[*np.concatenate(onehenc.categories_)]),
                    left_index=True,
                    right_index=True,
                )
            else:
                X_oh = onehenc.transform(X[cat_cols]).toarray()
                X = X.merge(
                    pd.DataFrame(X_oh, columns=[*np.concatenate(onehenc.categories_)]),
                    left_index=True,
                    right_index=True,
                )

        elif cat_method == "poly":
            global cat_comb
            if train:
                cat_comb = np.array(
                    np.meshgrid(*[X[cat].unique() for cat in cat_cols])
                ).T.reshape(-1, len(cat_cols))

            for combination in cat_comb:
                name = "_".join(map(str, combination))
                X[name] = 1
                for i in range(len(cat_cols)):
                    X[name] = X[name] & (X[cat_cols[i]] == combination[i]).astype(int)

    to_drop = to_drop + cat_cols

    global stats
    if train:
        if transform_stats is None:
            stats = X.describe().T
        else:
            stats = transform_stats

    for col in num_cols:
        if (not train) and (col not in X.columns):
            continue
        denom = stats.loc[col, "max"] - stats.loc[col, "min"]
        if denom == 0 or pd.isna(denom):
            X[col] = 0.0
        else:
            X[col] = (X[col] - stats.loc[col, "min"]) / denom

    global x_cols
    if train:
        Y = X["FVC"].dropna()

        if display_stats and "FVC" in X.columns:
            print(
                X.corr(numeric_only=True)["FVC"].abs().sort_values(ascending=False)[1:]
            )

        X = X.drop(to_drop, axis=1)
        x_cols = X.columns
        return X, Y
    else:
        X = X.drop(to_drop, axis=1, errors="ignore")
        X = X.reindex(columns=x_cols, fill_value=0.0)
        return X




## === cell 8
def augment_train_cosine(data, n_similar=3, threshold=0.25, display_sample=True):
    from sklearn.metrics.pairwise import cosine_similarity

    temp = base_shift(data.copy(), q=0)

    temp["present_minus_past"] = (
        temp.groupby("Patient")["FVC"].transform("diff").fillna(0)
    )
    temp["Week_diff"] = temp.groupby("Patient")["Weeks"].transform("diff").fillna(0)
    temp["pms"] = (
        (temp["present_minus_past"] / temp["Week_diff"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    temp["pms_min"] = temp.groupby("Patient")["pms"].transform("min")
    temp["pms_25"] = temp.groupby("Patient")["pms"].transform(
        lambda x: np.percentile(x, q=25)
    )
    temp["pms_mean"] = temp.groupby("Patient")["pms"].transform("mean")
    temp["pms_75"] = temp.groupby("Patient")["pms"].transform(
        lambda x: np.percentile(x, q=75)
    )
    temp["pms_max"] = temp.groupby("Patient")["pms"].transform("max")
    temp["pms_sum"] = temp.groupby("Patient")["pms"].transform("sum")

    temp["pmb_avg"] = (
        (temp["Percent"] - temp["Base_Percent"]).groupby(temp.Patient).transform("mean")
    )
    temp["p_std"] = temp.groupby("Patient")["Percent"].transform("std")

    temp = temp.merge(
        pd.concat(
            [
                temp.groupby("Patient")
                .apply(
                    lambda x: (x["Percent"].values[-1] - x["Percent"].values[0])
                    / (x["Weeks"].values[-1] - x["Weeks"].values[0])
                )
                .rename("Slope"),
                temp.groupby("Patient")
                .apply(lambda x: x["Week_Offset"].iloc[np.argmax(x["pms"])])
                .rename("pmsw_max"),
                temp.groupby("Patient")
                .apply(lambda x: x["Week_Offset"].iloc[np.argmin(x["pms"])])
                .rename("pmsw_min"),
            ],
            axis=1,
        ).reset_index(),
        on="Patient",
        how="left",
    )

    temp = temp.groupby("Patient").head(1).reset_index(drop=True)
    temp["factor"] = temp["Base_FVC"] / temp["Base_Percent"]

    train_cols = [
        "Patient",
        "pms_25",
        "pms_mean",
        "pms_75",
        "pms_sum",
        "pmsw_min",
        "pmsw_max",
        "pmb_avg",
        "Slope",
        "factor",
    ]

    cat_cols = np.intersect1d(["Sex", "SmokingStatus"], train_cols)
    temp = temp[train_cols]
    temp = pd.get_dummies(
        temp, columns=cat_cols, drop_first=True, prefix="", prefix_sep=""
    )

    n_similar += 1
    sim = cosine_similarity(
        temp.drop(columns=["Patient"]), temp.drop(columns=["Patient"])
    )
    groups = pd.DataFrame(np.argsort(sim)[:, -1 : -n_similar - 1 : -1])

    groups = groups[~pd.DataFrame(np.sort(groups.values, axis=1)).duplicated()]
    groups = (
        groups.applymap(lambda x: temp.Patient.to_dict()[x])
        .apply(list, axis=1)
        .to_dict()
    )

    aug_data = []
    for group in tqdm(groups.values(), disable=not display_sample):
        tmp = base_shift(train[train.Patient.isin(group)], q=0)

        tmp["base_per_diff_from_mean"] = (
            tmp["Base_Percent"] - tmp["Base_Percent"].unique().mean()
        )
        tmp["Percent_shifted"] = tmp["Percent"] - tmp["base_per_diff_from_mean"]

        tmp["base_week_diff_from_mean"] = (
            tmp["Base_Week"] - tmp["Base_Week"].unique().mean()
        )
        tmp["Week_shifted"] = tmp["Weeks"] - tmp["base_week_diff_from_mean"]

        tmp = pd.merge_ordered(
            tmp.drop(
                columns=["Age", "Sex", "SmokingStatus", "Base_Week", "Week_Offset"]
            ),
            (
                tmp.groupby("Week_shifted")["Percent_shifted"]
                .agg(["mean", "count"])
                .query(f"count > {n_similar * threshold}")
                .drop(columns=["count"])
            ),
            on="Week_shifted",
            left_by="Patient",
        )

        aug_data.append(tmp)

    tmp = pd.concat(aug_data).reset_index(drop=True)

    tmp[
        [
            "base_per_diff_from_mean",
            "base_week_diff_from_mean",
            "Base_Percent",
            "Base_FVC",
        ]
    ] = tmp.groupby("Patient")[
        [
            "base_per_diff_from_mean",
            "base_week_diff_from_mean",
            "Base_Percent",
            "Base_FVC",
        ]
    ].ffill()

    tmp["Week_aug"] = tmp["Week_shifted"] + tmp["base_week_diff_from_mean"]

    tmp["Percent_aug"] = np.where(
        tmp["Percent"].isna(),
        tmp["mean"] + tmp["base_per_diff_from_mean"],
        tmp["Percent"],
    )

    tmp = tmp[~tmp["Percent_aug"].isna()]

    test_ids = tmp["FVC"].isna()
    tmp.loc[test_ids, "FVC"] = (
        LinearRegression()
        .fit(
            tmp.loc[~test_ids, ["Base_Percent", "Percent_aug", "Base_FVC"]],
            tmp.loc[~test_ids, "FVC"],
        )
        .predict(tmp.loc[test_ids, ["Base_Percent", "Percent_aug", "Base_FVC"]])
    )

    tmp = tmp.groupby(["Patient", "Week_aug"]).mean(numeric_only=True).reset_index()

    tmp = tmp[["Patient", "Week_aug", "Percent_aug", "FVC"]]
    tmp = tmp.rename({"Week_aug": "Weeks", "Percent_aug": "Percent"}, axis=1)
    tmp["Weeks"] = tmp["Weeks"].astype(int)

    tmp = tmp.merge(
        data[["Patient", "Sex", "Age", "SmokingStatus"]]
        .groupby("Patient")
        .head(1)
        .reset_index(drop=True),
        on="Patient",
        how="left",
    )

    return tmp




## === cell 9
sub = sample_sub.Patient_Week.str.extract(r"(ID\w+)_(\-?\d+)").rename(
    {0: "Patient", 1: "Weeks"}, axis=1
)
sub["Weeks"] = sub["Weeks"].astype(int)
sub = pd.merge(sub, test[["Patient", "Sex", "SmokingStatus"]], on="Patient", how="left")
sub["Patient_Week"] = sub.Patient + "_" + sub.Weeks.astype(str)
sub.head()




## === cell 10
class GBR(RegressorMixin, BaseEstimator):
    def __init__(self, alpha=0.75, **params):
        self.alpha = alpha
        self.umodel = self._create_model(loss="quantile", q=self.alpha, **params)
        self.mmodel = self._create_model(loss="lad", **params)
        self.lmodel = self._create_model(loss="quantile", q=1 - self.alpha, **params)

    def _create_model(self, loss, q=0.75, **params):
        model = GradientBoostingRegressor(
            init=LinearRegression(),
            criterion="friedman_mse",
            n_estimators=50,
            max_depth=2,
            loss=loss,
            alpha=q,
            random_state=RANDOM_STATE,
            **params,
        )
        return model

    def fit(self, x, y):
        self.umodel.fit(x, y)
        self.mmodel.fit(x, y)
        self.lmodel.fit(x, y)
        return self

    def predict(self, X):
        return self.mmodel.predict(X)

    def predict_forecast(self, X, return_bounds=False):
        preds = self.mmodel.predict(X)
        upper = self.umodel.predict(X)
        lower = self.lmodel.predict(X)
        if return_bounds:
            return preds, upper, lower
        return preds, (upper - lower)




## === cell 11
stats = (
    multi_baseweek_frame(pd.concat([train, test], ignore_index=True), display=False)
    .describe()
    .T
)
stats.head()



## === cell 12
op = multi_baseweek_frame(
    augment_train_cosine(train, display_sample=False, n_similar=5, threshold=0.5),
    display=False,
)
op.shape, op.head()



## === cell 13
cat_cols = ["Sex", "SmokingStatus"]
to_drop = ["FVC", "Percent", "Weeks", "factor", "Base_Percent"]
num_cols = [
    "Weeks",
    "Week_Offset",
    "Base_Week",
    "Age",
    "Base_FVC",
    "Percent",
    "Base_Percent",
]
cat_method = "ord"
math = []
age_bins = 5

folds = 7
total_patients = train.Patient.unique().copy()
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(total_patients)
val_len = len(total_patients) // folds

temp_rows = []
pred_folds = []

X, Y = get_model_data(
    op,
    factor=True,
    num_cols=num_cols,
    cat_cols=cat_cols,
    age_bins=age_bins,
    to_drop=to_drop,
    display_stats=False,
    cat_method=cat_method,
    transform_stats=stats,
    math=math,
)

X_VAL = base_shift(train, q=0)
Y_VAL = X_VAL["FVC"].dropna()
X_VAL["Percent"] = X_VAL["Base_Percent"]
X_VAL = get_model_data(
    X_VAL,
    num_cols=num_cols,
    cat_cols=cat_cols,
    to_drop=to_drop,
    factor=True,
    cat_method=cat_method,
    train=False,
    age_bins=age_bins,
    math=math,
)

x_test = sub[["Patient", "Weeks"]].merge(
    test.rename(
        {"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}, axis=1
    ),
    on="Patient",
    how="left",
)
x_test["Week_Offset"] = x_test["Weeks"] - x_test["Base_Week"]
x_test["Percent"] = x_test["Base_Percent"]
x_test = get_model_data(
    x_test,
    cat_cols=cat_cols,
    num_cols=num_cols,
    to_drop=to_drop,
    math=math,
    factor=True,
    train=False,
    cat_method=cat_method,
    age_bins=age_bins,
)
x_test_nopat = x_test.drop(columns=["Patient"], errors="ignore")

for i in range(folds):
    val_patients = total_patients[(i) * val_len : (i + 1) * val_len]
    train_patients = np.setdiff1d(total_patients, val_patients)
    assert len(np.intersect1d(val_patients, train_patients)) == 0

    tr_mask = X.Patient.isin(train_patients)
    va_mask = X_VAL.Patient.isin(val_patients)

    x_tr = X.loc[tr_mask].drop(columns=["Patient"])
    y_tr = Y.loc[tr_mask]

    x_va = X_VAL.loc[va_mask].drop(columns=["Patient"])
    y_va = Y_VAL.loc[va_mask]

    model = GBR(alpha=0.75)
    model.fit(x_tr, y_tr)

    y_mid, y_up, y_lo = model.predict_forecast(x_va, return_bounds=True)
    y_mid_test, y_conf_test = model.predict_forecast(x_test_nopat)

    print(
        "For Fold #{} Val Score: {:.2f} @ 70 Confidence | {:.2f} @ Pred Confidence".format(
            i + 1,
            -laplace_log_likelihood(y_va, y_mid, 70),
            -laplace_log_likelihood(y_va, y_mid, (y_up - y_lo)),
        )
    )

    temp_rows.append(
        pd.DataFrame(
            {"upper": y_up, "lower": y_lo, "pred": y_mid, "actual": y_va.to_numpy()}
        )
    )
    pred_folds.append(pd.DataFrame({"pred": y_mid_test, "Confidence": y_conf_test}))

temp = pd.concat(temp_rows, ignore_index=True)
temp["Confidence"] = temp["upper"] - temp["lower"]

preds = pd.concat(pred_folds, axis=0).groupby(level=0).mean()
preds.head()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_55/2325002847.py in <cell line: 0>()
     89 
     90     model = GBR(alpha=0.75)
---> 91     model.fit(x_tr, y_tr)
     92 
     93     y_mid, y_up, y_lo = model.predict_forecast(x_va, return_bounds=True)

/tmp/ipykernel_55/3654468124.py in fit(self, x, y)
     21     def fit(self, x, y):
     22         self.umodel.fit(x, y)
---> 23         self.mmodel.fit(x, y)
     24         self.lmodel.fit(x, y)
     25         return self

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in fit(self, X, y, sample_weight, monitor)
    418             Fitted estimator.
    419         """
--> 420         self._validate_params()
    421 
    422         if not self.warm_start:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_params(self)
    598         accepted constraints.
    599         """
--> 600         validate_parameter_constraints(
    601             self._parameter_constraints,
    602             self.get_params(deep=False),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in validate_parameter_constraints(parameter_constraints, params, caller_name)
     95                 )
     96 
---> 97             raise InvalidParameterError(
     98                 f"The {param_name!r} parameter of {caller_name} must be"
     99                 f" {constraints_str}. Got {param_val!r} instead."

InvalidParameterError: The 'loss' parameter of GradientBoostingRegressor must be a str among {'absolute_error', 'quantile', 'huber', 'squared_error'}. Got 'lad' instead.

## === cell 14
pat_scores = (
    temp.reset_index(drop=True)
    .groupby(train.Patient, sort=False)
    .apply(lambda x: -laplace_log_likelihood(x["actual"], x["pred"], x["Confidence"]))
)
pat_scores = (
    pat_scores.rename("scores").reset_index().sort_values("scores", ascending=False)
)

pat_scores = pat_scores.head(8)
print("Worst Patient-Mean-Score: {:.2f}".format(pat_scores.scores.mean()))
pat_scores = pat_scores.Patient.values



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/277365291.py in <cell line: 0>()
      1 # Fix pandas rename error: groupby(...).apply returns Series -> name it directly
      2 pat_scores = (
----> 3     temp.reset_index(drop=True)
      4     .groupby(train.Patient, sort=False)
      5     .apply(lambda x: -laplace_log_likelihood(x["actual"], x["pred"], x["Confidence"]))

NameError: name 'temp' is not defined

## === cell 15
print(
    "\n|================== Summary ==================|\n"
    "Score on Total Dataset: {:.3f} @   70 Confidence\n"
    "Score on Total Dataset: {:.3f} @  225 Confidence\n"
    "Score on Total Dataset: {:.3f} @ Pred Confidence".format(
        -laplace_log_likelihood(temp["actual"], temp["pred"], 70),
        -laplace_log_likelihood(temp["actual"], temp["pred"], 225),
        -laplace_log_likelihood(temp["actual"], temp["pred"], temp["Confidence"]),
    )
)

try:
    f, ax = plt.subplots(figsize=(18, 18), nrows=4, ncols=2)
    ax = ax.ravel()
    for i, pat in enumerate(pat_scores):
        (
            temp.reset_index(drop=True)
            .loc[train.Patient == pat]
            .drop(columns=["Confidence"])
            .plot(ax=ax[i], legend=False)
        )
    f.suptitle("Model's Worst Predictions", size=16)
    f.tight_layout()
except Exception as e:
    print("Plot skipped:", repr(e))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/243563836.py in <cell line: 0>()
      4     "Score on Total Dataset: {:.3f} @  225 Confidence\n"
      5     "Score on Total Dataset: {:.3f} @ Pred Confidence".format(
----> 6         -laplace_log_likelihood(temp["actual"], temp["pred"], 70),
      7         -laplace_log_likelihood(temp["actual"], temp["pred"], 225),
      8         -laplace_log_likelihood(temp["actual"], temp["pred"], temp["Confidence"]),

NameError: name 'temp' is not defined

## === cell 16
sub_out = sub.copy()
sub_out["FVC"] = preds["pred"].values
sub_out["Confidence"] = preds["Confidence"].values

for i in range(len(test)):
    key = test.Patient.iloc[i] + "_" + str(int(test.Weeks.iloc[i]))
    m = sub_out["Patient_Week"] == key
    sub_out.loc[m, "FVC"] = float(test.FVC.iloc[i])
    sub_out.loc[m, "Confidence"] = 70.0

sub_out[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    sub_out[["Patient_Week", "FVC", "Confidence"]].shape,
)
sub_out.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1997724326.py in <cell line: 0>()
      2 # Also ensure baseline rows in test are exactly copied with confidence=70, as in original.
      3 sub_out = sub.copy()
----> 4 sub_out["FVC"] = preds["pred"].values
      5 sub_out["Confidence"] = preds["Confidence"].values
      6 

NameError: name 'preds' is not defined
