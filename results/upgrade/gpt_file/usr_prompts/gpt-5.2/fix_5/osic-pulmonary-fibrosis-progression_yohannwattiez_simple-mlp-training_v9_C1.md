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

No external packages required in the script and installed.

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

-6.876722047898075

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.64809) has done: 'I fix the feature-building merge for `X_prediction` so it creates the intended `Weeks` (prediction week) and `Base_week` (baseline week from test.csv) without column collisions that caused the KeyError. Then I make the custom `OneHotEncoder` robust to both sparse and dense outputs (newer sklearn returns dense arrays when `sparse=False`), and also catch the correct sklearn `NotFittedError` so the encoders fit on train before transforming. With those fixes, the downstream variables (`X_train`, `X_test`, `mu/sigma`, `sub`) be defined and the script run end-to-end and write a valid `submission.csv` in the required format.'
- What this solution (achieved -24.64809) has done: 'Your current score is far below the target (higher is better), so we should make a small, metric-aligned improvement without changing the overall modeling approach (still linear quantile regression with pinball loss and the same features). The largest issue for this competition is that predicting *every week* like the sample submission requires modeling the time trend relative to baseline; right now the model only sees absolute `Weeks` and `Base_week` separately, which makes learning the per-patient slope harder and hurts extrapolation. I add one derived feature `Week_delta = Weeks - Base_week` (minimal feature engineering consistent with your existing linear setup) and include it in `SELECTED_COLUMNS`. I also calibrate `Confidence` by using half the inter-quantile range (approximate sigma proxy) before applying the metric’s `max(.,70)` clip, which typically improves Laplace log-likelihood without changing the core prediction logic.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd




## === cell 1
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_all(20)



## === cell 2
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_raw = pd.read_csv(train_path)
raw_test = pd.read_csv(test_path)
X_prediction = pd.read_csv(sample_path)



## === cell 3
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.8  # kept for compatibility; training below uses only pinball (core idea: quantiles)

SELECTED_COLUMNS = [
    "Weeks",
    "Base_week",
    "Week_delta",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]



## === cell 4
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError as SklearnNotFittedError


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index="", name="", **kwargs):
        mat = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        if hasattr(mat, "toarray"):
            arr = mat.toarray()
        else:
            arr = np.asarray(mat)
        d_out = pd.DataFrame(arr, columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        new_columns = []
        for j in range(len(categories)):
            new_columns.append("{}_{}".format(name, categories[j]))
        return new_columns


class data_preparation:
    def __init__(self):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder(sparse=False, handle_unknown="ignore")

    def __call__(self, data_untransformed: pd.DataFrame) -> pd.DataFrame:
        data = data_untransformed.copy(deep=True)

        try:
            data["Sex"] = self.enc_sex.transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.transform(
                data["SmokingStatus"].values
            )
            oh = self.onehotenc_smok.transform(
                data["SmokingStatus"].values.reshape(-1, 1),
                categories=self.enc_smok.classes_,
                name="",
                index=data.index,
            ).astype(int)
            data = pd.concat([data.drop(columns=["SmokingStatus"]), oh], axis=1)

        except SklearnNotFittedError:
            data["Sex"] = self.enc_sex.fit_transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.fit_transform(
                data["SmokingStatus"].values
            )
            oh = self.onehotenc_smok.fit_transform(
                data["SmokingStatus"].values.reshape(-1, 1),
                categories=self.enc_smok.classes_,
                name="",
                index=data.index,
            ).astype(int)
            data = pd.concat([data.drop(columns=["SmokingStatus"]), oh], axis=1)

        return data




## === cell 5
base = (
    train_raw.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    .rename(
        columns={"Weeks": "Base_week", "FVC": "Base_FVC", "Percent": "Base_percent"}
    )
)

sample_tmp = pd.read_csv(sample_path)[["Patient_Week"]].copy()
sample_tmp["Patient"] = sample_tmp["Patient_Week"].str.extract(r"(.*)_.*")
sample_tmp["Weeks"] = sample_tmp["Patient_Week"].str.extract(r".*_(.*)").astype(int)
weeks_grid = sample_tmp[["Patient", "Weeks"]].drop_duplicates()

train_grid = weeks_grid.merge(
    train_raw[["Patient", "Weeks", "FVC"]],
    on=["Patient", "Weeks"],
    how="inner",
)

train = train_grid.merge(base, on="Patient", how="left", suffixes=("", "_base")).merge(
    train_raw[["Patient", "Age", "Sex", "SmokingStatus"]].drop_duplicates("Patient"),
    on="Patient",
    how="left",
    suffixes=("", "_clin"),
)

train = train[
    [
        "Patient",
        "Weeks",
        "FVC",
        "Base_week",
        "Base_FVC",
        "Base_percent",
        "Age",
        "Sex",
        "SmokingStatus",
    ]
].copy()

train["Week_delta"] = (train["Weeks"] - train["Base_week"]).astype(np.int32)



## === cell 6
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]].rename(
    columns={"Weeks": "Weeks"}
)

raw_test_base = raw_test.rename(
    columns={"Weeks": "Base_week", "FVC": "Base_FVC", "Percent": "Base_percent"}
)[["Patient", "Base_week", "Base_FVC", "Base_percent", "Age", "Sex", "SmokingStatus"]]

X_prediction = (
    X_prediction.merge(raw_test_base, how="left", on="Patient")
    .loc[
        :,
        [
            "Patient",
            "Base_week",
            "Base_FVC",
            "Base_percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ],
    ]
    .reset_index(drop=True)
)

X_prediction["Week_delta"] = (X_prediction["Weeks"] - X_prediction["Base_week"]).astype(
    np.int32
)



## === cell 7
data_prep = data_preparation()
train_p = data_prep(train).sort_values("Patient").reset_index(drop=True)
pred_p = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)

for c in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if c not in train_p.columns:
        train_p[c] = 0
    if c not in pred_p.columns:
        pred_p[c] = 0

X_train = train_p[SELECTED_COLUMNS].astype(np.float64).values
y_train = train_p["FVC"].astype(np.float64).values
X_test = pred_p[SELECTED_COLUMNS].astype(np.float64).values




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1193907873.py in __call__(self, data_untransformed)
     46         try:
---> 47             data["Sex"] = self.enc_sex.transform(data["Sex"].values)
     48             data["SmokingStatus"] = self.enc_smok.transform(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_label.py in transform(self, y)
    132         """
--> 133         check_is_fitted(self)
    134         y = column_or_1d(y, dtype=self.classes_.dtype, warn=True)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 

NotFittedError: This LabelEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3891860517.py in <cell line: 0>()
      1 data_prep = data_preparation()
----> 2 train_p = data_prep(train).sort_values("Patient").reset_index(drop=True)
      3 pred_p = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)
      4 
      5 for c in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:

/tmp/ipykernel_11/1193907873.py in __call__(self, data_untransformed)
     62                 data["SmokingStatus"].values
     63             )
---> 64             oh = self.onehotenc_smok.fit_transform(
     65                 data["SmokingStatus"].values.reshape(-1, 1),
     66                 categories=self.enc_smok.classes_,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/tmp/ipykernel_11/1193907873.py in fit_transform(self, X, categories, index, name, **kwargs)
     25 
     26     def fit_transform(self, X, categories, index, name, **kwargs):
---> 27         self.fit(X)
     28         return self.transform(X, categories=categories, index=index, name=name)
     29 

/tmp/ipykernel_11/1193907873.py in fit(self, X, **kwargs)
     10 
     11     def fit(self, X, **kwargs):
---> 12         out = super().fit(X)
     13         self.fit_flag = True
     14         return out

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_encoders.py in fit(self, X, y)
    876         self._check_infrequent_enabled()
    877 
--> 878         fit_results = self._fit(
    879             X,
    880             handle_unknown=self.handle_unknown,

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_encoders.py in _fit(self, X, handle_unknown, force_all_finite, return_counts)
     72         self._check_n_features(X, reset=True)
     73         self._check_feature_names(X, reset=True)
---> 74         X_list, n_samples, n_features = self._check_X(
     75             X, force_all_finite=force_all_finite
     76         )

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_encoders.py in _check_X(self, X, force_all_finite)
     44         if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
     45             # if not a dataframe, do normal check_array validation
---> 46             X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
     47             if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
     48                 X = check_array(X, dtype=object, force_all_finite=force_all_finite)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    929         n_samples = _num_samples(array)
    930         if n_samples < ensure_min_samples:
--> 931             raise ValueError(
    932                 "Found array with %d sample(s) (shape=%s) while a"
    933                 " minimum of %d is required%s."

ValueError: Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.

## === cell 8
def add_bias(X):
    return np.concatenate([np.ones((X.shape[0], 1), dtype=X.dtype), X], axis=1)


def pinball_grad(Xb, y, W, q_list):
    """
    Xb: (n, d+1)
    y: (n,)
    W: (d+1, 3)
    returns gradient dW of mean pinball loss over all quantiles
    """
    n = Xb.shape[0]
    Yhat = Xb @ W  # (n,3)
    dW = np.zeros_like(W)
    for j, q in enumerate(q_list):
        e = y - Yhat[:, j]  # (n,)
        dL_dyhat = np.where(e >= 0, -q, -(q - 1.0))  # (n,)
        dW[:, j] = (Xb.T @ dL_dyhat) / n
    return dW


def train_quantile_linear(X, y, q_list=(0.2, 0.5, 0.8), lr=0.05, epochs=400):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    Xb = add_bias(Xs)
    W = np.zeros((Xb.shape[1], len(q_list)), dtype=np.float64)

    for _ in range(epochs):
        g = pinball_grad(Xb, y, W, q_list)
        W -= lr * g

    return W, mu, sigma


W, mu, sigma = train_quantile_linear(
    X_train, y_train, q_list=tuple(PINBALL_QUANTILE), lr=0.05, epochs=400
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/423200248.py in <cell line: 0>()
     37 
     38 W, mu, sigma = train_quantile_linear(
---> 39     X_train, y_train, q_list=tuple(PINBALL_QUANTILE), lr=0.05, epochs=400
     40 )
     41 

NameError: name 'X_train' is not defined

## === cell 9
Xs_test = (X_test - mu) / sigma
Xb_test = add_bias(Xs_test)
yq = Xb_test @ W  # (n,3)

yq_sorted = np.sort(yq, axis=1)
q20 = yq_sorted[:, 0]
q50 = yq_sorted[:, 1]
q80 = yq_sorted[:, 2]

conf = (0.5 * (q80 - q20)).astype(np.float64)
conf = np.maximum(conf, 1e-6)
conf = np.maximum(conf, 70.0)

sub = pd.DataFrame(
    {
        "Patient_Week": pred_p["Patient_Week"].values,
        "FVC": q50.astype(np.float64),
        "Confidence": conf.astype(np.float64),
    }
)

sub = sub.merge(
    pd.read_csv(sample_path)[["Patient_Week"]], on="Patient_Week", how="right"
)

sub["FVC"] = sub["FVC"].astype(float)
sub["Confidence"] = sub["Confidence"].astype(float)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1936754958.py in <cell line: 0>()
----> 1 Xs_test = (X_test - mu) / sigma
      2 Xb_test = add_bias(Xs_test)
      3 yq = Xb_test @ W  # (n,3)
      4 
      5 # NOTE (score-improving, minimal): enforce non-crossing quantiles by sorting per-row.

NameError: name 'X_test' is not defined
