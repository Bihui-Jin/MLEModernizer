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

-7.66964

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.64809) has done: 'I fix the feature-building merge for `X_prediction` so it creates the intended `Weeks` (prediction week) and `Base_week` (baseline week from test.csv) without column collisions that caused the KeyError. Then I make the custom `OneHotEncoder` robust to both sparse and dense outputs (newer sklearn returns dense arrays when `sparse=False`), and also catch the correct sklearn `NotFittedError` so the encoders fit on train before transforming. With those fixes, the downstream variables (`X_train`, `X_test`, `mu/sigma`, `sub`) be defined and the script run end-to-end and write a valid `submission.csv` in the required format.'
- What this solution (achieved -24.64809) has done: 'Your current score is far below the target (higher is better), so we should make a small, metric-aligned improvement without changing the overall modeling approach (still linear quantile regression with pinball loss and the same features). The largest issue for this competition is that predicting *every week* like the sample submission requires modeling the time trend relative to baseline; right now the model only sees absolute `Weeks` and `Base_week` separately, which makes learning the per-patient slope harder and hurts extrapolation. I add one derived feature `Week_delta = Weeks - Base_week` (minimal feature engineering consistent with your existing linear setup) and include it in `SELECTED_COLUMNS`. I also calibrate `Confidence` by using half the inter-quantile range (approximate sigma proxy) before applying the metric’s `max(.,70)` clip, which typically improves Laplace log-likelihood without changing the core prediction logic.'
- What this solution (achieved -24.64809) has done: 'I fix the data-prep failure that causes `train` to become empty (so the encoders see 0 samples) by building the training set directly from `train.csv` instead of inner-joining to the sample-submission week grid. Then I make the encoder fitting deterministic and safe by explicitly fitting on train first and only transforming test, avoiding reliance on catching the wrong exception path. Finally, I keep your existing quantile linear pinball setup unchanged and ensure we always write a valid `submission.csv` with the required columns and correct alignment to `sample_submission.csv`.'
- What this solution (achieved -10.73551) has done: 'Your score is far below the target (higher is better), so we make minimal, metric-aligned improvements without changing the core “linear quantile regression with pinball loss” approach. The biggest gain with small risk is to (1) train on a target that is easier to extrapolate: predict residuals relative to each patient’s baseline FVC, then add baseline back at inference; this preserves the same model and loss, just a stable re-parameterization. We also (2) standardize the prediction features using the same train-time scaling (already done) but ensure the *baseline* is treated consistently by keeping `Base_FVC` out of the learned target (since it’s added back deterministically), reducing leakage of trivial identity mapping that harms slope learning. Finally, we (3) tune Confidence slightly upward via a single scalar multiplier before the required `max(.,70)` clip, which often improves the Laplace log-likelihood when errors are otherwise under-estimated—this is a minimal post-processing consistent with the metric.'
- What this solution (achieved -10.73551) has done: 'We make two minimal, metric-aligned adjustments that keep your core quantile-linear pinball setup intact but should improve the Laplace log-likelihood toward the target. First, we calibrate `Confidence` using the normal-approximation from the inter-quantile range (`sigma ≈ IQR/1.2816` for 20/80%), instead of `0.5*(q80-q20)`, because the metric directly rewards well-calibrated (not under/over) uncertainty. Second, we use a single global `CONF_MULT` chosen via a small, deterministic out-of-fold calibration on the training set (still the same model/loops/features), which typically improves score with low risk. These changes only affect post-processing of predicted quantiles into `(FVC, Confidence)` and do not alter your model architecture or training procedure.'
- What this solution (achieved -10.73551) has done: 'Your current score (-10.73551) is still materially worse than the target (-6.8767), so we should cautiously improve (higher is better) with minimal, metric-aligned changes while keeping the same linear-quantile pinball training. The biggest low-risk issue is that your quantile heads can cross, and you “fix” that by sorting per-row, which breaks the intended quantile meaning and can mis-calibrate Confidence; instead we enforce non-crossing quantiles with a minimal monotonic adjustment. Then we recalibrate the Confidence multiplier using a proper out-of-fold procedure (still the same model/loops/features) to better match the competition’s Laplace log-likelihood, avoiding overly optimistic in-sample calibration. These two changes typically move the score upward without changing the core modeling approach or adding heavy computation.'
- What this solution (achieved -10.73414) has done: 'We keep your quantile linear pinball setup unchanged and focus on one low-risk, metric-aligned improvement: add a per-patient “slope prior” by fitting a simple linear trend (FVC vs Weeks) on each training patient and using that slope as an additional numeric feature for both train and test rows. This helps extrapolate from baseline to future weeks (the core difficulty) without changing the model class or training loop. We also compute the test-time slope feature from the training-set average slope (since test has only one point), which is deterministic and avoids leakage. Everything else (residual target, non-crossing quantiles, OOF confidence calibration, submission writing) stays the same.'
- What this solution (achieved -10.73632) has done: 'Your current score (-10.73414) is worse than the target (-6.8767), so we should improve (higher is better) with minimal, metric-aligned changes. The biggest low-risk gain is to stop using a single global slope for all test rows and instead estimate a per-test-patient slope from similar training patients using only baseline metadata (Age/Sex/SmokingStatus/Base_percent), then use that as the `Slope_FVC_per_week` feature—this keeps the same model and training loop. We also fit the confidence multiplier with patient-grouped folds (instead of random row folds) to reduce leakage and better calibrate uncertainty, which the metric rewards directly. All paths stay the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved -10.73113) has done: 'Your current score (-10.73632) is still well below the target (-6.8767), so we should make a small, low-risk improvement that better matches the competition metric without changing your core quantile-linear pinball training. The main issue is that the metric clips confidence at 70 and rewards well-calibrated uncertainty; using only (q80-q20) can understate uncertainty for some cases, so we blend in a small “floor” based on the model’s global residual dispersion to reduce overconfidence. We also extend the confidence multiplier grid slightly upward (still chosen via your existing patient-grouped OOF calibration) to allow reaching a better uncertainty scale when needed. These are minimal post-processing/calibration changes: same features, same model, same training loop, just a more metric-aligned Confidence.'
- What this solution (achieved -7.66964) has done: 'We keep your quantile-linear pinball training exactly as-is and focus on metric-aligned calibration that can improve the Laplace log-likelihood toward the target. The smallest likely win is to calibrate both (a) the confidence multiplier and (b) a small additive sigma offset (in ml) using the same patient-grouped OOF procedure you already use, because the metric strongly penalizes mis-calibrated uncertainty. This does not change model architecture, features, loss, or training loop—only post-processing that maps predicted quantiles to `Confidence`. We also expand the multiplier grid slightly around your current best region and keep runtime bounded (still 5 folds, same epochs).'

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
sample_sub = pd.read_csv(sample_path)



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
    "Slope_FVC_per_week",
]



## === cell 4
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder


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
        self.onehotenc_smok = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )

    def fit(self, data_untransformed: pd.DataFrame):
        data = data_untransformed.copy(deep=True)
        self.enc_sex.fit(data["Sex"].astype(str).values)
        self.enc_smok.fit(data["SmokingStatus"].astype(str).values)
        smok_int = self.enc_smok.transform(data["SmokingStatus"].astype(str).values)
        self.onehotenc_smok.fit(smok_int.reshape(-1, 1))
        return self

    def transform(self, data_untransformed: pd.DataFrame) -> pd.DataFrame:
        data = data_untransformed.copy(deep=True)

        data["Sex"] = self.enc_sex.transform(data["Sex"].astype(str).values)
        data["SmokingStatus"] = self.enc_smok.transform(
            data["SmokingStatus"].astype(str).values
        )

        oh = self.onehotenc_smok.transform(
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

train = (
    train_raw.merge(base, on="Patient", how="left", suffixes=("", "_base"))
    .loc[
        :,
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
        ],
    ]
    .copy()
)
train["Week_delta"] = (train["Weeks"] - train["Base_week"]).astype(np.int32)


def _safe_slope(x_weeks: np.ndarray, y_fvc: np.ndarray) -> float:
    x = x_weeks.astype(np.float64)
    y = y_fvc.astype(np.float64)
    if x.size < 2:
        return 0.0
    x0 = x - x.mean()
    denom = float(np.dot(x0, x0))
    if denom <= 0:
        return 0.0
    return float(np.dot(x0, y - y.mean()) / denom)


slopes = (
    train_raw.groupby("Patient")
    .apply(lambda g: _safe_slope(g["Weeks"].values, g["FVC"].values))
    .rename("Slope_FVC_per_week")
    .reset_index()
)

global_slope = float(slopes["Slope_FVC_per_week"].mean())
train = train.merge(slopes, on="Patient", how="left")
train["Slope_FVC_per_week"] = (
    train["Slope_FVC_per_week"].fillna(global_slope).astype(np.float64)
)

train_base_for_slope = base.merge(slopes, on="Patient", how="left").copy()
train_base_for_slope["Slope_FVC_per_week"] = train_base_for_slope[
    "Slope_FVC_per_week"
].fillna(global_slope)



## === cell 6
X_prediction = sample_sub[["Patient_Week"]].copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)

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


def _prep_nn_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Sex"] = out["Sex"].astype(str)
    out["SmokingStatus"] = out["SmokingStatus"].astype(str)
    out["_is_male"] = (out["Sex"] == "Male").astype(np.float64)
    out["_is_female"] = (out["Sex"] == "Female").astype(np.float64)
    out["_smoke_current"] = (out["SmokingStatus"] == "Currently smokes").astype(
        np.float64
    )
    out["_smoke_ex"] = (out["SmokingStatus"] == "Ex-smoker").astype(np.float64)
    out["_smoke_never"] = (out["SmokingStatus"] == "Never smoked").astype(np.float64)
    for c in ["Age", "Base_percent"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(out[c].median())
    return out


train_nn = _prep_nn_features(
    train_base_for_slope[
        ["Patient", "Age", "Sex", "SmokingStatus", "Base_percent", "Slope_FVC_per_week"]
    ]
)
test_nn = _prep_nn_features(
    raw_test_base[["Patient", "Age", "Sex", "SmokingStatus", "Base_percent"]]
)

train_feat = train_nn[
    [
        "Age",
        "Base_percent",
        "_is_male",
        "_is_female",
        "_smoke_current",
        "_smoke_ex",
        "_smoke_never",
    ]
].to_numpy(np.float64)
test_feat = test_nn[
    [
        "Age",
        "Base_percent",
        "_is_male",
        "_is_female",
        "_smoke_current",
        "_smoke_ex",
        "_smoke_never",
    ]
].to_numpy(np.float64)

f_mu = train_feat.mean(axis=0)
f_sig = train_feat.std(axis=0)
f_sig[f_sig == 0] = 1.0
train_feat_z = (train_feat - f_mu) / f_sig
test_feat_z = (test_feat - f_mu) / f_sig

k = 15
sl = train_nn["Slope_FVC_per_week"].to_numpy(np.float64)

test_slopes = np.empty(test_feat_z.shape[0], dtype=np.float64)
for i in range(test_feat_z.shape[0]):
    d = train_feat_z - test_feat_z[i]
    dist2 = np.einsum("ij,ij->i", d, d)
    nn_idx = np.argpartition(dist2, kth=min(k, dist2.size - 1))[:k]
    w = 1.0 / (dist2[nn_idx] + 1e-6)
    test_slopes[i] = float(np.sum(w * sl[nn_idx]) / np.sum(w))

test_slope_map = dict(zip(test_nn["Patient"].values, test_slopes))
X_prediction["Slope_FVC_per_week"] = (
    X_prediction["Patient"].map(test_slope_map).fillna(global_slope).astype(np.float64)
)



## === cell 7
data_prep = data_preparation()
data_prep.fit(train)

train_p = data_prep.transform(train).sort_values("Patient").reset_index(drop=True)
pred_p = data_prep.transform(X_prediction).sort_values("Patient").reset_index(drop=True)

for c in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if c not in train_p.columns:
        train_p[c] = 0
    if c not in pred_p.columns:
        pred_p[c] = 0

if "Slope_FVC_per_week" not in train_p.columns:
    train_p["Slope_FVC_per_week"] = global_slope
if "Slope_FVC_per_week" not in pred_p.columns:
    pred_p["Slope_FVC_per_week"] = global_slope

X_train = train_p[SELECTED_COLUMNS].astype(np.float64).values

y_train = (
    train_p["FVC"].astype(np.float64).values
    - train_p["Base_FVC"].astype(np.float64).values
)

X_test = pred_p[SELECTED_COLUMNS].astype(np.float64).values




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




## === cell 9
def laplace_metric_np(y_true, y_pred, sigma):
    sigma_clipped = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return -(np.sqrt(2.0) * delta) / sigma_clipped - np.log(
        np.sqrt(2.0) * sigma_clipped
    )


def enforce_non_crossing(q20, q50, q80, eps=1e-6):
    q50_adj = np.maximum(q50, q20 + eps)
    q80_adj = np.maximum(q80, q50_adj + eps)
    return q20, q50_adj, q80_adj


IQR20_80_TO_SIGMA = 1.281551565545


def choose_conf_calibration_oof_grouped_by_patient(
    X,
    y,
    base_fvc,
    patients,
    q_list,
    lr,
    epochs,
    mult_grid,
    add_grid,
    sigma_floor,
    floor_weight,
    n_folds=5,
    seed=20,
):
    rng = np.random.RandomState(seed)

    uniq_pat = np.unique(patients)
    rng.shuffle(uniq_pat)
    pat_folds = np.array_split(uniq_pat, n_folds)

    all_q20 = np.empty(X.shape[0], dtype=np.float64)
    all_q50 = np.empty(X.shape[0], dtype=np.float64)
    all_q80 = np.empty(X.shape[0], dtype=np.float64)

    for k in range(n_folds):
        val_pats = set(pat_folds[k].tolist())
        val_mask = np.array([p in val_pats for p in patients], dtype=bool)
        tr_mask = ~val_mask

        Wk, muk, sigk = train_quantile_linear(
            X[tr_mask], y[tr_mask], q_list=q_list, lr=lr, epochs=epochs
        )
        Xb_val = add_bias((X[val_mask] - muk) / sigk)
        yq_val = Xb_val @ Wk  # (n_val,3)

        q20_abs = yq_val[:, 0] + base_fvc[val_mask]
        q50_abs = yq_val[:, 1] + base_fvc[val_mask]
        q80_abs = yq_val[:, 2] + base_fvc[val_mask]
        q20_abs, q50_abs, q80_abs = enforce_non_crossing(q20_abs, q50_abs, q80_abs)

        all_q20[val_mask] = q20_abs
        all_q50[val_mask] = q50_abs
        all_q80[val_mask] = q80_abs

    sigma_iqr = (all_q80 - all_q20) / IQR20_80_TO_SIGMA
    sigma_iqr = np.maximum(sigma_iqr, 1e-6)
    sigma_base = (1.0 - floor_weight) * sigma_iqr + floor_weight * sigma_floor
    sigma_base = np.maximum(sigma_base, 1e-6)

    y_true_abs = y + base_fvc
    best_score = -1e18
    best_mult = None
    best_add = None
    for add in add_grid:
        sigma_shifted = np.maximum(sigma_base + float(add), 1e-6)
        for m in mult_grid:
            s = sigma_shifted * float(m)
            score = float(laplace_metric_np(y_true_abs, all_q50, s).mean())
            if score > best_score:
                best_score = score
                best_mult = float(m)
                best_add = float(add)

    return best_mult, best_add, float(best_score)


resid_train = (
    train_p["FVC"].astype(np.float64).values
    - train_p["Base_FVC"].astype(np.float64).values
)
sigma_floor = float(np.std(resid_train))
sigma_floor = max(sigma_floor, 70.0)  # metric-relevant scale

FLOOR_WEIGHT = 0.10

mult_grid = np.array(
    [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2],
    dtype=np.float64,
)

add_grid = np.array([0.0, 10.0, 20.0, 35.0, 50.0, 70.0, 90.0], dtype=np.float64)

base_fvc_train = train_p["Base_FVC"].astype(np.float64).values
patients_train = train_p["Patient"].astype(str).values

best_mult, best_add, best_oof_score = choose_conf_calibration_oof_grouped_by_patient(
    X_train,
    y_train,
    base_fvc_train,
    patients_train,
    q_list=tuple(PINBALL_QUANTILE),
    lr=0.05,
    epochs=400,
    mult_grid=mult_grid,
    add_grid=add_grid,
    sigma_floor=sigma_floor,
    floor_weight=FLOOR_WEIGHT,
    n_folds=5,
    seed=20,
)

print(
    "Chosen calibration (OOF patient-grouped):",
    "CONF_MULT=",
    best_mult,
    "| SIGMA_ADD=",
    best_add,
    "| OOF metric=",
    best_oof_score,
    "| sigma_floor=",
    sigma_floor,
    "| floor_weight=",
    FLOOR_WEIGHT,
)



## === cell 10
Xs_test = (X_test - mu) / sigma
Xb_test = add_bias(Xs_test)
yq_resid = Xb_test @ W  # residual quantiles (n,3)

q20_r = yq_resid[:, 0]
q50_r = yq_resid[:, 1]
q80_r = yq_resid[:, 2]

base_fvc_test = pred_p["Base_FVC"].astype(np.float64).values
q20 = q20_r + base_fvc_test
q50 = q50_r + base_fvc_test
q80 = q80_r + base_fvc_test

q20, q50, q80 = enforce_non_crossing(q20, q50, q80)

sigma_iqr_test = (q80 - q20) / IQR20_80_TO_SIGMA
sigma_iqr_test = np.maximum(sigma_iqr_test, 1e-6)
sigma_base_test = (1.0 - FLOOR_WEIGHT) * sigma_iqr_test + FLOOR_WEIGHT * sigma_floor

sigma_base_test = np.maximum(sigma_base_test + best_add, 1e-6)

conf = (sigma_base_test * best_mult).astype(np.float64)
conf = np.maximum(conf, 1e-6)
conf = np.maximum(conf, 70.0)

sub = pd.DataFrame(
    {
        "Patient_Week": pred_p["Patient_Week"].values,
        "FVC": q50.astype(np.float64),
        "Confidence": conf.astype(np.float64),
    }
)

sub = sample_sub[["Patient_Week"]].merge(sub, on="Patient_Week", how="left")

if sub["FVC"].isna().any():
    sub["FVC"] = sub["FVC"].fillna(raw_test["FVC"].median())
if sub["Confidence"].isna().any():
    sub["Confidence"] = sub["Confidence"].fillna(70.0)

sub["FVC"] = sub["FVC"].astype(float)
sub["Confidence"] = sub["Confidence"].astype(float)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
