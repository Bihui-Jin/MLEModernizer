# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
imageio==2.37.0
imageio-ffmpeg==0.6.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
pillow==11.3.0
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
requests==2.32.5
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, random, gc, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


def seed_it_all(seed=7):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_it_all()

print("\n--- ENVIRONMENT READY ---\n")



## === cell 1
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
ss_df = pd.read_csv(SS_CSV)

print(f"Train rows: {len(train_df):,}, Test rows: {len(test_df):,}")




## === cell 2
def add_features(
    df,
    U_IN_N_FORWARD=7,
    U_IN_N_BACKWARD=7,
    U_OUT_N_FORWARD=2,
    U_OUT_N_BACKWARD=2,
    V_0=1,
    p_0=1,
    r_0=1,
    use_rc=True,
):
    """Add engineered features used in the original notebook."""
    df["measured_volume"] = (V_0 + df["u_in"] * df["time_step"].diff()).fillna(0) + V_0
    r_t = (3 * df["measured_volume"] / 4 / np.pi) ** (1 / 3)
    df["measured_pressure"] = (
        p_0 + (1 - (r_t / r_0) ** 6) * 1 / (r_t * r_0**2)
    ).fillna(0) + p_0

    df["uin_auc"] = (df["time_step"] * df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["uin_csum"] = df["u_in"].groupby(df["breath_id"]).cumsum()
    df["breath_id__u_in__max"] = df.groupby("breath_id")["u_in"].transform("max")
    df["breath_id__u_in__diffmax"] = df["breath_id__u_in__max"] - df["u_in"]
    df["breath_id__u_in__diffmean"] = (
        df.groupby("breath_id")["u_in"].transform("mean") - df["u_in"]
    )
    df["cross3"] = df["time_step"] * df["u_in"]
    df["cross3_sqd_1"] = df["time_step"] * df["u_in"] ** 2
    df["cross3_sqd_2"] = df["time_step"] ** 2 * df["u_in"]
    df["cross3_cubed_1"] = df["time_step"] * df["u_in"] ** 3
    df["cross3_cubed_2"] = df["time_step"] ** 3 * df["u_in"]

    for i in range(1, U_IN_N_BACKWARD + 1):
        df[f"u_in_{i}_back"] = df.groupby("breath_id")["u_in"].shift(i).fillna(0)
    for i in range(1, U_IN_N_FORWARD + 1):
        df[f"u_in_{i}_forw"] = df.groupby("breath_id")["u_in"].shift(-i).fillna(0)

    for i in range(1, U_OUT_N_BACKWARD + 1):
        df[f"u_out_{i}_back"] = df.groupby("breath_id")["u_out"].shift(i).fillna(0)
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_{i}_forw"] = df.groupby("breath_id")["u_out"].shift(-i).fillna(0)

    for i in range(1, U_IN_N_BACKWARD + 1):
        df[f"u_in_diff_{i}_back"] = df["u_in"] - df[f"u_in_{i}_back"]
    for i in range(1, U_OUT_N_BACKWARD + 1):
        df[f"u_out_diff_{i}_back"] = df["u_out"] - df[f"u_out_{i}_back"]
    for i in range(1, U_IN_N_FORWARD + 1):
        df[f"u_in_diff_{i}_forw"] = df["u_in"] - df[f"u_in_{i}_forw"]
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_diff_{i}_forw"] = df["u_out"] - df[f"u_out_{i}_forw"]

    if use_rc:
        df["R_C"] = df["R"].astype(str) + "_" + df["C"].astype(str)
        df["R"] = df["R"] / 50
        df["C"] = df["C"] / 50
        df = pd.get_dummies(df)

    for c in df.columns:
        if c == "u_out":
            df[c] = df[c].astype("uint8")
        elif df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    gc.collect()
    return df


train_df = add_features(train_df, use_rc=True)
test_df = add_features(test_df, use_rc=True)



## === cell 3
LABEL = "pressure"
GROUPBY = ["breath_id"]
IGNORE = ["id"]
FEATURES = [c for c in train_df.columns if c not in [LABEL] + GROUPBY + IGNORE]

scaler = RobustScaler()
scaler.fit(train_df[FEATURES].values)

X_train_scaled = scaler.transform(train_df[FEATURES].values)
X_test_scaled = scaler.transform(test_df[FEATURES].values)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train = poly.fit_transform(X_train_scaled)
X_test = poly.transform(X_test_scaled)

y_train = train_df[LABEL].values

model = LinearRegression()
model.fit(X_train, y_train)

test_pred = model.predict(X_test).astype(np.float32)

p_min, p_max = train_df[LABEL].min(), train_df[LABEL].max()
test_pred = np.clip(test_pred, p_min, p_max)



## === cell 4
ss_df["pressure"] = test_pred
submission_path = "submission.csv"
ss_df[["id", "pressure"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
display(ss_df.head())
