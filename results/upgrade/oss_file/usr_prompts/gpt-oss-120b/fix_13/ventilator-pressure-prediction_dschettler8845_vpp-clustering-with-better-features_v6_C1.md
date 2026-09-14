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

cudf-cu12==25.2.2
cudf-polars-cu12==25.6.0
cuml-cu12==25.2.1
cupy-cuda12x==13.6.0
dask-cudf-cu12==25.2.2
geopandas==0.14.4
google-api-python-client==2.177.0
imageio==2.37.0
imageio-ffmpeg==0.6.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
libcudf-cu12==25.2.2
libcuml-cu12==25.2.1
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numba==0.60.0
numba-cuda==0.2.0
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
pylibcudf-cu12==25.2.2
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

# 5. Target score

0.6658

# 6. Current score

1.16651

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.11597) has done: 'I adjust the feature‑building function to gracefully handle the test set (which lacks the `pressure` column) and ensure it returns `None` for `Y` when appropriate. I also add a `step` column to `test_df` so the predicted pressures can be merged correctly with the original rows, fixing the NameError and merge issues and producing a proper `submission.csv` file.'
- What this solution (achieved 0.98219) has done: 'I add the `time_step` sequence to the breath feature vector (as a normalized series) and slightly tighten the nearest‑neighbour weighting by using inverse‑distance‑squared weights. I also reduce the neighbour count from 10 to 8, which often improves KNN stability for this data. These small tweaks keep the overall KNN‑based pipeline unchanged while moving the MAE closer to the target 0.6658.'
- What this solution (achieved 1.06071) has done: 'I keep the overall K‑Nearest‑Neighbour pipeline unchanged but adjust the way the neighbour contributions are weighted.  
Instead of using inverse‑distance‑squared weights (which can overly favor a single neighbour), I use simple inverse‑distance weights and normalise them, which usually yields a modest improvement in MAE for this data while preserving the core logic. The rest of the code and file handling remain the same.'
- What this solution (achieved 1.05212) has done: 'I add a few simple breath‑level summary statistics (means and standard deviations of `u_in` and `u_out`) to the feature vector and increase the neighbour count slightly. I also revert to the inverse‑distance‑squared weighting (which performed better previously). These small, targeted changes keep the overall K‑NN pipeline unchanged while aiming to lower the MAE toward the target.'
- What this solution (achieved 1.02537) has done: 'I add a few extra breath‑level summary statistics (max/min of u_in and sum of u_out) to the feature vector, and slightly reduce the neighbour count from 12 to 10 so the K‑NN smoothing is a bit tighter. These small, targeted changes keep the overall pipeline unchanged while giving the model a richer representation that should lower the MAE toward the target.'
- What this solution (achieved 1.32599) has done: 'I keep the overall K‑Nearest‑Neighbour pipeline but improve distance handling: each breath‑level feature vector is L2‑normalized and the neighbour search now uses Euclidean distance with a slightly larger neighbour count (12). I also switch to simple inverse‑distance weighting (instead of squared) which usually gives smoother, more accurate predictions. These minimal, targeted changes should lower the MAE toward the target while preserving the original logic.'
- What this solution (achieved 1.16651) has done: 'I revert the recent changes that hurt performance: remove the L2‑normalisation of the breath feature vectors, use fewer neighbours (8 instead of 12) and switch back to inverse‑distance‑squared weighting, which the earlier experiments showed gives a lower MAE. These are the smallest adjustments that keep the overall K‑NN pipeline unchanged while moving the score closer to the target.'
- What this solution (achieved 1.30563) has done: 'I adjust the K‑Nearest Neighbour settings to better match the data distribution: increase the neighbour count from 8 to 10 and replace the inverse‑distance‑squared weighting with simple inverse‑distance weighting. These minimal tweaks keep the overall KNN pipeline unchanged while expectedly lowering the MAE toward the target.'
- What this solution (achieved 1.16651) has done: 'I revert the neighbor count to a smaller, more stable value (8) and switch the weighting back to inverse‑distance‑squared, which earlier experiments showed lowers MAE. These changes keep the overall K‑NN pipeline intact while moving the score closer to the target.'
- What this solution (achieved 1.35127) has done: 'I increase the neighbour count to 12 and switch the weighting from inverse‑distance‑squared to simple inverse‑distance (normalised). This small tweak keeps the K‑NN pipeline unchanged while giving a smoother contribution from more neighbours, which is expected to lower the MAE and move the score closer to the target.'
- What this solution (achieved 1.16651) has done: 'We reduce the neighbour count from 12 to 8 and switch back to inverse‑distance‑squared weighting, which gives stronger influence to the closest breaths while keeping the overall K‑NN pipeline unchanged. This minor tweak is expected to lower the MAE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.neighbors import NearestNeighbors
import random


def seed_all(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)


seed_all()



## === cell 1
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

train_df = train_df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
test_df = test_df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

ROWS_PER_BREATH = 80  # fixed in the competition


def build_breath_features(df: pd.DataFrame):
    """Return feature matrix X, pressure matrix Y (or None for test), and breath ids."""
    breaths = df.groupby("breath_id")
    u_in_series = breaths["u_in"].apply(lambda x: np.array(x))
    u_out_series = breaths["u_out"].apply(lambda x: np.array(x))
    time_series = breaths["time_step"].apply(lambda x: np.array(x) / 2.73)  # normalize

    if "pressure" in df.columns:
        pressure_series = breaths["pressure"].apply(lambda x: np.array(x))
        Y = np.stack(pressure_series.values, axis=0)  # (n_breaths, ROWS_PER_BREATH)
    else:
        Y = None

    R_series = breaths["R"].first()
    C_series = breaths["C"].first()

    feats = []
    for i in range(len(u_in_series)):
        u_in = u_in_series.values[i]
        u_out = u_out_series.values[i]
        t = time_series.values[i]

        u_in_mean = u_in.mean()
        u_in_std = u_in.std()
        u_in_max = u_in.max()
        u_in_min = u_in.min()

        u_out_mean = u_out.mean()
        u_out_std = u_out.std()
        u_out_sum = u_out.sum()

        breath_feat = np.concatenate(
            [
                u_in,
                u_out,
                t,
                [R_series.values[i] / 50.0, C_series.values[i] / 50.0],
                [
                    u_in_mean,
                    u_in_std,
                    u_in_max,
                    u_in_min,
                    u_out_mean,
                    u_out_std,
                    u_out_sum,
                ],
            ]
        )
        feats.append(breath_feat)

    X = np.stack(feats, axis=0)
    breath_ids = u_in_series.index.values
    return X, Y, breath_ids


print("Building training features...")
X_train, Y_train, train_breath_ids = build_breath_features(train_df)

print("Building test features...")
X_test, _, test_breath_ids = build_breath_features(test_df)




## === cell 2
NN_TO_USE = 8
print(f"Fitting NearestNeighbors (n_neighbors={NN_TO_USE}, metric='euclidean')...")
nn_model = NearestNeighbors(n_neighbors=NN_TO_USE, metric="euclidean")
nn_model.fit(X_train)

print("Predicting pressures for each test breath...")
predicted_pressures = np.zeros(
    (len(test_breath_ids), ROWS_PER_BREATH), dtype=np.float32
)

batch_iter = tqdm(enumerate(X_test), total=len(X_test), desc="Predicting")
for idx, x in batch_iter:
    distances, indices = nn_model.kneighbors(x.reshape(1, -1))
    neighbor_pressures = Y_train[indices[0]]  # (NN_TO_USE, 80)

    inv_dist = 1.0 / (distances[0] ** 2 + 1e-8)
    weights = inv_dist / inv_dist.sum()
    pred = np.average(neighbor_pressures, axis=0, weights=weights)
    predicted_pressures[idx] = pred




## === cell 3
print("Creating submission file...")

test_df["step"] = test_df.groupby("breath_id").cumcount()

pred_df = pd.DataFrame(
    {
        "breath_id": test_breath_ids,
        **{f"pred_{i}": predicted_pressures[:, i] for i in range(ROWS_PER_BREATH)},
    }
)
pred_long = pred_df.melt(id_vars="breath_id", var_name="step", value_name="pressure")
pred_long["step"] = pred_long["step"].str.replace("pred_", "").astype(int)

submission = test_df[["id", "breath_id", "step"]].merge(
    pred_long, on=["breath_id", "step"], how="left"
)
submission = submission[["id", "pressure"]]

submission["pressure"] = submission["pressure"].astype(np.float32)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
