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
libcudf-cu12==25.2.2
libcuml-cu12==25.2.1
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
pylibcudf-cu12==25.2.2
sklearn-pandas==2.2.0

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

1.7292

# 6. Current score

12.26237

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.87923) has done: 'Diagnosis: Cell 7 crashes in `plt.plot(np.arange(80), err, '-o')` because `err` is a list of CuPy 0-d arrays/scalars (`mae`), and Matplotlib cannot implicitly convert CuPy arrays to NumPy (`TypeError: Implicit conversion to a NumPy array is not allowed`). The rest of the computation uses CuPy arrays intentionally, so we should only convert the plotting data (not the training/inference logic) to a NumPy list/array explicitly right before plotting.

Patch summary: Keep the KNN/MAE logic unchanged, but append `float(mae.get())` to `err` (or equivalently convert `err` at plot time) so Matplotlib receives CPU-native numeric types. This is a minimal, localized fix in cell 7 that preserves all downstream variables and semantics.

Updated cells: Only cell 7 is modified.

Compatibility notes for cell k+1: `all_true_inhale`, `all_oof_inhale`, `all_true`, and `all_oof` remain lists of CuPy arrays exactly as before, so cell 8’s `cupy.concatenate(...)` continues to work unchanged. Only `err` becomes a list of Python floats for plotting, and `all_pred`/`all_test_id` are unchanged.

Assumptions: We assume CuPy is configured to disallow implicit NumPy conversion (current default behavior), and Matplotlib plotting is intended to run on CPU with NumPy-compatible inputs.'
- What this solution (achieved 8.87381) has done: 'Your current score (8.87923 MAE, lower-is-better) is far above the target (1.7292), so we need a modest, legitimate improvement without changing the overall KNN-by-(R,C) approach. The smallest high-impact fix is to use the nearest neighbor (index 0) rather than the second-nearest (index 1): your current code effectively ignores the closest match and uses a worse one, which inflates error. We keep the same model, metric, and loops, but switch to using the first neighbor for OOF and test predictions (and add a tiny guard to avoid accidental self-match by requesting one extra neighbor in the training-to-training query only). This should reduce MAE substantially and move you toward the target while preserving core logic and producing the same submission format.'
- What this solution (achieved 8.87381) has done: 'Diagnosis: The crash happens at `d0 = distances[:, 0]` because `cuml.NearestNeighbors.kneighbors` in this RAPIDS version returns `distances`/`indices` as cuDF DataFrames, and cuDF does not support NumPy-style 2D slicing (`[:, 0]`). That slicing triggers an internal index assignment on a 0-column intermediate and raises the “Length mismatch” ValueError. We need to select the first column using cuDF-compatible indexing and ensure we still produce a CuPy array for `cupy.where`, without changing the KNN logic.  

Patch summary: Replace the unsupported `distances[:, 0]` with `distances.iloc[:, 0]` (and convert to a CuPy array), leaving the rest of the neighbor-selection and downstream logic unchanged. This is localized to cell 7 and keeps all variable names and outputs compatible.  

Updated cells: cell 7 only.  

Compatibility notes for cell k+1: All lists (`all_true_inhale`, `all_oof_inhale`, `all_true`, `all_oof`) still accumulate CuPy arrays the same way, so `cupy.concatenate` in cell 8 remains valid. No interface/shape changes are introduced.  

Assumptions: `distances` and `indices` returned by cuML are cuDF DataFrames (not CuPy arrays) in this environment; `.iloc[:, 0]` is supported and `.values` yields a CuPy array.'
- What this solution (achieved 8.84356) has done: 'Your MAE is far worse than the target (lower-is-better), so we should make a small, legitimate change that improves accuracy without changing the overall KNN-by-(R,C) core logic. The biggest issue left is that test-time prediction always uses only the single nearest neighbor; switching to a simple mean over the top-2 neighbors (still KNN, same metric, same loops) typically reduces noise and should move the score substantially toward the target. To keep semantics stable, we also request 2 neighbors for test queries and average their corresponding `y_DO` pressures, while leaving the OOF self-neighbor guard logic unchanged. The submission writing stays identical and still produces a valid `submission_rapids_knn.csv`.'
- What this solution (achieved 12.2569) has done: 'Your current MAE (8.84356, lower-is-better) is still far above the target (1.7292), so we need a small but high-impact accuracy fix while staying within the same KNN-by-(R,C) core logic. The main issue is that the KNN is currently fit only on `u_in` time steps, ignoring `u_out` even though the metric scores only inspiratory phase and `u_out` strongly determines that phase boundary. We keep the exact same pipeline/loops/MAE evaluation, but expand the KNN feature vector to include both `u_in` and `(1-u_out)` sequences (i.e., `x_0..x_79` plus `z_0..z_79`) for both train and test queries. This should materially reduce neighbor mismatches and move the score toward the target while producing the same valid submission CSV.'
- What this solution (achieved 12.26237) has done: 'Your current MAE (12.2569, lower-is-better) is far above the target (1.7292), and the main cause is that the KNN is matching whole breaths using only control signals but then copying neighbor pressures directly, which tends to miscalibrate because pressure values live on a discrete grid. A minimal, competition-standard improvement that keeps the same KNN core logic is to snap (quantize) predicted pressures to the known set of discrete pressure levels observed in the training data. This is a pure post-processing step (no model/loop/feature changes) and typically reduces MAE substantially on this competition. I add extraction of the pressure grid from train and apply nearest-grid-value mapping right before writing the submission, leaving everything else unchanged and still producing `submission_rapids_knn.csv`.'
- What this solution (achieved 12.26237) has done: 'Your current MAE (12.262) is far above the target (1.729, lower-is-better), so we need a small accuracy improvement that keeps the same KNN-by-(R,C) logic. The most likely issue is that the KNN feature space omits `R` and `C`, while the pressure dynamics depend strongly on them; adding `R` and `C` as two extra features is a minimal change that usually makes nearest-neighbor matching much more consistent without changing the approach. I update the feature slices consistently for both training and test queries, keeping everything else (neighbors, averaging, quantization, submission format) the same. This should move the score materially toward the target while preserving the core pipeline.'
- What this solution (achieved 12.26237) has done: 'We need to move your MAE down toward 1.7292 (lower-is-better) from 12.262, so we should fix a single high-impact mismatch while keeping the same KNN-by-(R,C) approach. The biggest remaining issue is that your OOF self-match guard is incorrect: it checks `d0==0` from the first neighbor, but with `n_neighbors=NEIGHBORS+1` the first neighbor is always “self” (distance 0), so you always pick the second neighbor and never use the closest non-self match. I change the guard to drop the self neighbor by always taking neighbor #1 for OOF (the first non-self), without changing features, neighbors count for test, averaging, or quantization. This should materially reduce error while preserving the overall logic and submission format.'

# 9. Code solution

## === cell 0
import pandas as pd, numpy as np
import cudf, cupy
import matplotlib.pyplot as plt

print("RAPIDS version", cudf.__version__)



## === cell 1
train = cudf.read_csv("../input/ventilator-pressure-prediction/train.csv")
exhale = 80 - train.groupby("breath_id")[["u_out"]].agg("sum")
length = train.groupby("breath_id")[["time_step"]].agg("max")
print("Train shape:", train.shape)
train.head()



## === cell 2
series = train.groupby("breath_id").collect().reset_index()
for k in range(80):
    series[f"x_{k}"] = series.u_in.list.get(k)
for k in range(80):
    series[f"y_{k}"] = series.pressure.list.get(k)
for k in range(80):
    series[f"z_{k}"] = 1 - series.u_out.list.get(k)
series.R = series.R.list.get(0)
series.C = series.C.list.get(0)
series = series.drop(["id", "time_step", "u_in", "u_out", "pressure"], axis=1)
series = series.merge(exhale, on="breath_id", how="left")
series = series.merge(length, on="breath_id", how="left")
series = series.rename({"time_step": "time_length", "u_out": "expire"}, axis=1)
series = series.sort_values("breath_id").reset_index(drop=True)

print("Train as series shape:", series.shape)
print(
    "Min inhale length=",
    series["expire"].min(),
    ",Max inhale length=",
    series["expire"].max(),
    "Max breath length=",
    series["time_length"].max(),
)
series.head()



## === cell 3
from cuml.neighbors import NearestNeighbors

NEIGHBORS = 1000
TIME_STEPS = 80
SKIP = 0
IGNORE = 3

FEATURE_START = IGNORE + SKIP
FEATURE_END = IGNORE + 2 * TIME_STEPS + SKIP + 2  # x_0..x_79, z_0..z_79, R, C

model = NearestNeighbors(n_neighbors=NEIGHBORS, metric="l1")
model.fit(series.iloc[:, FEATURE_START:FEATURE_END])

SHOW = [47, 68]
CTS = [300, 300]

for USE, CT in zip(SHOW, CTS):
    distances, indices = model.kneighbors(
        series.iloc[USE : USE + 1, FEATURE_START:FEATURE_END]
    )

    plt.figure(figsize=(20, 4))
    plt.plot(
        np.arange(80), series.iloc[USE, IGNORE : IGNORE + 80].to_numpy(), label="u_in"
    )
    plt.plot(
        np.arange(80),
        series.iloc[USE, IGNORE + 80 : IGNORE + 160].to_numpy(),
        label="pressure",
    )
    y_max = plt.ylim()[1]
    if TIME_STEPS != 80:
        plt.plot(
            [SKIP + TIME_STEPS - 1, SKIP + TIME_STEPS - 1],
            [0, y_max],
            "--",
            color="gray",
        )
    if SKIP != 0:
        plt.plot([SKIP - 1, SKIP - 1], [0, y_max], "--", color="gray")
    exhale_v = series.loc[USE, "expire"]
    plt.plot([exhale_v, exhale_v], [0, y_max], "--", color="black", label="exhale")
    rr = series.loc[USE, "R"]
    cc = series.loc[USE, "C"]
    bb = series.loc[USE, "breath_id"]
    tt = series.loc[USE, "time_length"]
    plt.title(f"Breath_Id={bb}, R={rr}, C={cc}, Length={tt:.3}", size=16)
    plt.legend()

    temp2 = series.iloc[indices.iloc[0].values].reset_index(drop=True)
    cdict = {5: "yellow", 20: "orange", 50: "red"}
    plt.figure(figsize=(20, 10))
    legend = {5: 0, 20: 0, 50: 0}
    for r in range(CT):
        if r == 0:
            plt.plot(
                np.arange(80),
                temp2.iloc[r, IGNORE : IGNORE + 80].to_numpy(),
                color="blue",
                label="u_in",
            )
        else:
            plt.plot(
                np.arange(80),
                temp2.iloc[r, IGNORE : IGNORE + 80].to_numpy(),
                color="blue",
            )
        if legend[temp2.loc[r, "R"]] == 0:
            legend[temp2.loc[r, "R"]] = 1
            cc2 = temp2.loc[r, "R"]
            plt.plot(
                np.arange(80),
                temp2.iloc[r, IGNORE + 80 : IGNORE + 160].to_numpy(),
                c=cdict[temp2.loc[r, "R"]],
                label=f"R={cc2}",
            )
        else:
            plt.plot(
                np.arange(80),
                temp2.iloc[r, IGNORE + 80 : IGNORE + 160].to_numpy(),
                c=cdict[temp2.loc[r, "R"]],
            )
    y_max = plt.ylim()[1]
    if TIME_STEPS != 80:
        plt.plot(
            [SKIP + TIME_STEPS - 1, SKIP + TIME_STEPS - 1],
            [0, y_max],
            "--",
            color="black",
        )
    if SKIP != 0:
        plt.plot([SKIP - 1, SKIP - 1], [0, y_max], "--", color="black")
    plt.title(
        f"300 time series similar to breath_Id={bb}. For each new time series we plot and color code its pressure with respect to R variable",
        size=16,
    )
    plt.legend()
    plt.show()

    temp2 = series.iloc[indices.iloc[0].values].reset_index(drop=True)
    cdict = {10: "red", 20: "orange", 50: "yellow"}
    plt.figure(figsize=(20, 10))
    legend = {10: 0, 20: 0, 50: 0}
    for r in range(CT):
        if r == 0:
            plt.plot(
                np.arange(80),
                temp2.iloc[r, IGNORE : IGNORE + 80].to_numpy(),
                color="blue",
                label="u_in",
            )
        else:
            plt.plot(
                np.arange(80),
                temp2.iloc[r, IGNORE : IGNORE + 80].to_numpy(),
                color="blue",
            )
        if legend[temp2.loc[r, "C"]] == 0:
            legend[temp2.loc[r, "C"]] = 1
            cc2 = temp2.loc[r, "C"]
            plt.plot(
                np.arange(80),
                temp2.iloc[r, IGNORE + 80 : IGNORE + 160].to_numpy(),
                c=cdict[temp2.loc[r, "C"]],
                label=f"C={cc2}",
            )
        else:
            plt.plot(
                np.arange(80),
                temp2.iloc[r, IGNORE + 80 : IGNORE + 160].to_numpy(),
                c=cdict[temp2.loc[r, "C"]],
            )
    y_max = plt.ylim()[1]
    if TIME_STEPS != 80:
        plt.plot(
            [SKIP + TIME_STEPS - 1, SKIP + TIME_STEPS - 1],
            [0, y_max],
            "--",
            color="black",
        )
    if SKIP != 0:
        plt.plot([SKIP - 1, SKIP - 1], [0, y_max], "--", color="black")
    plt.title(
        f"300 time series similar to breath_Id={bb}. For each new time series we plot and color code its pressure with respect to C variable",
        size=16,
    )
    plt.legend()
    plt.show()



## === cell 4
from cuml.cluster import KMeans

model = KMeans(n_clusters=1000)
model.fit(series.iloc[:, IGNORE + SKIP : IGNORE + TIME_STEPS + SKIP])
series["cluster"] = model.labels_
idx = series.cluster.value_counts().index.values

SHOW = 50
DISPLAY = 32
SHOW_R = False
SHOW_C = True

for i in range(DISPLAY):

    if i >= DISPLAY // 2:
        SHOW_R = True
        SHOW_C = False

    k = idx[np.random.randint(0, 300)]
    temp2 = series.loc[series.cluster == k.item()].sample(SHOW).reset_index(drop=True)

    if SHOW_R:
        cdict = {5: "yellow", 20: "orange", 50: "red"}
        plt.figure(figsize=(20, 10))
        legend = {5: 0, 20: 0, 50: 0}
        for r in range(SHOW):
            if r != 0:
                plt.plot(
                    np.arange(80),
                    temp2.iloc[r, IGNORE : IGNORE + 80].to_numpy(),
                    color="blue",
                )
            else:
                plt.plot(
                    np.arange(80),
                    temp2.iloc[r, IGNORE : IGNORE + 80].to_numpy(),
                    color="blue",
                    label="u_in",
                )
            if legend[temp2.loc[r, "R"]] == 0:
                legend[temp2.loc[r, "R"]] = 1
                cc2 = temp2.loc[r, "R"]
                plt.plot(
                    np.arange(80),
                    temp2.iloc[r, IGNORE + 80 : IGNORE + 160].to_numpy(),
                    c=cdict[temp2.loc[r, "R"]],
                    label=f"R={cc2}",
                )
            else:
                plt.plot(
                    np.arange(80),
                    temp2.iloc[r, IGNORE + 80 : IGNORE + 160].to_numpy(),
                    c=cdict[temp2.loc[r, "R"]],
                )
        y_max = plt.ylim()[1]
        if TIME_STEPS != 80:
            plt.plot(
                [SKIP + TIME_STEPS - 1, SKIP + TIME_STEPS - 1],
                [0, y_max],
                "--",
                color="black",
            )
        if SKIP != 0:
            plt.plot([SKIP - 1, SKIP - 1], [0, y_max], "--", color="black")
        plt.title(
            f"Cluster {k.item()}, Blues are u_in, Other colors are pressure with R parameter",
            size=16,
        )
        plt.legend()
        plt.show()

    if SHOW_C:
        cdict = {10: "red", 20: "orange", 50: "yellow"}
        plt.figure(figsize=(20, 10))
        legend = {10: 0, 20: 0, 50: 0}
        for r in range(SHOW):
            if r != 0:
                plt.plot(
                    np.arange(80),
                    temp2.iloc[r, IGNORE : IGNORE + 80].to_numpy(),
                    color="blue",
                )
            else:
                plt.plot(
                    np.arange(80),
                    temp2.iloc[r, IGNORE : IGNORE + 80].to_numpy(),
                    color="blue",
                    label="u_in",
                )
            if legend[temp2.loc[r, "C"]] == 0:
                legend[temp2.loc[r, "C"]] = 1
                cc2 = temp2.loc[r, "C"]
                plt.plot(
                    np.arange(80),
                    temp2.iloc[r, IGNORE + 80 : IGNORE + 160].to_numpy(),
                    c=cdict[temp2.loc[r, "C"]],
                    label=f"C={cc2}",
                )
            else:
                plt.plot(
                    np.arange(80),
                    temp2.iloc[r, IGNORE + 80 : IGNORE + 160].to_numpy(),
                    c=cdict[temp2.loc[r, "C"]],
                )
        y_max = plt.ylim()[1]
        if TIME_STEPS != 80:
            plt.plot(
                [SKIP + TIME_STEPS - 1, SKIP + TIME_STEPS - 1],
                [0, y_max],
                "--",
                color="black",
            )
        if SKIP != 0:
            plt.plot([SKIP - 1, SKIP - 1], [0, y_max], "--", color="black")
        plt.title(
            f"Cluster {k.item()}, Blues are u_in, Others colors are pressure with C parameter",
            size=16,
        )
        plt.legend()
        plt.show()



## === cell 5
test = cudf.read_csv("../input/ventilator-pressure-prediction/test.csv")
exhale = 80 - test.groupby("breath_id")[["u_out"]].agg("sum")
length = test.groupby("breath_id")[["time_step"]].agg("max")
print("Test shape:", test.shape)
test.head()



## === cell 6
test_series = test.groupby("breath_id").collect().reset_index()
for k in range(80):
    test_series[f"x_{k}"] = test_series.u_in.list.get(k)
for k in range(80):
    test_series[f"z_{k}"] = 1 - test_series.u_out.list.get(k)
test_series.R = test_series.R.list.get(0)
test_series.C = test_series.C.list.get(0)
test_series = test_series.drop(["id", "time_step", "u_in", "u_out"], axis=1)
test_series = test_series.merge(exhale, on="breath_id", how="left")
test_series = test_series.merge(length, on="breath_id", how="left")
test_series = test_series.sort_values("breath_id").reset_index(drop=True).reset_index()
test_series = test_series.rename(
    {"time_step": "time_length", "u_out": "expire", "index": "row"}, axis=1
)

print("Test as series shape:", test_series.shape)
print(
    "Min inhale length=",
    test_series["expire"].min(),
    ",Max inhale length=",
    test_series["expire"].max(),
    "Max breath length=",
    test_series["time_length"].max(),
)
test_series.head()



## === cell 7
from cuml.neighbors import NearestNeighbors

NEIGHBORS = 2
SIZE = 80

TRAIN_FEAT_SLICE = slice(IGNORE, IGNORE + 2 * SIZE + 2)  # x_0..x_79, z_0..z_79, R, C
TEST_FEAT_SLICE = slice(IGNORE, IGNORE + 2 * SIZE + 2)

all_true = []
all_oof = []
all_true_inhale = []
all_oof_inhale = []

all_pred = []
all_test_id = []

for r in [5, 20, 50]:
    for c in [10, 20, 50]:

        temp2 = series.loc[(series.R == r) & (series.C == c)].reset_index(drop=True)
        model = NearestNeighbors(n_neighbors=NEIGHBORS, metric="l1")
        model.fit(temp2.iloc[:, TRAIN_FEAT_SLICE])

        err = []

        distances, indices = model.kneighbors(
            temp2.iloc[:, TRAIN_FEAT_SLICE], n_neighbors=NEIGHBORS + 1
        )

        nn_idx = indices.iloc[:, 1].values  # first non-self neighbor

        test_temp2 = test_series.loc[
            (test_series.R == r) & (test_series.C == c)
        ].reset_index(drop=True)

        test_distances, test_indices = model.kneighbors(
            test_temp2.iloc[:, TEST_FEAT_SLICE], n_neighbors=NEIGHBORS
        )
        test_nn0 = test_indices.iloc[:, 0].values
        test_nn1 = test_indices.iloc[:, 1].values

        for DO in range(80):

            true = temp2[f"y_{DO}"].values
            oof = temp2.loc[nn_idx, f"y_{DO}"].values

            mask = temp2[f"z_{DO}"].values

            preds0 = temp2.loc[test_nn0, f"y_{DO}"].values
            preds1 = temp2.loc[test_nn1, f"y_{DO}"].values
            preds = 0.5 * (preds0 + preds1)

            test_ids = test_temp2.row.values * 80 + DO + 1

            mae = cupy.mean(cupy.abs(true - oof))
            err.append(float(mae.get()))

            idx = cupy.where(mask == 1)[0]
            all_true_inhale.append(true[idx])
            all_oof_inhale.append(oof[idx])
            all_true.append(true)
            all_oof.append(oof)

            all_pred.append(preds)
            all_test_id.append(test_ids)

        plt.figure(figsize=(20, 5))
        plt.plot(np.arange(80), err, "-o")
        plt.ylabel("OOF RSME", size=14)
        plt.xlabel("Time Step", size=14)
        plt.title(f"OOF RSME for R={r} C={c}", size=16)
        plt.show()



## === cell 8
all_true2 = cupy.concatenate(all_true_inhale)
all_oof2 = cupy.concatenate(all_oof_inhale)
rsme = cupy.mean(cupy.abs(all_true2 - all_oof2))
print("RAPIDS KNN CV RSME=", rsme, "(inhale only)")

all_true2 = cupy.concatenate(all_true)
all_oof2 = cupy.concatenate(all_oof)
rsme = cupy.mean(cupy.abs(all_true2 - all_oof2))
print("RAPIDS KNN CV RSME=", rsme, "(all breath)")



## === cell 9
pressure_grid = cupy.asarray(
    np.sort(train["pressure"].to_pandas().unique()).astype(np.float32)
)
print("Pressure grid size:", int(pressure_grid.size))



## === cell 10
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

test_id = cupy.hstack(all_test_id)
pred = cupy.hstack(all_pred).astype(cupy.float32)

idx = cupy.searchsorted(pressure_grid, pred, side="left")
idx = cupy.clip(idx, 0, pressure_grid.size - 1)
idx0 = cupy.clip(idx - 1, 0, pressure_grid.size - 1)
p1 = pressure_grid[idx]
p0 = pressure_grid[idx0]
pred_q = cupy.where(cupy.abs(pred - p0) <= cupy.abs(pred - p1), p0, p1)

sub["id"] = cupy.asnumpy(test_id)
sub["pressure"] = cupy.asnumpy(pred_q)
sub = sub.sort_values("id").reset_index(drop=True)
sub.to_csv("submission_rapids_knn.csv", index=False)
print(sub.shape)
sub.head()
