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

17.65244

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 8.87923) has done: 'I fix the RAPIDS/cuDF API mismatches causing the `to_array()` plotting errors by converting row slices to NumPy safely, and I fix the CuPy→NumPy implicit conversion error in matplotlib by explicitly `.get()`-ing values. Then I fix the core submission-length bug by generating predictions for *all* test breaths (including any (R,C) combinations with zero matches) and by building the submission directly from `test` row order so it always has exactly 603600 rows aligned to `id`. These changes keep the core KNN-by-(R,C) logic intact and only adjust I/O, plotting safety, and edge-case handling so the notebook runs end-to-end and writes a valid `submission_rapids_knn.csv`. Finally, I ensure the output is sorted by `id` and has the required columns.'
- What this solution (achieved 17.65486) has done: 'You’re currently scoring far worse than target (lower-is-better), and the biggest minimal lever that keeps your KNN-by-(R,C) core logic intact is to make the neighbor choice and feature space better aligned with the evaluation: MAE is computed only for inspiratory phase, so we should avoid using exhalation-only steps as similarity signal and avoid using the *2nd* neighbor (index 1) which is both unstable and often unnecessarily worse than the closest match. I (1) include `u_out` into the neighbor features so matching respects valve state, (2) pick the nearest neighbor (index 0) instead of the 2nd, and (3) compute KNN distances with an inspiratory mask (set exhale steps to 0 so they don’t drive similarity), while leaving your overall approach (per-(R,C) KNN lookup of training pressures) unchanged. These are small, metric-aligned tweaks that should move MAE substantially down toward your 1.7292 target without changing the overall modeling paradigm or adding new training.'
- What this solution (achieved 8.71125) has done: 'I fix the cuML `kneighbors` return-type mismatch (it returns CuPy ndarrays here, not cuDF objects), so `.iloc` doesn’t work and the code crashes before filling `test_pred_matrix`. Then I make the OOF-collection robust so `cell 8` doesn’t crash when a split produces no inspiratory samples collected (or when earlier loops are skipped), while keeping the same KNN-by-(R,C) approach and feature construction. Finally, I ensure the prediction matrix is fully populated for every test breath (with a safe fallback if any (R,C) combination yields no rows) so the submission always has exactly 603600 rows aligned to `id`. These are execution/stability fixes plus minimal, metric-aligned safeguards, and should move the score back down toward the target by actually running the intended nearest-neighbor logic.'
- What this solution (achieved 8.71125) has done: 'I fix the plotting/diagnostic cell crash by making the neighbor-plot loop robust when there are fewer neighbors than the requested CT (this is what triggers the out-of-bounds gather). I also make cell 3 non-blocking (safe-guarded) so the training/prediction pipeline always runs end-to-end and writes a valid `submission_rapids_knn.csv`. Finally, I keep the core KNN-by-(R,C) logic intact and only adjust indexing conversions between cuML/cupy/cudf where needed for stability, without changing the model or evaluation semantics.'
- What this solution (achieved 8.70698) has done: 'Your current score (8.71125, lower-is-better) is far worse than the target (1.7292), so we should improve performance with the smallest metric-aligned change that preserves your KNN-by-(R,C) core logic. The biggest issue is that your KNN feature space is currently missing strong time-dynamics information (time_step) and uses raw-scale u_in without normalization, which makes L1 distances dominated by magnitude rather than shape; adding time_step as an extra per-step feature (masked by inspiratory phase like you already do) and standardizing u_in per-(R,C) group typically reduces MAE substantially without changing the approach. I keep your architecture/training loops intact (still cuML NearestNeighbors, same per-(R,C) loops, same neighbor=0), and only (1) add time_step features, (2) do simple z-score scaling of u_in using training-group mean/std (applied to both train/test), and (3) ensure the same changes are used in the global fallback. This should move the score down toward the target while remaining a minimal, legitimate adjustment consistent with the evaluation.'
- What this solution (achieved 8.70707) has done: 'We’re currently far worse than the target (8.70698 vs 1.7292, lower-is-better), so we need a small but meaningful improvement that keeps the same per-(R,C) KNN-by-breath core logic. The most leverage with minimal change is to (1) use the existing inspiratory mask `z_*` to also zero-out `u_out` in the KNN feature vector (so exhale doesn’t distort distances), and (2) weight the binary `u_out` feature higher so the nearest neighbor respects valve state strongly (this is aligned with the metric because only inspiratory phase is scored). Everything else (cuML NearestNeighbors, neighbor=0, per-(R,C) loops, submission alignment) stays the same, and it still writes `submission_rapids_knn.csv` with 603600 rows.'
- What this solution (achieved 8.70707) has done: 'Your current MAE (8.707) is far worse than the target (1.7292, lower-is-better), so we should make one small, metric-aligned change that preserves your per-(R,C) breath-level KNN core logic. The main issue is that your KNN currently matches breaths using only the control inputs/time/u_out, but then copies neighbor *pressures* directly; this can mismatch the discrete pressure grid and hurt MAE. A minimal, legitimate post-processing step for this competition is to snap predictions to the set of pressures observed in the training data (many strong baselines do this) without changing the model/training approach. I add this “pressure grid snapping” right before writing the submission, keeping all paths and the rest of your pipeline intact.'
- What this solution (achieved 8.65463) has done: 'Your current score (8.707, lower-is-better) is far above the target (1.7292), so we should make a small, metric-aligned improvement without changing the core per-(R,C) breath-level KNN logic. The biggest minimal lever here is that you’re currently copying the *entire* neighbor pressure curve, including expiratory steps that are not scored and can be very mismatched; instead, we keep your KNN matching exactly as-is but blend predictions so that expiratory steps use a stable fallback (the within-(R,C) mean pressure at that time step), which reduces noise without touching the scored inspiratory phase logic. This preserves your approach (KNN-by-breath, same features, same neighbor selection) and only changes post-processing at prediction assembly time. The submission writing remains identical and still snaps to the training pressure grid.'
- What this solution (achieved 9.72966) has done: 'Your current MAE (8.6546, lower-is-better) is still far above the target (1.7292), so we need a small but meaningful improvement that keeps your per-(R,C) breath-level KNN core logic intact. The biggest minimal issue is that you are selecting neighbors using only control/time/valve features but then copying the neighbor’s raw pressure curve; because pressure is a deterministic function of the control trajectory within an (R,C) setting, predicting pressure as a *function of the matched u_in trajectory* is often closer than copying the neighbor’s pressure directly. I keep your same KNN search and grouping, but change only the per-time-step prediction to use the neighbor’s (x→y) mapping: for each time step, find the closest u_in value in the neighbor breath and take its pressure (still blended on exhale as you already do). This is a minimal post-processing change inside the same KNN-by-(R,C) framework and should move MAE materially down toward the target while still producing the same valid submission file.'
- What this solution (achieved 9.1912) has done: 'We need to reduce MAE (lower-is-better) from 9.72966 toward 1.7292; the smallest, metric-aligned improvement that preserves your KNN-by-(R,C) breath matching core logic is to stop predicting noisy expiratory steps altogether and to make inspiratory predictions more consistent with the true pressure “grid”. I (1) force expiratory predictions to exactly 0 (the true label is unscored there, and this removes large errors if Kaggle’s scoring mask ever differs slightly from your `z_k`), and (2) replace the current per-step “nearest u_in within neighbor” mapping with a monotonic, binned mapping from `u_in`→`pressure` learned from the neighbor breath (still using the same neighbor breath chosen by KNN). This keeps the same model (cuML NearestNeighbors), same grouping, same feature construction, and same submission assembly, but makes the post-processing mapping substantially less brittle and typically lowers MAE a lot for this competition. Submission writing remains identical and still produces `submission_rapids_knn.csv` with exactly 603600 rows aligned to `id`.'
- What this solution (achieved 17.65244) has done: 'We’re far worse than the target (9.1912 vs 1.7292, lower-is-better), so we need a small, metric-aligned improvement that keeps your KNN-by-(R,C) breath matching core logic intact. The most impactful minimal fix is that your mapping currently predicts inspiratory pressure from **u_in only**, ignoring the strong, stepwise dependence on **time_step**; we keep the same neighbor selection but change the mapping to use **(time_step, u_in)** bins within the chosen neighbor breath, which better matches the true dynamics without changing the model/training loop. We also keep your “exhale=0” behavior and pressure-grid snapping, but we increase bin resolution slightly (still cheap) to reduce discretization error. These changes are localized to the post-processing mapping and should move MAE down materially toward the target while preserving the overall approach and producing the same valid submission file.'
- What this solution (achieved 17.65244) has done: 'I fix the shape bug in the binned (time_step, u_in) neighbor mapping that currently returns a (n, 80, 64) tensor instead of per-time-step predictions, which causes the broadcasting error when blending with `z_test`. The minimal correction is to collapse the mapping over the u_in-bin axis by gathering the correct bin per time step and returning an (n, 80) matrix, preserving your neighbor selection and overall KNN-by-(R,C) logic. I also keep prediction assembly and submission writing unchanged, only adjusting the mapping function and adding small type/shape guards to avoid similar runtime errors. This should both make the notebook run end-to-end and move MAE back down versus the currently broken pipeline (which can’t produce a valid score).'

# 9. Code solution

## === cell 0
import pandas as pd, numpy as np
import cudf, cupy
import matplotlib.pyplot as plt

print("RAPIDS version", cudf.__version__)


def _row_to_numpy(row_slice):
    """
    Accepts a cuDF Series (typically from df.iloc[row, col_start:col_end])
    and returns a NumPy array on host for plotting.
    """
    try:
        return row_slice.to_cupy().get()
    except Exception:
        try:
            return cupy.asnumpy(row_slice.values)
        except Exception:
            return np.asarray(row_slice.to_pandas())




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
for k in range(80):
    series[f"o_{k}"] = series.u_out.list.get(k)

for k in range(80):
    series[f"t_{k}"] = series.time_step.list.get(k)

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

model = NearestNeighbors(n_neighbors=NEIGHBORS, metric="l1")
model.fit(series.iloc[:, IGNORE + SKIP : IGNORE + TIME_STEPS + SKIP])

SHOW = [47, 68]
CTS = [300, 300]

for USE, CT in zip(SHOW, CTS):
    try:
        distances, indices = model.kneighbors(
            series.iloc[USE : USE + 1, IGNORE + SKIP : IGNORE + TIME_STEPS + SKIP]
        )

        plt.figure(figsize=(20, 4))
        plt.plot(
            np.arange(80),
            _row_to_numpy(series.iloc[USE, IGNORE : IGNORE + 80]),
            label="u_in",
        )
        plt.plot(
            np.arange(80),
            _row_to_numpy(series.iloc[USE, IGNORE + 80 : IGNORE + 160]),
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
        exhale_step = int(series.loc[USE, "expire"])
        plt.plot(
            [exhale_step, exhale_step], [0, y_max], "--", color="black", label="exhale"
        )
        rr = int(series.loc[USE, "R"])
        cc = int(series.loc[USE, "C"])
        bb = int(series.loc[USE, "breath_id"])
        tt = float(series.loc[USE, "time_length"])
        plt.title(f"Breath_Id={bb}, R={rr}, C={cc}, Length={tt:.3}", size=16)
        plt.legend()

        idxs = cupy.asnumpy(indices[0]).astype(np.int64)
        temp2 = series.iloc[idxs].reset_index(drop=True)

        CT_eff = int(min(CT, len(temp2)))
        if CT_eff <= 0:
            print(f"Diagnostic plot skipped for USE={USE}: no neighbors returned")
            continue

        cdict = {5: "yellow", 20: "orange", 50: "red"}
        plt.figure(figsize=(20, 10))
        legend = {5: 0, 20: 0, 50: 0}
        for r in range(CT_eff):
            if r == 0:
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE : IGNORE + 80]),
                    color="blue",
                    label="u_in",
                )
            else:
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE : IGNORE + 80]),
                    color="blue",
                )
            rr2 = int(temp2.loc[r, "R"])
            if legend[rr2] == 0:
                legend[rr2] = 1
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE + 80 : IGNORE + 160]),
                    c=cdict[rr2],
                    label=f"R={rr2}",
                )
            else:
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE + 80 : IGNORE + 160]),
                    c=cdict[rr2],
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
            f"{CT_eff} time series similar to breath_Id={bb}. Pressure colored by R",
            size=16,
        )
        plt.legend()
        plt.show()

        temp2 = series.iloc[idxs].reset_index(drop=True)
        cdict = {10: "red", 20: "orange", 50: "yellow"}
        plt.figure(figsize=(20, 10))
        legend = {10: 0, 20: 0, 50: 0}
        for r in range(CT_eff):
            if r == 0:
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE : IGNORE + 80]),
                    color="blue",
                    label="u_in",
                )
            else:
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE : IGNORE + 80]),
                    color="blue",
                )
            cc2 = int(temp2.loc[r, "C"])
            if legend[cc2] == 0:
                legend[cc2] = 1
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE + 80 : IGNORE + 160]),
                    c=cdict[cc2],
                    label=f"C={cc2}",
                )
            else:
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE + 80 : IGNORE + 160]),
                    c=cdict[cc2],
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
            f"{CT_eff} time series similar to breath_Id={bb}. Pressure colored by C",
            size=16,
        )
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Diagnostic neighbor-plot skipped for USE={USE} due to error: {repr(e)}")



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
                    _row_to_numpy(temp2.iloc[r, IGNORE : IGNORE + 80]),
                    color="blue",
                )
            else:
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE : IGNORE + 80]),
                    color="blue",
                    label="u_in",
                )
            rr2 = int(temp2.loc[r, "R"])
            if legend[rr2] == 0:
                legend[rr2] = 1
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE + 80 : IGNORE + 160]),
                    c=cdict[rr2],
                    label=f"R={rr2}",
                )
            else:
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE + 80 : IGNORE + 160]),
                    c=cdict[rr2],
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
        plt.title(f"Cluster {k.item()}, u_in (blue) and pressure colored by R", size=16)
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
                    _row_to_numpy(temp2.iloc[r, IGNORE : IGNORE + 80]),
                    color="blue",
                )
            else:
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE : IGNORE + 80]),
                    color="blue",
                    label="u_in",
                )
            cc2 = int(temp2.loc[r, "C"])
            if legend[cc2] == 0:
                legend[cc2] = 1
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE + 80 : IGNORE + 160]),
                    c=cdict[cc2],
                    label=f"C={cc2}",
                )
            else:
                plt.plot(
                    np.arange(80),
                    _row_to_numpy(temp2.iloc[r, IGNORE + 80 : IGNORE + 160]),
                    c=cdict[cc2],
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
        plt.title(f"Cluster {k.item()}, u_in (blue) and pressure colored by C", size=16)
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
for k in range(80):
    test_series[f"o_{k}"] = test_series.u_out.list.get(k)

for k in range(80):
    test_series[f"t_{k}"] = test_series.time_step.list.get(k)

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

all_true = []
all_oof = []
all_true_inhale = []
all_oof_inhale = []

n_test_breaths = int(len(test_series))
test_pred_matrix = cupy.zeros((n_test_breaths, 80), dtype=cupy.float32)


def _get_group_x_stats(train_df_like):
    """
    z-score u_in in feature space so L1 distances reflect shape, not magnitude.
    Stats are computed per (R,C) training subset and applied to both train/test subset.
    """
    x = train_df_like[[f"x_{k}" for k in range(80)]].values.astype(cupy.float32)
    mu = cupy.mean(x)
    sigma = cupy.std(x)
    sigma = cupy.maximum(sigma, cupy.float32(1e-6))
    return mu, sigma


def _make_knn_features(
    df_like,
    x_mu,
    x_sigma,
    t_scale=cupy.float32(10.0),
    o_scale=cupy.float32(5.0),
):
    """
    Core logic preserved: KNN on per-breath sequences with inspiratory masking.
    """
    x = df_like[[f"x_{k}" for k in range(80)]].values.astype(cupy.float32)
    o = df_like[[f"o_{k}" for k in range(80)]].values.astype(cupy.float32)
    z = df_like[[f"z_{k}" for k in range(80)]].values.astype(
        cupy.float32
    )  # 1=inspire,0=exhale
    t = df_like[[f"t_{k}" for k in range(80)]].values.astype(cupy.float32)

    x_std = (x - x_mu) / x_sigma
    x_masked = x_std * z
    t_masked = (t * t_scale) * z
    o_masked = (o * o_scale) * z

    return cupy.concatenate([x_masked, t_masked, o_masked], axis=1)  # (n, 240)


def _predict_from_neighbor_binned_mapping_tx(
    train_df_like, nn_idx_vec, test_x_mat, test_t_mat, n_bins_t=32, n_bins_x=64
):
    neigh_x = train_df_like.loc[
        nn_idx_vec, [f"x_{k}" for k in range(80)]
    ].values.astype(cupy.float32)
    neigh_t = train_df_like.loc[
        nn_idx_vec, [f"t_{k}" for k in range(80)]
    ].values.astype(cupy.float32)
    neigh_y = train_df_like.loc[
        nn_idx_vec, [f"y_{k}" for k in range(80)]
    ].values.astype(cupy.float32)

    x = cupy.clip(test_x_mat, cupy.float32(0.0), cupy.float32(100.0))
    t = cupy.clip(test_t_mat, cupy.float32(0.0), cupy.float32(3.0))

    nx = cupy.clip(neigh_x, cupy.float32(0.0), cupy.float32(100.0))
    nt = cupy.clip(neigh_t, cupy.float32(0.0), cupy.float32(3.0))

    bx = cupy.minimum(
        (x * (cupy.float32(n_bins_x) / cupy.float32(100.0))).astype(cupy.int32),
        cupy.int32(n_bins_x - 1),
    )
    bt = cupy.minimum(
        (t * (cupy.float32(n_bins_t) / cupy.float32(3.0))).astype(cupy.int32),
        cupy.int32(n_bins_t - 1),
    )

    bnx = cupy.minimum(
        (nx * (cupy.float32(n_bins_x) / cupy.float32(100.0))).astype(cupy.int32),
        cupy.int32(n_bins_x - 1),
    )
    bnt = cupy.minimum(
        (nt * (cupy.float32(n_bins_t) / cupy.float32(3.0))).astype(cupy.int32),
        cupy.int32(n_bins_t - 1),
    )

    n = neigh_y.shape[0]
    sum_y = cupy.zeros((n, n_bins_t, n_bins_x), dtype=cupy.float32)
    cnt = cupy.zeros((n, n_bins_t, n_bins_x), dtype=cupy.float32)

    rows = cupy.arange(n, dtype=cupy.int32)[
        :, None
    ]  # (n,1) for broadcasting with (n,80)
    cupy.add.at(sum_y, (rows, bnt, bnx), neigh_y)
    cupy.add.at(cnt, (rows, bnt, bnx), cupy.float32(1.0))

    neigh_mean = cupy.mean(neigh_y, axis=1, keepdims=True)  # (n,1)
    mean_y = cupy.where(
        cnt > 0,
        sum_y / cupy.maximum(cnt, cupy.float32(1.0)),
        neigh_mean[:, None, None],
    )

    pred = mean_y[rows, bt, bx]  # (n,80)
    return pred.astype(cupy.float32)


group_mean_curve = {}  # (R,C) -> cupy float32 [80]
for r in [5, 20, 50]:
    for c in [10, 20, 50]:
        temp_g = series.loc[(series.R == r) & (series.C == c)]
        if len(temp_g) == 0:
            continue
        means = []
        for DO in range(80):
            means.append(cupy.mean(temp_g[f"y_{DO}"].values.astype(cupy.float32)))
        group_mean_curve[(int(r), int(c))] = cupy.stack(means).astype(cupy.float32)

global_mean_curve = cupy.stack(
    [cupy.mean(series[f"y_{DO}"].values.astype(cupy.float32)) for DO in range(80)]
).astype(cupy.float32)

filled_test_rows = cupy.zeros((n_test_breaths,), dtype=cupy.int8)

for r in [5, 20, 50]:
    for c in [10, 20, 50]:
        temp2 = series.loc[(series.R == r) & (series.C == c)].reset_index(drop=True)
        test_temp2 = test_series.loc[
            (test_series.R == r) & (test_series.C == c)
        ].reset_index(drop=True)

        if len(test_temp2) == 0:
            continue

        if len(temp2) == 0:
            temp2 = series

        x_mu, x_sigma = _get_group_x_stats(temp2)

        X_train = _make_knn_features(temp2, x_mu, x_sigma)
        X_test = _make_knn_features(test_temp2, x_mu, x_sigma)

        model = NearestNeighbors(n_neighbors=NEIGHBORS, metric="l1")
        model.fit(X_train)

        err = []
        _, indices = model.kneighbors(X_train)
        _, test_indices = model.kneighbors(X_test)

        nn_train = indices[:, 0].astype(cupy.int32)
        nn_test = test_indices[:, 0].astype(cupy.int32)

        test_rows = test_temp2["row"].values.astype(cupy.int32)
        filled_test_rows[test_rows] = 1

        mean_curve = group_mean_curve.get((int(r), int(c)), global_mean_curve)

        test_x_mat = test_temp2[[f"x_{k}" for k in range(80)]].values.astype(
            cupy.float32
        )
        test_t_mat = test_temp2[[f"t_{k}" for k in range(80)]].values.astype(
            cupy.float32
        )

        preds_knn_mapped = _predict_from_neighbor_binned_mapping_tx(
            temp2, nn_test, test_x_mat, test_t_mat, n_bins_t=32, n_bins_x=64
        )  # (n_group_test, 80)

        for DO in range(80):
            true = temp2[f"y_{DO}"].values
            oof = temp2.loc[nn_train, f"y_{DO}"].values
            mask = temp2[f"z_{DO}"].values

            mae = cupy.mean(cupy.abs(true - oof))
            err.append(float(mae.get()))

            idx_inhale = cupy.where(mask == 1)[0]
            if int(idx_inhale.size) > 0:
                all_true_inhale.append(true[idx_inhale])
                all_oof_inhale.append(oof[idx_inhale])
            all_true.append(true)
            all_oof.append(oof)

            z_test = test_temp2[f"z_{DO}"].values.astype(
                cupy.float32
            )  # (n_group_test,)

            preds_blend = preds_knn_mapped[:, DO] * z_test + cupy.float32(0.0) * (
                cupy.float32(1.0) - z_test
            )
            test_pred_matrix[test_rows, DO] = preds_blend.astype(cupy.float32)

        plt.figure(figsize=(20, 5))
        plt.plot(np.arange(80), np.asarray(err), "-o")
        plt.ylabel("OOF MAE", size=14)
        plt.xlabel("Time Step", size=14)
        plt.title(f"OOF MAE for R={r} C={c}", size=16)
        plt.show()


if int(cupy.sum(filled_test_rows).get()) != n_test_breaths:
    missing = cupy.where(filled_test_rows == 0)[0]
    print(
        "Warning: missing test breath rows:",
        int(missing.size.get()),
        "- filling via global KNN fallback",
    )
    x_mu_all, x_sigma_all = _get_group_x_stats(series)
    X_train_all = _make_knn_features(series, x_mu_all, x_sigma_all)
    X_test_missing = _make_knn_features(
        test_series.iloc[cupy.asnumpy(missing)], x_mu_all, x_sigma_all
    )

    model_all = NearestNeighbors(n_neighbors=NEIGHBORS, metric="l1")
    model_all.fit(X_train_all)
    _, test_indices_miss = model_all.kneighbors(X_test_missing)
    nn_test_miss = test_indices_miss[:, 0].astype(cupy.int32)

    miss_df = test_series.iloc[cupy.asnumpy(missing)]
    miss_x_mat = miss_df[[f"x_{k}" for k in range(80)]].values.astype(cupy.float32)
    miss_t_mat = miss_df[[f"t_{k}" for k in range(80)]].values.astype(cupy.float32)

    preds_miss_mapped = _predict_from_neighbor_binned_mapping_tx(
        series, nn_test_miss, miss_x_mat, miss_t_mat, n_bins_t=32, n_bins_x=64
    )  # (n_missing, 80)

    for DO in range(80):
        z_test_m = miss_df[f"z_{DO}"].values.astype(cupy.float32)
        preds_blend_m = preds_miss_mapped[:, DO] * z_test_m + cupy.float32(0.0) * (
            cupy.float32(1.0) - z_test_m
        )
        test_pred_matrix[missing, DO] = preds_blend_m.astype(cupy.float32)

    filled_test_rows[missing] = 1



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1371285458.py in <cell line: 0>()
    192             )  # (n_group_test,)
    193 
--> 194             preds_blend = preds_knn_mapped[:, DO] * z_test + cupy.float32(0.0) * (
    195                 cupy.float32(1.0) - z_test
    196             )

cupy/_core/core.pyx in cupy._core.core._ndarray_base.__mul__()

cupy/_core/_kernel.pyx in cupy._core._kernel.ufunc.__call__()

cupy/_core/internal.pyx in cupy._core.internal._broadcast_core()

ValueError: operands could not be broadcast together with shapes (826, 64) (826,)

## === cell 8
if len(all_true_inhale) > 0:
    all_true2 = cupy.concatenate(all_true_inhale)
    all_oof2 = cupy.concatenate(all_oof_inhale)
    rsme = cupy.mean(cupy.abs(all_true2 - all_oof2))
    print("RAPIDS KNN CV MAE=", float(rsme.get()), "(inhale only)")
else:
    print("RAPIDS KNN CV MAE=(inhale only) unavailable: no inhale samples collected")

if len(all_true) > 0:
    all_true2 = cupy.concatenate(all_true)
    all_oof2 = cupy.concatenate(all_oof)
    rsme = cupy.mean(cupy.abs(all_true2 - all_oof2))
    print("RAPIDS KNN CV MAE=", float(rsme.get()), "(all breath)")
else:
    print("RAPIDS KNN CV MAE=(all breath) unavailable: no samples collected")



## === cell 9
test_sorted = test.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
n_test_rows = int(len(test_sorted))

time_index = test_sorted.groupby("breath_id").cumcount().astype("int32").values
breath_ids = test_sorted["breath_id"].values

test_breath_ids = test_series["breath_id"].values
bid_to_row = pd.Series(
    data=np.arange(len(test_breath_ids), dtype=np.int32),
    index=cupy.asnumpy(test_breath_ids),
)

breath_row = cupy.asarray(
    bid_to_row.loc[cupy.asnumpy(breath_ids)].values, dtype=cupy.int32
)
pred_pressure = test_pred_matrix[breath_row, time_index].astype(cupy.float32)

pressure_grid = cupy.asarray(
    np.sort(train["pressure"].unique().to_pandas().values.astype(np.float32))
)
pred_pressure_2d = pred_pressure.reshape((-1, 1))
grid_2d = pressure_grid.reshape((1, -1))
nearest_idx = cupy.argmin(cupy.abs(pred_pressure_2d - grid_2d), axis=1)
pred_pressure_snapped = pressure_grid[nearest_idx]

sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
sub["pressure"] = cupy.asnumpy(pred_pressure_snapped).astype(np.float32)
sub.to_csv("submission_rapids_knn.csv", index=False)
print(sub.shape)
sub.head()
