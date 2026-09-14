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

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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

NEIGHBORS = 1
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

        for DO in range(80):

            true = temp2[f"y_{DO}"].values
            oof = temp2.loc[nn_idx, f"y_{DO}"].values

            mask = temp2[f"z_{DO}"].values

            preds = temp2.loc[test_nn0, f"y_{DO}"].values

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



## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1401037228.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     39[0m             [0mtest_temp2[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0mTEST_FEAT_SLICE[0m[0;34m][0m[0;34m,[0m [0mn_neighbors[0m[0;34m=[0m[0mNEIGHBORS[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m         )
[0;32m---> 41[0;31m         [0mtest_nn0[0m [0;34m=[0m [0mtest_indices[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0;36m0[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     42[0m [0;34m[0m[0m
[1;32m     43[0m         [0;32mfor[0m [0mDO[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m80[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m     49[0m                     )
[1;32m     50[0m                 )
[0;32m---> 51[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     52[0m [0;34m[0m[0m
[1;32m     53[0m     [0;32mreturn[0m [0mwrapper[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/cudf/core/series.py[0m in [0;36m__getitem__[0;34m(self, arg)[0m
[1;32m    186[0m     [0;32mdef[0m [0m__getitem__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0marg[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    187[0m         indexing_spec = indexing_utils.parse_row_iloc_indexer(
[0;32m--> 188[0;31m             [0mindexing_utils[0m[0;34m.[0m[0mdestructure_series_iloc_indexer[0m[0;34m([0m[0marg[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_frame[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    189[0m             [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_frame[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    190[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/cudf/core/indexing_utils.py[0m in [0;36mdestructure_series_iloc_indexer[0;34m(key, frame)[0m
[1;32m    177[0m     [0mSingle[0m [0mkey[0m [0mthat[0m [0mwill[0m [0mindex[0m [0mthe[0m [0mrows[0m[0;34m[0m[0;34m[0m[0m
[1;32m    178[0m     """
[0;32m--> 179[0;31m     [0;34m([0m[0mrows[0m[0;34m,[0m[0;34m)[0m [0;34m=[0m [0mdestructure_iloc_key[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mframe[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    180[0m     [0;32mreturn[0m [0mrows[0m[0;34m[0m[0;34m[0m[0m
[1;32m    181[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/cudf/core/indexing_utils.py[0m in [0;36mdestructure_iloc_key[0;34m(key, frame)[0m
[1;32m    102[0m         [0mindexers[0m [0;34m=[0m [0mkey[0m [0;34m+[0m [0;34m([0m[0mslice[0m[0;34m([0m[0;32mNone[0m[0;34m)[0m[0;34m,[0m[0;34m)[0m [0;34m*[0m [0;34m([0m[0mn[0m [0;34m-[0m [0mlen[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    103[0m         [0;32mif[0m [0mlen[0m[0;34m([0m[0mindexers[0m[0;34m)[0m [0;34m>[0m [0mn[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 104[0;31m             raise IndexError(
[0m[1;32m    105[0m                 [0;34mf"Too many indexers: got {len(indexers)} expected {n}"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    106[0m             )

[0;31mIndexError[0m: Too many indexers: got 2 expected 1

## === cell 8
all_true2 = cupy.concatenate(all_true_inhale)
all_oof2 = cupy.concatenate(all_oof_inhale)
rsme = cupy.mean(cupy.abs(all_true2 - all_oof2))
print("RAPIDS KNN CV RSME=", rsme, "(inhale only)")

all_true2 = cupy.concatenate(all_true)
all_oof2 = cupy.concatenate(all_oof)
rsme = cupy.mean(cupy.abs(all_true2 - all_oof2))
print("RAPIDS KNN CV RSME=", rsme, "(all breath)")
