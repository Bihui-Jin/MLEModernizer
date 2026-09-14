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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.1449394944940853

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 10.73281) has done: 'I remove the hard dependency on missing external Kaggle datasets (the `keraslstm151`, `lstmfold10146`, `youdontknow`, and `../input/torch*` folders) which currently crash the notebook, and replace it with a robust fallback that trains a simple per-(R,C,time_step,u_in,u_out) median lookup model directly from `train.csv`. This preserves the original “median/rounding” post-processing semantics and ensures a valid `submission.csv` is always written. I also make the OOF/submission ensembling cells conditional: if those external artifacts exist they be used; otherwise the fallback generates predictions end-to-end. Finally, I fix the bug where `df1.to_csv` was used instead of the blended/processed `submission.to_csv`, ensuring the correct file is saved.'
- What this solution (achieved 9.06816) has done: 'Your current fallback is a very weak “exact key median lookup” on continuous `u_in`/`time_step`, so it misses most rows and collapses to a global median, causing the huge MAE. I keep the same core lookup+median+rounding semantics, but change the key to a breath-sequence-friendly form by adding lag features (`u_in_lag1`, `u_out_lag1`) and cumulative volume (`u_in_cumsum`) computed per `breath_id`, then take a median over that richer key. This remains a pure train→median→merge model (no new learning loop/architecture), but drastically reduces the fallback error toward your target. I also ensure exact alignment to `sample_submission` ids and keep your post-processing unchanged.'
- What this solution (achieved 3.4003) has done: 'I keep your current “train→median lookup→merge→round/clip” core logic intact, but make the lookup key match train/test more often so the fallback doesn’t default to the global median (which is driving the huge MAE). Specifically, I (1) make the sequence-derived features consistent and more merge-friendly by using fixed quantization and stable per-breath ordering, and (2) add a tiny hierarchical backoff: try the full rich key first, then progressively drop the noisiest components (cumsum/lag) before finally using the global median. This remains the same modeling approach (median tables + merge) and same post-processing, but should substantially reduce error toward your 0.145 target without changing architecture/training loops. I also ensure the merged predictions stay aligned to `sample_submission` ids exactly.'
- What this solution (achieved 3.1561) has done: 'Your current score (3.4003 MAE; lower is better) is far from the target (0.1449), so we need a real accuracy lift while keeping your “median-lookup with rounding/clip” core approach intact. The biggest issue is the keying/backoff: using quantized `time_step/u_in` plus lag/cumsum still leaves many misses and doesn’t explicitly use the competition’s “only inspiratory phase is scored” fact, so the lookup is trying to fit both phases. I keep the same pipeline (train→groupby median tables→merge with hierarchical backoff→round/clip), but (1) train all median tables only on inspiratory rows (`u_out==0`) and (2) add a final per-(R,C,time_step_q,u_out) backoff so expiratory rows don’t collapse to a global median. This is a minimal, metric-aligned change that should reduce MAE substantially toward your target without changing the overall modeling semantics.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

paths = glob.glob("../input/torch*")
print(f"Found {len(paths)} torch* paths")


## === cell 1
src_copy = "../input/youdontknow/submission_median_round_LB153.csv"
if os.path.exists(src_copy):
    os.system(f"cp -r {src_copy} ./")
    print("Copied external submission file.")
else:
    print("External submission file not found; skipping copy.")




## === cell 2
class config:
    paths = {
        "train": "../input/ventilator-pressure-prediction/train.csv",
        "test": "../input/ventilator-pressure-prediction/test.csv",
        "ss": "../input/ventilator-pressure-prediction/sample_submission.csv",
    }

    model_params = {
        "is_train": True,
        "debug": False,
        "EPOCH": 300,
        "BATCH_SIZE": 1024,
        "NUM_FOLDS": 10,
    }

    post_processing = {
        "max_pressure": 64.82099173863948,
        "min_pressure": -1.8957442945646408,
        "diff_pressure": 0.07030215,
    }




## === cell 3
p1 = "../input/keraslstm151/submission_median_round_LB153.csv"
p2 = "../input/lstmfold10146/submission_median_round_LB153.csv"

submission = None

if os.path.exists(p1) and os.path.exists(p2):
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)

    if not {"id", "pressure"}.issubset(df1.columns) or not {"id", "pressure"}.issubset(
        df2.columns
    ):
        raise ValueError("External submissions must contain columns: id, pressure")

    df1 = df1.sort_values("id").reset_index(drop=True)
    df2 = df2.sort_values("id").reset_index(drop=True)
    if not np.array_equal(df1["id"].values, df2["id"].values):
        df2 = df2.set_index("id").reindex(df1["id"].values).reset_index()

    df1["pressure"] = df1["pressure"] * 0.15 + df2["pressure"] * 0.85
    submission = df1[["id", "pressure"]].copy()

    submission["pressure"] = (
        np.round(
            (submission.pressure - config.post_processing["min_pressure"])
            / config.post_processing["diff_pressure"]
        )
        * config.post_processing["diff_pressure"]
        + config.post_processing["min_pressure"]
    )
    submission["pressure"] = np.clip(
        submission["pressure"],
        config.post_processing["min_pressure"],
        config.post_processing["max_pressure"],
    )

    submission.to_csv("sub.csv", index=False)
    print("Wrote blended external submission to sub.csv")
else:
    print(
        "External blend files not found; will generate predictions from train.csv/test.csv in later cells."
    )


## === cell 4
df = None
if len(paths) > 0:
    oof_files = [p + "/oof.csv" for p in paths if os.path.exists(p + "/oof.csv")]
    if len(oof_files) > 0:
        df = pd.concat([pd.read_csv(f) for f in oof_files], ignore_index=True)
        if "pred" in df.columns:
            df = df[df.pred != 0]
        print(f"Loaded OOF from {len(oof_files)} files: {df.shape}")
    else:
        print("No oof.csv found under ../input/torch* paths; skipping OOF concat.")
else:
    print("No ../input/torch* paths; skipping OOF concat.")


## === cell 5
if df is not None and {"pred", "pressure"}.issubset(df.columns):
    mae = np.mean(np.abs(df["pred"].values - df["pressure"].values))
    print("OOF MAE:", mae)
else:
    print("OOF MAE not computed (missing df or required columns).")


## === cell 6
if submission is None:
    train_path = config.paths["train"]
    test_path = config.paths["test"]
    ss_path = config.paths["ss"]

    usecols_train = [
        "id",
        "breath_id",
        "R",
        "C",
        "time_step",
        "u_in",
        "u_out",
        "pressure",
    ]
    usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    train = pd.read_csv(train_path, usecols=usecols_train)
    test = pd.read_csv(test_path, usecols=usecols_test)
    ss = pd.read_csv(ss_path, usecols=["id"])

    def add_seq_features(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values(["breath_id", "time_step", "id"]).reset_index(drop=True)

        u_in = d["u_in"].astype(np.float32)

        d["u_in_lag1"] = (
            d.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
        )
        d["u_out_lag1"] = (
            d.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
        )

        dt = d.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
        d["u_in_cumsum"] = (
            (u_in * dt).groupby(d["breath_id"]).cumsum().astype(np.float32)
        )

        d["time_step_q"] = np.rint(d["time_step"].astype(np.float32) / 0.01).astype(
            np.int16
        )
        d["u_in_q"] = np.rint(u_in / 0.1).astype(np.int16)
        d["u_in_lag1_q"] = np.rint(d["u_in_lag1"] / 0.1).astype(np.int16)
        d["u_in_cumsum_q"] = np.rint(d["u_in_cumsum"] / 0.1).astype(np.int32)

        return d

    train_f = add_seq_features(train)
    test_f = add_seq_features(test)

    train_insp = train_f[train_f["u_out"] == 0].copy()
    train_exp = train_f[train_f["u_out"] == 1].copy()

    global_median_insp = float(train_insp["pressure"].median())
    global_median_exp = (
        float(train_exp["pressure"].median())
        if len(train_exp)
        else float(train_f["pressure"].median())
    )
    global_median_all = float(train_f["pressure"].median())

    key_full = [
        "R",
        "C",
        "time_step_q",
        "u_in_q",
        "u_out",
        "u_in_lag1_q",
        "u_out_lag1",
        "u_in_cumsum_q",
    ]
    key_drop_cumsum = [
        "R",
        "C",
        "time_step_q",
        "u_in_q",
        "u_out",
        "u_in_lag1_q",
        "u_out_lag1",
    ]
    key_drop_lags = ["R", "C", "time_step_q", "u_in_q", "u_out"]
    key_basic = ["R", "C", "time_step_q", "u_in_q"]
    key_phase = ["R", "C", "time_step_q", "u_out"]

    def build_tables(d: pd.DataFrame):
        med_full = d.groupby(key_full, sort=False)["pressure"].median().reset_index()
        med_dc = (
            d.groupby(key_drop_cumsum, sort=False)["pressure"].median().reset_index()
        )
        med_dl = d.groupby(key_drop_lags, sort=False)["pressure"].median().reset_index()
        med_basic = d.groupby(key_basic, sort=False)["pressure"].median().reset_index()
        return med_full, med_dc, med_dl, med_basic

    insp_tables = build_tables(train_insp)
    exp_tables = build_tables(train_exp) if len(train_exp) else insp_tables

    med_phase = (
        train_f.groupby(key_phase, sort=False)["pressure"].median().reset_index()
    )

    out = test_f[["id"] + key_full].copy()
    is_exp_mask = out["u_out"].values.astype(np.int8) == 1

    pred_series = pd.Series(np.nan, index=out.index, dtype="float64")

    def predict_with_backoff(
        out_part: pd.DataFrame, tables, phase_fill_df: pd.DataFrame, global_fill: float
    ) -> pd.Series:
        med_full, med_dc, med_dl, med_basic = tables

        p = out_part.merge(med_full, on=key_full, how="left")["pressure"]

        if p.isna().any():
            p2 = out_part.loc[p.isna(), key_drop_cumsum].merge(
                med_dc, on=key_drop_cumsum, how="left"
            )["pressure"]
            p.loc[p.isna()] = p2.values

        if p.isna().any():
            p3 = out_part.loc[p.isna(), key_drop_lags].merge(
                med_dl, on=key_drop_lags, how="left"
            )["pressure"]
            p.loc[p.isna()] = p3.values

        if p.isna().any():
            p4 = out_part.loc[p.isna(), key_basic].merge(
                med_basic, on=key_basic, how="left"
            )["pressure"]
            p.loc[p.isna()] = p4.values

        if p.isna().any():
            p5 = out_part.loc[p.isna(), key_phase].merge(
                phase_fill_df, on=key_phase, how="left"
            )["pressure"]
            p.loc[p.isna()] = p5.values

        if p.isna().any():
            p = p.fillna(global_fill)

        return p.astype(np.float64)

    if (~is_exp_mask).any():
        idx_insp = np.where(~is_exp_mask)[0]
        out_insp = out.iloc[idx_insp].copy()
        pred_series.iloc[idx_insp] = predict_with_backoff(
            out_insp, insp_tables, med_phase, global_median_insp
        ).values

    if is_exp_mask.any():
        idx_exp = np.where(is_exp_mask)[0]
        out_exp = out.iloc[idx_exp].copy()
        pred_series.iloc[idx_exp] = predict_with_backoff(
            out_exp, exp_tables, med_phase, global_median_exp
        ).values

    pred = pred_series.values

    pred = (
        np.round(
            (pred - config.post_processing["min_pressure"])
            / config.post_processing["diff_pressure"]
        )
        * config.post_processing["diff_pressure"]
        + config.post_processing["min_pressure"]
    )
    pred = np.clip(
        pred,
        config.post_processing["min_pressure"],
        config.post_processing["max_pressure"],
    )

    ss_sorted = ss.sort_values("id").reset_index(drop=True)
    id_to_pred = pd.DataFrame({"id": test_f["id"].values, "pressure": pred}).set_index(
        "id"
    )
    ss_sorted["pressure"] = id_to_pred.reindex(ss_sorted["id"].values)[
        "pressure"
    ].values
    submission = ss_sorted[["id", "pressure"]].copy()

    print(
        "Generated fallback submission via phase-specific median lookup with sequence features + hierarchical backoff + rounding/clip."
    )


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
IndexingError                             Traceback (most recent call last)
/tmp/ipykernel_11/1241301034.py in <cell line: 0>()
    147         idx_insp = np.where(~is_exp_mask)[0]
    148         out_insp = out.iloc[idx_insp].copy()
--> 149         pred_series.iloc[idx_insp] = predict_with_backoff(
    150             out_insp, insp_tables, med_phase, global_median_insp
    151         ).values

/tmp/ipykernel_11/1241301034.py in predict_with_backoff(out_part, tables, phase_fill_df, global_fill)
    115 
    116         if p.isna().any():
--> 117             p2 = out_part.loc[p.isna(), key_drop_cumsum].merge(
    118                 med_dc, on=key_drop_cumsum, how="left"
    119             )["pressure"]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1018                 continue
   1019 
-> 1020             retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1411             return self._get_slice_axis(key, axis=axis)
   1412         elif com.is_bool_indexer(key):
-> 1413             return self._getbool_axis(key, axis=axis)
   1414         elif is_list_like_indexer(key):
   1415             # an iterable multi-selection

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getbool_axis(self, key, axis)
   1207         # caller is responsible for ensuring non-None axis
   1208         labels = self.obj._get_axis(axis)
-> 1209         key = check_bool_indexer(labels, key)
   1210         inds = key.nonzero()[0]
   1211         return self.obj._take_with_is_copy(inds, axis=axis)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in check_bool_indexer(index, key)
   2660         indexer = result.index.get_indexer_for(index)
   2661         if -1 in indexer:
-> 2662             raise IndexingError(
   2663                 "Unalignable boolean Series provided as "
   2664                 "indexer (index of the boolean Series and of "

IndexingError: Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).

## === cell 7
if len(paths) > 0:
    sub_files = [
        p + "/submission.csv" for p in paths if os.path.exists(p + "/submission.csv")
    ]
    if len(sub_files) > 0:
        base = submission[["id"]].copy().sort_values("id").reset_index(drop=True)
        preds = []
        for f in sub_files:
            tmp = pd.read_csv(f)
            tmp = tmp[["id", "pressure"]].sort_values("id").reset_index(drop=True)
            if not np.array_equal(tmp["id"].values, base["id"].values):
                tmp = tmp.set_index("id").reindex(base["id"].values).reset_index()
            preds.append(tmp["pressure"].values.astype(np.float64))
        base["pressure"] = np.mean(np.vstack(preds), axis=0)

        base["pressure"] = (
            np.round(
                (base.pressure - config.post_processing["min_pressure"])
                / config.post_processing["diff_pressure"]
            )
            * config.post_processing["diff_pressure"]
            + config.post_processing["min_pressure"]
        )
        base["pressure"] = np.clip(
            base["pressure"],
            config.post_processing["min_pressure"],
            config.post_processing["max_pressure"],
        )
        submission = base
        print(f"Ensembled {len(sub_files)} torch* submissions.")
    else:
        print(
            "No submission.csv found under ../input/torch* paths; using current submission."
        )
else:
    print("No ../input/torch* paths; using current submission.")


## === cell 8
submission = submission[["id", "pressure"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission.shape)
print(submission.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3332190030.py in <cell line: 0>()
----> 1 submission = submission[["id", "pressure"]].copy()
      2 submission.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv:", submission.shape)
      4 print(submission.head())

TypeError: 'NoneType' object is not subscriptable
