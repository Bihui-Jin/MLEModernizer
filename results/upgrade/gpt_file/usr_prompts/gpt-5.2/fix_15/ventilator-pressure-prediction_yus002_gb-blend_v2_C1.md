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

3.9

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

0.4136034816750526

# 6. Current score

4.16538

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'I remove the dependency on missing external blend files (which causes the `FileNotFoundError`) and instead generate a valid baseline submission directly from the provided competition data. To keep changes minimal and score-neutral-but-valid, the fix read `sample_submission.csv` and write it back out as `submission.csv` (correct columns, correct `.csv` suffix). I also make the data path robust by checking both `/kaggle/input/ventilator-pressure-prediction/` and `/kaggle/data/` locations used in your environment. This ensures the notebook runs end-to-end and always produces a valid submission file.'
- What this solution (achieved 8.13508) has done: 'Your current score is very far from the target (17.65 vs 0.41, lower is better) because the fallback writes the all-zero `sample_submission.csv`, which is a valid file but a very weak predictor. To move the score toward the target with minimal change and without introducing new modeling, I keep your `blend()` interface but change the fallback to a simple physics-inspired baseline derived directly from `train.csv`: map each test `(R,C,u_in,u_out)` to the mean `pressure` learned from train, and predict 0 when `u_out==1` (expiratory phase is not scored, and this avoids absurd values). This preserves the “no external files needed” behavior while producing a substantially better submission and should reduce MAE dramatically toward your target band. I also make sure we always align predictions by `id` and still write `submission.csv` with the required columns.'
- What this solution (achieved 6.41508) has done: 'Your current MAE (8.13508, lower is better) is still far from the target (0.4136), so we need a small but materially stronger baseline without changing the overall “train-to-mean-map then predict” core logic. The biggest issue is that grouping by exact floating `u_in` creates sparse/unseen keys in test, causing heavy fallback to a global mean and hurting MAE. I keep the same mean-lookup approach but make it robust by (1) rounding `u_in` to 1 decimal (matching the dataset’s typical discretization) before grouping/merging, and (2) adding a hierarchical fallback: if the exact `(R,C,u_in,u_out)` key is missing, back off to `(R,C,u_out)`, then `(u_out)`, then global mean. I also keep setting `u_out==1` predictions to 0 (not scored phase), and ensure the submission remains correctly aligned by `id` and written as `submission.csv`.'
- What this solution (achieved 6.49836) has done: 'Your current MAE (6.415, lower is better) is still far from the target (0.414), so we need a modest but meaningful improvement without changing the overall “train-derived lookup then predict” approach. The largest remaining weakness is that a straight mean lookup doesn’t respect the discrete pressure levels and tends to blur them; snapping predictions to the nearest allowed pressure value from train is a minimal post-processing step aligned with the metric and typically improves MAE a lot for this competition. I keep your hierarchical backoff and `u_out==1 -> 0` behavior intact, but (1) build the set of allowed pressures from train, (2) quantize predicted pressures to the nearest allowed level, and (3) make `u_in` rounding slightly finer (2 decimals) to reduce key mismatch while still keeping grouping robust. The script still run end-to-end and write `submission.csv` with `id,pressure`.'
- What this solution (achieved 6.4149) has done: 'I keep your same “train-derived lookup with hierarchical backoff + snap to allowed pressures” core logic, but make two minimal, score-relevant fixes that typically reduce MAE for this competition. First, I stop forcing `u_out==1` predictions to 0 (those rows are simply ignored in scoring, and setting them to 0 can only hurt if any are accidentally scored or if masks differ), while leaving the rest of your pipeline unchanged. Second, I make the lookup less sparse by using `u_in` binning (round to 1 decimal) for the group keys while still snapping to discrete pressure levels afterward; this keeps the same approach but improves match rate between train/test keys. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.13268) has done: 'Your current score (6.4149 MAE, lower is better) is still far above the target (0.4136), so we need a small but meaningful improvement while keeping the same “train-derived lookup + hierarchical backoff + snap to allowed pressures” core logic. The main issue is that mean-by-key is too blunt for this competition; a very common minimal upgrade is to use the median (more robust) and, crucially, add a time-dependent feature derived from the sequence index within each breath (approximated via `id % 80` since each breath has 80 time steps). We keep the exact same lookup/backoff structure, just extend the key set with this within-breath step and use median aggregates, then still snap to allowed pressure levels. This typically reduces MAE substantially for ventilator pressure prediction without introducing any new model or training loop, and it stays fast enough under the runtime limit.'
- What this solution (achieved 4.13268) has done: 'We keep your existing “train-derived aggregation lookup + hierarchical backoff + snap-to-allowed pressures” core logic intact, but fix the biggest source of error: `t_in_breath` is currently computed from the global `id`, which does not reliably encode position within each breath. Instead, we compute `t_in_breath` from `time_step` (available in both train/test) by ranking within each `breath_id`, which matches the true 80-step sequence structure and should move MAE materially toward your target. To keep behavior stable and minimal, we still round `u_in` the same way, still use median aggregates, still do the same backoff chain, and still snap predictions to the discrete pressure grid from train. Runtime stays within limits by reading only needed columns and using groupby rank (vectorized).'
- What this solution (achieved 4.13268) has done: 'I keep your existing “train-derived aggregation lookup + hierarchical backoff + snap-to-allowed pressures” approach unchanged, but fix two score-relevant alignment issues that can strongly hurt MAE. First, `t_in_breath` should be computed by ordering `time_step` **within each breath** (sort + `cumcount`) rather than `rank()`, which can mis-assign step indices when there are floating ties/ordering quirks and can therefore break the key match. Second, I ensure `id` alignment is correct by sorting `test` by `id` before the final merge and by using an index-preserving merge, so predictions can’t drift across rows. These are minimal changes, keep the same semantics, and should move your MAE down toward the target.'
- What this solution (achieved 3.47755) has done: 'I keep your exact “train-derived aggregation lookup + hierarchical backoff + snap-to-allowed pressures” approach, but make the lookup keys more faithful to the underlying dynamics with a minimal feature: cumulative inspired volume (`u_in` integrated over time) per breath. This is a small extension of your existing key set (no new model/training loop) and typically reduces MAE a lot for this competition because pressure depends strongly on delivered volume, not just instantaneous `u_in`. To avoid harming generalization via key sparsity, I add it in a controlled way (rounded) and insert it into the backoff chain between your strongest key and weaker fallbacks. Everything remains aligned by `id`, runs end-to-end, and writes a valid `submission.csv`.'
- What this solution (achieved 3.92662) has done: 'We keep your exact aggregation-lookup + hierarchical backoff + snap-to-allowed-pressures pipeline, but make the added cumulative inspired volume feature more faithful to the underlying dynamics by integrating using the original (unrounded) `u_in` while still using rounded `u_in` only for the lookup keys. This is a minimal change that typically reduces key noise and improves match quality, moving MAE down toward your target without changing the approach or adding any model/training loop. We also slightly increase the resolution of `v_in` rounding (to 3 decimals) to reduce collision error while keeping the same backoff structure to avoid sparsity. All paths and output format remain the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 3.92662) has done: 'Your current MAE (3.92662, lower is better) is still far above the target (0.4136), so we should improve the same lookup/backoff approach without introducing any new model. The smallest high-impact change for this competition is to respect the fact that pressure is only evaluated when `u_out == 0`: we build all aggregation tables using only inspiratory rows (`u_out==0`) so the learned medians aren’t “polluted” by expiratory behavior. We also compute `v_in` (cumulative inspired volume) only over inspiratory flow by integrating `u_in` when `u_out==0` (and 0 otherwise), which better matches the physics while preserving your existing keys, backoff chain, and snapping to allowed pressures. Everything else (paths, merges, snapping, submission writing) stays the same and still produces `submission.csv` end-to-end.'
- What this solution (achieved 1.82631) has done: 'To move MAE down toward your target while keeping the same “train-derived aggregation lookup + hierarchical backoff + snap-to-allowed-pressures” core logic, I make the lookup keys less sparse and more physically consistent with two minimal feature tweaks. First, I compute `v_in` using a fixed `dt=0.033` per step (80 steps per breath) and cumulative sum of `u_in` during inspiration; this avoids noise from tiny floating `time_step` differences and improves train/test key matches. Second, I quantize `v_in` to a coarser but more stable grid (0.5) and slightly reduce `u_in` rounding (to 0 decimals) to further increase match rate; the hierarchical backoff and pressure snapping remain unchanged. These changes should improve generalization and reduce MAE without introducing any new model or training loop, and the script still write a valid `submission.csv`.'
- What this solution (achieved 2.75472) has done: 'We keep your exact aggregation-lookup + hierarchical backoff + snap-to-allowed-pressures pipeline, but make the lookup keys less sparse in a way that is still physically consistent and typically lowers MAE for this competition. Specifically, we (1) stop over-coarsening `u_in` (round to 1 decimal instead of 0) and (2) use a slightly finer `v_in` quantization grid (0.25 instead of 0.5) while keeping the fixed `dt=0.033` integration and inspiratory-only aggregation unchanged. These two minimal tweaks should increase train/test key match rate for the strongest tables (especially `means_6` and `means_5`) and reduce fallback usage, moving your score down toward the 0.41 target. Output paths and submission schema remain identical and the script still writes `submission.csv` end-to-end within the runtime limit.'
- What this solution (achieved 4.16538) has done: 'I keep your current aggregation-lookup + hierarchical backoff + snap-to-allowed-pressures pipeline, but make the keys slightly less sparse in a targeted way to reduce fallback usage (which is currently driving the MAE to 2.75 vs the 0.41 target). Concretely, I (1) quantize `v_in` a bit finer (0.20 vs 0.25) and (2) round `u_in` a bit finer (2 decimals vs 1), while keeping the same fixed-`dt` inspiratory-only integration, the same set of backoff tables, and the same snapping step. These are minimal parameter tweaks that should improve train/test key match rate without changing the core approach or adding any new modeling. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os




## === cell 1
def blend(a, b, c, d):
    """
    Original intent: blend 4 existing submission files.
    If any input files are missing (as in this environment), fall back to a valid baseline submission.

    Minimal score-improving changes (preserve core logic: train->lookup + hierarchical backoff + snapping):
    - Keep using ONLY inspiratory-phase rows (u_out==0) to build the aggregation lookup tables.
    - Compute v_in as fixed-step integration (dt=0.033) using cumulative sum of u_in during inspiration.
    - KEY CHANGE (score-relevant, minimal): reduce key sparsity a bit while keeping robustness:
        * round u_in to 2 decimals (from 1) to better preserve the control signal and reduce collisions
        * quantize v_in to grid 0.20 (from 0.25) to retain more state information without exploding unique keys
      This should reduce fallback frequency and lower MAE toward the target.
    """
    paths = [a, b, c, d]
    if all(isinstance(p, str) and os.path.exists(p) for p in paths):
        a_df = pd.read_csv(a)
        b_df = pd.read_csv(b)
        c_df = pd.read_csv(c)
        d_df = pd.read_csv(d)

        for name, df in [("a", a_df), ("b", b_df), ("c", c_df), ("d", d_df)]:
            if not {"id", "pressure"}.issubset(df.columns):
                raise ValueError(
                    f"Input submission {name} must contain columns: id, pressure"
                )

        a_df = a_df.sort_values("id").reset_index(drop=True)
        b_df = b_df.sort_values("id").reset_index(drop=True)
        c_df = c_df.sort_values("id").reset_index(drop=True)
        d_df = d_df.sort_values("id").reset_index(drop=True)

        if not (
            a_df["id"].equals(b_df["id"])
            and a_df["id"].equals(c_df["id"])
            and a_df["id"].equals(d_df["id"])
        ):
            raise ValueError("Input submissions have mismatched id ordering/values.")

        a_df["pressure"] = (
            a_df["pressure"] * 0.4
            + b_df["pressure"] * 0.3
            + c_df["pressure"] * 0.2
            + d_df["pressure"] * 0.1
        )
        out_path = "submission.csv"
        a_df[["id", "pressure"]].to_csv(out_path, index=False)
        return out_path

    candidate_roots = [
        "/kaggle/input/ventilator-pressure-prediction",
        "/kaggle/data/ventilator-pressure-prediction",
        "/kaggle/data",
        "/kaggle/input",
    ]

    def find_file(filename: str) -> str:
        for root in candidate_roots:
            p = os.path.join(root, filename)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"Could not find {filename} in expected Kaggle paths. Checked: "
            + ", ".join(candidate_roots)
        )

    train_path = find_file("train.csv")
    test_path = find_file("test.csv")
    sample_path = find_file("sample_submission.csv")

    train = pd.read_csv(
        train_path,
        usecols=["id", "breath_id", "time_step", "R", "C", "u_in", "u_out", "pressure"],
    )
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "time_step", "R", "C", "u_in", "u_out"]
    )

    train = train.sort_values(["breath_id", "time_step", "id"]).reset_index(drop=True)
    test = test.sort_values(["breath_id", "time_step", "id"]).reset_index(drop=True)
    train["t_in_breath"] = (
        train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    )
    test["t_in_breath"] = (
        test.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    )

    def add_cum_volume_inspiratory_fixed_dt(
        df: pd.DataFrame, dt: float = 0.033
    ) -> pd.Series:
        uin_eff = df["u_in"].astype(np.float64) * (df["u_out"].to_numpy() == 0).astype(
            np.float64
        )
        v = (uin_eff * dt).groupby(df["breath_id"], sort=False).cumsum()
        return v

    train["v_in"] = add_cum_volume_inspiratory_fixed_dt(train)
    test["v_in"] = add_cum_volume_inspiratory_fixed_dt(test)

    v_grid = 0.20
    train["v_in"] = (np.round(train["v_in"] / v_grid) * v_grid).astype(np.float32)
    test["v_in"] = (np.round(test["v_in"] / v_grid) * v_grid).astype(np.float32)

    train["u_in"] = train["u_in"].round(2)
    test["u_in"] = test["u_in"].round(2)

    train_insp = train.loc[train["u_out"] == 0].copy()
    aggfunc = "median"

    means_6 = (
        train_insp.groupby(
            ["R", "C", "t_in_breath", "v_in", "u_in", "u_out"], sort=False
        )["pressure"]
        .agg(aggfunc)
        .reset_index()
        .rename(columns={"pressure": "pressure_6"})
    )

    means_5 = (
        train_insp.groupby(["R", "C", "t_in_breath", "u_in", "u_out"], sort=False)[
            "pressure"
        ]
        .agg(aggfunc)
        .reset_index()
        .rename(columns={"pressure": "pressure_5"})
    )

    means_4 = (
        train_insp.groupby(["R", "C", "u_in", "u_out"], sort=False)["pressure"]
        .agg(aggfunc)
        .reset_index()
        .rename(columns={"pressure": "pressure_4"})
    )

    means_3 = (
        train_insp.groupby(["R", "C", "u_out"], sort=False)["pressure"]
        .agg(aggfunc)
        .reset_index()
        .rename(columns={"pressure": "pressure_3"})
    )

    means_2 = (
        train_insp.groupby(["t_in_breath", "u_out"], sort=False)["pressure"]
        .agg(aggfunc)
        .reset_index()
        .rename(columns={"pressure": "pressure_2"})
    )

    means_1 = (
        train_insp.groupby(["u_out"], sort=False)["pressure"]
        .agg(aggfunc)
        .reset_index()
        .rename(columns={"pressure": "pressure_1"})
    )

    global_fill = float(train_insp["pressure"].median())

    test = test.merge(
        means_6, on=["R", "C", "t_in_breath", "v_in", "u_in", "u_out"], how="left"
    )
    test = test.merge(
        means_5, on=["R", "C", "t_in_breath", "u_in", "u_out"], how="left"
    )
    test = test.merge(means_4, on=["R", "C", "u_in", "u_out"], how="left")
    test = test.merge(means_3, on=["R", "C", "u_out"], how="left")
    test = test.merge(means_2, on=["t_in_breath", "u_out"], how="left")
    test = test.merge(means_1, on=["u_out"], how="left")

    test["pressure"] = test["pressure_6"]
    test["pressure"] = test["pressure"].fillna(test["pressure_5"])
    test["pressure"] = test["pressure"].fillna(test["pressure_4"])
    test["pressure"] = test["pressure"].fillna(test["pressure_3"])
    test["pressure"] = test["pressure"].fillna(test["pressure_2"])
    test["pressure"] = test["pressure"].fillna(test["pressure_1"])
    test["pressure"] = test["pressure"].fillna(global_fill)

    allowed_pressures = np.sort(train["pressure"].unique()).astype(np.float64)

    def snap_to_allowed(values: np.ndarray, allowed: np.ndarray) -> np.ndarray:
        values = values.astype(np.float64)
        idx = np.searchsorted(allowed, values, side="left")
        idx = np.clip(idx, 0, len(allowed) - 1)
        left_idx = np.clip(idx - 1, 0, len(allowed) - 1)
        right = allowed[idx]
        left = allowed[left_idx]
        choose_left = np.abs(values - left) <= np.abs(values - right)
        return np.where(choose_left, left, right)

    test["pressure"] = snap_to_allowed(test["pressure"].to_numpy(), allowed_pressures)

    test = test.sort_values("id").reset_index(drop=True)

    sub = (
        pd.read_csv(sample_path, usecols=["id"])
        .sort_values("id")
        .reset_index(drop=True)
    )

    out = sub.merge(test[["id", "pressure"]], on="id", how="left").sort_values("id")

    if out["pressure"].isna().any():
        out["pressure"] = out["pressure"].fillna(global_fill)
        out["pressure"] = snap_to_allowed(out["pressure"].to_numpy(), allowed_pressures)

    out_path = "submission.csv"
    out[["id", "pressure"]].to_csv(out_path, index=False)
    return out_path




## === cell 2
submission_path = blend(
    "../input/gb-blending/0.455.csv",
    "../input/gb-blending/0.538.csv",
    "../input/gb-blending/0.634.csv",
    "../input/gb-blending/0.675.csv",
)

print("Wrote submission to:", submission_path)
print(pd.read_csv(submission_path).head())
print("Rows:", len(pd.read_csv(submission_path)))
