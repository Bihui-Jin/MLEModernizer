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
Predict the `scalar_coupling_constant` between atom pairs in molecules, given the two atom types (e.g., C and H), the coupling type (e.g., `2JHC`), and any features you are able to create from the molecule structure (`xyz`) files.

## Metric
Log of the Mean Absolute Error, calculated for each scalar coupling type, and then averaged across types.

## Submission Format
```
id,scalar_coupling_constant
2324604,0.0
2324605,0.0
2324606,0.0
etc.
```

## Dataset
The training and test splits are by *molecule*, so that no molecule in the training data is found in the test data.

- **train.csv** - the training set, where the first column (`molecule_name`) is the name of the molecule where the coupling constant originates (the corresponding XYZ file is located at ./structures/.xyz), the second (`atom_index_0`) and third column (`atom_index_1`) is the atom indices of the atom-pair creating the coupling and the fourth column (`scalar_coupling_constant`) is the scalar coupling constant that we want to be able to predict
- **test.csv** - the test set; same info as train, without the target variable
- **sample_submission.csv** - a sample submission file in the correct format
- **structures.zip** - folder containing molecular structure (xyz) files, where the first line is the number of atoms in the molecule, followed by a blank line, and then a line for every atom, where the first column contains the atomic element (H for hydrogen, C for carbon etc.) and the remaining columns contain the X, Y and Z cartesian coordinates (a standard format for chemists and molecular visualization programs)
- **structures.csv** - this file contains the **same** information as the individual xyz structure files, but in a single file
- **dipole_moments.csv** - contains the molecular electric dipole moments. These are three dimensional vectors that indicate the charge distribution in the molecule. The first column (`molecule_name`) are the names of the molecule, the second to fourth column are the `X`, `Y` and `Z` components respectively of the dipole moment.
- **magnetic_shielding_tensors.csv** - contains the magnetic shielding tensors for all atoms in the molecules. The first column (`molecule_name`) contains the molecule name, the second column (`atom_index`) contains the index of the atom in the molecule, the third to eleventh columns contain the `XX`, `YX`, `ZX`, `XY`, `YY`, `ZY`, `XZ`, `YZ` and `ZZ` elements of the tensor/matrix respectively.
- **mulliken_charges.csv** - contains the mulliken charges for all atoms in the molecules. The first column (`molecule_name`) contains the name of the molecule, the second column (`atom_index`) contains the index of the atom in the molecule, the third column (`mulliken_charge`) contains the mulliken charge of the atom.
- **potential_energy.csv** - contains the potential energy of the molecules. The first column (`molecule_name`) contains the name of the molecule, the second column (`potential_energy`) contains the potential energy of the molecule.
- **scalar_coupling_contributions.csv** - The scalar coupling constants in `train.csv` (or corresponding files) are a sum of four terms. `scalar_coupling_contributions.csv` contain all these terms. The first column (`molecule_name`) are the name of the molecule, the second (`atom_index_0`) and third column (`atom_index_1`) are the atom indices of the atom-pair, the fourth column indicates the type of coupling, the fifth column (`fc`) is the Fermi Contact contribution, the sixth column (`sd`) is the Spin-dipolar contribution, the seventh column (`pso`) is the Paramagnetic spin-orbit contribution and the eighth column (`dso`) is the Diamagnetic spin-orbit contribution.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (111 lines)
            dipole_moments.csv (76511 lines)
            dipole_moments.csv.zip (892.4 kB)
            magnetic_shielding_tensors.csv (1379965 lines)
            magnetic_shielding_tensors.csv.zip (47.9 MB)
            mulliken_charges.csv (1379965 lines)
            mulliken_charges.csv.zip (9.5 MB)
            potential_energy.csv (76511 lines)
            potential_energy.csv.zip (641.9 kB)
            sample_submission.csv (467814 lines)
            sample_submission.csv.zip (846.9 kB)
            scalar_coupling_contributions.csv (4191264 lines)
            scalar_coupling_contributions.csv.zip (90.0 MB)
            structures.csv (1379965 lines)
            structures.csv.zip (33.0 MB)
            structures.zip (44.3 MB)
            test.csv (467814 lines)
            test.csv.zip (2.6 MB)
            train.csv (4191264 lines)
            train.csv.zip (43.6 MB)
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
            structures/
                dsgdb9nsd_000001.xyz (212 Bytes)
                dsgdb9nsd_000002.xyz (171 Bytes)
                ... and 76508 other files
        input/
            description.md (111 lines)
            dipole_moments.csv (76511 lines)
            dipole_moments.csv.zip (892.4 kB)
            magnetic_shielding_tensors.csv (1379965 lines)
            magnetic_shielding_tensors.csv.zip (47.9 MB)
            mulliken_charges.csv (1379965 lines)
            mulliken_charges.csv.zip (9.5 MB)
            potential_energy.csv (76511 lines)
            potential_energy.csv.zip (641.9 kB)
            sample_submission.csv (467814 lines)
            sample_submission.csv.zip (846.9 kB)
            scalar_coupling_contributions.csv (4191264 lines)
            scalar_coupling_contributions.csv.zip (90.0 MB)
            structures.csv (1379965 lines)
            structures.csv.zip (33.0 MB)
            structures.zip (44.3 MB)
            test.csv (467814 lines)
            test.csv.zip (2.6 MB)
            train.csv (4191264 lines)
            train.csv.zip (43.6 MB)
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
            structures/
                dsgdb9nsd_000001.xyz (212 Bytes)
                dsgdb9nsd_000002.xyz (171 Bytes)
                ... and 76508 other files
        working/
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
```

-> data/champs-scalar-coupling/dipole_moments.csv has 76510 rows and 4 columns.
The columns are: molecule_name, X, Y, Z

-> data/champs-scalar-coupling/magnetic_shielding_tensors.csv has 1379964 rows and 11 columns.
The columns are: molecule_name, atom_index, XX, YX, ZX, XY, YY, ZY, XZ, YZ, ZZ

-> data/champs-scalar-coupling/mulliken_charges.csv has 1379964 rows and 3 columns.
The columns are: molecule_name, atom_index, mulliken_charge

-> data/champs-scalar-coupling/potential_energy.csv has 76510 rows and 2 columns.
The columns are: molecule_name, potential_energy

-> data/champs-scalar-coupling/sample_submission.csv has 467813 rows and 2 columns.
The columns are: id, scalar_coupling_constant

-> data/champs-scalar-coupling/scalar_coupling_contributions.csv has 4191263 rows and 8 columns.
The columns are: molecule_name, atom_index_0, atom_index_1, type, fc, sd, pso, dso

-> data/champs-scalar-coupling/structures.csv has 1379964 rows and 6 columns.
The columns are: molecule_name, atom_index, atom, x, y, z

-> data/champs-scalar-coupling/test.csv has 467813 rows and 5 columns.
The columns are: id, molecule_name, atom_index_0, atom_index_1, type

-> data/champs-scalar-coupling/train.csv has 4191263 rows and 6 columns.
The columns are: id, molecule_name, atom_index_0, atom_index_1, type, scalar_coupling_constant

-> (stopped after 10 files for performance)

# 5. Target score

-2.872362622891446

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I fix the runtime failures by (1) removing hard dependencies on missing `../input/...` prediction files and making each external prediction optional with safe fallbacks. Then I ensure every added prediction column aligns by `id` (not by row order/index), which avoids silent misalignment and also prevents downstream `KeyError` for missing columns. Finally, I always compute `final_preds` and write a valid `ensemble_sub.csv` with the required columns; if `stack15` is unavailable, the script still produce a submission using the available ensemble components (score may differ, but it run end-to-end).'
- What this solution (achieved 1.18497) has done: 'Your current score (1.99777, lower-is-better) is far worse than the target (-2.87236), so we need a real (but still minimal) improvement rather than calibration tweaks. The biggest issue is that this notebook is effectively a “load external OOF/test predictions” ensemble; when most `../input/...` files are missing, it falls back to zeros/partial blends, which score terribly. To move toward the target while preserving the same ensemble logic, I add a small, deterministic in-notebook fallback model that produces reasonable predictions when external files are absent: a per-`type` median learned from `train.csv`, plus a light distance-based correction using `structures.csv` (both are consistent with the competition and fast). This keeps your blending/stacking semantics intact (it just fills missing base predictors with a much better baseline), and still writes a valid `ensemble_sub.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far from the target (-2.87236), so we need a real improvement while keeping the same “external-preds ensemble with fallback” core logic. The smallest high-impact fix is to make the fallback stronger but still simple: use per-`type` median as before, plus a per-`type` distance→coupling fit that is learned on a deterministic subset of train and then blended back toward the median to avoid overfitting/outliers. This keeps the overall ensemble structure identical (we only improve what happens when external prediction files are missing/weak) and stays fast by fitting on a capped sample per type. Finally, we keep submission formatting/alignment by `id` unchanged and always write `ensemble_sub.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far from the target (-2.87236), so we need to substantially improve predictions while keeping your existing “external-preds ensemble + fallback” structure intact. The biggest gain with minimal semantic change is to make the fallback much stronger by (1) adding a per-`type` baseline that depends on inter-atomic distance using binned train medians (very robust) and (2) adding a tiny per-type linear correction on top, trained deterministically on capped rows to stay fast. Then we increase the fallback’s blend weight when external prediction files are missing (which is your likely runtime environment), while leaving the ensemble logic unchanged when many strong external predictors are available. Finally, we keep strict `id` alignment and still write `ensemble_sub.csv` with the required columns.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is extremely far from the target (-2.87236), so we need a meaningful but still minimal improvement while keeping your “external predictions ensemble + fallback” logic intact. The most impactful issue is that your fallback is trained on raw `scalar_coupling_constant`, while the competition metric averages MAE by `type` in log space—so errors on small-magnitude types can dominate; we improve the fallback by learning and predicting residuals relative to the per-`type` median (and then add the median back), which better matches the metric without changing the overall approach. We also make the distance binning more adaptive per `type` via quantile bins (still just robust medians, not a new model family) so the fallback generalizes better across coupling types. Finally, we slightly increase the fallback blend weight when external predictors are sparse (your likely environment), to move the score substantially toward the target while preserving the same ensembling structure.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is still extremely far from the target (-2.87236), and the main limiter is that in this environment most of the strong external prediction files are missing, so your ensemble relies heavily on the fallback. To move the score substantially toward the target without changing your ensemble structure, I strengthen the fallback using only allowed competition files by adding two very lightweight, robust feature merges (Mulliken charge and magnetic shielding tensors) for the two atoms and learning per-`type` binned median residual corrections on top of your existing per-`type` median + distance logic. This keeps the same semantics (robust median baselines + residual corrections; no new “big model”/training loop), but gives the fallback much more signal and should dramatically reduce MAE by type. Finally, I slightly increase the fallback blend weight when external predictors are sparse (which is the likely case here), to ensure the improved fallback actually influences `final_preds`.'

# 9. Code solution

## === cell 0
import os, glob
import numpy as np
import pandas as pd

base_input = "../input"
print("Listing ../input (top-level):")
print(sorted(os.listdir(base_input))[:50])

for p in ["1-mpn", "nnet-b-seed-10", "nnet-b-seed-11", "nnet-b-seed-12", "champ-preds"]:
    full = os.path.join(base_input, p)
    if os.path.exists(full):
        print(f"\nFound {full}, sample contents:")
        print(sorted(os.listdir(full))[:20])
    else:
        print(f"\nMissing {full} (will skip).")




## === cell 1
import numpy as np
import pandas as pd

DATA_DIR = "../input/champs-scalar-coupling"
TARGET = "scalar_coupling_constant"

test = pd.read_csv(
    f"{DATA_DIR}/test.csv",
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)
print("test shape:", test.shape)
print(test.head())


def load_pred_series(path, target_col=TARGET, id_col="id", test_ids=None):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if id_col in df.columns:
        s = df.set_index(id_col)[target_col]
        if test_ids is not None:
            s = s.reindex(test_ids)
        return s
    df2 = pd.read_csv(path, index_col=0)
    if target_col in df2.columns:
        s = df2[target_col]
        if test_ids is not None:
            s = s.reindex(test_ids)
        return s
    if df.shape[1] == 1:
        s = df.iloc[:, 0]
        if test_ids is not None and len(s) == len(test_ids):
            s = pd.Series(s.values, index=test_ids)
        return s
    return None


def get_median_from_files(files, test_ids):
    series_list = []
    for f in files:
        s = load_pred_series(f, target_col=TARGET, test_ids=test_ids)
        if s is None:
            print(f"Missing/unreadable pred file (skipping): {f}")
            continue
        series_list.append(s.astype("float64"))
    if len(series_list) == 0:
        return None
    mat = pd.concat(series_list, axis=1)
    return mat.median(axis=1)


test_ids = test["id"].values
pred_cols = {}




## === cell 2
conservative_path = (
    "../input/champs-ensemble-conservative/ensemble_sub_conservative.csv"
)
s = load_pred_series(conservative_path, target_col=TARGET, test_ids=test_ids)
if s is not None:
    pred_cols["conservative"] = s
else:
    print(f"Not found: {conservative_path} (skipping conservative)")

n1_files = [
    "../input/champ-preds/gnn_median_2302.csv",
    "../input/champ-preds/gnn_median_2301.csv",
    "../input/champ-preds/gnn_median_adjusted_1JHC_2296.csv",
]
n1 = get_median_from_files(n1_files, test_ids=test_ids)
if n1 is not None:
    gnn_2312 = load_pred_series(
        "../input/champ-preds/gnn_median_65_68_2312.csv",
        target_col=TARGET,
        test_ids=test_ids,
    )
    lastgnn = load_pred_series(
        "../input/champ-preds/gnn_median_69_73.csv",
        target_col=TARGET,
        test_ids=test_ids,
    )
    if (gnn_2312 is not None) and (lastgnn is not None):
        pred_cols["n1"] = lastgnn * 0.5 + gnn_2312 * 0.3 + n1 * 0.2
    else:
        pred_cols["n1"] = n1
else:
    print("No n1 sources available; skipping n1")

s = load_pred_series(
    "../input/champ-preds/gnn0_median_2068.csv", target_col=TARGET, test_ids=test_ids
)
if s is not None:
    pred_cols["n2"] = s
else:
    print("Missing n2 source; skipping n2")

lgb_a_files = [
    "../input/champ-preds/submission_type_2100.csv",
    "../input/champ-preds/submission_type_2085.csv",
]
s = get_median_from_files(lgb_a_files, test_ids=test_ids)
if s is not None:
    pred_cols["lgb_a"] = s
else:
    print("No lgb_a sources available; skipping lgb_a")

lgb_m_files = [
    "../input/champ-preds/lgb_type_full_f286_10.csv",
    "../input/champ-preds/lgb_type_full_f262_10.csv",
]
s = get_median_from_files(lgb_m_files, test_ids=test_ids)
if s is not None:
    pred_cols["lgb_m"] = s
else:
    print("No lgb_m sources available; skipping lgb_m")

nnet_files = [
    "../input/nnpvals-seed-20/nnet_sub_s11.csv",
    "../input/champ-preds/nnet_sub.csv",
    "../input/nn-seed-10/nnet_sub.csv",
    "../input/nn-seed-11/nnet_sub.csv",
    "../input/nnet-c-seed-10/lgb_type_cv-1.7126_mae0.23572_fd5_10.csv",
    "../input/nnet-c-seed-11/lgb_type_cv-1.70994_mae0.23497_fd5_11.csv",
    "../input/nnet-c-seed-12/lgb_type_cv-1.71029_mae0.23523_fd5_12.csv",
    "../input/nnet-b-seed-10/lgb_type_cv-1.72296_mae0.24019_bags-1_f120_fd5_10.csv",
    "../input/nnet-b-seed-11/lgb_type_cv-1.70647_mae0.2376_bags-1_f120_fd5_11.csv",
    "../input/nnet-b-seed-12/lgb_type_cv-1.69833_mae0.24106_bags-1_f120_fd5_12.csv",
    "../input/nnet-try-seed-11/lgb_type_cv-1.64944_mae0.23965_bags-1_f120_fd5_11.csv",
]
s = get_median_from_files(nnet_files, test_ids=test_ids)
if s is not None:
    pred_cols["nnet"] = s
else:
    print("No nnet sources available; skipping nnet")

nnet_cont_files = [
    "../input/nncont-seed-23-p/nnetCont_sub_only_predict.csv",
    "../input/nncont-seed-22/nnetCont_sub-1.8213.csv",
]
s = get_median_from_files(nnet_cont_files, test_ids=test_ids)
if s is not None:
    nn_contof = load_pred_series(
        "../input/nncont-seed-26/nnetCont_sub_-2.6677.csv",
        target_col=TARGET,
        test_ids=test_ids,
    )
    if nn_contof is not None:
        pred_cols["nnet_cont"] = nn_contof * 0.6 + s * 0.4
    else:
        pred_cols["nnet_cont"] = s
else:
    print("No nnet_cont sources available; skipping nnet_cont")

s = load_pred_series(
    "../input/champ-preds/final_mpnn.csv", target_col=TARGET, test_ids=test_ids
)
if s is not None:
    pred_cols["final_mpnn"] = s
else:
    print("Missing final_mpnn; skipping final_mpnn")

s = load_pred_series(
    "../input/champ-preds/mpnn_5_fold_pseudo_1449.csv",
    target_col=TARGET,
    test_ids=test_ids,
)
if s is not None:
    pred_cols["mpnn"] = s
else:
    print("Missing mpnn; skipping mpnn")

s = load_pred_series(
    "../input/chemistry-of-best-models-1-895/stack_median.csv",
    target_col=TARGET,
    test_ids=test_ids,
)
if s is not None:
    pred_cols["lb"] = s
else:
    print("Missing lb stack; skipping lb")

s = load_pred_series(
    "../input/15th-stacking/test_stacking.csv", target_col=TARGET, test_ids=test_ids
)
if s is not None:
    pred_cols["stack15"] = s
else:
    print("Missing stack15; skipping stack15")

for k, s in pred_cols.items():
    test[k] = s.values

print("Available prediction columns:", sorted(pred_cols.keys()))
print(
    test[
        [
            c
            for c in test.columns
            if c not in ["molecule_name", "atom_index_0", "atom_index_1", "type"]
        ]
    ].head()
)




## === cell 3
def build_fallback_predictions(test_df, data_dir=DATA_DIR, target_col=TARGET):
    train_y = pd.read_csv(
        f"{data_dir}/train.csv",
        usecols=["molecule_name", "atom_index_0", "atom_index_1", "type", target_col],
    )
    type_median = train_y.groupby("type")[target_col].median()
    global_median = float(train_y[target_col].median())

    train_y = train_y.copy()
    train_y["type_med"] = train_y["type"].map(type_median).astype("float64")
    train_y["resid"] = (
        train_y[target_col].astype("float64") - train_y["type_med"]
    ).astype("float64")

    base_med = test_df["type"].map(type_median).astype("float64").fillna(global_median)
    base = base_med.copy()

    try:
        structures = pd.read_csv(
            f"{data_dir}/structures.csv",
            usecols=["molecule_name", "atom_index", "x", "y", "z"],
        )
        s0 = structures.rename(
            columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"}
        )
        s1 = structures.rename(
            columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"}
        )

        train_full = train_y.merge(
            s0, on=["molecule_name", "atom_index_0"], how="left"
        ).merge(s1, on=["molecule_name", "atom_index_1"], how="left")

        dx = (train_full["x0"] - train_full["x1"]).astype("float64")
        dy = (train_full["y0"] - train_full["y1"]).astype("float64")
        dz = (train_full["z0"] - train_full["z1"]).astype("float64")
        train_full["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype("float64")

        mull = pd.read_csv(
            f"{data_dir}/mulliken_charges.csv",
            usecols=["molecule_name", "atom_index", "mulliken_charge"],
        )
        m0 = mull.rename(
            columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"}
        )
        m1 = mull.rename(
            columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"}
        )
        train_full = train_full.merge(
            m0, on=["molecule_name", "atom_index_0"], how="left"
        ).merge(m1, on=["molecule_name", "atom_index_1"], how="left")
        train_full["qdiff"] = (train_full["q0"] - train_full["q1"]).astype("float64")
        train_full["qsum"] = (train_full["q0"] + train_full["q1"]).astype("float64")

        mst = pd.read_csv(
            f"{data_dir}/magnetic_shielding_tensors.csv",
            usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
        )
        mst["trace"] = (
            mst["XX"].astype("float64")
            + mst["YY"].astype("float64")
            + mst["ZZ"].astype("float64")
        ).astype("float64")
        t0 = mst[["molecule_name", "atom_index", "trace"]].rename(
            columns={"atom_index": "atom_index_0", "trace": "t0"}
        )
        t1 = mst[["molecule_name", "atom_index", "trace"]].rename(
            columns={"atom_index": "atom_index_1", "trace": "t1"}
        )
        train_full = train_full.merge(
            t0, on=["molecule_name", "atom_index_0"], how="left"
        ).merge(t1, on=["molecule_name", "atom_index_1"], how="left")
        train_full["tdiff"] = (train_full["t0"] - train_full["t1"]).astype("float64")
        train_full["tsum"] = (train_full["t0"] + train_full["t1"]).astype("float64")

        per_type_cap = 140000
        train_full = (
            train_full.groupby("type", group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), per_type_cap), random_state=42))
            .reset_index(drop=True)
        )

        eps = 1e-6
        train_full = train_full[
            np.isfinite(train_full["dist"].values) & (train_full["dist"].values > eps)
        ].copy()

        n_bins = 16
        type_bin_resid_median = {}
        for t, g in train_full.groupby("type", sort=False):
            d = g["dist"].values.astype("float64")
            r = g["resid"].values.astype("float64")
            m = np.isfinite(d) & np.isfinite(r) & (d > eps)
            if m.sum() < 2000:
                continue
            dm = d[m]
            rm = r[m]
            qs = np.quantile(dm, np.linspace(0, 1, n_bins + 1))
            qs[0] = 0.0
            qs = np.maximum.accumulate(qs)
            qs[-1] = max(qs[-1], qs[-2] + 1e-9)
            b = np.digitize(dm, qs, right=False).astype("int16")
            b = np.clip(b, 1, n_bins + 1)
            df_tmp = pd.DataFrame({"b": b, "r": rm})
            med = df_tmp.groupby("b")["r"].median()
            type_bin_resid_median[t] = (qs.astype("float64"), med)

        n_qbins = 8
        type_bin2_qdiff = {}
        type_bin2_tsum = {}
        for t, g in train_full.groupby("type", sort=False):
            if len(g) < 5000:
                continue
            d = g["dist"].values.astype("float64")
            r = g["resid"].values.astype("float64")

            qd = g["qdiff"].values.astype("float64")
            m = np.isfinite(d) & np.isfinite(qd) & np.isfinite(r) & (d > eps)
            if m.sum() >= 5000:
                dm = d[m]
                qdm = qd[m]
                rm = r[m]
                d_edges = np.quantile(dm, np.linspace(0, 1, n_bins + 1))
                d_edges[0] = 0.0
                d_edges = np.maximum.accumulate(d_edges)
                d_edges[-1] = max(d_edges[-1], d_edges[-2] + 1e-9)

                q_edges = np.quantile(qdm, np.linspace(0, 1, n_qbins + 1))
                q_edges = np.maximum.accumulate(q_edges)
                q_edges[-1] = max(q_edges[-1], q_edges[-2] + 1e-12)

                db = np.digitize(dm, d_edges, right=False).astype("int16")
                db = np.clip(db, 1, n_bins + 1)
                qb = np.digitize(qdm, q_edges, right=False).astype("int16")
                qb = np.clip(qb, 1, n_qbins + 1)

                tmp = pd.DataFrame({"db": db, "qb": qb, "r": rm})
                med2 = tmp.groupby(["db", "qb"])["r"].median()
                type_bin2_qdiff[t] = (
                    d_edges.astype("float64"),
                    q_edges.astype("float64"),
                    med2,
                )

            ts = g["tsum"].values.astype("float64")
            m = np.isfinite(d) & np.isfinite(ts) & np.isfinite(r) & (d > eps)
            if m.sum() >= 5000:
                dm = d[m]
                tsm = ts[m]
                rm = r[m]
                d_edges = np.quantile(dm, np.linspace(0, 1, n_bins + 1))
                d_edges[0] = 0.0
                d_edges = np.maximum.accumulate(d_edges)
                d_edges[-1] = max(d_edges[-1], d_edges[-2] + 1e-9)

                t_edges = np.quantile(tsm, np.linspace(0, 1, n_qbins + 1))
                t_edges = np.maximum.accumulate(t_edges)
                t_edges[-1] = max(t_edges[-1], t_edges[-2] + 1e-12)

                db = np.digitize(dm, d_edges, right=False).astype("int16")
                db = np.clip(db, 1, n_bins + 1)
                tb = np.digitize(tsm, t_edges, right=False).astype("int16")
                tb = np.clip(tb, 1, n_qbins + 1)

                tmp = pd.DataFrame({"db": db, "tb": tb, "r": rm})
                med2 = tmp.groupby(["db", "tb"])["r"].median()
                type_bin2_tsum[t] = (
                    d_edges.astype("float64"),
                    t_edges.astype("float64"),
                    med2,
                )

        params = {}
        min_rows = 5000
        for t, g in train_full.groupby("type", sort=False):
            if len(g) < min_rows:
                continue
            d = g["dist"].values.astype("float64")
            y = g["resid"].values.astype("float64")
            m = np.isfinite(d) & np.isfinite(y) & (d > eps)
            if m.sum() < min_rows:
                continue
            dm = d[m]
            ym = y[m]
            X = np.column_stack([np.ones(dm.shape[0], dtype="float64"), dm, 1.0 / dm])
            try:
                beta = np.linalg.lstsq(X, ym, rcond=None)[0]
            except Exception:
                continue
            if np.all(np.isfinite(beta)):
                params[t] = beta  # a,b,c for residual

        test_full = test_df.merge(
            s0, on=["molecule_name", "atom_index_0"], how="left"
        ).merge(s1, on=["molecule_name", "atom_index_1"], how="left")
        dx = (test_full["x0"] - test_full["x1"]).astype("float64")
        dy = (test_full["y0"] - test_full["y1"]).astype("float64")
        dz = (test_full["z0"] - test_full["z1"]).astype("float64")
        dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype("float64")

        test_full = test_full.merge(
            m0, on=["molecule_name", "atom_index_0"], how="left"
        ).merge(m1, on=["molecule_name", "atom_index_1"], how="left")
        test_full["qdiff"] = (test_full["q0"] - test_full["q1"]).astype("float64")
        test_full["qsum"] = (test_full["q0"] + test_full["q1"]).astype("float64")

        test_full = test_full.merge(
            t0, on=["molecule_name", "atom_index_0"], how="left"
        ).merge(t1, on=["molecule_name", "atom_index_1"], how="left")
        test_full["tdiff"] = (test_full["t0"] - test_full["t1"]).astype("float64")
        test_full["tsum"] = (test_full["t0"] + test_full["t1"]).astype("float64")

        pred = base.copy()

        tvals = test_df["type"].values
        dvals = dist.values.astype("float64")

        for t in np.unique(tvals):
            if t not in type_bin_resid_median:
                continue
            edges, med = type_bin_resid_median[t]
            mask = (tvals == t) & np.isfinite(dvals) & (dvals > eps)
            if not mask.any():
                continue
            bins = np.digitize(dvals[mask], edges, right=False).astype("int16")
            bins = np.clip(bins, 1, n_bins + 1)
            resid_med = med.reindex(bins).values.astype("float64")
            good = np.isfinite(resid_med)
            idxs = np.where(mask)[0]
            if good.any():
                pred.values[idxs[good]] = pred.values[idxs[good]] + resid_med[good]

        qdiff_vals = test_full["qdiff"].values.astype("float64")
        q_w = 0.45
        for t in np.unique(tvals):
            if t not in type_bin2_qdiff:
                continue
            d_edges, q_edges, med2 = type_bin2_qdiff[t]
            mask = (
                (tvals == t)
                & np.isfinite(dvals)
                & (dvals > eps)
                & np.isfinite(qdiff_vals)
            )
            if not mask.any():
                continue
            db = np.digitize(dvals[mask], d_edges, right=False).astype("int16")
            db = np.clip(db, 1, n_bins + 1)
            qb = np.digitize(qdiff_vals[mask], q_edges, right=False).astype("int16")
            qb = np.clip(qb, 1, n_qbins + 1)
            idxs = np.where(mask)[0]
            keys = list(zip(db.tolist(), qb.tolist()))
            r = np.array([med2.get(k, np.nan) for k in keys], dtype="float64")
            good = np.isfinite(r)
            if good.any():
                pred.values[idxs[good]] = pred.values[idxs[good]] + q_w * r[good]

        tsum_vals = test_full["tsum"].values.astype("float64")
        t_w = 0.35
        for t in np.unique(tvals):
            if t not in type_bin2_tsum:
                continue
            d_edges, t_edges, med2 = type_bin2_tsum[t]
            mask = (
                (tvals == t)
                & np.isfinite(dvals)
                & (dvals > eps)
                & np.isfinite(tsum_vals)
            )
            if not mask.any():
                continue
            db = np.digitize(dvals[mask], d_edges, right=False).astype("int16")
            db = np.clip(db, 1, n_bins + 1)
            tb = np.digitize(tsum_vals[mask], t_edges, right=False).astype("int16")
            tb = np.clip(tb, 1, n_qbins + 1)
            idxs = np.where(mask)[0]
            keys = list(zip(db.tolist(), tb.tolist()))
            r = np.array([med2.get(k, np.nan) for k in keys], dtype="float64")
            good = np.isfinite(r)
            if good.any():
                pred.values[idxs[good]] = pred.values[idxs[good]] + t_w * r[good]

        shrink = 0.50
        for t, beta in params.items():
            mask = (tvals == t) & np.isfinite(dvals) & (dvals > eps)
            if mask.any():
                d = dvals[mask]
                raw = beta[0] + beta[1] * d + beta[2] * (1.0 / d)  # residual
                pred.values[mask] = pred.values[mask] + shrink * raw

        pred = pd.Series(pred, index=test_df.index).astype("float64")
        pred = (
            pred.replace([np.inf, -np.inf], np.nan).fillna(base).fillna(global_median)
        )
        return pred
    except Exception as e:
        print(
            "Fallback enriched model failed; using type-median only. Reason:", repr(e)
        )
        return base.astype("float64")


test["fallback"] = build_fallback_predictions(test)
print("Fallback stats:", test["fallback"].describe())




## === cell 4
def col_or_none(df, col):
    return df[col] if col in df.columns else None


nnet = col_or_none(test, "nnet")
nnet_cont = col_or_none(test, "nnet_cont")
lgb_a = col_or_none(test, "lgb_a")
lgb_m = col_or_none(test, "lgb_m")

if (nnet_cont is not None) and (nnet is not None):
    test["nnet_ens"] = nnet_cont * 0.6 + nnet * 0.4
elif nnet is not None:
    test["nnet_ens"] = nnet
elif nnet_cont is not None:
    test["nnet_ens"] = nnet_cont

if (lgb_a is not None) and (lgb_m is not None):
    test["lgb_ens"] = lgb_a * 0.8 + lgb_m * 0.2
elif lgb_a is not None:
    test["lgb_ens"] = lgb_a
elif lgb_m is not None:
    test["lgb_ens"] = lgb_m

terms = []
weights = []


def add_term(col, w):
    s = col_or_none(test, col)
    if s is not None:
        terms.append(s.astype("float64"))
        weights.append(float(w))


add_term("n1", 0.5)
add_term("n2", 0.1)
add_term("lgb_ens", 0.15)
add_term("nnet_ens", 0.1)
add_term("lb", 0.1)
add_term("final_mpnn", 0.03)
add_term("mpnn", 0.02)

n_ext = len([k for k in pred_cols.keys() if k not in ["stack15"]])

if "fallback" in test.columns:
    if n_ext == 0:
        add_term("fallback", 1.0)
    elif n_ext <= 2:
        add_term("fallback", 0.92)
    elif n_ext <= 4:
        add_term("fallback", 0.55)
    else:
        add_term("fallback", 0.12)

if len(terms) == 0:
    test["final_preds"] = 0.0
else:
    wsum = sum(weights)
    blended = sum(t * w for t, w in zip(terms, weights)) / wsum
    test["final_preds"] = blended

pred_cols_present = [
    c
    for c in [
        "n1",
        "n2",
        "lgb_ens",
        "nnet_ens",
        "lb",
        "final_mpnn",
        "mpnn",
        "fallback",
        "final_preds",
        "stack15",
    ]
    if c in test.columns
]
print("Columns used/produced:", pred_cols_present)
print(test[["id"] + pred_cols_present].head())




## === cell 5
submission = pd.DataFrame({"id": test["id"].astype(np.int64)})

if "stack15" in test.columns:
    submission[TARGET] = test["final_preds"] * 0.1 + test["stack15"] * 0.9
else:
    submission[TARGET] = test["final_preds"]

if submission[TARGET].isna().any():
    submission[TARGET] = submission[TARGET].fillna(submission[TARGET].median())

submission_path = "ensemble_sub.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission.shape)
print(submission.head())




## === cell 6
submission.head(20)
