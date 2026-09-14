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

-1.32544

# 6. Current score

1.99777

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'Diagnosis: Cell 1 crashes because it tries to read several CSVs from `../input/blender/`, but that directory (and those files) do not exist in the provided environment. This is an external-input dependency issue, not a pandas bug. To unblock execution while preserving downstream blending semantics in cell 2, we should load whichever of those files actually exist and otherwise fall back to the competition’s provided `sample_submission.csv` (with its required columns) so that `sub1..sub6` and their `scalar_coupling_constant` columns exist.

Patch summary: In cell 1 only, add a small helper that attempts to read each expected blender file from a list of plausible input roots and, if not found, falls back to `../input/champs-scalar-coupling/sample_submission.csv` and renames/ensures the prediction column. Keep the same variable names (`sub1`..`sub6`) and the same describe() prints so cell 2 remains compatible.

Updated cells: Only cell 1 is changed.

Compatibility notes for cell k+1: `sub1`, `sub2`, `sub3`, `sub4`, `sub5`, `sub6` remain pandas DataFrames and all contain a numeric `scalar_coupling_constant` column, so cell 2’s weighted blending and `to_csv` run unchanged.

Assumptions: If none of the blender submission files are present anywhere under `../input`, using `sample_submission.csv` (with constant/zero predictions) is acceptable as a safe fallback to prevent crashing; if any blender files are present, they be used preferentially.'
- What this solution (achieved 1.99777) has done: 'Diagnosis: Cell 3 crashes because newer pandas versions disallow passing the plot “kind” as a positional argument to `Series.plot()`. The code uses `sub1['scalar_coupling_constant'].plot('hist', bins=100)`, which now raises a `TypeError`.  
Patch summary: Update the call to pass `kind='hist'` as a keyword argument, preserving identical plotting behavior and leaving `sub1` unchanged.  
Updated cells: Only cell 3 is modified.  
Compatibility notes for cell k+1: No variables or outputs are renamed/removed; `sub1` remains the same DataFrame, so downstream cells (if any) behave identically.  
Assumptions: A plotting backend (e.g., matplotlib) is available as in typical Kaggle notebook environments; this patch only addresses the pandas API change.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777, lower-is-better) is far worse than the target (-1.32544), so we should improve legitimately with minimal changes. The biggest issue is that your “blender” files are missing, so you’re effectively submitting the zero-filled sample submission, which scores poorly. I keep the same blending logic but make cell 1 actually find and load the best available predictions: first try the specified blender paths, then try any CSVs under the competition input folder whose filenames look like submissions and select a few diverse candidates. This keeps the approach identical (weighted blending of multiple submissions) while giving it real prediction inputs, and still produces a valid `submission1236.csv`.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777, lower-is-better) is far from the target (-1.32544), and the main reason is that you’re blending mostly fallback/zero predictions because the referenced “blender” files aren’t present. I keep your exact blending approach (same 0.3/0.3/0.15/0.25 weights and same output file) but make cell 1 reliably discover and load real submission-like CSVs from the available dataset folder, instead of defaulting to sample_submission zeros. I also ensure all loaded candidates align to the exact sample submission `id` set/order, so the blend is correctly matched and not silently misaligned. This should legitimately improve the score while preserving your core logic (a weighted blend of four submissions).'
- What this solution (achieved 1.99777) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that you’re still likely blending mostly fallback/near-constant predictions because there are no real “blender” submissions available in the provided dataset. To move the score toward the target without changing your core approach (weighted blending of multiple submissions), I keep the same blend but make cell 1 generate several non-trivial, model-like baseline predictions from the official CHAMPS data (structures + train/test), so `sub1/sub2/sub3/sub6` become meaningful instead of zeros. This uses a per-`type` median of the target plus a distance-based linear adjustment (fit on train only) and aligns predictions to the sample submission `id` order to avoid any silent mismatch. The rest of your code (cell 2 blending weights and output filename) stays the same, but now the blend has signal and should substantially reduce the log-MAE.'
- What this solution (achieved 1.99777) has done: 'Your current submission is blending multiple internally-generated variants of the same simple distance-based baseline, which likely isn’t strong enough to approach the target; the easiest legitimate improvement (without changing the blending logic in cell 2) is to make each blended component more informative. I keep your exact “median + linear distance slope” approach, but compute the distance feature in a more CHAMPS-relevant way by enriching it with atom identity (atomic numbers for atom_0/atom_1) and fitting the same closed-form slope per type on that augmented feature set. This preserves your training approach (no iterative training loops, no new models) while providing a stronger signal than raw distance alone, and keeps the output schema/filename unchanged. I also keep id alignment to sample submission order to avoid silent mismatches.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))




## === cell 1
def _coerce_to_submission_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" not in df.columns:
        for c in df.columns:
            if str(c).lower().startswith("unnamed"):
                df = df.rename(columns={c: "id"})
                break
    if "scalar_coupling_constant" not in df.columns:
        if "prediction" in df.columns:
            df = df.rename(columns={"prediction": "scalar_coupling_constant"})
        else:
            candidate_cols = [c for c in df.columns if c != "id"]
            if len(candidate_cols) >= 1:
                df = df.rename(columns={candidate_cols[0]: "scalar_coupling_constant"})
    if "id" in df.columns and "scalar_coupling_constant" in df.columns:
        df = df[["id", "scalar_coupling_constant"]]
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["scalar_coupling_constant"] = pd.to_numeric(
        df["scalar_coupling_constant"], errors="coerce"
    ).fillna(0.0)
    return df


def _read_and_align_to_sample(path, sample_df):
    d = pd.read_csv(path)
    d = _coerce_to_submission_schema(d)
    if len(d) == 0:
        return None
    aligned = sample_df[["id"]].merge(d, on="id", how="left")
    miss = aligned["scalar_coupling_constant"].isna().mean()
    if miss > 0.01:
        return None
    aligned["scalar_coupling_constant"] = aligned["scalar_coupling_constant"].fillna(
        0.0
    )
    if aligned["scalar_coupling_constant"].std() <= 1e-9:
        return None
    return aligned


_ATOMIC_NUMBER = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


def _build_distance_feature(train_df, test_df, structures_df):
    s = structures_df[["molecule_name", "atom_index", "atom", "x", "y", "z"]].copy()
    s["Z"] = s["atom"].map(_ATOMIC_NUMBER).astype(np.float32)

    a0 = s.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "Z": "Z0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    a1 = s.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "Z": "Z1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )

    cols = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
    tr = (
        train_df[cols + ["scalar_coupling_constant"]]
        .merge(a0, on=["molecule_name", "atom_index_0"], how="left")
        .merge(a1, on=["molecule_name", "atom_index_1"], how="left")
    )
    te = (
        test_df[cols]
        .merge(a0, on=["molecule_name", "atom_index_0"], how="left")
        .merge(a1, on=["molecule_name", "atom_index_1"], how="left")
    )

    for d in (tr, te):
        dx = d["x0"] - d["x1"]
        dy = d["y0"] - d["y1"]
        dz = d["z0"] - d["z1"]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
        d["dist"] = dist

        zprod = (d["Z0"].astype(np.float32) * d["Z1"].astype(np.float32)).fillna(0.0)
        d["dist_z"] = (dist * zprod).astype(np.float32)

    return tr, te


def _fit_typewise_median_plus_dist_slope(train_feat: pd.DataFrame):
    grp = train_feat.groupby("type", sort=False)
    type_median = grp["scalar_coupling_constant"].median()

    f_mean = grp["dist_z"].mean()

    slopes = {}
    for t, df_t in grp:
        x = (df_t["dist_z"] - f_mean.loc[t]).to_numpy(dtype=np.float64)
        y = (df_t["scalar_coupling_constant"] - type_median.loc[t]).to_numpy(
            dtype=np.float64
        )
        var = np.mean(x * x)
        if var <= 1e-18:
            slopes[t] = 0.0
        else:
            slopes[t] = float(np.mean(x * y) / var)

    slopes = pd.Series(slopes)
    return type_median, f_mean, slopes


def _predict_median_plus_dist(test_feat: pd.DataFrame, type_median, f_mean, slopes):
    global_med = float(type_median.median())

    t = test_feat["type"]
    med = t.map(type_median).fillna(global_med).to_numpy(dtype=np.float64)
    fmean = t.map(f_mean).fillna(float(f_mean.mean())).to_numpy(dtype=np.float64)
    sl = t.map(slopes).fillna(0.0).to_numpy(dtype=np.float64)

    feat = test_feat["dist_z"].to_numpy(dtype=np.float64)
    pred = med + sl * (feat - fmean)
    return pred.astype(np.float32)


def _make_internal_submission_variants():
    data_root = "../input/champs-scalar-coupling"
    train = pd.read_csv(f"{data_root}/train.csv")
    test = pd.read_csv(f"{data_root}/test.csv")
    sample = pd.read_csv(f"{data_root}/sample_submission.csv")[["id"]].copy()
    sample["id"] = pd.to_numeric(sample["id"], errors="coerce").astype("Int64")

    structures = pd.read_csv(
        f"{data_root}/structures.csv",
        usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    )

    tr_feat, te_feat = _build_distance_feature(train, test, structures)
    type_median, f_mean, slopes = _fit_typewise_median_plus_dist_slope(tr_feat)

    base_pred = _predict_median_plus_dist(te_feat, type_median, f_mean, slopes)

    p1 = base_pred
    p2 = _predict_median_plus_dist(
        te_feat,
        type_median,
        f_mean,
        slopes * 0.9,
    )
    p3 = _predict_median_plus_dist(
        te_feat,
        type_median,
        f_mean,
        slopes * 1.1,
    )

    global_med = float(type_median.median())
    p6 = te_feat["type"].map(type_median).fillna(global_med).to_numpy(dtype=np.float32)

    def to_sub(preds):
        df = pd.DataFrame(
            {
                "id": te_feat["id"].astype("Int64").values,
                "scalar_coupling_constant": preds,
            }
        )
        aligned = sample.merge(df, on="id", how="left")
        aligned["scalar_coupling_constant"] = (
            aligned["scalar_coupling_constant"].fillna(0.0).astype(np.float32)
        )
        return aligned[["id", "scalar_coupling_constant"]]

    return (
        to_sub(p1),
        to_sub(p2),
        to_sub(p3),
        to_sub(p6),
        sample,
    )


def _load_best_effort_submissions():
    sample = pd.read_csv("../input/champs-scalar-coupling/sample_submission.csv")[
        ["id", "scalar_coupling_constant"]
    ].copy()
    sample["id"] = pd.to_numeric(sample["id"], errors="coerce").astype("Int64")
    sample["scalar_coupling_constant"] = pd.to_numeric(
        sample["scalar_coupling_constant"], errors="coerce"
    ).fillna(0.0)

    expected = [
        "../input/blender/LGB_2019-07-11_-1.4378.csv",
        "../input/blender/submission-2.csv",
        "../input/blender/stack_minmax_median.csv",
        "../input/blender/stack_mean.csv",
        "../input/blender/stack_median.csv",
        "../input/blender/workingsubmission-test.csv",
    ]

    loaded = []
    for p in expected:
        if os.path.exists(p):
            a = _read_and_align_to_sample(p, sample)
            if a is not None:
                loaded.append((p, a))

    if len(loaded) < 4:
        sub_int1, sub_int2, sub_int3, sub_int6, _sample_ids = (
            _make_internal_submission_variants()
        )
        loaded.extend(
            [
                ("internal_median_plus_distZ", sub_int1),
                ("internal_median_plus_distZ_0.9", sub_int2),
                ("internal_median_plus_distZ_1.1", sub_int3),
                ("internal_type_median", sub_int6),
            ]
        )

    while len(loaded) < 6:
        loaded.append(("fallback_sample_submission", sample.copy()))

    return loaded[:6]


picked = _load_best_effort_submissions()
sub1 = picked[0][1]
sub2 = picked[1][1]
sub3 = picked[2][1]
sub4 = picked[3][1]
sub5 = picked[4][1]
sub6 = picked[5][1]

print(sub1["scalar_coupling_constant"].describe())
print(sub2["scalar_coupling_constant"].describe())
print(sub3["scalar_coupling_constant"].describe())
print(sub3["scalar_coupling_constant"].describe())
print(sub4["scalar_coupling_constant"].describe())
print(sub5["scalar_coupling_constant"].describe())
print(sub6["scalar_coupling_constant"].describe())



## === cell 2
base_ids = sub1[["id"]].copy()
sub2a = base_ids.merge(sub2, on="id", how="left")
sub3a = base_ids.merge(sub3, on="id", how="left")
sub6a = base_ids.merge(sub6, on="id", how="left")

sub1["scalar_coupling_constant"] = (
    0.3 * sub1["scalar_coupling_constant"].values
    + 0.3 * sub2a["scalar_coupling_constant"].fillna(0.0).values
    + 0.15 * sub3a["scalar_coupling_constant"].fillna(0.0).values
    + 0.25 * sub6a["scalar_coupling_constant"].fillna(0.0).values
)
sub1.to_csv("submission1236.csv", index=False)



## === cell 3
sub1["scalar_coupling_constant"].plot(kind="hist", bins=100)
