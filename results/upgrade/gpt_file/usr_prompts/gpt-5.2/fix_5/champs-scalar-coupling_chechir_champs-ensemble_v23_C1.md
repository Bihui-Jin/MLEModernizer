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

-2.4128442927016835

# 6. Current score

1.99777

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I fix the crash during the median ensemble by ensuring we only aggregate truly numeric prediction columns (coercing to numeric and dropping non-numeric series), which prevents strings like molecule names from entering the median computation. I also tighten the candidate-file selection so we don’t accidentally treat dataset CSVs (with non-prediction columns) as prediction files, while keeping the same “median over found predictions with sample-submission fallback” core logic. Finally, I make submission creation robust by guaranteeing `final_preds` always exists (even if no usable prediction files are found) and writing a valid `.csv` with the required columns.'
- What this solution (achieved 1.99777) has done: 'Your current score is far worse than the target (lower-is-better, and 1.99777 vs -2.4128), which strongly suggests your “ensemble” is effectively just the `sample_submission` (all zeros) because no real prediction files are being found/used. The smallest meaningful improvement is to stop relying on external files and instead train a simple per-`type` baseline on the provided `train.csv`, then predict each test row with that type’s median (with a global-median fallback for any unseen type). This preserves your overall “produce a valid submission from available CSVs” approach, but ensures non-trivial predictions using only competition-provided data, which should move the score substantially toward the target without changing any modeling architecture/loops (you currently have none). I keep your existing external-median logic as an optional override, but default to the per-type median baseline when no usable external predictions exist.'
- What this solution (achieved 1.99777) has done: 'Your current score (1.99777, lower-is-better) is far worse than the target (-2.4128), which strongly indicates your submission is effectively a weak baseline (e.g., per-type median) rather than a feature-based model. With minimal changes and without introducing a new model/training loop, the most direct way to move the score toward the target is to enrich the baseline using only officially provided side tables: add per-molecule `potential_energy` and `dipole_moments`, and per-atom `mulliken_charge`, then predict by the median target within groups defined by `(type, atom_0, atom_1)` plus coarse bins of distance/energy/dipole and charge summaries. This keeps the “median lookup” core logic but makes predictions substantially more informative, while retaining a safe fallback chain to ensure a valid submission is always written. External prediction-file ensembling remains supported and override if usable files are found.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np

INPUT_ROOT = "../input"
COMP_ROOT = os.path.join(INPUT_ROOT, "champs-scalar-coupling")
TARGET = "scalar_coupling_constant"

print("INPUT_ROOT exists:", os.path.exists(INPUT_ROOT))
print("COMP_ROOT exists:", os.path.exists(COMP_ROOT))
print(
    "Some input dirs:",
    sorted([p for p in glob.glob(os.path.join(INPUT_ROOT, "*")) if os.path.isdir(p)])[
        :20
    ],
)



## === cell 1
test_path = os.path.join(COMP_ROOT, "test.csv")
sample_path = os.path.join(COMP_ROOT, "sample_submission.csv")
train_path = os.path.join(COMP_ROOT, "train.csv")

test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

assert "id" in test.columns
assert list(sample.columns) == ["id", TARGET]

print("test shape:", test.shape)
print("sample shape:", sample.shape)
test.head()




## === cell 2
def _read_pred_file_as_series(path: str, test_ids: pd.Series) -> pd.Series:
    """
    Read a prediction CSV and return a float Series aligned to test_ids (index = id).
    Only returns usable numeric predictions; otherwise raises.
    """
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})

    if "id" not in df.columns:
        if df.shape[1] >= 2:
            df2 = df.copy()
            df2.columns = ["id"] + list(df2.columns[1:])
            df = df2
        else:
            raise ValueError(
                f"No 'id' column and not enough columns to infer in {path}"
            )

    if TARGET in df.columns:
        pred_col = TARGET
    else:
        cand_cols = [c for c in df.columns if c != "id"]
        if len(cand_cols) == 1:
            pred_col = cand_cols[0]
        else:
            raise ValueError(
                f"Cannot uniquely infer prediction column in {path}; columns={df.columns.tolist()}"
            )

    s = df.set_index("id")[pred_col]
    s = pd.to_numeric(s, errors="coerce")
    s = s.reindex(test_ids.values)

    coverage = float(s.notna().mean())
    if coverage < 0.90:
        raise ValueError(f"Low numeric coverage ({coverage:.3f}) in {path}")

    s.name = os.path.basename(path)
    return s


def get_median_from_files(files, test_ids: pd.Series) -> pd.Series:
    """
    Compute per-id median across a list of prediction files.
    Skips files that can't be read/aligned as numeric predictions.
    """
    usable = []
    for f in files:
        try:
            s = _read_pred_file_as_series(f, test_ids)
            usable.append(s)
        except Exception:
            continue

    print(f"Requested {len(files)} files; usable {len(usable)}")
    if len(usable) == 0:
        return pd.Series(
            [np.nan] * len(test_ids), index=test_ids.values, name="median_pred"
        )

    concat = pd.concat(usable, axis=1)
    concat = concat.apply(pd.to_numeric, errors="coerce")
    med = concat.median(axis=1, skipna=True)
    med.name = "median_pred"
    return med


all_csvs = glob.glob(os.path.join(INPUT_ROOT, "**", "*.csv"), recursive=True)
exclude = {
    os.path.join(COMP_ROOT, "train.csv"),
    os.path.join(COMP_ROOT, "test.csv"),
    os.path.join(COMP_ROOT, "sample_submission.csv"),
    os.path.join(COMP_ROOT, "structures.csv"),
    os.path.join(COMP_ROOT, "scalar_coupling_contributions.csv"),
    os.path.join(COMP_ROOT, "dipole_moments.csv"),
    os.path.join(COMP_ROOT, "magnetic_shielding_tensors.csv"),
    os.path.join(COMP_ROOT, "mulliken_charges.csv"),
    os.path.join(COMP_ROOT, "potential_energy.csv"),
}

name_keys = ("sub", "submission", "pred", "oof", "blend", "ensemble")
pred_csvs = [
    p
    for p in all_csvs
    if p not in exclude and any(k in os.path.basename(p).lower() for k in name_keys)
]

print("Total CSVs found:", len(all_csvs))
print("Candidate pred CSVs:", len(pred_csvs))
print("Example candidates:", pred_csvs[:20])



## === cell 3
test_ids = test["id"]
test["ens_median"] = get_median_from_files(pred_csvs, test_ids)

fallback = sample.set_index("id")[TARGET].reindex(test_ids.values)

if test["ens_median"].isna().all():
    print(
        "No usable external prediction files found; using enriched grouped-median baseline from train + side tables."
    )

    structures_path = os.path.join(COMP_ROOT, "structures.csv")
    mulliken_path = os.path.join(COMP_ROOT, "mulliken_charges.csv")
    dipole_path = os.path.join(COMP_ROOT, "dipole_moments.csv")
    pe_path = os.path.join(COMP_ROOT, "potential_energy.csv")

    train = pd.read_csv(
        train_path,
        usecols=["molecule_name", "atom_index_0", "atom_index_1", "type", TARGET],
    )
    test_min = test[
        ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
    ].copy()

    structs = pd.read_csv(
        structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
    )
    s0 = structs.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]
    s1 = structs.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]

    def add_geom(df):
        df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
        df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
        dx = df["x0"] - df["x1"]
        dy = df["y0"] - df["y1"]
        dz = df["z0"] - df["z1"]
        df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
        return df

    train_f = add_geom(train)
    test_f = add_geom(test_min)

    mull = pd.read_csv(
        mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
    )
    m0 = mull.rename(columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"})[
        ["molecule_name", "atom_index_0", "q0"]
    ]
    m1 = mull.rename(columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"})[
        ["molecule_name", "atom_index_1", "q1"]
    ]

    for df in (train_f, test_f):
        df.merge(m0, on=["molecule_name", "atom_index_0"], how="left", copy=False)

    train_f = train_f.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
    train_f = train_f.merge(m1, on=["molecule_name", "atom_index_1"], how="left")
    test_f = test_f.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
    test_f = test_f.merge(m1, on=["molecule_name", "atom_index_1"], how="left")

    train_f["q_abs_sum"] = train_f["q0"].abs() + train_f["q1"].abs()
    train_f["q_diff"] = (train_f["q0"] - train_f["q1"]).abs()
    test_f["q_abs_sum"] = test_f["q0"].abs() + test_f["q1"].abs()
    test_f["q_diff"] = (test_f["q0"] - test_f["q1"]).abs()

    pe = pd.read_csv(pe_path, usecols=["molecule_name", "potential_energy"])
    dip = pd.read_csv(dipole_path, usecols=["molecule_name", "X", "Y", "Z"])
    dip["dipole_norm"] = np.sqrt(dip["X"] ** 2 + dip["Y"] ** 2 + dip["Z"] ** 2)
    dip = dip[["molecule_name", "dipole_norm"]]

    train_f = train_f.merge(pe, on="molecule_name", how="left")
    train_f = train_f.merge(dip, on="molecule_name", how="left")
    test_f = test_f.merge(pe, on="molecule_name", how="left")
    test_f = test_f.merge(dip, on="molecule_name", how="left")

    def qbin_edges(s, qs):
        s = pd.to_numeric(s, errors="coerce")
        vals = s.dropna().values
        if vals.size == 0:
            return None
        edges = np.unique(np.quantile(vals, qs))
        if edges.size < 2:
            return None
        return edges

    dist_edges = qbin_edges(train_f["dist"], qs=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    pe_edges = qbin_edges(train_f["potential_energy"], qs=[0, 0.25, 0.5, 0.75, 1.0])
    dn_edges = qbin_edges(train_f["dipole_norm"], qs=[0, 0.25, 0.5, 0.75, 1.0])
    qa_edges = qbin_edges(train_f["q_abs_sum"], qs=[0, 0.25, 0.5, 0.75, 1.0])
    qd_edges = qbin_edges(train_f["q_diff"], qs=[0, 0.25, 0.5, 0.75, 1.0])

    def apply_bins(df):
        if dist_edges is not None:
            df["dist_bin"] = pd.cut(
                df["dist"], bins=dist_edges, include_lowest=True, labels=False
            )
        else:
            df["dist_bin"] = 0
        if pe_edges is not None:
            df["pe_bin"] = pd.cut(
                df["potential_energy"], bins=pe_edges, include_lowest=True, labels=False
            )
        else:
            df["pe_bin"] = 0
        if dn_edges is not None:
            df["dn_bin"] = pd.cut(
                df["dipole_norm"], bins=dn_edges, include_lowest=True, labels=False
            )
        else:
            df["dn_bin"] = 0
        if qa_edges is not None:
            df["qa_bin"] = pd.cut(
                df["q_abs_sum"], bins=qa_edges, include_lowest=True, labels=False
            )
        else:
            df["qa_bin"] = 0
        if qd_edges is not None:
            df["qd_bin"] = pd.cut(
                df["q_diff"], bins=qd_edges, include_lowest=True, labels=False
            )
        else:
            df["qd_bin"] = 0
        for c in ["dist_bin", "pe_bin", "dn_bin", "qa_bin", "qd_bin"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int16)
        return df

    train_f = apply_bins(train_f)
    test_f = apply_bins(test_f)

    global_median = float(train_f[TARGET].median())
    type_median = train_f.groupby("type")[TARGET].median()

    gcols1 = [
        "type",
        "atom_0",
        "atom_1",
        "dist_bin",
        "pe_bin",
        "dn_bin",
        "qa_bin",
        "qd_bin",
    ]
    med1 = train_f.groupby(gcols1)[TARGET].median()

    gcols2 = ["type", "atom_0", "atom_1", "dist_bin"]
    med2 = train_f.groupby(gcols2)[TARGET].median()

    gcols3 = ["type", "dist_bin"]
    med3 = train_f.groupby(gcols3)[TARGET].median()

    idx1 = pd.MultiIndex.from_frame(test_f[gcols1])
    idx2 = pd.MultiIndex.from_frame(test_f[gcols2])
    idx3 = pd.MultiIndex.from_frame(test_f[gcols3])

    p1 = med1.reindex(idx1).to_numpy()
    p2 = med2.reindex(idx2).to_numpy()
    p3 = med3.reindex(idx3).to_numpy()
    p4 = test_f["type"].map(type_median).to_numpy(dtype=float)

    final = np.where(
        ~np.isnan(p1), p1, np.where(~np.isnan(p2), p2, np.where(~np.isnan(p3), p3, p4))
    )
    final = np.where(np.isnan(final), global_median, final)

    test["final_preds"] = final.astype(float)
else:
    test["final_preds"] = test["ens_median"].fillna(fallback).values

test[["id", "type", "final_preds"]].head()



## === cell 4
submission = pd.DataFrame(
    {
        "id": test["id"].astype(int),
        TARGET: pd.to_numeric(test["final_preds"], errors="coerce"),
    }
)

if submission[TARGET].isna().any():
    fb = sample.set_index("id")[TARGET].reindex(submission["id"].values).values
    submission[TARGET] = submission[TARGET].fillna(pd.Series(fb))

submission = submission.sort_values("id").reset_index(drop=True)

out_path = "ensemble_sub.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", submission.shape)
submission.head()



## === cell 5
assert submission.columns.tolist() == ["id", TARGET]
assert submission["id"].isna().sum() == 0
assert submission[TARGET].isna().sum() == 0
assert submission.shape[0] == test.shape[0]
print(submission.describe(include="all"))



## === cell 6
submission.head(20)
