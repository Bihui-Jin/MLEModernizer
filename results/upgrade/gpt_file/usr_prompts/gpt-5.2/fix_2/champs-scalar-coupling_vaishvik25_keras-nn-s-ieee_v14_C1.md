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
numpy==1.26.4
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

-1.4419498537699864

# 6. Current score

1.99777

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.99777) has done: 'I remove the dependency on missing Kaggle “../input/top-mol” and other external kernel directories by switching to the files that actually exist in your environment (the provided CHAMPS dataset). Since no model is present here to generate predictions, I make the pipeline run end-to-end by producing a valid baseline submission from `sample_submission.csv` (all zeros), ensuring correct columns, `.csv` suffix, and row alignment. I also fix notebook-only syntax (`%matplotlib inline`) and deprecated NumPy usage (`np.bool`) so the script runs as a plain Python script. The plotting/stacking cells are kept but guarded so they don’t crash when optional blend inputs are absent.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/input/champs-scalar-coupling"

print("Using DATA_DIR:", DATA_DIR)
print("Files in DATA_DIR (first 20):", sorted(os.listdir(DATA_DIR))[:20])



## === cell 1

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

sample_sub = pd.read_csv(sample_path)
test = pd.read_csv(test_path, usecols=["id"])

sub = sample_sub.merge(test, on="id", how="right", validate="one_to_one")
sub = sub[["id", "scalar_coupling_constant"]]

sub["scalar_coupling_constant"] = pd.to_numeric(
    sub["scalar_coupling_constant"], errors="coerce"
).fillna(0.0)

out_path = "submission.csv"
sub.to_csv(out_path, index=False, float_format="%.6f")

print("Wrote", out_path, "shape=", sub.shape)
print(sub.head())



## === cell 2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

for p in ["/kaggle/input", "/kaggle/data", "../input"]:
    if os.path.exists(p):
        print(f"ls {p} ->", sorted(os.listdir(p))[:20])



## === cell 3
sub_path = "../input/top-mol"
if os.path.isdir(sub_path):
    all_files = os.listdir(sub_path)
else:
    all_files = []
print("sub_path exists:", os.path.isdir(sub_path), "| n_files:", len(all_files))



## === cell 4
if all_files:
    outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
    ncol = concat_sub.shape[1]
    print(concat_sub.head())
else:
    concat_sub = None
    ncol = 0
    print("Skipping stacking: no files found in", sub_path)



## === cell 5
if concat_sub is not None:
    _ = concat_sub.iloc[:, 1:ncol].corr()
    print(_.iloc[:5, :5])
else:
    print("Skipping correlation: concat_sub is None")



## === cell 6
if concat_sub is not None:
    corr = concat_sub.iloc[:, 1 : min(7, ncol)].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )
    plt.close(f)
    print("Heatmap created (not displayed in script mode).")
else:
    print("Skipping heatmap: concat_sub is None")



## === cell 7
if concat_sub is not None:
    concat_sub["m_max"] = concat_sub.iloc[:, 1:ncol].max(axis=1)
    concat_sub["m_min"] = concat_sub.iloc[:, 1:ncol].min(axis=1)
    concat_sub["m_mean"] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
    concat_sub["m_median"] = concat_sub.iloc[:, 1:ncol].median(axis=1)
    print(concat_sub[["m_max", "m_min", "m_mean", "m_median"]].head())
else:
    print("Skipping stacking stats: concat_sub is None")



## === cell 8
if concat_sub is not None:
    print(concat_sub.describe().T.head(15))
else:
    print("Skipping describe: concat_sub is None")



## === cell 9
cutoff_lo = 0.8
cutoff_hi = 0.2
print("cutoff_lo, cutoff_hi:", cutoff_lo, cutoff_hi)



## === cell 10
pass



## === cell 11
if concat_sub is not None and "m_mean" in concat_sub.columns:
    concat_sub["scalar_coupling_constant"] = concat_sub["m_mean"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_mean.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_mean.csv")
else:
    print("Skipping stack_mean.csv: concat_sub not available")



## === cell 12
pass



## === cell 13
if concat_sub is not None and "m_median" in concat_sub.columns:
    concat_sub["scalar_coupling_constant"] = concat_sub["m_median"]
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_median.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_median.csv")
else:
    print("Skipping stack_median.csv: concat_sub not available")



## === cell 14
pass



## === cell 15
if concat_sub is not None and ncol > 1 and "m_median" in concat_sub.columns:
    concat_sub["scalar_coupling_constant"] = np.where(
        np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
        1,
        np.where(
            np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
            0,
            concat_sub["m_median"],
        ),
    )
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_pushout_median.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_pushout_median.csv")
else:
    print("Skipping stack_pushout_median.csv: concat_sub not available")



## === cell 16
pass



## === cell 17
if concat_sub is not None and ncol > 1:
    concat_sub["scalar_coupling_constant"] = np.where(
        np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
        concat_sub["m_max"],
        np.where(
            np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
            concat_sub["m_min"],
            concat_sub["m_mean"],
        ),
    )
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_minmax_mean.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_minmax_mean.csv")
else:
    print("Skipping stack_minmax_mean.csv: concat_sub not available")



## === cell 18
pass



## === cell 19
if concat_sub is not None and ncol > 1:
    concat_sub["scalar_coupling_constant"] = np.where(
        np.all(concat_sub.iloc[:, 1:ncol] > cutoff_lo, axis=1),
        concat_sub["m_max"],
        np.where(
            np.all(concat_sub.iloc[:, 1:ncol] < cutoff_hi, axis=1),
            concat_sub["m_min"],
            concat_sub["m_median"],
        ),
    )
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_minmax_median.csv", index=False, float_format="%.6f"
    )
    print("Wrote stack_minmax_median.csv")
else:
    print("Skipping stack_minmax_median.csv: concat_sub not available")



## === cell 20
pass



## === cell 21
if concat_sub is not None and all(
    c in concat_sub.columns for c in ["mol0", "mol1", "mol2"]
):
    concat_sub["scalar_coupling_constant"] = (
        concat_sub["mol0"].rank(method="min")
        + concat_sub["mol1"].rank(method="min")
        + concat_sub["mol2"].rank(method="min")
    )
    s = concat_sub["scalar_coupling_constant"]
    concat_sub["scalar_coupling_constant"] = (s - s.min()) / (s.max() - s.min())
    concat_sub[["id", "scalar_coupling_constant"]].to_csv(
        "stack_rank.csv", index=False, float_format="%.8f"
    )
    print("Wrote stack_rank.csv")
else:
    print("Skipping stack_rank.csv: need mol0,mol1,mol2 columns")



## === cell 22
pass



## === cell 23
paths = {
    "sub1": "../input/another-one/stackers_blend.csv",
    "sub2": "../input/lgb-public-kernels-plus-more-features/sub_lgb_model_individual.csv",
    "sub3": "../input/yet-another-one/stackers_blend.csv",
    "sub4": "../input/giba-r-data-table-simple-features-1-17-lb/submission-giba-1.csv",
}
loaded = {}
for k, p in paths.items():
    if os.path.exists(p):
        loaded[k] = pd.read_csv(p)
        print("Loaded", k, "from", p, "shape=", loaded[k].shape)
    else:
        print("Missing", k, "path:", p)

temp = None
if "sub1" in loaded:
    temp = loaded["sub1"].copy()



## === cell 24
if temp is not None and "sub3" in loaded and "sub4" in loaded:
    sub3 = loaded["sub3"]
    sub4 = loaded["sub4"]
    temp["scalar_coupling_constant"] = (
        0.6 * sub3["scalar_coupling_constant"] + 0.4 * sub4["scalar_coupling_constant"]
    )
    temp.to_csv("submission4.csv", index=False)
    print("Wrote submission4.csv")
else:
    print("Skipping submission4 blend: required files not available")



## === cell 25
if temp is not None and "scalar_coupling_constant" in temp.columns:
    ax = sns.histplot(temp["scalar_coupling_constant"], bins=50, kde=True)
    fig = ax.get_figure()
    fig.savefig("temp_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved temp_distribution.png")
else:
    print("Skipping distribution plot: temp not available")
