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

3.7

# 2. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

pd.options.display.precision = 15

import gc
import warnings

warnings.filterwarnings("ignore")



## === cell 1
INPUT_BASE = "../input"
if not os.path.exists(os.path.join(INPUT_BASE, "train.csv")):
    if os.path.exists(os.path.join(INPUT_BASE, "champs-scalar-coupling", "train.csv")):
        INPUT_BASE = os.path.join(INPUT_BASE, "champs-scalar-coupling")

os.listdir(INPUT_BASE)



## === cell 2
train = pd.read_csv(os.path.join(INPUT_BASE, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_BASE, "test.csv"))
sub = pd.read_csv(os.path.join(INPUT_BASE, "sample_submission.csv"))



## === cell 3
train.head()



## === cell 4
structures = pd.read_csv(os.path.join(INPUT_BASE, "structures.csv"))


def map_atom_info(df, atom_idx):
    df = df.copy()
    df["_row_id__"] = np.arange(len(df), dtype=np.int64)

    merged = pd.merge(
        df,
        structures,
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
        sort=False,
        validate="many_to_one",  # each (molecule_name, atom_index) should map to exactly one structure row
    )

    merged = merged.drop("atom_index", axis=1)
    merged = merged.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
        }
    )

    merged = merged.sort_values("_row_id__", kind="mergesort").drop(
        columns=["_row_id__"]
    )
    return merged


train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)



## === cell 5
train_p_0 = train[["x_0", "y_0", "z_0"]].values
train_p_1 = train[["x_1", "y_1", "z_1"]].values
test_p_0 = test[["x_0", "y_0", "z_0"]].values
test_p_1 = test[["x_1", "y_1", "z_1"]].values

train["dist"] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test["dist"] = np.linalg.norm(test_p_0 - test_p_1, axis=1)



## === cell 6
type_mean_dist = train.groupby("type", sort=False)["dist"].mean()
global_mean_dist = float(train["dist"].mean())

train_div = train["type"].map(type_mean_dist).fillna(global_mean_dist)
test_div = test["type"].map(type_mean_dist).fillna(global_mean_dist)

train["dist_to_type_mean"] = train["dist"] / train_div
test["dist_to_type_mean"] = test["dist"] / test_div

for df in (train, test):
    df["dist_to_type_mean"] = df["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan)
    df["dist_to_type_mean"] = df["dist_to_type_mean"].fillna(1.0).astype(np.float64)



## === cell 7
for f in ["atom_0", "atom_1"]:
    lbl = LabelEncoder()
    lbl.fit(list(train[f].values) + list(test[f].values))
    train[f] = lbl.transform(list(train[f].values))
    test[f] = lbl.transform(list(test[f].values))

TYPE_TO_GP = {
    "1JHC": 0,
    "1JHN": 1,
    "2JHC": 2,
    "2JHH": 3,
    "2JHN": 4,
    "3JHC": 5,
    "3JHH": 6,
    "3JHN": 7,
}

train["type_code"] = train["type"].map(TYPE_TO_GP).astype(np.int16)
test["type_code"] = test["type"].map(TYPE_TO_GP).astype(np.int16)

if train["type_code"].isna().any() or test["type_code"].isna().any():
    missing_train = sorted(set(train.loc[train["type_code"].isna(), "type"].unique()))
    missing_test = sorted(set(test.loc[test["type_code"].isna(), "type"].unique()))
    raise RuntimeError(
        f"Found unknown coupling types not in TYPE_TO_GP. "
        f"missing_train={missing_train}, missing_test={missing_test}"
    )




## === cell 8
def GP0(data):
    return (
        94.962006
        + 1.0
        * (
            (
                (data["atom_index_1"])
                - (
                    (
                        (
                            (
                                (
                                    ((((1.0) <= (data["dist_to_type_mean"])) * 1.0))
                                    + ((((1.0) <= (data["dist_to_type_mean"])) * 1.0))
                                )
                            )
                            * ((4.90689849853515625))
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (((14.0) * ((((data["dist_to_type_mean"]) <= (0.995526)) * 1.0))))
                + (np.minimum(((-1.0)), ((((14.0) - (data["atom_index_0"]))))))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (data["dist_to_type_mean"])
                                                    <= (0.991079)
                                                )
                                                * 1.0
                                            )
                                        )
                                        * 2.0
                                    )
                                )
                                * 2.0
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 0.948217
        * (
            (
                (
                    (
                        -1.0
                        * (
                            (
                                (
                                    (
                                        np.minimum(
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        ((6.0))
                                                                        > (data["x_0"])
                                                                    )
                                                                    * 1.0
                                                                )
                                                            )
                                                            > (
                                                                data[
                                                                    "dist_to_type_mean"
                                                                ]
                                                            )
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                            ),
                                            ((data["dist_to_type_mean"])),
                                        )
                                    )
                                    * 2.0
                                )
                            )
                        )
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (data["dist"])
                                        * (
                                            (
                                                ((((data["dist"]) > (1.100827)) * 1.0))
                                                * 2.0
                                            )
                                        )
                                    )
                                )
                                * (data["dist"])
                            )
                        )
                        * 2.0
                    )
                )
                * (data["dist"])
            )
        )
        + 1.0
        * (
            (
                -1.0
                * (
                    (
                        (
                            ((((data["dist_to_type_mean"]) > (0.997387)) * 1.0))
                            - ((((12.0) > (data["atom_index_0"])) * 1.0))
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (((0.318310) * ((11.50162220001220703))))
                * ((((0.997117) > (data["dist_to_type_mean"])) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        ((((((data["dist_to_type_mean"]) <= (0.986908)) * 1.0)) * 2.0))
                        * 2.0
                    )
                )
                - (
                    (
                        (
                            ((((data["dist_to_type_mean"]) > (0.986908)) * 1.0))
                            > (data["dist_to_type_mean"])
                        )
                        * 1.0
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (0.984850)
                * (
                    (
                        ((8.0))
                        * (
                            (
                                (0.984850)
                                * (
                                    (
                                        ((8.0))
                                        * (
                                            (
                                                (
                                                    (0.984850)
                                                    > (data["dist_to_type_mean"])
                                                )
                                                * 1.0
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (data["dist_to_type_mean"])
                            > (
                                np.maximum(
                                    ((1.0)), ((((data["atom_index_1"]) - (2.152171))))
                                )
                            )
                        )
                        * 1.0
                    )
                )
                - ((((1.013243) > (data["dist_to_type_mean"])) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        ((((1.013243) <= (data["dist_to_type_mean"])) * 1.0))
                        * ((10.21938037872314453))
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            ((-1.0 * (((((((data["y_0"]) > (-1.0)) * 1.0)) / 2.0)))))
                            + ((((data["atom_index_0"]) <= (12.0)) * 1.0))
                        )
                        / 2.0
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (-0.072825)
                + (
                    (
                        (-0.072825)
                        + (
                            (
                                (-0.072825)
                                + (
                                    (
                                        (
                                            (0.998071)
                                            > (
                                                np.maximum(
                                                    ((data["dist_to_type_mean"])),
                                                    ((data["y_1"])),
                                                )
                                            )
                                        )
                                        * 1.0
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        + 0.922325
        * (
            (
                (
                    ((5.0))
                    > (
                        (
                            (
                                (
                                    np.minimum(
                                        (((((data["y_0"]) + (data["z_1"])) / 2.0))),
                                        ((data["y_0"])),
                                    )
                                )
                                + (data["atom_index_0"])
                            )
                            / 2.0
                        )
                    )
                )
                * 1.0
            )
        )
        + 0.977528
        * (
            (
                (
                    (
                        (-0.543418)
                        * (
                            (
                                (
                                    (data["x_0"])
                                    <= ((((data["x_1"]) + (-0.283168)) / 2.0))
                                )
                                * 1.0
                            )
                        )
                    )
                )
                * ((((-2.0) <= (data["x_1"])) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    ((-1.0 * ((((((1.0)) > (data["dist_to_type_mean"])) * 1.0)))))
                    + (((((6.0)) > (((((data["y_1"]) * 2.0)) * 2.0))) * 1.0))
                )
                / 2.0
            )
        )
    )


def GP1(data):
    return (
        47.511509
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        ((-1.0 * ((data["dist"]))))
                                        + (
                                            (
                                                (
                                                    (
                                                        (
                                                            (data["dist_to_type_mean"])
                                                            <= (0.998991)
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                                * 2.0
                                            )
                                        )
                                    )
                                )
                                * 2.0
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (3.141593)
                * (
                    (
                        (1.003744)
                        * (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (1.003744)
                                                    > (data["dist_to_type_mean"])
                                                )
                                                * 1.0
                                            )
                                        )
                                        * 2.0
                                    )
                                )
                                - (1.387789)
                            )
                        )
                    )
                )
            )
        )
        + 0.979971
        * (
            np.minimum(
                ((((4.740009) - (data["dist"])))),
                (
                    (
                        (
                            (1.570796)
                            * (
                                (
                                    (data["atom_index_1"])
                                    * (
                                        (
                                            ((0.994081) > (data["dist_to_type_mean"]))
                                            * 1.0
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
            )
        )
        + 0.950660
        * (
            (
                (0.076053)
                - (
                    (
                        ((((((((data["atom_index_0"]) + (-2.0)) / 2.0)) / 2.0)) / 2.0))
                        * ((((data["dist_to_type_mean"]) <= (0.998991)) * 1.0))
                    )
                )
            )
        )
        + 1.0
        * (
            np.minimum(
                (
                    (
                        (
                            ((((0.994834) > (data["dist_to_type_mean"])) * 1.0))
                            * (((0.636620) + (0.994834)))
                        )
                    )
                ),
                ((np.maximum(((data["atom_index_1"])), ((0.636620))))),
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            -1.0
                            * (
                                (
                                    (
                                        (
                                            (
                                                np.maximum(
                                                    ((data["z_1"])),
                                                    (((-1.0 * ((data["z_1"]))))),
                                                )
                                            )
                                            > (0.110836)
                                        )
                                        * 1.0
                                    )
                                )
                            )
                        )
                    )
                    + ((((data["atom_index_1"]) > (7.0)) * 1.0))
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (data["atom_index_1"])
                                                    + (
                                                        (
                                                            (data["dist_to_type_mean"])
                                                            * 2.0
                                                        )
                                                    )
                                                )
                                            )
                                            > (data["atom_index_0"])
                                        )
                                        * 1.0
                                    )
                                )
                                * 2.0
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (data["atom_index_0"])
                    <= (
                        (
                            (
                                np.maximum(
                                    ((3.141593)),
                                    (
                                        (
                                            (
                                                (data["atom_index_1"])
                                                + (
                                                    (
                                                        (
                                                            (3.141593)
                                                            > (
                                                                (
                                                                    (
                                                                        data[
                                                                            "atom_index_1"
                                                                        ]
                                                                    )
                                                                    / 2.0
                                                                )
                                                            )
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                            )
                                        )
                                    ),
                                )
                            )
                            * 2.0
                        )
                    )
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (
                    np.minimum(
                        (
                            (
                                (
                                    (
                                        (
                                            -1.0
                                            * (
                                                (
                                                    (
                                                        (
                                                            (((data["y_0"]) / 2.0))
                                                            + (data["z_1"])
                                                        )
                                                        / 2.0
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        ),
                        ((((3.0) + (data["y_0"])))),
                    )
                )
                / 2.0
            )
        )
        + 0.916952
        * (
            np.minimum(
                (((((1.0) > (data["z_0"])) * 1.0))),
                (
                    (
                        (
                            (
                                ((((0.0) + (data["z_0"])) / 2.0))
                                + ((((data["dist_to_type_mean"]) > (1.0)) * 1.0))
                            )
                            / 2.0
                        )
                    )
                ),
            )
        )
        + 1.0
        * (
            (
                ((((((1.021625) > (data["dist"])) * 1.0)) - (data["dist"])))
                * ((((1.021625) + (((1.021625) * 2.0))) / 2.0))
            )
        )
        + 1.0
        * (
            (
                ((data["dist_to_type_mean"]) <= ((((data["dist"]) > (1.011806)) * 1.0)))
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (
                    (0.998991)
                    > (
                        (
                            (
                                (
                                    (
                                        (
                                            (0.998991)
                                            + (
                                                (
                                                    (0.998991)
                                                    * (data["dist_to_type_mean"])
                                                )
                                            )
                                        )
                                        / 2.0
                                    )
                                )
                                > ((((data["dist_to_type_mean"]) > (0.998991)) * 1.0))
                            )
                            * 1.0
                        )
                    )
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        np.minimum(
                            ((data["dist_to_type_mean"])),
                            ((((((data["atom_index_0"]) / 2.0)) / 2.0))),
                        )
                    )
                    <= ((((1.011806) <= (data["dist"])) * 1.0))
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                ((-1.0 * (((((0.318310) > (data["z_1"])) * 1.0)))))
                * (
                    (
                        (
                            ((((1.011806) > (data["dist"])) * 1.0))
                            > (data["atom_index_1"])
                        )
                        * 1.0
                    )
                )
            )
        )
        + 0.922814
        * (
            np.minimum(
                (((((1.008356) > (data["dist"])) * 1.0))),
                (((((((1.199319) * (data["z_1"]))) <= (((0.072832) * 2.0))) * 1.0))),
            )
        )
    )


def GP2(data):
    return (
        -0.268229
        + 0.937958
        * (
            (
                -1.0
                * (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (data["atom_index_1"])
                                                            > (
                                                                (
                                                                    (
                                                                        (data["y_0"])
                                                                        <= (
                                                                            (
                                                                                (
                                                                                    1.194078
                                                                                )
                                                                                / 2.0
                                                                            )
                                                                        )
                                                                    )
                                                                    * 1.0
                                                                )
                                                            )
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                                * 2.0
                                            )
                                        )
                                        > (data["atom_index_1"])
                                    )
                                    * 1.0
                                )
                            )
                            * 2.0
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            ((0.994044) <= (data["dist_to_type_mean"]))
                                            * 1.0
                                        )
                                    )
                                    * 2.0
                                )
                            )
                            - (2.0)
                        )
                    )
                    + ((((data["dist_to_type_mean"]) > (1.0)) * 1.0))
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                ((((1.032724) <= (data["dist_to_type_mean"])) * 1.0))
                - (
                    np.minimum(
                        (
                            (
                                (
                                    (
                                        (data["dist_to_type_mean"])
                                        > (
                                            (
                                                (
                                                    ((12.80295181274414062))
                                                    > (data["atom_index_0"])
                                                )
                                                * 1.0
                                            )
                                        )
                                    )
                                    * 1.0
                                )
                            )
                        ),
                        ((0.166034)),
                    )
                )
            )
        )
        + 0.869565
        * (
            np.minimum(
                ((data["dist_to_type_mean"])),
                (
                    (
                        (
                            (data["atom_index_1"])
                            * (
                                (
                                    (0.950018)
                                    - (
                                        (
                                            ((0.950018) <= (data["dist_to_type_mean"]))
                                            * 1.0
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
            )
        )
        + 0.926234
        * (
            np.minimum(
                ((((((1.570796) / 2.0)) / 2.0))),
                (
                    (
                        (
                            ((((data["y_0"]) > (data["atom_index_1"])) * 1.0))
                            * (data["atom_index_1"])
                        )
                    )
                ),
            )
        )
        + 1.0 * (((((((data["dist_to_type_mean"]) - (1.0))) * 2.0)) * 2.0))
        + 0.790914
        * (
            np.minimum(
                ((data["atom_index_1"])),
                (
                    (
                        (
                            (
                                (data["atom_index_1"])
                                <= (
                                    np.minimum(
                                        (
                                            (
                                                (
                                                    (
                                                        (data["y_0"])
                                                        > (
                                                            (
                                                                (((data["z_0"]) / 2.0))
                                                                / 2.0
                                                            )
                                                        )
                                                    )
                                                    * 1.0
                                                )
                                            )
                                        ),
                                        ((data["x_1"])),
                                    )
                                )
                            )
                            * 1.0
                        )
                    )
                ),
            )
        )
        + 0.850024
        * (
            (
                (data["x_0"])
                * (
                    (
                        (
                            (
                                -1.0
                                * (
                                    (
                                        (
                                            (
                                                (0.166034)
                                                <= (
                                                    (
                                                        (
                                                            (data["atom_index_1"])
                                                            <= (((0.166034) / 2.0))
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                            )
                                            * 1.0
                                        )
                                    )
                                )
                            )
                        )
                        / 2.0
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (3.0)
                * (
                    (
                        (0.982967)
                        - (
                            (
                                ((data["atom_index_0"]) > ((((7.0)) - (data["y_0"]))))
                                * 1.0
                            )
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        -1.0
                        * (
                            (
                                (
                                    (
                                        (((data["dist_to_type_mean"]) * (6.0)))
                                        <= (
                                            (
                                                (
                                                    ((10.40789985656738281))
                                                    + (
                                                        (
                                                            (
                                                                (6.0)
                                                                > (data["atom_index_0"])
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                )
                                                / 2.0
                                            )
                                        )
                                    )
                                    * 1.0
                                )
                            )
                        )
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (data["y_0"])
                * (
                    (
                        (
                            (0.950018)
                            + (
                                (
                                    (
                                        np.minimum(
                                            ((((-2.0) - (data["dist_to_type_mean"])))),
                                            ((0.832110)),
                                        )
                                    )
                                    + (2.080729)
                                )
                            )
                        )
                        / 2.0
                    )
                )
            )
        )
        + 0.999023
        * (
            (
                (
                    (
                        (
                            (data["x_0"])
                            > (
                                (
                                    (
                                        (
                                            (
                                                (data["atom_index_0"])
                                                + (
                                                    (
                                                        (
                                                            (data["y_0"])
                                                            > (
                                                                (
                                                                    (
                                                                        (data["x_0"])
                                                                        + (data["z_0"])
                                                                    )
                                                                    / 2.0
                                                                )
                                                            )
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                            )
                                            / 2.0
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            np.minimum(
                (((((data["dist"]) <= (2.080729)) * 1.0))),
                (
                    (
                        (
                            (
                                (data["dist"])
                                > (
                                    (
                                        (data["atom_index_1"])
                                        - (
                                            (
                                                ((2.080729) <= (data["atom_index_1"]))
                                                * 1.0
                                            )
                                        )
                                    )
                                )
                            )
                            * 1.0
                        )
                    )
                ),
            )
        )
        + 1.0
        * (
            (
                -1.0
                * (
                    (
                        (
                            ((((data["dist_to_type_mean"]) <= (0.982494)) * 1.0))
                            * (((data["dist"]) * (0.034596)))
                        )
                    )
                )
            )
        )
        + 0.929653
        * (
            (
                ((-1.0 * (((((((1.0) * 2.0)) <= (data["dist"])) * 1.0)))))
                * ((((((0.982494) > (data["dist_to_type_mean"])) * 1.0)) / 2.0))
            )
        )
        + 0.999511
        * (
            (
                (
                    (
                        (
                            ((((data["dist"]) <= (2.080729)) * 1.0))
                            + (
                                (
                                    (
                                        (
                                            (
                                                ((((data["z_0"]) <= (0.204195)) * 1.0))
                                                + (0.204195)
                                            )
                                            / 2.0
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
    )


def GP3(data):
    return (
        -10.275834
        + 1.0
        * (
            (
                -1.0
                * (
                    (
                        (
                            (
                                np.minimum(
                                    ((0.636620)),
                                    (
                                        (
                                            (
                                                (
                                                    (data["dist_to_type_mean"])
                                                    <= (
                                                        (
                                                            (
                                                                ((4.59200382232666016))
                                                                <= (
                                                                    data["atom_index_0"]
                                                                )
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                )
                                                * 1.0
                                            )
                                        )
                                    ),
                                )
                            )
                            * 2.0
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    np.minimum(
                                        (
                                            (
                                                (
                                                    (
                                                        (data["dist_to_type_mean"])
                                                        > (1.004117)
                                                    )
                                                    * 1.0
                                                )
                                            )
                                        ),
                                        ((data["dist_to_type_mean"])),
                                    )
                                )
                                * (data["dist_to_type_mean"])
                            )
                        )
                        * (2.0)
                    )
                )
                * (data["dist"])
            )
        )
        + 0.974597
        * (
            (
                (
                    (
                        (
                            (
                                (((data["dist_to_type_mean"]) * 2.0))
                                * ((((1.796855) <= (data["dist"])) * 1.0))
                            )
                        )
                        * 2.0
                    )
                )
                - (((((1.570796) / 2.0)) / 2.0))
            )
        )
        + 0.787005
        * (
            (
                (
                    (
                        (
                            (
                                (data["dist_to_type_mean"])
                                <= ((((1.765520) <= (data["dist"])) * 1.0))
                            )
                            * 1.0
                        )
                    )
                    + ((((((1.765520) <= (data["dist"])) * 1.0)) - (data["dist"])))
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                ((((-0.443103) <= (data["y_1"])) * 1.0))
                                                > (data["dist_to_type_mean"])
                                            )
                                            * 1.0
                                        )
                                    )
                                    / 2.0
                                )
                            )
                            + ((((1.200606) > (data["y_1"])) * 1.0))
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 0.994138
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            np.minimum(
                                                ((data["dist_to_type_mean"])),
                                                (
                                                    (
                                                        (
                                                            (
                                                                (data["dist"])
                                                                <= (1.719267)
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                ),
                                            )
                                        )
                                        * (data["dist"])
                                    )
                                )
                                * 2.0
                            )
                        )
                        * 2.0
                    )
                )
                * (data["dist_to_type_mean"])
            )
        )
        + 0.788471
        * (
            (
                (((data["dist_to_type_mean"]) - (1.988152)))
                + (
                    np.maximum(
                        (((((data["y_0"]) > (3.0)) * 1.0))),
                        (((((1.004117) > (data["dist_to_type_mean"])) * 1.0))),
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (data["dist"])
                    > (
                        (
                            (1.570796)
                            - (
                                (
                                    (
                                        (
                                            -1.0
                                            * (
                                                (
                                                    (
                                                        (((3.0) / 2.0))
                                                        * (((0.170694) * 2.0))
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        )
                    )
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            np.minimum(
                                ((0.318310)),
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (data["dist_to_type_mean"])
                                                            - ((1.0))
                                                        )
                                                    )
                                                    * 2.0
                                                )
                                            )
                                            * 2.0
                                        )
                                    )
                                ),
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 0.804592
        * (
            (
                (
                    (
                        (
                            (
                                (data["y_0"])
                                - (
                                    (
                                        (data["dist_to_type_mean"])
                                        * (((1.004117) * (data["y_0"])))
                                    )
                                )
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 0.981925
        * (
            (
                (0.147920)
                * (
                    (
                        (
                            ((((((data["y_0"]) / 2.0)) > (0.147920)) * 1.0))
                            <= ((((((data["y_0"]) / 2.0)) > (1.004117)) * 1.0))
                        )
                        * 1.0
                    )
                )
            )
        )
        + 0.890572
        * (
            (
                (
                    (
                        (
                            (
                                -1.0
                                * (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (data["y_1"])
                                                                * (data["y_1"])
                                                            )
                                                        )
                                                        > (1.570796)
                                                    )
                                                    * 1.0
                                                )
                                            )
                                            / 2.0
                                        )
                                    )
                                )
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    ((-1.0 * ((1.004117))))
                    > (
                        (
                            -1.0
                            * (
                                (
                                    (
                                        (data["dist_to_type_mean"])
                                        - (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (1.947232)
                                                                <= (
                                                                    data[
                                                                        "dist_to_type_mean"
                                                                    ]
                                                                )
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                    + (0.054476)
                                                )
                                                / 2.0
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                * 1.0
            )
        )
        + 0.894480
        * (
            (
                -1.0
                * (
                    (
                        np.minimum(
                            ((0.636620)),
                            (((((data["dist_to_type_mean"]) > (1.004117)) * 1.0))),
                        )
                    )
                )
            )
        )
        + 0.870542
        * (
            (
                (
                    (
                        (
                            (
                                (((((data["dist"]) - (1.770059))) * (data["dist"])))
                                * (data["dist"])
                            )
                        )
                        * (data["dist"])
                    )
                )
                * (data["dist"])
            )
        )
        + 1.0
        * (
            np.minimum(
                (((((11.0) <= (data["atom_index_0"])) * 1.0))),
                ((((0.318310) * ((((((data["z_1"]) / 2.0)) <= (data["y_1"])) * 1.0))))),
            )
        )
    )


def GP4(data):
    return (
        3.128881
        + 1.0
        * (
            (
                ((((2.264586) > (data["dist"])) * 1.0))
                * (
                    (
                        ((-1.0 * ((data["dist"]))))
                        + (((((-1.0 * ((1.207241)))) + (data["atom_index_1"])) / 2.0))
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        np.minimum(
                            ((((data["y_0"]) + (data["atom_index_1"])))), ((0.318310))
                        )
                    )
                    <= (
                        (
                            ((data["atom_index_0"]) <= (((data["atom_index_1"]) * 2.0)))
                            * 1.0
                        )
                    )
                )
                * 1.0
            )
        )
        + 0.949194
        * (
            (
                (
                    (
                        ((((3.141593) <= (data["x_0"])) * 1.0))
                        - ((((-0.441697) <= (data["y_0"])) * 1.0))
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            np.maximum(
                (((((data["atom_index_1"]) <= (data["y_1"])) * 1.0))),
                (
                    (
                        np.minimum(
                            (((((-2.0) > (data["z_0"])) * 1.0))),
                            (
                                (
                                    (
                                        (
                                            (data["atom_index_1"])
                                            <= ((5.21898651123046875))
                                        )
                                        * 1.0
                                    )
                                )
                            ),
                        )
                    )
                ),
            )
        )
        + 1.0
        * (
            (
                -1.0
                * (
                    (
                        (
                            (
                                ((((data["dist_to_type_mean"]) <= (1.022687)) * 1.0))
                                <= (
                                    (
                                        -1.0
                                        * (
                                            (
                                                (
                                                    (
                                                        (data["atom_index_1"])
                                                        <= (data["dist_to_type_mean"])
                                                    )
                                                    * 1.0
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            * 1.0
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (((data["dist_to_type_mean"]) / 2.0))
                                    <= ((((0.24176722764968872)) * 2.0))
                                )
                                * 1.0
                            )
                        )
                        * (((data["dist_to_type_mean"]) * 2.0))
                    )
                )
                * (((data["dist_to_type_mean"]) * 2.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            np.minimum(
                                (((((data["dist"]) > (2.264586)) * 1.0))),
                                (
                                    (
                                        (
                                            (
                                                (data["dist_to_type_mean"])
                                                <= (data["atom_index_1"])
                                            )
                                            * 1.0
                                        )
                                    )
                                ),
                            )
                        )
                        * (data["dist_to_type_mean"])
                    )
                )
                * (data["dist_to_type_mean"])
            )
        )
        + 1.0
        * (
            np.minimum(
                (
                    (
                        np.maximum(
                            ((data["y_1"])),
                            (
                                (
                                    (
                                        (
                                            (
                                                -1.0
                                                * (
                                                    (
                                                        (
                                                            (
                                                                (data["atom_index_0"])
                                                                > (((14.0) - (-1.0)))
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                        / 2.0
                                    )
                                )
                            ),
                        )
                    )
                ),
                ((0.023229)),
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (((data["z_0"]) - (1.866194)))
                            > (
                                (
                                    (-2.0)
                                    * ((((((data["z_0"]) * 2.0)) <= (0.318310)) * 1.0))
                                )
                            )
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        np.maximum(
                            ((data["x_1"])),
                            (
                                (
                                    (
                                        ((((1.034271) > (data["x_1"])) * 1.0))
                                        - (data["z_0"])
                                    )
                                )
                            ),
                        )
                    )
                    > (np.maximum(((data["atom_index_1"])), ((data["dist"]))))
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                ((((((-1.0 * (((((1.929792) > (data["dist"])) * 1.0))))) * 2.0)) * 2.0))
                - ((((data["dist"]) <= (1.929792)) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (data["x_0"])
                    <= (
                        (
                            (
                                (
                                    -1.0
                                    * (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (data["atom_index_0"])
                                                                + ((-1.0 * ((2.0))))
                                                            )
                                                            / 2.0
                                                        )
                                                    )
                                                    + ((-1.0 * ((2.0))))
                                                )
                                                / 2.0
                                            )
                                        )
                                    )
                                )
                            )
                            * 2.0
                        )
                    )
                )
                * 1.0
            )
        )
        + 0.929653
        * (
            (
                (
                    (
                        (0.523343)
                        - (
                            (
                                (
                                    (data["dist_to_type_mean"])
                                    > ((((1.0) + ((((1.0)) - (0.029367)))) / 2.0))
                                )
                                * 1.0
                            )
                        )
                    )
                )
                / 2.0
            )
        )
        + 0.971177
        * (
            (
                (
                    (
                        (
                            (
                                np.minimum(
                                    ((data["x_0"])),
                                    (
                                        (
                                            (
                                                (((3.594635) - (data["atom_index_1"])))
                                                + (3.594635)
                                            )
                                        )
                                    ),
                                )
                            )
                            > (1.138006)
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                -1.0
                                * (
                                    (
                                        (
                                            (
                                                (0.318310)
                                                > (
                                                    (
                                                        (
                                                            (
                                                                (0.318310)
                                                                * (
                                                                    (
                                                                        (
                                                                            (
                                                                                data[
                                                                                    "dist_to_type_mean"
                                                                                ]
                                                                            )
                                                                            + (0.0)
                                                                        )
                                                                        / 2.0
                                                                    )
                                                                )
                                                            )
                                                        )
                                                        * 2.0
                                                    )
                                                )
                                            )
                                            * 1.0
                                        )
                                    )
                                )
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (data["x_0"])
                            <= (
                                (
                                    (
                                        np.minimum(
                                            ((data["x_1"])),
                                            (
                                                (
                                                    np.minimum(
                                                        ((((data["y_0"]) / 2.0))),
                                                        ((0.318310)),
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                    - (data["dist_to_type_mean"])
                                )
                            )
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
    )


def GP5(data):
    return (
        3.690675
        + 0.868100
        * (
            (
                (data["dist"])
                * (
                    (
                        (
                            (
                                (
                                    ((((3.142503) > (data["dist"])) * 1.0))
                                    <= ((((data["dist"]) > (3.141593)) * 1.0))
                                )
                                * 1.0
                            )
                        )
                        - (0.636620)
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (1.0)
                            <= (
                                np.maximum(
                                    (((-1.0 * ((data["y_0"]))))),
                                    (((((-0.206448) <= (data["y_0"])) * 1.0))),
                                )
                            )
                        )
                        * 1.0
                    )
                )
                * (((0.148571) / 2.0))
            )
        )
        + 0.885686 * ((((2.719738) > (data["dist"])) * 1.0))
        + 0.888129
        * (
            (
                (
                    (
                        (np.maximum(((data["dist_to_type_mean"])), ((0.826032))))
                        - ((((data["dist_to_type_mean"]) > (0.826032)) * 1.0))
                    )
                )
                * 2.0
            )
        )
        + 0.887640
        * (
            (
                (data["dist_to_type_mean"])
                * (
                    (
                        -1.0
                        * (
                            (
                                (
                                    (
                                        (
                                            (
                                                ((data["y_1"]) > (((-0.228787) / 2.0)))
                                                * 1.0
                                            )
                                        )
                                        > ((((data["dist"]) <= (3.141593)) * 1.0))
                                    )
                                    * 1.0
                                )
                            )
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            np.minimum(
                (((((data["dist_to_type_mean"]) > (1.090505)) * 1.0))),
                (
                    (
                        np.minimum(
                            (((((data["z_0"]) <= (data["dist_to_type_mean"])) * 1.0))),
                            (((((data["x_0"]) <= (data["dist_to_type_mean"])) * 1.0))),
                        )
                    )
                ),
            )
        )
        + 1.0
        * (
            (
                (
                    ((((-0.181841) + (((-0.228787) / 2.0))) / 2.0))
                    + (
                        (
                            (
                                (0.927964)
                                > (
                                    np.maximum(
                                        ((data["atom_index_1"])),
                                        ((data["dist_to_type_mean"])),
                                    )
                                )
                            )
                            * 1.0
                        )
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (((data["y_1"]) * 2.0))
                                                    > (data["y_1"])
                                                )
                                                * 1.0
                                            )
                                        )
                                        + (((1.570796) / 2.0))
                                    )
                                    / 2.0
                                )
                            )
                            > (data["dist_to_type_mean"])
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (0.018557)
                        * (
                            (
                                (
                                    np.maximum(
                                        ((data["x_0"])),
                                        ((((data["y_0"]) * (data["dist"])))),
                                    )
                                )
                                - (((1.570796) * 2.0))
                            )
                        )
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            np.minimum(
                (
                    (
                        (
                            (
                                (
                                    ((1.570796) > (((data["dist_to_type_mean"]) * 2.0)))
                                    * 1.0
                                )
                            )
                            * 2.0
                        )
                    )
                ),
                ((1.570796)),
            )
        )
        + 0.999511
        * (
            (
                (
                    (
                        (
                            np.minimum(
                                ((((((data["atom_index_1"]) + (-0.267775))) + (-1.0)))),
                                (((((1.570796) <= (data["z_0"])) * 1.0))),
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            np.minimum(
                (((((((0.00275373528711498)) * 2.0)) + ((0.00275373528711498))))),
                ((data["atom_index_1"])),
            )
        )
        + 1.0
        * (
            (
                (
                    ((((((3.647356) > (data["dist"])) * 1.0)) * (data["dist"])))
                    + ((-1.0 * ((data["dist"]))))
                )
                / 2.0
            )
        )
        + 0.999023
        * (
            np.maximum(
                (((((0.0) + ((((data["dist"]) + (-3.0)) / 2.0))) / 2.0))), ((0.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (((0.0) / 2.0))
                                        - (
                                            (
                                                (
                                                    (data["x_0"])
                                                    > (((data["x_0"]) * (data["x_0"])))
                                                )
                                                * 1.0
                                            )
                                        )
                                    )
                                )
                                / 2.0
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            np.maximum(
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (data["dist_to_type_mean"])
                                                        <= (0.913097)
                                                    )
                                                    * 1.0
                                                )
                                            )
                                            / 2.0
                                        )
                                    )
                                    / 2.0
                                )
                            )
                            / 2.0
                        )
                    )
                ),
                ((((0.913097) - (data["dist_to_type_mean"])))),
            )
        )
    )


def GP6(data):
    return (
        4.772715
        + 1.0
        * (
            (
                (
                    (
                        (0.636620)
                        - (
                            (
                                (
                                    (data["dist"])
                                    <= (np.maximum(((3.0)), ((data["y_0"]))))
                                )
                                * 1.0
                            )
                        )
                    )
                )
                - ((((data["dist"]) <= (3.0)) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    ((((-3.0) + (data["dist"])) / 2.0))
                                    > ((((data["y_1"]) > (((0.218830) * 2.0))) * 1.0))
                                )
                                * 1.0
                            )
                        )
                        * 2.0
                    )
                )
                * 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        ((data["dist_to_type_mean"]) <= ((((1.0) + (0.922659)) / 2.0)))
                        * 1.0
                    )
                )
                - ((((2.492783) <= (data["dist"])) * 1.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        ((((data["dist"]) > (3.0)) * 1.0))
                        - ((((data["dist"]) > (3.102541)) * 1.0))
                    )
                )
                - ((((data["dist"]) > (3.102541)) * 1.0))
            )
        )
        + 0.962384
        * (
            (
                (2.453189)
                - (
                    (
                        (data["dist"])
                        + ((-1.0 * (((((data["dist"]) <= (2.453189)) * 1.0)))))
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (1.129494)
                * (
                    (
                        (1.129494)
                        * (
                            (
                                (1.129494)
                                * (
                                    (
                                        (
                                            (np.maximum(((3.051597)), ((data["z_1"]))))
                                            <= (data["dist"])
                                        )
                                        * 1.0
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        + 0.882755
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            np.minimum(
                                                ((0.636620)), ((((data["y_1"]) / 2.0)))
                                            )
                                        )
                                        - ((((data["y_1"]) <= (0.636620)) * 1.0))
                                    )
                                )
                                / 2.0
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                ((((1.146451) > (data["dist_to_type_mean"])) * 1.0))
                - (data["dist_to_type_mean"])
            )
        )
        + 1.0
        * (
            (
                (
                    (-2.0)
                    > (
                        (
                            -1.0
                            * (
                                (
                                    (
                                        (data["dist_to_type_mean"])
                                        * (
                                            (
                                                (data["dist_to_type_mean"])
                                                * (
                                                    (
                                                        (data["dist_to_type_mean"])
                                                        * (data["y_0"])
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (((((data["y_0"]) - ((((data["y_0"]) <= ((1.0))) * 1.0)))) * 2.0))
                * (((data["dist_to_type_mean"]) - ((1.0))))
            )
        )
        + 1.0
        * (
            (
                ((((((3.090264) <= (data["dist"])) * 1.0)) * 2.0))
                * ((-1.0 * ((data["dist_to_type_mean"]))))
            )
        )
        + 1.0
        * (
            (
                (((2.268008) / 2.0))
                * ((((((data["dist_to_type_mean"]) > (1.132620)) * 1.0)) / 2.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (-0.278111)
                    + (
                        (
                            (
                                (
                                    (
                                        (
                                            (-1.0)
                                            + (
                                                (
                                                    (
                                                        (data["dist_to_type_mean"])
                                                        <= (0.941507)
                                                    )
                                                    * 1.0
                                                )
                                            )
                                        )
                                        / 2.0
                                    )
                                )
                                + ((((data["dist_to_type_mean"]) <= (0.941507)) * 1.0))
                            )
                            / 2.0
                        )
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    np.minimum(
                        ((((((-0.047501) * (data["y_0"]))) - (-0.047501)))),
                        ((-0.047501)),
                    )
                )
                * (((data["y_0"]) / 2.0))
            )
        )
        + 1.0
        * (
            (
                (
                    (-1.0)
                    + (
                        (
                            (
                                (0.922659)
                                <= (
                                    np.maximum(
                                        ((data["x_1"])),
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            np.maximum(
                                                                ((data["z_1"])),
                                                                (
                                                                    (
                                                                        data[
                                                                            "dist_to_type_mean"
                                                                        ]
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                        + (data["dist_to_type_mean"])
                                                    )
                                                    / 2.0
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                            * 1.0
                        )
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (3.046956)
                                    <= (np.maximum(((data["dist"])), ((data["z_0"]))))
                                )
                                * 1.0
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
    )


def GP7(data):
    return (
        0.990498
        + 0.938935
        * (
            np.minimum(
                (
                    (
                        (
                            (0.318310)
                            - (
                                (
                                    (data["dist_to_type_mean"])
                                    * (
                                        (
                                            ((data["dist_to_type_mean"]) <= (1.015225))
                                            * 1.0
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                ((((data["dist"]) - (3.141593)))),
            )
        )
        + 0.933073
        * (
            (
                -1.0
                * (
                    (
                        np.maximum(
                            (
                                (
                                    (
                                        (((((data["y_1"]) * 2.0)) * (0.011385)))
                                        * (data["dist"])
                                    )
                                )
                            ),
                            (((((data["dist"]) <= ((2.32503938674926758))) * 1.0))),
                        )
                    )
                )
            )
        )
        + 0.932096
        * (
            np.minimum(
                ((data["atom_index_1"])),
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                ((-1.0 * ((data["dist_to_type_mean"]))))
                                                + (
                                                    np.maximum(
                                                        ((0.983454)),
                                                        ((data["dist_to_type_mean"])),
                                                    )
                                                )
                                            )
                                            / 2.0
                                        )
                                    )
                                    * 2.0
                                )
                            )
                            * 2.0
                        )
                    )
                ),
            )
        )
        + 0.982413
        * (
            (
                (
                    np.minimum(
                        ((((-0.059342) / 2.0))),
                        ((((((((-0.059342) / 2.0)) / 2.0)) * (data["atom_index_1"])))),
                    )
                )
                * (data["dist_to_type_mean"])
            )
        )
        + 1.0
        * (
            (
                -1.0
                * (
                    (
                        (
                            (-0.108994)
                            * (
                                (
                                    (
                                        np.maximum(
                                            ((data["y_0"])),
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                ((-1.0 * ((3.141593))))
                                                                / 2.0
                                                            )
                                                        )
                                                        / 2.0
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        )
                    )
                )
            )
        )
        + 0.980948
        * (
            (
                (((((1.0)) > (data["atom_index_1"])) * 1.0))
                * (((data["dist"]) * (((1.055259) * (((-0.110218) / 2.0))))))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            ((0.04956008121371269))
                            > (((1.055259) * (((data["x_0"]) * (data["x_1"])))))
                        )
                        * 1.0
                    )
                )
                * ((0.04956008121371269))
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (data["atom_index_0"])
                                                <= ((5.20004987716674805))
                                            )
                                            * 1.0
                                        )
                                    )
                                    + (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (data["atom_index_1"])
                                                            <= (data["dist"])
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                                / 2.0
                                            )
                                        )
                                        / 2.0
                                    )
                                )
                                / 2.0
                            )
                        )
                        + (-0.059342)
                    )
                    / 2.0
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (data["dist_to_type_mean"])
                    <= (
                        (
                            (-0.210597)
                            + (
                                (
                                    (
                                        (data["y_1"])
                                        > (
                                            (
                                                (
                                                    (
                                                        (
                                                            (data["y_1"])
                                                            > (
                                                                (
                                                                    (
                                                                        data[
                                                                            "dist_to_type_mean"
                                                                        ]
                                                                    )
                                                                    / 2.0
                                                                )
                                                            )
                                                        )
                                                        * 1.0
                                                    )
                                                )
                                                * 2.0
                                            )
                                        )
                                    )
                                    * 1.0
                                )
                            )
                        )
                    )
                )
                * 1.0
            )
        )
        + 0.917929
        * (
            (
                -1.0
                * (
                    (
                        (
                            (
                                (
                                    (
                                        (data["dist_to_type_mean"])
                                        <= (((1.570796) / 2.0))
                                    )
                                    * 1.0
                                )
                            )
                            / 2.0
                        )
                    )
                )
            )
        )
        + 1.0
        * (
            (
                (
                    (data["dist"])
                    > (
                        np.maximum(
                            ((3.0)),
                            (
                                (
                                    (
                                        (
                                            (data["atom_index_0"])
                                            + (((-1.0) - ((-1.0 * ((data["x_1"]))))))
                                        )
                                        / 2.0
                                    )
                                )
                            ),
                        )
                    )
                )
                * 1.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                ((((1.177861) + ((-1.0 * ((data["y_0"])))))) <= (-1.0))
                                * 1.0
                            )
                        )
                        * (-0.108994)
                    )
                )
                * (data["dist_to_type_mean"])
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        -1.0
                        * (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    np.maximum(
                                                        ((data["y_0"])),
                                                        ((((data["dist"]) * 2.0))),
                                                    )
                                                )
                                                > (7.0)
                                            )
                                            * 1.0
                                        )
                                    )
                                    / 2.0
                                )
                            )
                        )
                    )
                )
                / 2.0
            )
        )
        + 0.999023
        * (
            (
                (
                    (
                        (
                            (((data["atom_index_1"]) / 2.0))
                            <= (((data["dist"]) - (data["atom_index_0"])))
                        )
                        * 1.0
                    )
                )
                / 2.0
            )
        )
        + 0.706400
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (data["y_1"])
                                                > (
                                                    (
                                                        ((-0.108994) + (data["y_0"]))
                                                        / 2.0
                                                    )
                                                )
                                            )
                                            * 1.0
                                        )
                                    )
                                    > (
                                        (
                                            (((5.40717029571533203)) + (data["y_0"]))
                                            / 2.0
                                        )
                                    )
                                )
                                * 1.0
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
        + 1.0
        * (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (3.0)
                                            <= (
                                                (
                                                    (data["z_0"])
                                                    - (
                                                        (
                                                            (
                                                                (data["x_1"])
                                                                <= (data["y_0"])
                                                            )
                                                            * 1.0
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                        * 1.0
                                    )
                                )
                                / 2.0
                            )
                        )
                        / 2.0
                    )
                )
                / 2.0
            )
        )
    )


def GP(data):
    preds = np.zeros(len(data), dtype=np.float64)
    for t, fn in [
        (0, GP0),
        (1, GP1),
        (2, GP2),
        (3, GP3),
        (4, GP4),
        (5, GP5),
        (6, GP6),
        (7, GP7),
    ]:
        m = data["type_code"].values == t
        if m.any():
            preds[m] = fn(data.loc[m])
    return pd.DataFrame({"id": data["id"].values, "scalar_coupling_constant": preds})




## === cell 9
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


predictions = GP(train)
group_mean_log_mae(
    train.scalar_coupling_constant, predictions.scalar_coupling_constant, train.type
)



## === cell 10
test_pred = GP(test)

sub_id = sub[["id"]].copy()
sub_id["id"] = pd.to_numeric(sub_id["id"], errors="raise").astype(np.int64)

test_pred = test_pred.copy()
test_pred["id"] = pd.to_numeric(test_pred["id"], errors="raise").astype(np.int64)

if not test_pred["id"].is_unique:
    raise RuntimeError("test_pred ids are not unique; cannot safely align submission.")

if not sub_id["id"].is_unique:
    raise RuntimeError(
        "sample_submission ids are not unique; cannot safely align submission."
    )

missing_ids = np.setdiff1d(
    sub_id["id"].values, test_pred["id"].values, assume_unique=False
)
extra_ids = np.setdiff1d(
    test_pred["id"].values, sub_id["id"].values, assume_unique=False
)
if len(missing_ids) > 0 or len(extra_ids) > 0:
    raise RuntimeError(
        f"ID mismatch between sample_submission and predictions. "
        f"missing_in_pred={len(missing_ids)}, extra_in_pred={len(extra_ids)}"
    )

aligned = sub_id.merge(test_pred, on="id", how="left", validate="one_to_one")

if aligned["scalar_coupling_constant"].isna().any():
    raise RuntimeError(
        "NaNs found after alignment; refusing to write a corrupted submission."
    )

submission = sub_id.copy()
submission["scalar_coupling_constant"] = aligned["scalar_coupling_constant"].astype(
    np.float64
)

assert submission["id"].is_unique, "Submission ids are not unique."
assert len(submission) == len(
    sub
), "Submission row count mismatch with sample_submission."
assert (
    submission["id"].values == sub_id["id"].values
).all(), "Submission id order differs."

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("submission.csv written with rows:", len(submission))


## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3223973186.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     33[0m [0;34m[0m[0m
[1;32m     34[0m [0;32mif[0m [0maligned[0m[0;34m[[0m[0;34m"scalar_coupling_constant"[0m[0;34m][0m[0;34m.[0m[0misna[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0many[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 35[0;31m     raise RuntimeError(
[0m[1;32m     36[0m         [0;34m"NaNs found after alignment; refusing to write a corrupted submission."[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m     )

[0;31mRuntimeError[0m: NaNs found after alignment; refusing to write a corrupted submission.
