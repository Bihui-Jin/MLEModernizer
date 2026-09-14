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

No external packages required in the script and installed.

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

1.999039056096077

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, warnings, psutil
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks, backend as K
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

BASE_PATH = "/kaggle/input/champs-scalar-coupling"
print("Files in input folder:", os.listdir(BASE_PATH))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
df_struct = pd.read_csv(os.path.join(BASE_PATH, "structures.csv"))
df_train_sub_charge = pd.read_csv(os.path.join(BASE_PATH, "mulliken_charges.csv"))
df_train_sub_tensor = pd.read_csv(
    os.path.join(BASE_PATH, "magnetic_shielding_tensors.csv")
)




## === cell 2
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith("int"):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df


df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
df_struct = reduce_mem_usage(df_struct)
df_train_sub_charge = reduce_mem_usage(df_train_sub_charge)
df_train_sub_tensor = reduce_mem_usage(df_train_sub_tensor)

print("Shapes after loading:", df_train.shape, df_test.shape, df_struct.shape)




## === cell 3
def map_atom_info(df_main, df_aux, atom_idx):
    """Merge atom‑level information (structure, charge, tensor) into main df."""
    merged = pd.merge(
        df_main,
        df_aux.drop_duplicates(subset=["molecule_name", "atom_index"]),
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
    )
    return merged.drop("atom_index", axis=1)


def show_ram_usage():
    proc = psutil.Process(os.getpid())
    print("RAM usage: {:.2f} GB".format(proc.memory_info()[0] / 2.0**30))


show_ram_usage()

for atom_idx in [0, 1]:
    df_train = map_atom_info(df_train, df_struct, atom_idx)
    df_train = map_atom_info(df_train, df_train_sub_charge, atom_idx)
    df_train = map_atom_info(df_train, df_train_sub_tensor, atom_idx)

    df_train = df_train.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
            "mulliken_charge": f"charge_{atom_idx}",
            "XX": f"XX_{atom_idx}",
            "YX": f"YX_{atom_idx}",
            "ZX": f"ZX_{atom_idx}",
            "XY": f"XY_{atom_idx}",
            "YY": f"YY_{atom_idx}",
            "ZY": f"ZY_{atom_idx}",
            "XZ": f"XZ_{atom_idx}",
            "YZ": f"YZ_{atom_idx}",
            "ZZ": f"ZZ_{atom_idx}",
        }
    )

    df_test = map_atom_info(df_test, df_struct, atom_idx)
    df_test = df_test.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
        }
    )

df_struct["c_x"] = df_struct.groupby("molecule_name")["x"].transform("mean")
df_struct["c_y"] = df_struct.groupby("molecule_name")["y"].transform("mean")
df_struct["c_z"] = df_struct.groupby("molecule_name")["z"].transform("mean")
df_struct["atom_n"] = df_struct.groupby("molecule_name")["atom_index"].transform("max")

show_ram_usage()
print("After mapping:", df_train.shape, df_test.shape)




## === cell 4
def make_features(df):
    df["dx"] = df["x_1"] - df["x_0"]
    df["dy"] = df["y_1"] - df["y_0"]
    df["dz"] = df["z_1"] - df["z_0"]
    df["distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2)
    return df


df_train = make_features(df_train)
df_test = make_features(df_test)


def get_dist(df):
    cols = [
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "distance",
        "x_0",
        "y_0",
        "z_0",
        "x_1",
        "y_1",
        "z_1",
    ]
    df_temp = df[cols].copy()
    df_temp_rev = df_temp.rename(
        columns={
            "atom_index_0": "atom_index_1",
            "atom_index_1": "atom_index_0",
            "x_0": "x_1",
            "y_0": "y_1",
            "z_0": "z_1",
            "x_1": "x_0",
            "y_1": "y_0",
            "z_1": "z_0",
        }
    )
    df_all = pd.concat([df_temp, df_temp_rev], axis=0)

    df_all["min_distance"] = df_all.groupby(["molecule_name", "atom_index_0"])[
        "distance"
    ].transform("min")
    df_all["max_distance"] = df_all.groupby(["molecule_name", "atom_index_0"])[
        "distance"
    ].transform("max")

    df_min = df_all[df_all["min_distance"] == df_all["distance"]].copy()
    df_min = df_min.drop(["x_0", "y_0", "z_0", "min_distance"], axis=1)
    df_min = df_min.rename(
        columns={
            "atom_index_0": "atom_index",
            "atom_index_1": "atom_index_closest",
            "distance": "distance_closest",
            "x_1": "x_closest",
            "y_1": "y_closest",
            "z_1": "z_closest",
        }
    )

    df_max = df_all[df_all["max_distance"] == df_all["distance"]].copy()
    df_max = df_max.drop(["x_0", "y_0", "z_0", "max_distance"], axis=1)
    df_max = df_max.rename(
        columns={
            "atom_index_0": "atom_index",
            "atom_index_1": "atom_index_farthest",
            "distance": "distance_farthest",
            "x_1": "x_farthest",
            "y_1": "y_farthest",
            "z_1": "z_farthest",
        }
    )

    for atom_idx in [0, 1]:
        df = map_atom_info(df, df_min, atom_idx)
        df = df.rename(
            columns={
                "atom_index_closest": f"atom_index_closest_{atom_idx}",
                "distance_closest": f"distance_closest_{atom_idx}",
                "x_closest": f"x_closest_{atom_idx}",
                "y_closest": f"y_closest_{atom_idx}",
                "z_closest": f"z_closest_{atom_idx}",
            }
        )
        df = map_atom_info(df, df_max, atom_idx)
        df = df.rename(
            columns={
                "atom_index_farthest": f"atom_index_farthest_{atom_idx}",
                "distance_farthest": f"distance_farthest_{atom_idx}",
                "x_farthest": f"x_farthest_{atom_idx}",
                "y_farthest": f"y_farthest_{atom_idx}",
                "z_farthest": f"z_farthest_{atom_idx}",
            }
        )
    return df


df_train = get_dist(df_train)
df_test = get_dist(df_test)

print("After distance features:", df_train.shape, df_test.shape)
show_ram_usage()




## === cell 5
def add_features(df):
    df["distance_center0"] = np.sqrt(
        (df["x_0"] - df["c_x"]) ** 2
        + (df["y_0"] - df["c_y"]) ** 2
        + (df["z_0"] - df["c_z"]) ** 2
    )
    df["distance_center1"] = np.sqrt(
        (df["x_1"] - df["c_x"]) ** 2
        + (df["y_1"] - df["c_y"]) ** 2
        + (df["z_1"] - df["c_z"]) ** 2
    )

    df["distance_c0"] = np.sqrt(
        (df["x_0"] - df["x_closest_0"]) ** 2
        + (df["y_0"] - df["y_closest_0"]) ** 2
        + (df["z_0"] - df["z_closest_0"]) ** 2
    )
    df["distance_c1"] = np.sqrt(
        (df["x_1"] - df["x_closest_1"]) ** 2
        + (df["y_1"] - df["y_closest_1"]) ** 2
        + (df["z_1"] - df["z_closest_1"]) ** 2
    )

    df["distance_f0"] = np.sqrt(
        (df["x_0"] - df["x_farthest_0"]) ** 2
        + (df["y_0"] - df["y_farthest_0"]) ** 2
        + (df["z_0"] - df["z_farthest_0"]) ** 2
    )
    df["distance_f1"] = np.sqrt(
        (df["x_1"] - df["x_farthest_1"]) ** 2
        + (df["y_1"] - df["y_farthest_1"]) ** 2
        + (df["z_1"] - df["z_farthest_1"]) ** 2
    )

    df["vec_center0_x"] = (df["x_0"] - df["c_x"]) / (df["distance_center0"] + 1e-10)
    df["vec_center0_y"] = (df["y_0"] - df["c_y"]) / (df["distance_center0"] + 1e-10)
    df["vec_center0_z"] = (df["z_0"] - df["c_z"]) / (df["distance_center0"] + 1e-10)

    df["vec_center1_x"] = (df["x_1"] - df["c_x"]) / (df["distance_center1"] + 1e-10)
    df["vec_center1_y"] = (df["y_1"] - df["c_y"]) / (df["distance_center1"] + 1e-10)
    df["vec_center1_z"] = (df["z_1"] - df["c_z"]) / (df["distance_center1"] + 1e-10)

    df["vec_c0_x"] = (df["x_0"] - df["x_closest_0"]) / (df["distance_c0"] + 1e-10)
    df["vec_c0_y"] = (df["y_0"] - df["y_closest_0"]) / (df["distance_c0"] + 1e-10)
    df["vec_c0_z"] = (df["z_0"] - df["z_closest_0"]) / (df["distance_c0"] + 1e-10)

    df["vec_c1_x"] = (df["x_1"] - df["x_closest_1"]) / (df["distance_c1"] + 1e-10)
    df["vec_c1_y"] = (df["y_1"] - df["y_closest_1"]) / (df["distance_c1"] + 1e-10)
    df["vec_c1_z"] = (df["z_1"] - df["z_closest_1"]) / (df["distance_c1"] + 1e-10)

    df["vec_f0_x"] = (df["x_0"] - df["x_farthest_0"]) / (df["distance_f0"] + 1e-10)
    df["vec_f0_y"] = (df["y_0"] - df["y_farthest_0"]) / (df["distance_f0"] + 1e-10)
    df["vec_f0_z"] = (df["z_0"] - df["z_farthest_0"]) / (df["distance_f0"] + 1e-10)

    df["vec_f1_x"] = (df["x_1"] - df["x_farthest_1"]) / (df["distance_f1"] + 11e-10)
    df["vec_f1_y"] = (df["y_1"] - df["y_farthest_1"]) / (df["distance_f1"] + 1e-10)
    df["vec_f1_z"] = (df["z_1"] - df["z_farthest_1"]) / (df["distance_f1"] + 1e-10)

    df["vec_x"] = (df["x_1"] - df["x_0"]) / df["distance"]
    df["vec_y"] = (df["y_1"] - df["y_0"]) / df["distance"]
    df["vec_z"] = (df["z_1"] - df["z_0"]) / df["distance"]

    df["cos_c0_c1"] = (
        df["vec_c0_x"] * df["vec_c1_x"]
        + df["vec_c0_y"] * df["vec_c1_y"]
        + df["vec_c0_z"] * df["vec_c1_z"]
    )
    df["cos_f0_f1"] = (
        df["vec_f0_x"] * df["vec_f1_x"]
        + df["vec_f0_y"] * df["vec_f1_y"]
        + df["vec_f0_z"] * df["vec_f1_z"]
    )
    df["cos_center0_center1"] = (
        df["vec_center0_x"] * df["vec_center1_x"]
        + df["vec_center0_y"] * df["vec_center1_y"]
        + df["vec_center0_z"] * df["vec_center1_z"]
    )
    df["cos_c0"] = (
        df["vec_c0_x"] * df["vec_x"]
        + df["vec_c0_y"] * df["vec_y"]
        + df["vec_c0_z"] * df["vec_z"]
    )
    df["cos_c1"] = (
        df["vec_c1_x"] * df["vec_x"]
        + df["vec_c1_y"] * df["vec_y"]
        + df["vec_c1_z"] * df["vec_z"]
    )
    df["cos_f0"] = (
        df["vec_f0_x"] * df["vec_x"]
        + df["vec_f0_y"] * df["vec_y"]
        + df["vec_f0_z"] * df["vec_z"]
    )
    df["cos_f1"] = (
        df["vec_f1_x"] * df["vec_x"]
        + df["vec_f1_y"] * df["vec_y"]
        + df["vec_f1_z"] * df["vec_z"]
    )
    df["cos_center0"] = (
        df["vec_center0_x"] * df["vec_x"]
        + df["vec_center0_y"] * df["vec_y"]
        + df["vec_center0_z"] * df["vec_z"]
    )
    df["cos_center1"] = (
        df["vec_center1_x"] * df["vec_x"]
        + df["vec_center1_y"] * df["vec_y"]
        + df["vec_center1_z"] * df["vec_z"]
    )

    df = df.drop(
        [
            "vec_c0_x",
            "vec_c0_y",
            "vec_c0_z",
            "vec_c1_x",
            "vec_c1_y",
            "vec_c1_z",
            "vec_f0_x",
            "vec_f0_y",
            "vec_f0_z",
            "vec_f1_x",
            "vec_f1_y",
            "vec_f1_z",
            "vec_center0_x",
            "vec_center0_y",
            "vec_center0_z",
            "vec_center1_x",
            "vec_center1_y",
            "vec_center1_z",
            "vec_x",
            "vec_y",
            "vec_z",
        ],
        axis=1,
    )
    return df


df_train = add_features(df_train)
df_test = add_features(df_test)

print("After adding engineered features:", df_train.shape, df_test.shape)
show_ram_usage()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'c_x'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4060785857.py in <cell line: 0>()
    140 
    141 
--> 142 df_train = add_features(df_train)
    143 df_test = add_features(df_test)
    144 

/tmp/ipykernel_11/4060785857.py in add_features(df)
      2     # centre of molecule (pre‑computed in df_struct and merged via get_dist)
      3     df["distance_center0"] = np.sqrt(
----> 4         (df["x_0"] - df["c_x"]) ** 2
      5         + (df["y_0"] - df["c_y"]) ** 2
      6         + (df["z_0"] - df["c_z"]) ** 2

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'c_x'

## === cell 6
def create_nn_model(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(256)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)

    out1 = Dense(2, activation="linear")(x)  # mulliken charges
    out2 = Dense(6, activation="linear")(x)  # tensor diagonal
    out3 = Dense(12, activation="linear")(x)  # remaining tensor elements

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)

    out = Dense(1, activation="linear")(x)  # scalar coupling constant
    model = Model(inputs=inp, outputs=[out, out1, out2, out3])
    return model


def plot_history(history, label):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"Loss for {label}")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()




## === cell 7
from datetime import datetime

mol_types = df_train["type"].unique()
cv_score = []
cv_score_total = 0.0
epoch_n = 5  # reduced for quick execution; can be increased later
verbose = 0
batch_size = 1024
retrain = False

config = tf.compat.v1.ConfigProto(device_count={"GPU": 1, "CPU": 2})
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

start_time = datetime.now()

for mol_type in mol_types:
    print(f"\nTraining molecule type: {mol_type}")
    df_train_ = df_train[df_train["type"] == mol_type].reset_index(drop=True)
    df_test_ = df_test[df_test["type"] == mol_type].reset_index(drop=True)

    input_features = [
        "x_0",
        "y_0",
        "z_0",
        "x_1",
        "y_1",
        "z_1",
        "c_x",
        "c_y",
        "c_z",
        "x_closest_0",
        "y_closest_0",
        "z_closest_0",
        "x_closest_1",
        "y_closest_1",
        "z_closest_1",
        "distance",
        "distance_center0",
        "distance_center1",
        "distance_c0",
        "distance_c1",
        "distance_f0",
        "distance_f1",
        "cos_c0_c1",
        "cos_f0_f1",
        "cos_center0_center1",
        "cos_c0",
        "cos_c1",
        "cos_f0",
        "cos_f1",
        "cos_center0",
        "cos_center1",
        "atom_n",
    ]

    scaler = StandardScaler()
    all_features = pd.concat(
        [df_train_[input_features], df_test_[input_features]], axis=0
    )
    scaler.fit(all_features)
    input_data = scaler.transform(all_features)

    train_input = input_data[: len(df_train_)]
    test_input = input_data[len(df_train_) :]

    target_main = df_train_["scalar_coupling_constant"].values
    target_charge = df_train_[["charge_0", "charge_1"]].values
    target_diag = df_train_[["XX_0", "YY_0", "ZZ_0", "XX_1", "YY_1", "ZZ_1"]].values
    target_off = df_train_[
        [
            "YX_0",
            "ZX_0",
            "XY_0",
            "ZY_0",
            "XZ_0",
            "YZ_0",
            "YX_1",
            "ZX_1",
            "XY_1",
            "ZY_1",
            "XZ_1",
            "YZ_1",
        ]
    ].values

    target_charge = 1 * StandardScaler().fit_transform(target_charge)
    target_diag = 4 * StandardScaler().fit_transform(target_diag)
    target_off = 1 * StandardScaler().fit_transform(target_off)

    idx = np.arange(len(df_train_))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=111)

    nn_model = create_nn_model(train_input.shape[1])

    if not retrain and os.path.exists(
        f"/kaggle/working/molecule_model_{mol_type}.hdf5"
    ):
        nn_model = load_model(f"/kaggle/working/molecule_model_{mol_type}.hdf5")

    nn_model.compile(loss="mae", optimizer=Adam())

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=5e-4,
        patience=8,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )
    rlr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=7, min_lr=1e-6, verbose=1, mode="auto"
    )
    ckpt = callbacks.ModelCheckpoint(
        f"/kaggle/working/molecule_model_{mol_type}.hdf5",
        monitor="val_loss",
        save_best_only=True,
        verbose=0,
    )

    history = nn_model.fit(
        train_input[train_idx],
        [
            target_main[train_idx],
            target_charge[train_idx],
            target_diag[train_idx],
            target_off[train_idx],
        ],
        validation_data=(
            train_input[val_idx],
            [
                target_main[val_idx],
                target_charge[val_idx],
                target_diag[val_idx],
                target_off[val_idx],
            ],
        ),
        epochs=epoch_n,
        batch_size=batch_size,
        callbacks=[es, rlr, ckpt],
        verbose=verbose,
    )

    val_pred = nn_model.predict(train_input[val_idx])[0].ravel()
    mae = np.mean(np.abs(target_main[val_idx] - val_pred))
    cv_score.append(np.log(mae))
    cv_score_total += np.log(mae)

    test_pred = nn_model.predict(test_input)[0].ravel()
    df_test.loc[df_test["type"] == mol_type, "pred_tmp"] = test_pred
    K.clear_session()

cv_score_total /= len(mol_types)
print(f"Overall CV log‑MAE: {cv_score_total:.5f}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/401303117.py in <cell line: 0>()
     13 config.gpu_options.allow_growth = True
     14 sess = tf.compat.v1.Session(config=config)
---> 15 K.set_session(sess)
     16 
     17 start_time = datetime.now()

AttributeError: module 'keras._tf_keras.keras.backend' has no attribute 'set_session'

## === cell 8
test_prediction = df_test.sort_values("id")["pred_tmp"].values


def submit(predictions):
    sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
    submit_df = pd.read_csv(sub_path)
    print("Submission shape:", submit_df.shape, "Predictions length:", len(predictions))
    submit_df["scalar_coupling_constant"] = predictions
    out_path = "/kaggle/working/workingsubmission-test.csv"
    submit_df.to_csv(out_path, index=False)
    print("Saved submission to:", out_path)


submit(test_prediction)

print("Total training time:", datetime.now() - start_time)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'pred_tmp'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1820236553.py in <cell line: 0>()
      1 # Prepare final predictions array in the original test order
----> 2 test_prediction = df_test.sort_values("id")["pred_tmp"].values
      3 
      4 
      5 def submit(predictions):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'pred_tmp'

## === cell 9
import tensorflow.keras as keras

print("Keras version:", keras.__version__)
