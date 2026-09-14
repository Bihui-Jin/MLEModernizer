# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, warnings, psutil, datetime
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Dense,
        Input,
        BatchNormalization,
        Dropout,
        LeakyReLU,
    )
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras import callbacks, backend as K
    from tensorflow.keras.optimizers import Adam
except Exception:
    tf = None
    K = None
    warnings.warn("TensorFlow import failed – will use sklearn models instead.")

warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

BASE_PATH = "/kaggle/input/champs-scalar-coupling"
print("Files in input folder:", os.listdir(BASE_PATH))




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

mol_center = df_struct[
    ["molecule_name", "c_x", "c_y", "c_z", "atom_n"]
].drop_duplicates()

df_train = df_train.merge(mol_center, on="molecule_name", how="left")
df_test = df_test.merge(mol_center, on="molecule_name", how="left")

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

    df["vec_f1_x"] = (df["x_1"] - df["x_farthest_1"]) / (df["distance_f1"] + 1e-10)
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




## === cell 6
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import math

mol_types = df_train["type"].unique()
cv_scores = []
batch_size = 1024
retrain = False

start_time = datetime.datetime.now()

for mol_type in mol_types:
    print(f"\nTraining molecule type: {mol_type}")
    df_tr = df_train[df_train["type"] == mol_type].reset_index(drop=True)
    df_te = df_test[df_test["type"] == mol_type].reset_index(drop=True)

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
    all_feat = pd.concat([df_tr[input_features], df_te[input_features]], axis=0)
    scaler.fit(all_feat)
    X_all = scaler.transform(all_feat)

    X_train = X_all[: len(df_tr)]
    X_test = X_all[len(df_tr) :]

    y = df_tr["scalar_coupling_constant"].values

    idx = np.arange(len(df_tr))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=111)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        n_jobs=4,
        random_state=42,
        verbose=0,
    )
    rf.fit(X_train[train_idx], y[train_idx])

    val_pred = rf.predict(X_train[val_idx])
    mae = mean_absolute_error(y[val_idx], val_pred)
    cv_scores.append(math.log(mae))

    test_pred = rf.predict(X_test)
    df_test.loc[df_test["type"] == mol_type, "pred_tmp"] = test_pred

    del rf, X_train, X_test
    if tf is not None and K is not None:
        K.clear_session()

overall_log_mae = np.mean(cv_scores)
print(f"Overall CV log‑MAE: {overall_log_mae:.5f}")




## === cell 7
if "pred_tmp" not in df_test.columns:
    df_test["pred_tmp"] = np.nan
df_test["pred_tmp"].fillna(df_test["pred_tmp"].mean(), inplace=True)

test_prediction = df_test.sort_values("id")["pred_tmp"].values


def submit(predictions):
    sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
    submit_df = pd.read_csv(sub_path)
    print("Submission shape:", submit_df.shape, "Predictions length:", len(predictions))
    submit_df["scalar_coupling_constant"] = predictions
    out_path = "/kaggle/working/submission.csv"
    submit_df.to_csv(out_path, index=False)
    print("Saved submission to:", out_path)


submit(test_prediction)

print("Total training time:", datetime.datetime.now() - start_time)




## === cell 8
import sklearn

print("scikit-learn version:", sklearn.__version__)
