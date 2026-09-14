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

geopandas==0.14.4
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

-1.572091784711615

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import hashlib
import numpy as np
import pandas as pd

CANDIDATE_DIRS = [
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/working/champs-scalar-coupling",
    "/kaggle/working/champs-scalar-coupling/champs-scalar-coupling",
]
DATA_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        f"Could not find train.csv/test.csv in any of: {CANDIDATE_DIRS}"
    )

print("Using DATA_DIR:", DATA_DIR)
print("Listing DATA_DIR (truncated):", sorted(os.listdir(DATA_DIR))[:20])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

train = pd.read_csv(
    train_path,
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)
structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "x", "y", "z"],
)

train["atom_index_0"] = train["atom_index_0"].astype(np.int16)
train["atom_index_1"] = train["atom_index_1"].astype(np.int16)
test["atom_index_0"] = test["atom_index_0"].astype(np.int16)
test["atom_index_1"] = test["atom_index_1"].astype(np.int16)
structures["atom_index"] = structures["atom_index"].astype(np.int16)

train["type"] = train["type"].astype("category")
test["type"] = test["type"].astype("category")
all_types = pd.Index(train["type"].astype(str).unique()).union(
    pd.Index(test["type"].astype(str).unique())
)
train["type"] = pd.Categorical(train["type"].astype(str), categories=all_types)
test["type"] = pd.Categorical(test["type"].astype(str), categories=all_types)

structures["x"] = structures["x"].astype(np.float32)
structures["y"] = structures["y"].astype(np.float32)
structures["z"] = structures["z"].astype(np.float32)

structures = structures.drop_duplicates(
    subset=["molecule_name", "atom_index"], keep="first"
)

s = structures
s0 = s.rename(columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"})
s1 = s.rename(columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"})


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    if df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=None):
        bad = df[df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=1)].head(10)
        raise ValueError(
            "Missing coordinates after merge; cannot compute distances. "
            f"Examples:\n{bad[['molecule_name','atom_index_0','atom_index_1','type']].to_string(index=False)}"
        )

    dx = df["x0"].to_numpy(dtype=np.float32) - df["x1"].to_numpy(dtype=np.float32)
    dy = df["y0"].to_numpy(dtype=np.float32) - df["y1"].to_numpy(dtype=np.float32)
    dz = df["z0"].to_numpy(dtype=np.float32) - df["z1"].to_numpy(dtype=np.float32)
    df["dist"] = (np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12).astype(np.float32)
    return df


train_feat = add_distance(train.copy())
test_feat = add_distance(test.copy())

print("train_feat shape:", train_feat.shape, "test_feat shape:", test_feat.shape)
print(
    "dist stats (train):",
    float(train_feat["dist"].min()),
    float(train_feat["dist"].max()),
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1524719399.py in <cell line: 0>()
     72 
     73 train_feat = add_distance(train.copy())
---> 74 test_feat = add_distance(test.copy())
     75 
     76 print("train_feat shape:", train_feat.shape, "test_feat shape:", test_feat.shape)

/tmp/ipykernel_11/1524719399.py in add_distance(df)
     59     if df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=None):
     60         bad = df[df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=1)].head(10)
---> 61         raise ValueError(
     62             "Missing coordinates after merge; cannot compute distances. "
     63             f"Examples:\n{bad[['molecule_name','atom_index_0','atom_index_1','type']].to_string(index=False)}"

ValueError: Missing coordinates after merge; cannot compute distances. Examples:
   molecule_name  atom_index_0  atom_index_1 type
dsgdb9nsd_071451             9             0 1JHC
dsgdb9nsd_071451             9             1 2JHC
dsgdb9nsd_071451             9             4 3JHC
dsgdb9nsd_071451             9             5 3JHC
dsgdb9nsd_071451             9            10 2JHH
dsgdb9nsd_071451             9            11 2JHH
dsgdb9nsd_071451            10             0 1JHC
dsgdb9nsd_071451            10             1 2JHC
dsgdb9nsd_071451            10             4 3JHC
dsgdb9nsd_071451            10             5 3JHC

## === cell 2
type_median = train_feat.groupby("type")["scalar_coupling_constant"].median()
global_median = float(train_feat["scalar_coupling_constant"].median())

slopes = {}
intercepts = {}
bias = {}
clip_lo = {}
clip_hi = {}


def stable_seed_from_string(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def fit_slope_intercept_ls(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Per-type least-squares line fit (y = a + b*x).
    """
    n = x.size
    if n < 2:
        a = float(np.median(y)) if n == 1 else float(global_median)
        return 0.0, a

    xm = float(np.mean(x))
    ym = float(np.mean(y))
    xc = x - xm
    denom = float(np.dot(xc, xc))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0, float(np.median(y))
    b = float(np.dot(xc, (y - ym)) / denom)
    a = float(ym - b * xm)
    return b, a


for t, g in train_feat.groupby("type", sort=False):
    seed_t = stable_seed_from_string(str(t))
    rng = np.random.RandomState(seed_t)
    n = len(g)
    max_n = 250_000
    if n > max_n:
        take = rng.choice(n, size=max_n, replace=False)
        gg = g.iloc[take]
    else:
        gg = g

    x = gg["dist"].to_numpy(dtype=np.float64)
    y = gg["scalar_coupling_constant"].to_numpy(dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size < 2 or np.allclose(x.var(), 0.0):
        b = 0.0
        a = float(type_median.loc[t])
    else:
        b, a = fit_slope_intercept_ls(x, y)

    gfull_x = g["dist"].to_numpy(dtype=np.float64)
    gfull_y = g["scalar_coupling_constant"].to_numpy(dtype=np.float64)
    mm = np.isfinite(gfull_x) & np.isfinite(gfull_y)
    gfull_x = gfull_x[mm]
    gfull_y = gfull_y[mm]

    if gfull_y.size > 0:
        base_pred = a + b * gfull_x
        resid = gfull_y - base_pred
        b0 = float(np.median(resid))
        lo, hi = np.quantile(gfull_y, [0.005, 0.995])
        clip_lo[str(t)] = float(lo)
        clip_hi[str(t)] = float(hi)
    else:
        b0 = 0.0
        clip_lo[str(t)] = global_median
        clip_hi[str(t)] = global_median

    slopes[str(t)] = float(b)
    intercepts[str(t)] = float(a)
    bias[str(t)] = float(b0)

print("Fitted per-type params:", len(slopes), "types")




## === cell 3
def base_predict(df: pd.DataFrame) -> np.ndarray:
    t_str = df["type"].astype(str).to_numpy()
    dist = df["dist"].to_numpy(dtype=np.float64)
    out = np.empty(len(df), dtype=np.float64)

    uniq = pd.unique(t_str)
    for tt in uniq:
        m = t_str == tt
        if not np.any(m):
            continue
        b = slopes.get(tt, 0.0)
        a = intercepts.get(tt, global_median)
        bb = bias.get(tt, 0.0)
        pred = a + b * dist[m] + bb
        lo = clip_lo.get(tt, -np.inf)
        hi = clip_hi.get(tt, np.inf)
        pred = np.minimum(np.maximum(pred, lo), hi)
        out[m] = pred
    return out


calib_x = {}
calib_y = {}
q = np.linspace(0.01, 0.99, 51)

train_pred_base = base_predict(train_feat)

for t, idx in train_feat.groupby("type", sort=False).indices.items():
    tt = str(t)
    yp = train_pred_base[idx].astype(np.float64, copy=False)
    yt = train_feat.loc[idx, "scalar_coupling_constant"].to_numpy(dtype=np.float64)
    m = np.isfinite(yp) & np.isfinite(yt)
    yp = yp[m]
    yt = yt[m]
    if yp.size < 50:
        continue
    xq = np.quantile(yp, q)
    yq = np.quantile(yt, q)
    xq = np.maximum.accumulate(xq)
    calib_x[tt] = xq.astype(np.float64)
    calib_y[tt] = yq.astype(np.float64)

test_pred_base = base_predict(test_feat)

test_types = test_feat["type"].astype(str).to_numpy()
pred = test_pred_base.copy()

for tt in np.unique(test_types):
    m = test_types == tt
    if not np.any(m):
        continue
    if tt in calib_x:
        pred[m] = np.interp(pred[m], calib_x[tt], calib_y[tt])
    lo = clip_lo.get(tt, -np.inf)
    hi = clip_hi.get(tt, np.inf)
    pred[m] = np.minimum(np.maximum(pred[m], lo), hi)

submission = pd.DataFrame(
    {
        "id": test_feat["id"].astype(np.int64),
        "scalar_coupling_constant": pred.astype(np.float64),
    }
)

submission = submission.sort_values("id").reset_index(drop=True)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Columns:", submission.columns.tolist())
assert out_path.endswith(".csv")
assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id", "scalar_coupling_constant"]
assert np.isfinite(submission["scalar_coupling_constant"].to_numpy()).all()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2864238355.py in <cell line: 0>()
     41     calib_y[tt] = yq.astype(np.float64)
     42 
---> 43 test_pred_base = base_predict(test_feat)
     44 
     45 test_types = test_feat["type"].astype(str).to_numpy()

NameError: name 'test_feat' is not defined
