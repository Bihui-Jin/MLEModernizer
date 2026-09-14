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

category_encoders==2.7.0
geopandas==0.14.4
lightgbm==4.6.0
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
import gc
import lightgbm as lgbm
from sklearn.model_selection import GroupKFold
import os

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("Listing data dir:", DATA_DIR)
print(sorted(os.listdir(DATA_DIR))[:50])

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
gc.collect()



## === cell 2
train = pd.read_csv(
    f"{DATA_DIR}/train.csv",
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    f"{DATA_DIR}/test.csv",
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", dtype={"id": np.int32})
structures = pd.read_csv(
    f"{DATA_DIR}/structures.csv",
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"structures.shape: {structures.shape}")
print(f"sample_sub.shape: {sample_sub.shape}")



## === cell 3
X_train = train.drop(columns=["scalar_coupling_constant"])
y_train = train["scalar_coupling_constant"]
X_test = test

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## === cell 4
test_id = X_test["id"].copy()

X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])




## === cell 5
def convert_object_to_categories(X_train_df, X_test_df):
    for col in X_train_df.columns:
        if X_train_df[col].dtype == "O":
            X_train_df[col] = X_train_df[col].astype("category")
            X_test_df[col] = X_test_df[col].astype("category")
    return X_train_df, X_test_df


X_train, X_test = convert_object_to_categories(X_train, X_test)




## === cell 6
def calc_score_from_type(type_arr, y_true, y_pred):
    type_codes, inv = np.unique(type_arr, return_inverse=True)
    err = np.abs(y_true - y_pred)

    sums = np.bincount(inv, weights=err)
    cnts = np.bincount(inv)
    means = sums / cnts
    return float(np.log(means).mean())


def calc_score(X_df, y_true, y_pred):
    t = X_df["type"].to_numpy()
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return calc_score_from_type(t, yt, yp)


print("Train types:", X_train["type"].unique())
print("Test types:", X_test["type"].unique())



## === cell 7
s_cols = ["molecule_name", "atom_index", "atom", "x", "y", "z"]
st = structures[s_cols].copy()

st0 = st.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
    }
)
X_train = X_train.merge(
    st0, on=["molecule_name", "atom_index_0"], how="left", copy=False
)
X_test = X_test.merge(st0, on=["molecule_name", "atom_index_0"], how="left", copy=False)

st1 = st.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
    }
)
X_train = X_train.merge(
    st1, on=["molecule_name", "atom_index_1"], how="left", copy=False
)
X_test = X_test.merge(st1, on=["molecule_name", "atom_index_1"], how="left", copy=False)

dx = (X_train["atom_index_0_x"] - X_train["atom_index_1_x"]).to_numpy()
dy = (X_train["atom_index_0_y"] - X_train["atom_index_1_y"]).to_numpy()
dz = (X_train["atom_index_0_z"] - X_train["atom_index_1_z"]).to_numpy()
X_train["distance"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

dx = (X_test["atom_index_0_x"] - X_test["atom_index_1_x"]).to_numpy()
dy = (X_test["atom_index_0_y"] - X_test["atom_index_1_y"]).to_numpy()
dz = (X_test["atom_index_0_z"] - X_test["atom_index_1_z"]).to_numpy()
X_test["distance"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

X_train, X_test = convert_object_to_categories(X_train, X_test)

print("After features:")
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

del structures, st, st0, st1
gc.collect()



## === cell 8
LGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    force_col_wise=True,
)

RUN_CV = False


def cross_val_by_type(X_df, y, group_col="molecule_name", type_col="type"):
    types = X_df[type_col]
    groups_all = X_df[group_col]
    y_arr = y.to_numpy()

    unique_types = (
        types.cat.categories
        if pd.api.types.is_categorical_dtype(types)
        else np.sort(types.unique())
    )
    scores = []

    splitter = GroupKFold(n_splits=5)

    cat_cols = [c for c in X_df.columns if pd.api.types.is_categorical_dtype(X_df[c])]
    cat_idxs = [X_df.columns.get_loc(c) for c in cat_cols]

    for t in [str(x) for x in unique_types]:
        mask = types.astype(str).to_numpy() == t
        if not mask.any():
            continue

        X_t = X_df.loc[mask]
        y_t = y_arr[mask]
        groups_t = groups_all.loc[mask]

        fold_scores = []
        for tr_idx, va_idx in splitter.split(X_t, y_t, groups=groups_t):
            model = lgbm.LGBMRegressor(**LGB_PARAMS)
            model.fit(
                X_t.iloc[tr_idx, :],
                y_t[tr_idx],
                categorical_feature=cat_idxs,
            )
            y_va = model.predict(X_t.iloc[va_idx, :])
            s = calc_score_from_type(
                X_t.iloc[va_idx, :]["type"].to_numpy(), y_t[va_idx], y_va
            )
            fold_scores.append(s)

        t_mean = float(np.mean(fold_scores))
        scores.append(t_mean)
        print(
            f"type={t}  CV(mean over folds)={t_mean:.6f}  folds_std={float(np.std(fold_scores)):.6f}"
        )

    print(
        f"By-type CV mean (avg over types): {float(np.mean(scores)):.6f}  std_over_types: {float(np.std(scores)):.6f}"
    )
    return scores


if RUN_CV:
    cv_scores = cross_val_by_type(
        X_train, y_train, group_col="molecule_name", type_col="type"
    )



## === cell 9
cat_cols_align = [
    c for c in X_train.columns if pd.api.types.is_categorical_dtype(X_train[c])
]
for c in cat_cols_align:
    if c in X_test.columns and pd.api.types.is_categorical_dtype(X_test[c]):
        train_cats = X_train[c].cat.categories
        test_cats = X_test[c].cat.categories
        union_cats = train_cats.union(test_cats)
        X_train[c] = X_train[c].cat.set_categories(union_cats)
        X_test[c] = X_test[c].cat.set_categories(union_cats)

type_models = {}
pred_test = np.empty(shape=(len(X_test),), dtype=np.float32)

train_type_str = X_train["type"].astype(str).to_numpy()
test_type_str = X_test["type"].astype(str).to_numpy()

unique_types = np.unique(train_type_str)
unique_types.sort()

train_idx_all = np.arange(len(X_train))
test_idx_all = np.arange(len(X_test))

train_indices_by_type = {t: train_idx_all[train_type_str == t] for t in unique_types}
test_indices_by_type = {t: test_idx_all[test_type_str == t] for t in unique_types}

for t in unique_types:
    tr_idx = train_indices_by_type[t]
    te_idx = test_indices_by_type.get(t, None)

    X_tr_t = X_train.iloc[tr_idx, :].copy()
    X_te_t = None
    if te_idx is not None and len(te_idx) > 0:
        X_te_t = X_test.iloc[te_idx, :].copy()

    cat_cols_t = [
        c for c in X_tr_t.columns if pd.api.types.is_categorical_dtype(X_tr_t[c])
    ]
    cat_idxs_t = [X_tr_t.columns.get_loc(c) for c in cat_cols_t]

    if X_te_t is not None and len(X_te_t) > 0:
        for c in cat_cols_t:
            if c in X_te_t.columns and pd.api.types.is_categorical_dtype(X_te_t[c]):
                union_cats = X_tr_t[c].cat.categories.union(X_te_t[c].cat.categories)
                X_tr_t[c] = X_tr_t[c].cat.set_categories(union_cats)
                X_te_t[c] = X_te_t[c].cat.set_categories(union_cats)

    model = lgbm.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr_t,
        y_train.to_numpy()[tr_idx],
        categorical_feature=cat_idxs_t,
    )
    type_models[t] = model

    if X_te_t is not None and len(X_te_t) > 0:
        pred_test[te_idx] = model.predict(X_te_t, validate_features=False).astype(
            np.float32
        )

pred_train = np.empty(shape=(len(X_train),), dtype=np.float32)
y_train_arr = y_train.to_numpy()
for t, model in type_models.items():
    tr_idx = train_indices_by_type[t]
    pred_train[tr_idx] = model.predict(
        X_train.iloc[tr_idx, :], validate_features=False
    ).astype(np.float32)

print(
    f"training score (by-type models): {calc_score(X_train, y_train_arr, pred_train):.6f}"
)

y_pred_arr = np.asarray(pred_test).reshape(-1)

if len(y_pred_arr) == 0:
    y_pred_arr = np.full(shape=(len(sample_sub),), fill_value=np.nan, dtype=float)
elif len(y_pred_arr) != len(sample_sub):
    raise ValueError(
        f"y_predict length ({len(y_pred_arr)}) does not match sample_sub length ({len(sample_sub)})."
    )

sub = sample_sub.copy()
sub["scalar_coupling_constant"] = y_pred_arr
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())


## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/956605208.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     58[0m [0;34m[0m[0m
[1;32m     59[0m     [0;32mif[0m [0mX_te_t[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32mand[0m [0mlen[0m[0;34m([0m[0mX_te_t[0m[0;34m)[0m [0;34m>[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 60[0;31m         pred_test[te_idx] = model.predict(X_te_t, validate_features=False).astype(
[0m[1;32m     61[0m             [0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m[0m[0;34m[0m[0m
[1;32m     62[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py[0m in [0;36mpredict[0;34m(self, X, raw_score, start_iteration, num_iteration, pred_leaf, pred_contrib, validate_features, **kwargs)[0m
[1;32m   1142[0m         [0mpredict_params[0m[0;34m[[0m[0;34m"num_threads"[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_process_n_jobs[0m[0;34m([0m[0mpredict_params[0m[0;34m[[0m[0;34m"num_threads"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1143[0m [0;34m[0m[0m
[0;32m-> 1144[0;31m         return self._Booster.predict(  # type: ignore[union-attr]
[0m[1;32m   1145[0m             [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1146[0m             [0mraw_score[0m[0;34m=[0m[0mraw_score[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36mpredict[0;34m(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features, **kwargs)[0m
[1;32m   4765[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4766[0m                 [0mnum_iteration[0m [0;34m=[0m [0;34m-[0m[0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4767[0;31m         return predictor.predict(
[0m[1;32m   4768[0m             [0mdata[0m[0;34m=[0m[0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4769[0m             [0mstart_iteration[0m[0;34m=[0m[0mstart_iteration[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36mpredict[0;34m(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features)[0m
[1;32m   1156[0m [0;34m[0m[0m
[1;32m   1157[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mpd_DataFrame[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1158[0;31m             data = _data_from_pandas(
[0m[1;32m   1159[0m                 [0mdata[0m[0;34m=[0m[0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1160[0m                 [0mfeature_name[0m[0;34m=[0m[0;34m"auto"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_data_from_pandas[0;34m(data, feature_name, categorical_feature, pandas_categorical)[0m
[1;32m    849[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    850[0m         [0;32mif[0m [0mlen[0m[0;34m([0m[0mcat_cols[0m[0;34m)[0m [0;34m!=[0m [0mlen[0m[0;34m([0m[0mpandas_categorical[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 851[0;31m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"train and valid dataset categorical_feature do not match."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    852[0m         [0;32mfor[0m [0mcol[0m[0;34m,[0m [0mcategory[0m [0;32min[0m [0mzip[0m[0;34m([0m[0mcat_cols[0m[0;34m,[0m [0mpandas_categorical[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    853[0m             [0;32mif[0m [0mlist[0m[0;34m([0m[0mdata[0m[0;34m[[0m[0mcol[0m[0;34m][0m[0;34m.[0m[0mcat[0m[0;34m.[0m[0mcategories[0m[0;34m)[0m [0;34m!=[0m [0mlist[0m[0;34m([0m[0mcategory[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: train and valid dataset categorical_feature do not match.
