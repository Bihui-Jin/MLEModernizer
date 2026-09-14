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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

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
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.9981346771070628

# 6. Current score

0.71012

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.7006) has done: 'Your code already writes a valid submission, so the “Not yielded” likely came from runtime/memory limits rather than formatting. I make two minimal performance-engineering changes that keep the same modeling logic: (1) avoid creating a huge one-hot matrix for the concatenated train+test by encoding categories via integer codes (still a deterministic, label-free encoding), and (2) compute GaussianNB/logreg inputs from the same encoded dense matrix to reduce RAM. This should make the notebook complete within the time/memory budget so you can obtain a Kaggle score, while preserving the same three-model ensemble and probability semantics.'
- What this solution (achieved 0.71012) has done: 'Your current 0.7006 AUC is far below the target 0.9981, so we need a real modeling-quality lift while keeping your overall “3-model probability ensemble” logic intact. The biggest issue is that your categorical handling collapses the important `f_27` hex-string signal into arbitrary integer codes, which is known to destroy AUC on this dataset; I minimally fix this by adding deterministic, label-free feature extraction from `f_27` (hex to 16 nibbles + a few simple aggregates) while leaving the rest of your pipeline (logreg GD, GaussianNB, binned NB, and weighted averaging) unchanged. I also keep your existing numeric imputation/scaling, and just append these new numeric columns to both train/test before fitting. Finally, I leave your ensemble weights as-is (so changes are localized), only ensuring the new features are included consistently and the submission stays valid.'

# 9. Code solution

## === cell 0
import os, warnings

warnings.filterwarnings("ignore")



## === cell 1
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 2
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"
sample_path = "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

assert "target" in train.columns
assert "id" in train.columns and "id" in test.columns
assert list(sample_sub.columns) == ["id", "target"]




## === cell 3
def add_f27_hex_features(df: pd.DataFrame) -> pd.DataFrame:
    if "f_27" not in df.columns:
        return df

    s = df["f_27"].fillna("0" * 16).astype(str).str.lower()

    s = s.str.pad(16, side="right", fillchar="0").str.slice(0, 16)

    for i in range(16):
        df[f"f27_n{i:02d}"] = (
            s.str[i]
            .map(
                lambda ch: int(ch, 16) if ("0" <= ch <= "9" or "a" <= ch <= "f") else 0
            )
            .astype(np.int16)
        )

    nib_cols = [f"f27_n{i:02d}" for i in range(16)]
    nib = df[nib_cols].to_numpy(dtype=np.int16)

    df["f27_sum"] = nib.sum(axis=1).astype(np.int16)
    df["f27_mean"] = (nib.mean(axis=1)).astype(np.float32)
    df["f27_std"] = (nib.std(axis=1)).astype(np.float32)
    df["f27_min"] = nib.min(axis=1).astype(np.int16)
    df["f27_max"] = nib.max(axis=1).astype(np.int16)
    df["f27_nonzero"] = (nib > 0).sum(axis=1).astype(np.int16)

    return df




## === cell 4
X = train.drop(columns=["target"])
y = train["target"].astype(np.int8)

X = add_f27_hex_features(X)
test = add_f27_hex_features(test)

all_data = pd.concat([X, test], axis=0, ignore_index=True)

cat_cols = all_data.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in all_data.columns if c not in cat_cols]

for c in num_cols:
    all_data[c] = all_data[c].fillna(all_data[c].median())
for c in cat_cols:
    all_data[c] = all_data[c].fillna("NA")

for c in cat_cols:
    all_data[c] = all_data[c].astype("category")
    all_data[c] = all_data[c].cat.codes.astype(np.int32)

X_enc = all_data.iloc[: len(X), :].copy()
test_enc = all_data.iloc[len(X) :, :].copy()

train_id = X_enc["id"].values if "id" in X_enc.columns else train["id"].values
test_id = test_enc["id"].values if "id" in test_enc.columns else test["id"].values

if "id" in X_enc.columns:
    X_enc = X_enc.drop(columns=["id"])
if "id" in test_enc.columns:
    test_enc = test_enc.drop(columns=["id"])

X_enc = X_enc.reindex(sorted(X_enc.columns), axis=1)
test_enc = test_enc.reindex(X_enc.columns, axis=1, fill_value=0)




## === cell 5
def _sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg_gd(X, y, lr=0.2, l2=1e-4, n_iter=120):
    """
    Simple L2-regularized logistic regression via batch GD.
    Core semantics match: linear model -> sigmoid -> probability.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n, d = X.shape
    w = np.zeros(d, dtype=np.float32)
    b = np.float32(0.0)

    for _ in range(n_iter):
        p = _sigmoid(X @ w + b)
        grad_w = (X.T @ (p - y)) / n + l2 * w
        grad_b = np.mean(p - y)
        w -= lr * grad_w.astype(np.float32)
        b -= np.float32(lr * grad_b)
    return w, b


def predict_logreg(X, w, b):
    X = np.asarray(X, dtype=np.float32)
    return _sigmoid(X @ w + b)


def fit_gaussian_nb(X, y, var_smoothing=1e-9):
    """
    Gaussian NB for (mostly) numeric; for one-hot it still works as a crude approximation.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    idx0 = y == 0
    idx1 = ~idx0
    prior1 = float(idx1.mean())
    prior0 = 1.0 - prior1

    mu0 = X[idx0].mean(axis=0)
    mu1 = X[idx1].mean(axis=0)
    var0 = X[idx0].var(axis=0) + var_smoothing
    var1 = X[idx1].var(axis=0) + var_smoothing

    return (prior0, prior1, mu0, mu1, var0, var1)


def predict_gaussian_nb(X, params):
    prior0, prior1, mu0, mu1, var0, var1 = params
    X = np.asarray(X, dtype=np.float32)

    logp0 = np.log(prior0 + 1e-12) - 0.5 * (
        np.log(2.0 * np.pi * var0).sum() + (((X - mu0) ** 2) / var0).sum(axis=1)
    )
    logp1 = np.log(prior1 + 1e-12) - 0.5 * (
        np.log(2.0 * np.pi * var1).sum() + (((X - mu1) ** 2) / var1).sum(axis=1)
    )
    m = np.maximum(logp0, logp1)
    p1 = np.exp(logp1 - m) / (np.exp(logp0 - m) + np.exp(logp1 - m))
    return p1.astype(np.float64)


def fit_binned_multinomial_nb(X_num, y, n_bins=32, alpha=0.5):
    """
    Minimal replacement for a tree-ish nonlinear model:
    discretize numeric features into bins and apply multinomial NB on the resulting one-hot bins.
    """
    X_num = np.asarray(X_num, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    n, d = X_num.shape

    edges = []
    for j in range(d):
        col = X_num[:, j]
        qs = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        e = np.quantile(col, qs)
        e = np.unique(e)
        if e.size < 3:
            mn, mx = float(col.min()), float(col.max())
            e = np.array([mn - 1e-6, (mn + mx) / 2.0, mx + 1e-6], dtype=np.float32)
        edges.append(e)

    counts = []
    class_counts = np.array([np.sum(y == 0), np.sum(y == 1)], dtype=np.float64)
    for j in range(d):
        e = edges[j]
        bj = np.searchsorted(e[1:-1], X_num[:, j], side="right")
        nbj = len(e) - 1
        c = np.zeros((nbj, 2), dtype=np.float64)
        for cls in (0, 1):
            mask = y == cls
            if mask.any():
                binc = np.bincount(bj[mask], minlength=nbj).astype(np.float64)
                c[:, cls] = binc
        counts.append(c)

    log_prob = []
    for j in range(d):
        c = counts[j]
        nbj = c.shape[0]
        lp = np.zeros_like(c)
        for cls in (0, 1):
            denom = c[:, cls].sum() + alpha * nbj
            lp[:, cls] = np.log((c[:, cls] + alpha) / denom)
        log_prob.append(lp)

    priors = (class_counts / class_counts.sum()).astype(np.float64)
    log_priors = np.log(priors + 1e-12)
    return edges, log_prob, log_priors


def predict_binned_multinomial_nb(X_num, params):
    edges, log_prob, log_priors = params
    X_num = np.asarray(X_num, dtype=np.float32)
    n, d = X_num.shape
    logp = np.tile(log_priors, (n, 1))
    for j in range(d):
        e = edges[j]
        bj = np.searchsorted(e[1:-1], X_num[:, j], side="right")
        lp = log_prob[j]
        logp[:, 0] += lp[bj, 0]
        logp[:, 1] += lp[bj, 1]
    m = np.max(logp, axis=1, keepdims=True)
    p = np.exp(logp - m)
    p = p / p.sum(axis=1, keepdims=True)
    return p[:, 1]




## === cell 6
X_dense = X_enc.to_numpy(dtype=np.float32)
T_dense = test_enc.to_numpy(dtype=np.float32)

X_num_filled = all_data.loc[: len(X) - 1, num_cols].to_numpy(dtype=np.float32)
T_num_filled = all_data.loc[len(X) :, num_cols].to_numpy(dtype=np.float32)

mean = X_dense.mean(axis=0)
std = X_dense.std(axis=0) + 1e-6
X_scaled = (X_dense - mean) / std
T_scaled = (T_dense - mean) / std

w_lr, b_lr = fit_logreg_gd(X_scaled, y, lr=0.25, l2=2e-4, n_iter=140)
grv_vsn_pred = predict_logreg(T_scaled, w_lr, b_lr)

gbdt_params = fit_binned_multinomial_nb(X_num_filled, y, n_bins=48, alpha=0.8)
gbdt_pred = predict_binned_multinomial_nb(T_num_filled, gbdt_params)

gnb_params = fit_gaussian_nb(X_dense, y, var_smoothing=1e-6)
nn_pred = predict_gaussian_nb(T_dense, gnb_params)

grv_vsn = pd.DataFrame({"id": test["id"].values, "target": grv_vsn_pred})
gbdt = pd.DataFrame({"id": test["id"].values, "target": gbdt_pred})
nn = pd.DataFrame({"id": test["id"].values, "target": nn_pred})



## === cell 7
print(grv_vsn.head())
print(gbdt.head())
print(nn.head())

assert grv_vsn.shape[0] == test.shape[0]
assert gbdt.shape[0] == test.shape[0]
assert nn.shape[0] == test.shape[0]



## === cell 8
grv_vsn = grv_vsn.sort_values("id").reset_index(drop=True)
gbdt = gbdt.sort_values("id").reset_index(drop=True)
nn = nn.sort_values("id").reset_index(drop=True)

assert (grv_vsn["id"].values == gbdt["id"].values).all()
assert (grv_vsn["id"].values == nn["id"].values).all()



## === cell 9
weights = [0.01, 0.04, 0.95]
ensamble = grv_vsn.copy()
ensamble["target"] = (
    weights[0] * grv_vsn["target"].values
    + weights[1] * gbdt["target"].values
    + weights[2] * nn["target"].values
)
ensamble.head()



## === cell 10
sub = sample_sub.sort_values("id").reset_index(drop=True)

ensamble_sorted = ensamble.sort_values("id").reset_index(drop=True)
assert (sub["id"].values == ensamble_sorted["id"].values).all()

pred = ensamble_sorted["target"].to_numpy(dtype=np.float64)
pred = np.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0)

sub["target"] = np.clip(pred, 0.0, 1.0)

out_path = "/kaggle/working/my_ensamble_052222.csv"
sub = sub[["id", "target"]]
assert sub.shape[0] == test.shape[0]
assert sub["target"].notna().all()

sub.to_csv(out_path, index=False)
print(f"Wrote submission to: {out_path}")
print(sub.head())
print(sub.describe(include="all"))
print(
    "File exists:",
    os.path.exists(out_path),
    "Size:",
    os.path.getsize(out_path) if os.path.exists(out_path) else None,
)
