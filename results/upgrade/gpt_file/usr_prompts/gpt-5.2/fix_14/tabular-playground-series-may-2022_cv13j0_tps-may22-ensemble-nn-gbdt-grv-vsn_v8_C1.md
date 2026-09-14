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

0.72658

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.7006) has done: 'Your code already writes a valid submission, so the “Not yielded” likely came from runtime/memory limits rather than formatting. I make two minimal performance-engineering changes that keep the same modeling logic: (1) avoid creating a huge one-hot matrix for the concatenated train+test by encoding categories via integer codes (still a deterministic, label-free encoding), and (2) compute GaussianNB/logreg inputs from the same encoded dense matrix to reduce RAM. This should make the notebook complete within the time/memory budget so you can obtain a Kaggle score, while preserving the same three-model ensemble and probability semantics.'
- What this solution (achieved 0.71012) has done: 'Your current 0.7006 AUC is far below the target 0.9981, so we need a real modeling-quality lift while keeping your overall “3-model probability ensemble” logic intact. The biggest issue is that your categorical handling collapses the important `f_27` hex-string signal into arbitrary integer codes, which is known to destroy AUC on this dataset; I minimally fix this by adding deterministic, label-free feature extraction from `f_27` (hex to 16 nibbles + a few simple aggregates) while leaving the rest of your pipeline (logreg GD, GaussianNB, binned NB, and weighted averaging) unchanged. I also keep your existing numeric imputation/scaling, and just append these new numeric columns to both train/test before fitting. Finally, I leave your ensemble weights as-is (so changes are localized), only ensuring the new features are included consistently and the submission stays valid.'
- What this solution (achieved 0.70358) has done: 'Your current AUC (0.710) is far below the target (0.998), so we need a genuine modeling lift while keeping your 3-model ensemble and custom implementations intact. The single biggest issue is that `f_27`’s hex string is being treated as a categorical code, which destroys its strong signal; I minimally add deterministic, label-free numeric features derived from `f_27` (16 nibbles plus a few aggregates) and then **drop the raw `f_27`** so it can’t dilute the models. I also ensure `num_cols` is recomputed after feature creation so your binned-NB actually uses the new numeric columns, and I standardize GaussianNB on the same scaled matrix as logistic regression (same core model, better numerical behavior). Everything else (training loops, losses, ensemble structure, output format/path) remains the same.'
- What this solution (achieved 0.68296) has done: 'Your current score (0.7036) is far below the target (0.9981), so we need a real lift while keeping your 3-model ensemble and custom training code intact. The biggest remaining signal in this competition is still the `f_27` hex string, so I keep your nibble features but *minimally strengthen them* by adding a few deterministic interaction/hash-style aggregates (no labels used) and by treating `f_07`..`f_10` (known categorical-like integer columns) as categorical codes instead of scaled continuous, which better matches the dataset’s structure. I also ensure the binned-NB uses the expanded numeric set (including the new aggregates) and keep your ensemble structure identical (same three models, same weighted averaging) so evaluation semantics remain the same. These changes are localized to feature construction/column typing and should move AUC substantially upward without altering your core modeling approach.'
- What this solution (achieved 0.73456) has done: 'Your current AUC is far below the target, so we need a real lift while keeping the exact same 3-model ensemble structure and training loops. The biggest remaining issue is that integer-coding of *all* categorical columns makes both your GD-logreg and GaussianNB treat category IDs as ordered continuous values, which is very harmful on this dataset; I minimally switch to a compact, deterministic one-hot encoding **only for the categorical columns** (including the forced categorical f_07..f_10), while keeping numeric columns (including your f_27 nibble/aggregate features) as-is. This preserves your core semantics (logreg+GNB on a dense feature matrix, plus the binned-NB on numeric-only), but gives the linear/probabilistic models the right representation to capture the strong categorical signal and should move AUC substantially upward toward the target. I keep paths, submission formatting, and the ensemble weights unchanged.'
- What this solution (achieved 0.67597) has done: 'Your current AUC (0.73456) is far below the target (0.99813), so we need a clear modeling-quality lift while keeping your exact 3-model ensemble structure and training loops. The single biggest remaining gap is that the dense one-hot encoding blows up dimensionality and makes your custom GD-logreg and GaussianNB hard to fit well; on this competition, a strong minimal baseline is to use **deterministic target-agnostic hashing / statistical encoding** for categoricals (especially `f_28..f_31`) and keep your existing `f_27` nibble features. I therefore replace the huge one-hot with compact frequency-based numeric encodings (count + log-count) computed on train+test (no label leakage), keep scaling/logreg/GNB semantics the same, and leave your binned-NB on numeric-only untouched. This is a localized feature-representation change that should substantially increase AUC toward the target without changing your model definitions, losses, or ensembling.'
- What this solution (achieved 0.66286) has done: 'Your current AUC (0.67597) is far below the target (0.99813), so we need a real lift while keeping your exact 3-model ensemble and training loops intact. The biggest remaining issue is that your categorical columns are only represented by global frequency counts/log-counts, which often underfits this dataset; we can add a compact, deterministic “folded target-agnostic hashing” encoding (still label-free) to give the linear/GNB models more separative signal without exploding dimensionality. I keep your `f_27` nibble features and all three models unchanged, and simply append a few hashed categorical features (computed on train+test) to the same dense matrix you already scale and feed to logreg/GNB. Finally, I keep your submission writing exactly the same to ensure a valid CSV is produced.'
- What this solution (achieved 0.72648) has done: 'Your current AUC (0.66286) is far below the target (0.99813), so we need a meaningful lift while keeping your 3-model ensemble and training loops intact. The largest win with minimal disruption is to treat the high-cardinality categoricals (especially `f_28`–`f_31`) with a compact but more expressive representation than counts+hashed-bin indices: we add deterministic **hashed one-hot (signed) features** via `FeatureHasher`-like behavior implemented with `numpy` (no new packages), which keeps memory bounded and preserves the same downstream models (GD-logreg + GaussianNB on dense matrix; binned-NB on numeric-only). This change is label-free (computed from strings only) and only appends features, so it preserves evaluation semantics while giving the linear/Gaussian models the separative power they currently lack. I also slightly rebalance ensemble weights toward the model most helped by the new representation (the GD logistic regression), which should move AUC upward toward your target without altering any model definitions.'
- What this solution (achieved 0.72658) has done: 'Your current AUC (0.726) is far below the 0.998 target, so we need a real lift while keeping your same 3-model ensemble and training loops intact. The biggest low-risk gain here is to stop feeding the *raw hashed-bin indices* (`__h*_bin`) into the linear/Gaussian models; those indices are arbitrary numbers and can actively hurt AUC, while your signed hashed one-hot already provides the intended separative signal. I keep your feature extraction and models identical, but exclude those `__h*_bin` columns from the dense matrix used by GD-logreg and GaussianNB (they remain available if needed elsewhere), which should improve ranking quality without changing evaluation semantics. I also make the feature-column selection deterministic and robust by selecting columns via explicit prefix rules (minimal change, avoids accidental inclusion of harmful columns).'

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
    """
    Keep same core logic: deterministic, label-free numeric feature extraction from f_27 hex string.
    """
    if "f_27" not in df.columns:
        return df

    s = df["f_27"].fillna("0" * 16).astype(str).str.lower()
    s = s.str.pad(16, side="right", fillchar="0").str.slice(0, 16)

    hex_map = {ch: i for i, ch in enumerate("0123456789abcdef")}

    for i in range(16):
        df[f"f27_n{i:02d}"] = s.str[i].map(hex_map).fillna(0).astype(np.int16)

    nib_cols = [f"f27_n{i:02d}" for i in range(16)]
    nib = df[nib_cols].to_numpy(dtype=np.int16)

    df["f27_sum"] = nib.sum(axis=1).astype(np.int16)
    df["f27_mean"] = nib.mean(axis=1).astype(np.float32)
    df["f27_std"] = nib.std(axis=1).astype(np.float32)
    df["f27_min"] = nib.min(axis=1).astype(np.int16)
    df["f27_max"] = nib.max(axis=1).astype(np.int16)
    df["f27_nonzero"] = (nib > 0).sum(axis=1).astype(np.int16)

    df["f27_sum_even"] = nib[:, ::2].sum(axis=1).astype(np.int16)
    df["f27_sum_odd"] = nib[:, 1::2].sum(axis=1).astype(np.int16)
    df["f27_diff_abs_sum"] = (
        np.abs(np.diff(nib.astype(np.int16), axis=1)).sum(axis=1).astype(np.int16)
    )
    w1 = np.array([1, 3, 5, 7, 11, 13, 17, 19] * 2, dtype=np.int16)
    w2 = np.array([2, 4, 6, 8, 9, 10, 12, 14] * 2, dtype=np.int16)
    df["f27_proj1_mod97"] = ((nib * w1).sum(axis=1) % 97).astype(np.int16)
    df["f27_proj2_mod89"] = ((nib * w2).sum(axis=1) % 89).astype(np.int16)

    return df




## === cell 4
X = train.drop(columns=["target"])
y = train["target"].astype(np.int8)

X = add_f27_hex_features(X)
test = add_f27_hex_features(test)

if "f_27" in X.columns:
    X = X.drop(columns=["f_27"])
if "f_27" in test.columns:
    test = test.drop(columns=["f_27"])

all_data = pd.concat([X, test], axis=0, ignore_index=True)

forced_cat = [c for c in ["f_07", "f_08", "f_09", "f_10"] if c in all_data.columns]
for c in forced_cat:
    all_data[c] = all_data[c].astype("Int64").astype("string")

cat_cols = all_data.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()
num_cols = [c for c in all_data.columns if c not in cat_cols]

for c in num_cols:
    all_data[c] = all_data[c].fillna(all_data[c].median())
for c in cat_cols:
    all_data[c] = all_data[c].fillna("NA")

train_id = (
    all_data.iloc[: len(X), :]["id"].values
    if "id" in all_data.columns
    else train["id"].values
)
test_id = (
    all_data.iloc[len(X) :, :]["id"].values
    if "id" in all_data.columns
    else test["id"].values
)




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
    Gaussian NB for numeric features.
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
    Discretize numeric features into bins and apply multinomial NB on per-feature bin counts.
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
if "id" in all_data.columns:
    features_df = all_data.drop(columns=["id"])
else:
    features_df = all_data.copy()

cat_cols = features_df.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()
num_cols = [c for c in features_df.columns if c not in cat_cols]

for c in cat_cols:
    col = features_df[c].astype("string").fillna("NA")
    vc = col.value_counts(dropna=False)
    cnt = col.map(vc).astype(np.float32)
    features_df[f"{c}__cnt"] = cnt
    features_df[f"{c}__logcnt"] = np.log1p(cnt).astype(np.float32)


def _hash_uint64(s: pd.Series, seed: int) -> np.ndarray:
    h = pd.util.hash_pandas_object(s.astype("string"), index=False).to_numpy(
        dtype=np.uint64
    )
    return (h ^ np.uint64(seed)).astype(np.uint64)


def add_hashed_onehot_signed(
    df: pd.DataFrame,
    cat_cols: list[str],
    n_features: int = 2048,
    n_seeds: int = 2,
    seed0: int = 2022,
) -> pd.DataFrame:
    """
    Deterministic, label-free hashed one-hot with sign trick.
    Produces dense float32 columns (still compact: n_seeds*n_features total),
    so we can keep your same GD-logreg and GaussianNB on dense matrices.
    """
    n = df.shape[0]
    out = df

    H = np.zeros((n, n_features * n_seeds), dtype=np.float32)

    for c in cat_cols:
        col = out[c].astype("string").fillna("NA")
        for r in range(n_seeds):
            h = _hash_uint64(col, seed=seed0 + 104729 * r + (abs(hash(c)) % 10007))
            idx = (h % np.uint64(n_features)).astype(np.int32) + r * n_features
            sgn = (((h >> np.uint64(1)) & np.uint64(1)) * 2 - 1).astype(np.int8)
            H[np.arange(n), idx] += sgn.astype(np.float32)

    for j in range(H.shape[1]):
        out[f"cat_hash_{j:04d}"] = H[:, j]
    return out


features_df = add_hashed_onehot_signed(
    features_df, cat_cols=cat_cols, n_features=1024, n_seeds=2, seed0=2022
)


def _hash_series_to_bins(s: pd.Series, n_bins: int, seed: int) -> np.ndarray:
    h = pd.util.hash_pandas_object(s.astype("string"), index=False).to_numpy(
        dtype=np.uint64
    )
    h = h ^ np.uint64(seed)
    return (h % np.uint64(n_bins)).astype(np.int32)


HASH_BINS = 1024
HASH_REP = 3

for c in cat_cols:
    col = features_df[c].astype("string").fillna("NA")
    for r in range(HASH_REP):
        bins = _hash_series_to_bins(
            col, HASH_BINS, seed=1337 + 97 * r + (abs(hash(c)) % 1000)
        )

        features_df[f"{c}__h{r}_bin"] = (
            bins.astype(np.float32) / float(HASH_BINS - 1)
        ).astype(np.float32)
        bc = np.bincount(bins, minlength=HASH_BINS).astype(np.float32)
        features_df[f"{c}__h{r}_logbinfreq"] = np.log1p(bc[bins]).astype(np.float32)

for c in num_cols:
    features_df[c] = pd.to_numeric(features_df[c], errors="coerce").astype(np.float32)
    features_df[c] = features_df[c].fillna(features_df[c].median())

dense_cols = []
for c in features_df.columns:
    if c in cat_cols:
        continue
    if c.endswith("_bin") and "__h" in c and c.startswith("f_"):
        continue
    dense_cols.append(c)

dense_cols = sorted(dense_cols)  # deterministic column order

full_features = features_df[dense_cols].astype(np.float32)

X_enc = full_features.iloc[: len(X), :].reset_index(drop=True)
test_enc = full_features.iloc[len(X) :, :].reset_index(drop=True)

X_num_filled = features_df.iloc[: len(X), :][num_cols].to_numpy(
    dtype=np.float32, copy=True
)
T_num_filled = features_df.iloc[len(X) :, :][num_cols].to_numpy(
    dtype=np.float32, copy=True
)

X_dense = X_enc.to_numpy(dtype=np.float32, copy=True)
T_dense = test_enc.to_numpy(dtype=np.float32, copy=True)

mean = X_dense.mean(axis=0, dtype=np.float64).astype(np.float32)
std = (X_dense.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)

X_scaled = (X_dense - mean) / std
T_scaled = (T_dense - mean) / std

w_lr, b_lr = fit_logreg_gd(X_scaled, y, lr=0.25, l2=2e-4, n_iter=140)
grv_vsn_pred = predict_logreg(T_scaled, w_lr, b_lr)

gbdt_params = fit_binned_multinomial_nb(X_num_filled, y, n_bins=48, alpha=0.8)
gbdt_pred = predict_binned_multinomial_nb(T_num_filled, gbdt_params)

gnb_params = fit_gaussian_nb(X_scaled, y, var_smoothing=1e-6)
nn_pred = predict_gaussian_nb(T_scaled, gnb_params)

grv_vsn = pd.DataFrame({"id": test_id, "target": grv_vsn_pred})
gbdt = pd.DataFrame({"id": test_id, "target": gbdt_pred})
nn = pd.DataFrame({"id": test_id, "target": nn_pred})



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
weights = [0.20, 0.05, 0.75]
ensamble = grv_vsn.copy()
ensamble["target"] = (
    weights[0] * grv_vsn["target"].values
    + weights[1] * gbdt["target"].values
    + weights[2] * nn["target"].values
)



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
