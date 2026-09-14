# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

cudf-polars-cu12==25.6.0
geopandas==0.14.4
lightgbm==4.6.0
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
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import re
import copy
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgbm
from tqdm.auto import tqdm
from lightgbm import log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, accuracy_score

from scipy import sparse
from scipy.optimize import minimize

import warnings

warnings.filterwarnings("ignore")

try:
    from sklearnex import patch_sklearn  # scikit-learn-intelex

    patch_sklearn()
except Exception:
    pass

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)

pl.Config.set_tbl_rows(5)
pl.Config.set_tbl_cols(50)

N_JOBS = int(os.environ.get("OMP_NUM_THREADS", "0")) or (os.cpu_count() or 4)



## === cell 1
path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"


def load_db(name):
    df = pl.read_csv(path + name + ".csv")
    print(f"< {name} DataFrame Info >")
    print(df.head(1))
    return df




## === cell 2
df_train = load_db("train")
df_test = load_db("test")



## === cell 3
_HTML_RE = r"<.*?>"
_MENTION_RE = r"@\w+"
_QUOTED_DIGITS_RE = r"'\d+"
_DIGITS_RE = r"\d+"
_HTTP_RE = r"http\w+"
_WS_RE = r"\s+"
_DOTS_RE = r"\.+"
_COMMAS_RE = r"\,+"


def Data_clearning(x):
    x = str(x).lower()
    x = re.compile(r"<.*?>").sub(r"", x)
    x = re.sub(r"@\w+", "", x)
    x = re.sub(r"'\d+", "", x)
    x = re.sub(r"\d+", "", x)
    x = re.sub(r"http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


def clean_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(_HTML_RE, "")
        .str.replace_all(_MENTION_RE, "")
        .str.replace_all(_QUOTED_DIGITS_RE, "")
        .str.replace_all(_DIGITS_RE, "")
        .str.replace_all(_HTTP_RE, "")
        .str.replace_all(_WS_RE, " ")
        .str.replace_all(_DOTS_RE, ".")
        .str.replace_all(_COMMAS_RE, ",")
        .str.strip_chars()
    )


df_train = (
    df_train.lazy()
    .with_columns(clean_expr("full_text").alias("full_text_clean"))
    .collect(streaming=True)
)
df_test = (
    df_test.lazy()
    .with_columns(clean_expr("full_text").alias("full_text_clean"))
    .collect(streaming=True)
)



## === cell 4
_clean_re = re.compile(r"(<.*?>)|(@\w+)|('\d+)|(\d+)|(http\w+)|(\s+)|(\.+)|(,+)")

_ws_re = re.compile(r"\s+")
_html_re = re.compile(_HTML_RE)
_mention_re = re.compile(_MENTION_RE)
_quoted_digits_re = re.compile(_QUOTED_DIGITS_RE)
_digits_re = re.compile(_DIGITS_RE)
_http_re = re.compile(_HTTP_RE)
_dots_re = re.compile(_DOTS_RE)
_commas_re = re.compile(_COMMAS_RE)


def clean_text_py(x: str) -> str:
    x = str(x).lower()
    x = _html_re.sub("", x)
    x = _mention_re.sub("", x)
    x = _quoted_digits_re.sub("", x)
    x = _digits_re.sub("", x)
    x = _http_re.sub("", x)
    x = _ws_re.sub(" ", x)
    x = _dots_re.sub(".", x)
    x = _commas_re.sub(",", x)
    return x.strip()


def _safe_stats(arr: np.ndarray):
    return (
        float(arr.max()),
        float(arr.mean()),
        float(arr.min()),
        float(arr[0]),
        float(arr[-1]),
    )


def get_feature(df: pl.DataFrame) -> pl.DataFrame:
    out = df.select(["essay_id", "full_text"]).head(1)
    return out


add_f = ["par_len", "par_sentence_cnt", "par_word_cnt"]
tmp = get_feature(df_train)



## === cell 5
print(tmp)



## === cell 6
import matplotlib.pyplot as plt
import seaborn as sns

DO_PLOTS = False

if DO_PLOTS:
    pass




## === cell 7
def add_satis(df_exploded_par: pl.DataFrame, add_f):
    raise RuntimeError(
        "add_satis should not be called with exploded data after optimization."
    )


def add_satis_from_text(essay_ids: np.ndarray, full_texts: np.ndarray, add_f):
    len_col = add_f[0]
    thresholds = np.arange(0, 701, 25).astype(int)

    cols = []
    for fea in add_f:
        cols.extend(
            [
                f"{fea}_max",
                f"{fea}_mean",
                f"{fea}_min",
                f"{fea}_first",
                f"{fea}_last",
            ]
        )
    thr_cols = [f"par_{int(t)}_cnt" for t in thresholds]
    all_cols = ["essay_id"] + cols + thr_cols

    n = len(essay_ids)
    data = {c: np.zeros(n, dtype=np.float32) for c in cols}
    thr_data = {c: np.zeros(n, dtype=np.int32) for c in thr_cols}

    for i in range(n):
        txt = "" if full_texts[i] is None else str(full_texts[i])
        parts = txt.split("\n\n")
        pars = [clean_text_py(p) for p in parts]
        par_len = np.fromiter((len(p) for p in pars), count=len(pars), dtype=np.int32)
        par_sentence_cnt = np.fromiter(
            (p.count(".") + 1 for p in pars), count=len(pars), dtype=np.int32
        )
        par_word_cnt = np.fromiter(
            (p.count(" ") + 1 for p in pars), count=len(pars), dtype=np.int32
        )

        feats = {
            "par_len": par_len.astype(np.float32, copy=False),
            "par_sentence_cnt": par_sentence_cnt.astype(np.float32, copy=False),
            "par_word_cnt": par_word_cnt.astype(np.float32, copy=False),
        }

        for fea in add_f:
            arr = feats[fea]
            mx, me, mn, fi, la = _safe_stats(arr)
            data[f"{fea}_max"][i] = mx
            data[f"{fea}_mean"][i] = me
            data[f"{fea}_min"][i] = mn
            data[f"{fea}_first"][i] = fi
            data[f"{fea}_last"][i] = la

        len_arr = feats[len_col].astype(np.int32, copy=False)
        for t in thresholds:
            thr_data[f"par_{int(t)}_cnt"][i] = int((len_arr >= t).sum())

    out = pd.DataFrame({"essay_id": essay_ids})
    for c in cols:
        out[c] = data[c]
    for c in thr_cols:
        out[c] = thr_data[c]
    return out


train_ids = df_train["essay_id"].to_numpy()
train_texts = df_train["full_text"].to_numpy()
train_df2 = add_satis_from_text(train_ids, train_texts, add_f)

train_scores = (
    df_train.select(["essay_id", "score"])
    .sort("essay_id")
    .to_pandas()["score"]
    .to_numpy()
)
train_df2 = train_df2.sort_values("essay_id").reset_index(drop=True)
train_scores = (
    df_train.select(["essay_id", "score"])
    .sort("essay_id")
    .to_pandas()["score"]
    .to_numpy()
)
train_df2["score"] = train_scores

print("Feature cnt: ", train_df2.shape[1])
train_df2.head(3)




## === cell 8
def Sentence_process(df: pl.DataFrame) -> pl.DataFrame:
    raise RuntimeError("Sentence_process should not be called after optimization.")


sen_fea = ["sen_len", "sen_word_cnt"]


def Sentence_addf(df_exploded_sen: pl.DataFrame, sen_fea):
    raise RuntimeError("Sentence_addf should not be called after optimization.")


def Sentence_addf_from_clean_text(
    essay_ids: np.ndarray, clean_texts: np.ndarray, sen_fea
):
    thr = [15]
    thr.extend(np.arange(50, 301, 50).astype(int))
    thr = [int(x) for x in thr]

    cols = []
    for fea in sen_fea:
        cols.extend(
            [
                f"{fea}_max",
                f"{fea}_mean",
                f"{fea}_min",
                f"{fea}_first",
                f"{fea}_last",
            ]
        )
    thr_cols = [f"sen_{int(t)}_cnt" for t in thr]

    n = len(essay_ids)
    data = {c: np.zeros(n, dtype=np.float32) for c in cols}
    thr_data = {c: np.zeros(n, dtype=np.int32) for c in thr_cols}

    for i in range(n):
        txt = "" if clean_texts[i] is None else str(clean_texts[i])
        sens_all = txt.split(".")
        lens = np.fromiter(
            (len(s) for s in sens_all), count=len(sens_all), dtype=np.int32
        )
        mask = lens >= 15
        if not mask.any():
            continue

        sens = [sens_all[j] for j in np.nonzero(mask)[0]]
        sen_len = lens[mask].astype(np.float32, copy=False)
        sen_word_cnt = np.fromiter(
            (s.count(" ") + 1 for s in sens), count=len(sens), dtype=np.int32
        ).astype(np.float32, copy=False)

        feats = {"sen_len": sen_len, "sen_word_cnt": sen_word_cnt}

        for fea in sen_fea:
            arr = feats[fea]
            mx, me, mn, fi, la = _safe_stats(arr)
            data[f"{fea}_max"][i] = mx
            data[f"{fea}_mean"][i] = me
            data[f"{fea}_min"][i] = mn
            data[f"{fea}_first"][i] = fi
            data[f"{fea}_last"][i] = la

        len_int = feats["sen_len"].astype(np.int32, copy=False)
        for t in thr:
            thr_data[f"sen_{int(t)}_cnt"][i] = int((len_int >= t).sum())

    out = pd.DataFrame({"essay_id": essay_ids})
    for c in cols:
        out[c] = data[c]
    for c in thr_cols:
        out[c] = thr_data[c]
    return out


train_clean_texts = df_train["full_text_clean"].to_numpy()
train_df = train_df2.merge(
    Sentence_addf_from_clean_text(train_ids, train_clean_texts, sen_fea),
    on="essay_id",
    how="left",
)

print("train_df's feature number: ", train_df.shape[1])
train_df.head(3)




## === cell 9
def Word_process(df: pl.DataFrame) -> pl.DataFrame:
    raise RuntimeError("Word_process should not be called after optimization.")


def Word_addf(df_exploded_word: pl.DataFrame):
    raise RuntimeError("Word_addf should not be called after optimization.")


def Word_addf_from_clean_text(essay_ids: np.ndarray, clean_texts: np.ndarray):
    thresholds = np.arange(15).astype(int)  # i -> len >= i+1

    cols = [
        "word_len_max",
        "word_len_mean",
        "word_len_min",
        "word_len_std",
        "word_len_q1",
        "word_len_q2",
        "word_len_q3",
    ] + [f"word_{int(i)}_cnt" for i in thresholds]

    n = len(essay_ids)
    out = {c: np.zeros(n, dtype=np.float32) for c in cols[:7]}
    cnt = {c: np.zeros(n, dtype=np.int32) for c in cols[7:]}

    for i in range(n):
        txt = "" if clean_texts[i] is None else str(clean_texts[i])
        words = txt.split(" ")
        lens = np.fromiter((len(w) for w in words), count=len(words), dtype=np.int32)
        lens = lens[lens != 0]
        if lens.size == 0:
            continue

        lf = lens.astype(np.float32, copy=False)
        out["word_len_max"][i] = float(lf.max())
        out["word_len_mean"][i] = float(lf.mean())
        out["word_len_min"][i] = float(lf.min())
        out["word_len_std"][i] = float(lf.std(ddof=1)) if lf.size > 1 else 0.0
        out["word_len_q1"][i] = float(np.quantile(lf, 0.25, method="linear"))
        out["word_len_q2"][i] = float(np.quantile(lf, 0.5, method="linear"))
        out["word_len_q3"][i] = float(np.quantile(lf, 0.75, method="linear"))

        for t in thresholds:
            cnt[f"word_{int(t)}_cnt"][i] = int((lens >= (int(t) + 1)).sum())

    df_out = pd.DataFrame({"essay_id": essay_ids})
    for c in cols[:7]:
        df_out[c] = out[c]
    for c in cols[7:]:
        df_out[c] = cnt[c]
    return df_out


train_df = train_df.merge(
    Word_addf_from_clean_text(train_ids, train_clean_texts),
    on="essay_id",
    how="left",
)

print("train_df's feature number: ", train_df.shape[1])
train_df.head(3)



## === cell 10
train_clean = df_train["full_text_clean"].to_numpy()
test_clean = df_test["full_text_clean"].to_numpy()



## === cell 11
vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    dtype=np.float32,
    token_pattern=r"(?u)\b\w\w+\b",
)

X_tfid_train = vectorizer.fit_transform(train_clean)
X_tfid_test = vectorizer.transform(test_clean)

print("TF-IDF train shape:", X_tfid_train.shape, "nnz:", X_tfid_train.nnz)
print("TF-IDF test shape :", X_tfid_test.shape, "nnz:", X_tfid_test.nnz)



## === cell 12
train_df = train_df.loc[:, ~train_df.columns.duplicated()]



## === cell 13
num_cols = [c for c in train_df.columns if c not in ["essay_id"]]
for c in num_cols:
    if pd.api.types.is_numeric_dtype(train_df[c]):
        train_df[c] = train_df[c].fillna(0)



## === cell 14
a = 2.94972423
b = 1.092


def quard_w_kappa_sklearn(y_true, y_pred):
    y_true = np.asarray(y_true) + a
    y_pred = (np.asarray(y_pred) + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj_sklearn(y_true, y_pred):
    labels = np.asarray(y_true) + a
    preds = (np.asarray(y_pred) + a).clip(1, 6)
    df_ = preds - labels
    dg = preds - a
    f = 0.5 * np.sum((df_) ** 2)
    g = 0.5 * np.sum((dg) ** 2 + b)
    grad = (df_ / g - f * df_ / (g**2)) * len(labels)
    hess = np.ones(len(labels), dtype=np.float32)
    return grad, hess




## === cell 15
def apply_thresholds(pred_cont, th):
    th = np.asarray(th, dtype=float)
    th = np.sort(th)
    bins = [-np.inf, th[0], th[1], th[2], th[3], th[4], np.inf]
    cls = np.digitize(pred_cont, bins[1:-1], right=False) + 1
    return np.clip(cls, 1, 6).astype(int)


def fit_thresholds(y_true_int, pred_cont):
    y_true_int = np.asarray(y_true_int).astype(int)
    pred_cont = np.asarray(pred_cont).astype(float)

    init = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=float)

    def objective(th):
        y_pred_int = apply_thresholds(pred_cont, th)
        return -cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")

    res = minimize(
        objective,
        init,
        method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4},
    )
    th_best = np.sort(res.x)
    return th_best, -res.fun




## === cell 16
LOAD = False
models = []
fold_num = 5

feature_names_dense = [c for c in train_df.columns if c not in ["essay_id", "score"]]

X_dense_train = sparse.csr_matrix(
    np.ascontiguousarray(
        train_df[feature_names_dense].to_numpy(dtype=np.float32, copy=False)
    )
)
X_train_all = sparse.hstack([X_dense_train, X_tfid_train], format="csr")

y = train_df["score"].to_numpy()


def get_score(ms, X_all, y_true_score, best_itr, val_indices_by_fold):
    for k in range(fold_num):
        val_idx = val_indices_by_fold[k]
        pred = ms[k].predict(X_all[val_idx], num_iteration=best_itr[k]) + a
        y_true = y_true_score[val_idx]
        y_pred = np.clip(np.rint(np.clip(pred, 1, 6)), 1, 6).astype(int)
        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        print(f"model_{k}'s score")
        print("acc: ", acc)
        print("kappa: ", kappa)
        print("=" * 30)


if LOAD:
    path_model = "/kaggle/input/mymodel1/"
    best_itr = [639, 649, 581, 386, 444]
    for i in range(fold_num):
        models.append(lgbm.Booster(model_file=f"{path_model}fold_{i}.txt"))
else:
    best_itr = []
    val_indices_by_fold = []
    kfold = StratifiedKFold(n_splits=fold_num, random_state=0, shuffle=True)

    callbacks = [log_evaluation(period=25)]

    oof_pred_cont = np.zeros(len(y), dtype=np.float32)

    splits = list(kfold.split(np.zeros(len(y)), y.astype(str)))

    for fold_id, (train_idx, val_idx) in tqdm(enumerate(splits), total=fold_num):
        model = lgbm.LGBMRegressor(
            objective=qwk_obj_sklearn,
            metrics="None",
            learning_rate=0.08,
            max_depth=5,
            num_leaves=10,
            colsample_bytree=0.5,
            reg_alpha=0.1,
            reg_lambda=0.8,
            n_estimators=1024,
            random_state=0,
            verbosity=-1,
            n_jobs=N_JOBS,
        )

        X_tr = X_train_all[train_idx]
        y_tr = y[train_idx] - a
        X_va = X_train_all[val_idx]
        y_va = y[val_idx] - a

        print(f'{fold_id+1}_Fold Training {"="*50}')

        lgbm_model = model.fit(
            X_tr,
            y_tr,
            eval_names=["val"],
            eval_set=[(X_va, y_va)],
            eval_metric=quard_w_kappa_sklearn,
            callbacks=callbacks,
        )

        best_itr.append(
            lgbm_model.best_iteration_
            if lgbm_model.best_iteration_ is not None
            else 1024
        )
        models.append(lgbm_model.booster_)
        val_indices_by_fold.append(val_idx)
        lgbm_model.booster_.save_model(f"fold_{fold_id}.txt")

        oof_pred_cont[val_idx] = (
            models[-1].predict(X_va, num_iteration=best_itr[-1]) + a
        ).astype(np.float32)

    get_score(models, X_train_all, y, best_itr, val_indices_by_fold)

    thresholds, oof_kappa = fit_thresholds(
        y_true_int=y.astype(int), pred_cont=oof_pred_cont
    )
    print("Fitted thresholds:", thresholds)
    print("OOF QWK after thresholding:", oof_kappa)



## === cell 17
test_ids = df_test["essay_id"].to_numpy()
test_texts = df_test["full_text"].to_numpy()
test_clean_texts = df_test["full_text_clean"].to_numpy()

test_feats = add_satis_from_text(test_ids, test_texts, add_f)
test_feats = test_feats.merge(
    Sentence_addf_from_clean_text(test_ids, test_clean_texts, sen_fea),
    on="essay_id",
    how="left",
)
test_feats = test_feats.merge(
    Word_addf_from_clean_text(test_ids, test_clean_texts),
    on="essay_id",
    how="left",
)

test_feats = test_feats.sort_values("essay_id").reset_index(drop=True)

for c in feature_names_dense:
    if c not in test_feats.columns:
        test_feats[c] = 0.0

extra_cols = [
    c for c in test_feats.columns if c not in (["essay_id"] + feature_names_dense)
]
if len(extra_cols) > 0:
    test_feats = test_feats.drop(columns=extra_cols)

for c in feature_names_dense:
    if pd.api.types.is_numeric_dtype(test_feats[c]):
        test_feats[c] = test_feats[c].fillna(0)

X_dense_test = sparse.csr_matrix(
    np.ascontiguousarray(
        test_feats[feature_names_dense].to_numpy(dtype=np.float32, copy=False)
    )
)
X_test_all = sparse.hstack([X_dense_test, X_tfid_test], format="csr")

print("Dense engineered feature count:", len(feature_names_dense))
print("Final X_test_all shape:", X_test_all.shape)



## === cell 18
prediction = test_feats[["essay_id"]].copy()

pred_sum = np.zeros(X_test_all.shape[0], dtype=np.float64)
for i in range(fold_num):
    pred_sum += models[i].predict(X_test_all, num_iteration=best_itr[i]) + a
pred_test = pred_sum / fold_num

print(pred_test[:10])



## === cell 19
pred_test = np.clip(pred_test, 1, 6)

if "thresholds" in globals() and thresholds is not None and len(thresholds) == 5:
    pred_label = apply_thresholds(pred_test, thresholds)
else:
    pred_label = np.rint(pred_test).astype(int)

pred_label = np.clip(pred_label, 1, 6).astype(int)

prediction["score"] = pred_label
prediction.to_csv("submission.csv", index=False)

print(prediction.head(3))
print("Wrote submission.csv with shape:", prediction.shape)
print("Columns:", prediction.columns.tolist())
