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

# 5. Target score

0.8034006002900264

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.67901) has done: 'I fix the Polars/Pandas mismatch that causes `.apply()` to fail by explicitly converting the needed text columns to Python lists (or Pandas Series) before TF-IDF. I also remove the dependency on missing pre-trained model files by switching `LOAD` to `False` so the script trains the LightGBM models in-notebook and then predicts. To keep the core logic unchanged, I not alter the feature engineering, model objective, or CV training loop—only make type/IO fixes and ensure test features align to train features. Finally, I ensure a valid `submission.csv` with the required `essay_id,score` columns is written.'
- What this solution (achieved 0.68651) has done: 'Your score gap to the target is large (0.67901 vs 0.8034), so we need a meaningful but still core-logic-preserving improvement. The biggest issue is that you’re converting the TF‑IDF sparse matrix to a dense DataFrame (`toarray()`), which both hurts performance and effectively pushes the model to rely on a tiny set of very frequent ngrams due to `min_df=0.05`; we keep TF‑IDF but keep it sparse and stack it with the engineered numeric features, training LightGBM on the combined matrix. We also remove early stopping callbacks (they were disallowed by your requirements) while keeping the same training loop, objective, and model hyperparameters. Finally, we ensure test features are aligned in the exact same way and still write a valid `submission.csv` with `essay_id,score`.'
- What this solution (achieved 0.6873) has done: 'We keep your architecture and training loop intact, but align the TF‑IDF preprocessing with what the vectorizer expects: right now you pass full cleaned strings while forcing `tokenizer=lambda x: x`, which makes the vectorizer treat each character as a token and severely hurts QWK. The minimal fix is to tokenize the cleaned text into word tokens (lists of strings) so the existing TF‑IDF settings behave correctly. We also make the tokenization deterministic and keep all paths and submission schema unchanged. This should move the score substantially upward toward the 0.8034 target without changing the model, loss, or CV procedure.'

# 9. Code solution

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
def get_feature(df: pl.DataFrame) -> pl.DataFrame:
    x = (
        df.lazy()
        .select(["essay_id", "full_text"])
        .with_columns(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))
        .explode("paragraph")
        .with_columns(clean_expr("paragraph").alias("paragraph"))
        .with_columns(
            pl.col("paragraph").str.len_chars().alias("par_len"),
            (pl.col("paragraph").str.count_matches(r"\.") + 1).alias(
                "par_sentence_cnt"
            ),
            (pl.col("paragraph").str.count_matches(r" ") + 1).alias("par_word_cnt"),
        )
        .select(
            ["essay_id", "paragraph", "par_len", "par_sentence_cnt", "par_word_cnt"]
        )
        .collect(streaming=True)
    )
    return x


add_f = ["par_len", "par_sentence_cnt", "par_word_cnt"]
tmp = get_feature(df_train)




## === cell 5
print(tmp.select(add_f).describe())




## === cell 6
import matplotlib.pyplot as plt
import seaborn as sns

DO_PLOTS = False

if DO_PLOTS:
    tmp_pd = tmp.select(add_f).to_pandas()
    f, ax = plt.subplots(3, 1, figsize=(13, 15))
    for i, col in enumerate(add_f):
        sns.histplot(data=tmp_pd, x=col, bins=100, kde=True, ax=ax[i])
        ax[i].set_title(f"Distribution: {col}")
    plt.show()




## === cell 7
def add_satis(df_exploded_par: pl.DataFrame, add_f):
    len_col = add_f[0]
    thresholds = np.arange(0, 701, 25)

    aggs = [
        pl.col("paragraph")
        .filter(pl.col(len_col) >= int(i))
        .count()
        .alias(f"par_{int(i)}_cnt")
        for i in thresholds
    ]
    for fea in add_f:
        aggs.extend(
            [
                pl.col(fea).max().alias(f"{fea}_max"),
                pl.col(fea).mean().alias(f"{fea}_mean"),
                pl.col(fea).min().alias(f"{fea}_min"),
                pl.col(fea).first().alias(f"{fea}_first"),
                pl.col(fea).last().alias(f"{fea}_last"),
            ]
        )

    out = (
        df_exploded_par.lazy()
        .group_by(["essay_id"], maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .collect(streaming=True)
    )
    return out.to_pandas()


train_df2 = add_satis(tmp, add_f)

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
    x = (
        df.lazy()
        .select(["essay_id", "full_text_clean"])
        .with_columns(pl.col("full_text_clean").str.split(by=".").alias("sen"))
        .explode("sen")
        .with_columns(pl.col("sen").str.len_chars().alias("sen_len"))
        .filter(pl.col("sen_len") >= 15)
        .with_columns((pl.col("sen").str.count_matches(r" ") + 1).alias("sen_word_cnt"))
        .select(["essay_id", "sen", "sen_len", "sen_word_cnt"])
        .collect(streaming=True)
    )
    return x


sen_fea = ["sen_len", "sen_word_cnt"]


def Sentence_addf(df_exploded_sen: pl.DataFrame, sen_fea):
    thr = [15]
    thr.extend(np.arange(50, 301, 50))

    aggs = [
        pl.col("sen")
        .filter(pl.col("sen_len") >= int(i))
        .count()
        .alias(f"sen_{int(i)}_cnt")
        for i in thr
    ]
    for fea in sen_fea:
        aggs.extend(
            [
                pl.col(fea).max().alias(f"{fea}_max"),
                pl.col(fea).mean().alias(f"{fea}_mean"),
                pl.col(fea).min().alias(f"{fea}_min"),
                pl.col(fea).first().alias(f"{fea}_first"),
                pl.col(fea).last().alias(f"{fea}_last"),
            ]
        )

    out = (
        df_exploded_sen.lazy()
        .group_by(["essay_id"], maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .collect(streaming=True)
    )
    return out.to_pandas()


tmp_sen = Sentence_process(df_train)
train_df = train_df2.merge(Sentence_addf(tmp_sen, sen_fea), on="essay_id", how="left")

print("train_df's feature number: ", train_df.shape[1])
train_df.head(3)




## === cell 9
def Word_process(df: pl.DataFrame) -> pl.DataFrame:
    x = (
        df.lazy()
        .select(["essay_id", "full_text_clean"])
        .with_columns(pl.col("full_text_clean").str.split(by=" ").alias("word"))
        .explode("word")
        .with_columns(pl.col("word").str.len_chars().alias("word_len"))
        .filter(pl.col("word_len") != 0)
        .select(["essay_id", "word", "word_len"])
        .collect(streaming=True)
    )
    return x


def Word_addf(df_exploded_word: pl.DataFrame):
    aggs = [
        pl.col("word")
        .filter(pl.col("word_len") >= int(i) + 1)
        .count()
        .alias(f"word_{int(i)}_cnt")
        for i in np.arange(15)
    ]
    aggs.extend(
        [
            pl.col("word_len").max().alias("word_len_max"),
            pl.col("word_len").mean().alias("word_len_mean"),
            pl.col("word_len").min().alias("word_len_min"),
            pl.col("word_len").std().alias("word_len_std"),
            pl.col("word_len").quantile(0.25).alias("word_len_q1"),
            pl.col("word_len").quantile(0.5).alias("word_len_q2"),
            pl.col("word_len").quantile(0.75).alias("word_len_q3"),
        ]
    )
    out = (
        df_exploded_word.lazy()
        .group_by(["essay_id"], maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .collect(streaming=True)
    )
    return out.to_pandas()


tmp_word = Word_process(df_train)
train_df = train_df.merge(Word_addf(tmp_word), on="essay_id", how="left")

print("train_df's feature number: ", train_df.shape[1])
train_df.head(3)




## === cell 10
train_clean = df_train["full_text_clean"].to_list()
test_clean = df_test["full_text_clean"].to_list()




## === cell 11
vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    dtype=np.float32,
    n_jobs=N_JOBS,
)

X_tfid_train = vectorizer.fit_transform(train_clean)
X_tfid_test = vectorizer.transform(test_clean)

print("TF-IDF train shape:", X_tfid_train.shape, "nnz:", X_tfid_train.nnz)
print("TF-IDF test shape :", X_tfid_test.shape, "nnz:", X_tfid_test.nnz)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3575229961.py in <cell line: 0>()
      1 # Speed: use default tokenization (word analyzer) rather than python-level list tokens.
      2 # Also enable parallelism in TF-IDF building.
----> 3 vectorizer = TfidfVectorizer(
      4     strip_accents="unicode",
      5     analyzer="word",

TypeError: TfidfVectorizer.__init__() got an unexpected keyword argument 'n_jobs'

## === cell 12
train_df = train_df.loc[:, ~train_df.columns.duplicated()]




## === cell 13
num_cols = [c for c in train_df.columns if c not in ["essay_id"]]
for c in num_cols:
    if pd.api.types.is_numeric_dtype(train_df[c]):
        train_df[c] = train_df[c].fillna(0)




## === cell 14
def quard_w_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = (y_pred + a).clip(1, 6)
    df_ = preds - labels
    dg = preds - a
    f = 1 / 2 * np.sum((df_) ** 2)
    g = 1 / 2 * np.sum((dg) ** 2 + b)
    grad = (df_ / g - f * df_ / g**2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess


a = 2.94972423
b = 1.092




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
    train_df[feature_names_dense].to_numpy(dtype=np.float32, copy=False)
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
            objective=qwk_obj,
            metrics="None",
            learning_rate=0.08,
            max_depth=5,
            num_leaves=10,
            colsample_bytree=0.5,
            reg_alpha=0.1,
            reg_lambda=0.8,
            n_estimators=1024,
            class_weight="balanced",
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
            eval_metric=quard_w_kappa,
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




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1801500790.py in <cell line: 0>()
      9     train_df[feature_names_dense].to_numpy(dtype=np.float32, copy=False)
     10 )
---> 11 X_train_all = sparse.hstack([X_dense_train, X_tfid_train], format="csr")
     12 
     13 y = train_df["score"].to_numpy()

NameError: name 'X_tfid_train' is not defined

## === cell 17
tmp = get_feature(df_test)
test_feats = add_satis(tmp, add_f)

tmp = Sentence_process(df_test)
test_feats = test_feats.merge(Sentence_addf(tmp, sen_fea), on="essay_id", how="left")

tmp = Word_process(df_test)
test_feats = test_feats.merge(Word_addf(tmp), on="essay_id", how="left")

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
    test_feats[feature_names_dense].to_numpy(dtype=np.float32, copy=False)
)
X_test_all = sparse.hstack([X_dense_test, X_tfid_test], format="csr")

print("Dense engineered feature count:", len(feature_names_dense))
print("Final X_test_all shape:", X_test_all.shape)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2455235738.py in <cell line: 0>()
     25     test_feats[feature_names_dense].to_numpy(dtype=np.float32, copy=False)
     26 )
---> 27 X_test_all = sparse.hstack([X_dense_test, X_tfid_test], format="csr")
     28 
     29 print("Dense engineered feature count:", len(feature_names_dense))

NameError: name 'X_tfid_test' is not defined

## === cell 18
prediction = test_feats[["essay_id"]].copy()

pred_sum = np.zeros(X_test_all.shape[0], dtype=np.float64)
for i in range(fold_num):
    pred_sum += models[i].predict(X_test_all) + a
pred_test = pred_sum / fold_num

print(pred_test[:10])




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2174573943.py in <cell line: 0>()
      2 
      3 # Speed: vectorized fold averaging without Python loop overhead on intermediate arrays.
----> 4 pred_sum = np.zeros(X_test_all.shape[0], dtype=np.float64)
      5 for i in range(fold_num):
      6     pred_sum += models[i].predict(X_test_all) + a

NameError: name 'X_test_all' is not defined

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

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/705823136.py in <cell line: 0>()
----> 1 pred_test = np.clip(pred_test, 1, 6)
      2 
      3 if "thresholds" in globals() and thresholds is not None and len(thresholds) == 5:
      4     pred_label = apply_thresholds(pred_test, thresholds)
      5 else:

NameError: name 'pred_test' is not defined
