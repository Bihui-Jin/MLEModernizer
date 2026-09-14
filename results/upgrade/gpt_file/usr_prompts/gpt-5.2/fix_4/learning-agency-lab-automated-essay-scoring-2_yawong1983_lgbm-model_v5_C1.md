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

0.6873

# 7. Whether higher score is better

Higher is better.

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
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, accuracy_score

from scipy import sparse

import warnings

warnings.filterwarnings("ignore")



## === cell 1
path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"


def load_db(name):
    columns = [(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))]
    df = pl.read_csv(path + name + ".csv").with_columns(columns)
    print(f"< {name} DataFrame Info >")
    print(df.head(1))
    return df




## === cell 2
df_train = load_db("train")
df_test = load_db("test")



## === cell 3
np.random.seed(0)




## === cell 4
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


def get_feature(x):
    x = x.explode("paragraph")
    x = x.with_columns(pl.col("paragraph").map_elements(Data_clearning))
    x = x.with_columns(
        pl.col("paragraph").map_elements(lambda t: len(t)).alias("par_len")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda t: len(t.split(".")))
        .alias("par_sentence_cnt")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda t: len(t.split(" ")))
        .alias("par_word_cnt")
    )
    return x




## === cell 5
tmp = get_feature(df_train)
add_f = ["par_len", "par_sentence_cnt", "par_word_cnt"]
tmp = tmp.to_pandas()



## === cell 6
print(tmp.describe())



## === cell 7
import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(3, 1, figsize=(13, 15))
for i, col in enumerate(add_f):
    sns.histplot(data=tmp, x=col, bins=100, kde=True, ax=ax[i])
    ax[i].set_title(f"Distribution: {col}")
plt.show()




## === cell 8
def add_satis(df, add_f):
    aggs = [
        *[
            pl.col("paragraph")
            .filter(pl.col(add_f[0]) >= i)
            .count()
            .alias(f"par_{i}_cnt")
            for i in np.arange(0, 701, 25)
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in add_f],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in add_f],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in add_f],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in add_f],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in add_f],
    ]
    df = df.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df


tmp = get_feature(df_train)
train_df2 = add_satis(tmp, add_f)

train_df2["score"] = df_train.sort("essay_id")["score"].to_list()

print("Feature cnt: ", train_df2.shape[1])
train_df2.head(3)




## === cell 9
def Sentence_process(df):
    df = df.with_columns(
        pl.col("full_text").map_elements(Data_clearning).str.split(by=".").alias("sen")
    )
    df = df.explode("sen")
    df = df.with_columns(pl.col("sen").map_elements(lambda t: len(t)).alias("sen_len"))

    sns.histplot(x="sen_len", data=df.to_pandas(), stat="percent", bins=100)
    plt.show()

    df = df.filter(pl.col("sen_len") >= 15)
    df = df.with_columns(
        pl.col("sen").map_elements(lambda t: len(t.split(" "))).alias("sen_word_cnt")
    )
    return df


sen_fea = ["sen_len", "sen_word_cnt"]


def Sentence_addf(df, sen_fea):
    _ = [15]
    _.extend(np.arange(50, 301, 50))
    aggs = [
        *[
            pl.col("sen").filter(pl.col("sen_len") >= i).count().alias(f"sen_{i}_cnt")
            for i in _
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in sen_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sen_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in sen_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in sen_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in sen_fea],
    ]
    df = df.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df


tmp = Sentence_process(df_train)
train_df = train_df2.merge(Sentence_addf(tmp, sen_fea), on="essay_id", how="left")

print("train_df's feature number: ", train_df.shape[1])
train_df.head(3)




## === cell 10
def Word_process(df):
    df = df.with_columns(
        pl.col("full_text").map_elements(Data_clearning).str.split(by=" ").alias("word")
    )
    df = df.explode("word")
    df = df.with_columns(
        pl.col("word").map_elements(lambda t: len(t)).alias("word_len")
    )

    sns.histplot(x="word_len", data=df.to_pandas(), stat="percent", bins=100)
    plt.show()

    df = df.filter(pl.col("word_len") != 0)
    return df


def Word_addf(df):
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i + 1)
            .count()
            .alias(f"word_{i}_cnt")
            for i in np.arange(15)
        ],
        pl.col("word_len").max().alias("word_len_max"),
        pl.col("word_len").mean().alias("word_len_mean"),
        pl.col("word_len").min().alias("word_len_min"),
        pl.col("word_len").std().alias("word_len_std"),
        pl.col("word_len").quantile(0.25).alias("word_len_q1"),
        pl.col("word_len").quantile(0.5).alias("word_len_q2"),
        pl.col("word_len").quantile(0.75).alias("word_len_q3"),
    ]
    df = df.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df


tmp = Word_process(df_train)
train_df = train_df.merge(Word_addf(tmp), on="essay_id", how="left")

print("train_df's feature number: ", train_df.shape[1])
train_df.head(3)



## === cell 11
_token_re = re.compile(r"[a-z]+(?:'[a-z]+)?")  # deterministic simple word tokenizer


def to_tokens(txt: str):
    txt = Data_clearning(txt)
    return _token_re.findall(txt)


train_text = [to_tokens(x) for x in df_train["full_text"].to_list()]
test_text = [to_tokens(x) for x in df_test["full_text"].to_list()]



## === cell 12
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 3),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,
)

X_tfid_train = vectorizer.fit_transform(train_text)
X_tfid_test = vectorizer.transform(test_text)

print("TF-IDF train shape:", X_tfid_train.shape, "nnz:", X_tfid_train.nnz)
print("TF-IDF test shape :", X_tfid_test.shape, "nnz:", X_tfid_test.nnz)



## === cell 13
train_df = train_df.loc[:, ~train_df.columns.duplicated()].copy()



## === cell 14
num_cols = [c for c in train_df.columns if c not in ["essay_id"]]
train_df[num_cols] = train_df[num_cols].apply(pd.to_numeric, errors="ignore")
for c in train_df.columns:
    if c not in ["essay_id"] and pd.api.types.is_numeric_dtype(train_df[c]):
        train_df[c] = train_df[c].fillna(0)




## === cell 15
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



## === cell 16
LOAD = False
models = []
fold_num = 5

feature_names_dense = [c for c in train_df.columns if c not in ["essay_id", "score"]]
X_dense_train = train_df[feature_names_dense].astype(np.float32).values
X_dense_train = sparse.csr_matrix(X_dense_train)
X_train_all = sparse.hstack([X_dense_train, X_tfid_train], format="csr")

y = train_df["score"].values


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

    for fold_id, (train_idx, val_idx) in tqdm(
        enumerate(kfold.split(np.zeros(len(y)), y.astype(str))), total=fold_num
    ):
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
        )

        X_tr = X_train_all[train_idx]
        y_tr = y[train_idx] - a
        X_va = X_train_all[val_idx]
        y_va = y[val_idx] - a

        print(f'{fold_id+1}_Fold Training {"="*50}')

        lgbm_model = model.fit(
            X_tr,
            y_tr,
            eval_names=["train", "val"],
            eval_set=[(X_tr, y_tr), (X_va, y_va)],
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

    get_score(models, X_train_all, y, best_itr, val_indices_by_fold)



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

X_dense_test = test_feats[feature_names_dense].astype(np.float32).values
X_dense_test = sparse.csr_matrix(X_dense_test)
X_test_all = sparse.hstack([X_dense_test, X_tfid_test], format="csr")

print("Dense engineered feature count:", len(feature_names_dense))
print("Final X_test_all shape:", X_test_all.shape)



## === cell 18
prediction = test_feats[["essay_id"]].copy()

pred_test = models[0].predict(X_test_all) + a
for i in range(1, fold_num):
    pred_test = pred_test + (models[i].predict(X_test_all) + a)
pred_test = pred_test / fold_num

print(pred_test[:10])



## === cell 19
pred_test = np.clip(pred_test, 1, 6)
pred_test = np.rint(pred_test).astype(int)

prediction["score"] = pred_test
prediction.to_csv("submission.csv", index=False)

print(prediction.head(3))
print("Wrote submission.csv with shape:", prediction.shape)
print("Columns:", prediction.columns.tolist())
