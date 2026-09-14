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

catboost==1.2.8
cudf-polars-cu12==25.6.0
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
optuna==4.5.0
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

0.8003551363465263

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.02059) has done: 'Your code doesn’t currently guarantee a valid submission because the provided notebook starts at `cell 0` (your runner expects sequential cells), and it also risks a silent feature-name collision bug in `Word_aggregation` where word-length count features are mistakenly named `sentence_{i}_cnt`. I (1) reindex cells to start at 1 so it runs end-to-end, (2) fix that word-feature column naming collision (this is a real modeling bug and should improve QWK without changing the overall approach), and (3) ensure train/test feature columns are aligned deterministically before prediction so the submission rows map correctly to `essay_id`. These are minimal changes that preserve your model/training loop and should move the score upward toward your target.'

# 9. Code solution

## === cell 0
try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(x):
        print(x)


import os
import gc
import ctypes
import random
import time
import string
import re
import pickle
from typing import List

import pandas as pd, numpy as np
import polars as pl

import matplotlib.pyplot as plt

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LinearRegression

import catboost
from catboost import CatBoostRegressor, Pool

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




## === cell 1
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass




## === cell 2
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None
    LOAD_FEATURES_FROM = None
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 3
if not os.path.exists(CFG.BASE_PATH):
    CFG.BASE_PATH = "/kaggle/input/"
print("BASE_PATH:", CFG.BASE_PATH)




## === cell 4
Clean = True


def clean_memory():
    if Clean:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        gc.collect()


clean_memory()




## === cell 5
def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()




## === cell 6
df_train = pd.read_csv(os.path.join(CFG.BASE_PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(CFG.BASE_PATH, "test.csv"))

print("Shape of Train: ", df_train.shape)
display(df_train.head())
print("Shape of Test: ", df_test.shape)
display(df_test.head())

train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)

schema_train = train.schema
schema_test = test.schema




## === cell 7
_HTML_RE = re.compile(r"<.*?>")
_AT_RE = re.compile(r"@\w+")
_APOS_NUM_RE = re.compile(r"'\d+")
_NUM_RE = re.compile(r"\d+")
_HTTP_RE = re.compile(r"http\w+")
_WS_RE = re.compile(r"\s+")
_DOTS_RE = re.compile(r"\.+")
_COMMAS_RE = re.compile(r"\,+")


def removeHTML(x: str) -> str:
    return _HTML_RE.sub(r"", x)


def dataPreprocessing(x: str) -> str:
    x = x.lower()
    x = removeHTML(x)
    x = _AT_RE.sub("", x)
    x = _APOS_NUM_RE.sub("", x)
    x = _NUM_RE.sub("", x)
    x = _HTTP_RE.sub("", x)
    x = _WS_RE.sub(" ", x)
    x = _DOTS_RE.sub(".", x)
    x = _COMMAS_RE.sub(".", x)
    x = x.strip()
    return x


def dataPreprocessing2(x: str) -> str:
    x = x.lower()
    x = removeHTML(x)
    x = _AT_RE.sub("", x)
    x = _HTTP_RE.sub("", x)
    x = _WS_RE.sub(" ", x)
    x = x.strip()
    return x




## === cell 8
def _polars_preprocess_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"<.*?>", "")
        .str.replace_all(r"@\w+", "")
        .str.replace_all(r"'\d+", "")
        .str.replace_all(r"\d+", "")
        .str.replace_all(r"http\w+", "")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"\.+", ".")
        .str.replace_all(r"\,+", ".")
        .str.strip_chars()
    )


def _polars_preprocess2_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"<.*?>", "")
        .str.replace_all(r"@\w+", "")
        .str.replace_all(r"http\w+", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def add_clean_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        _polars_preprocess_expr("full_text").alias("full_text_clean"),
        _polars_preprocess2_expr("full_text").alias("full_text_clean2"),
    )


train = add_clean_columns(train)
test = add_clean_columns(test)




## === cell 9
paragraph_features = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]


def Paragraph_Features(x: pl.DataFrame) -> pl.DataFrame:
    x = x.explode("paragraph")
    x = x.with_columns(_polars_preprocess_expr("paragraph").alias("paragraph"))
    x = x.with_columns(
        pl.col("paragraph").str.len_chars().alias("paragraph_len"),
        (pl.col("paragraph").str.count_matches(r"\.") + 1).alias(
            "paragraph_sentence_cnt"
        ),
        (pl.col("paragraph").str.count_matches(" ") + 1).alias("paragraph_word_cnt"),
    )
    return x


def Paragraph_aggregation(x: pl.DataFrame) -> pd.DataFrame:
    aggs = [
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_len") >= i)
            .count()
            .alias(f"paragraph_{i}_cnt")
            for i in [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
        ],
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in paragraph_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in paragraph_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in paragraph_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in paragraph_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in paragraph_features],
    ]
    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()




## === cell 10
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(x: pl.DataFrame) -> pl.DataFrame:
    x = x.with_columns(pl.col("full_text_clean").str.split(".").alias("sentence"))
    x = x.explode("sentence")
    x = x.with_columns(
        pl.col("sentence").str.len_chars().alias("sentence_len"),
        (pl.col("sentence").str.count_matches(" ") + 1).alias("sentence_word_cnt"),
    )
    return x


def Sentence_aggregation(x: pl.DataFrame) -> pd.DataFrame:
    aggs = [
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") >= i)
            .count()
            .alias(f"sentence_{i}_cnt")
            for i in [5, 10, 15, 25, 30, 40, 50, 60, 70, 80, 90, 100]
        ],
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in sentence_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in sentence_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in sentence_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in sentence_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in sentence_features],
    ]
    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()




## === cell 11
word_features = ["word_len"]


def Word_Features(x: pl.DataFrame) -> pl.DataFrame:
    x = x.with_columns(pl.col("full_text_clean").str.split(" ").alias("word"))
    x = x.explode("word")
    x = x.with_columns(pl.col("word").str.len_chars().alias("word_len"))
    return x


def Word_aggregation(x: pl.DataFrame) -> pd.DataFrame:
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i)
            .count()
            .alias(f"word_{i}_cnt")
            for i in [2, 3, 4, 5, 6]
        ],
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in word_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in word_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in word_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in word_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in word_features],
    ]
    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()




## === cell 12
lexical_features = ["unique_word_text", "num_puncts_text"]


def Lexical_Features(x: pl.DataFrame) -> pl.DataFrame:
    punct_chars = re.escape(string.punctuation)
    x = x.with_columns(
        pl.col("full_text_clean2")
        .str.count_matches(f"[{punct_chars}]")
        .cast(pl.Int64)
        .alias("num_puncts_text"),
        pl.col("full_text_clean2")
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        .list.unique()
        .list.len()
        .cast(pl.Int64)
        .alias("unique_word_text"),
    )
    return x


def lexical_aggregation(x: pl.DataFrame) -> pd.DataFrame:
    aggs = [
        *[
            pl.col("full_text_clean2")
            .filter(pl.col("unique_word_text") >= i)
            .count()
            .alias(f"unique_word_{i}_cnt")
            for i in [50, 100, 125, 150, 175, 200]
        ],
        *[
            pl.col("full_text_clean2")
            .filter(pl.col("num_puncts_text") >= i)
            .count()
            .alias(f"num_puncts_{i}_cnt")
            for i in [20, 30, 35, 40, 45, 50, 60]
        ],
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in lexical_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in lexical_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in lexical_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in lexical_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in lexical_features],
    ]
    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()




## === cell 13
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 3),
    min_df=0.10,
    max_df=0.95,
    max_features=30000,
    sublinear_tf=True,
)

train_tokens: List[List[str]] = [
    s.split(" ") for s in train["full_text_clean"].to_list()
]
train_tfid = vectorizer.fit_transform(train_tokens)
train_essay_id = df_train["essay_id"].values




## === cell 14
if CFG.LOAD_FEATURES_FROM is None:
    print("Build train_feats from scratch (optimized)")
    t0 = time.time()

    train_feats1 = Paragraph_aggregation(
        Paragraph_Features(train.select(["essay_id", "paragraph"]))
    )
    train_feats2 = Sentence_aggregation(
        Sentence_Features(train.select(["essay_id", "full_text_clean"]))
    )
    train_feats3 = Word_aggregation(
        Word_Features(train.select(["essay_id", "full_text_clean"]))
    )
    train_feats4 = lexical_aggregation(
        Lexical_Features(train.select(["essay_id", "full_text_clean2"]))
    )

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats4, on="essay_id", how="left")
    train_feats = train_feats.merge(
        df_train[["essay_id", "score"]], on="essay_id", how="left"
    )

    if train_feats.columns.duplicated().any():
        train_feats = train_feats.loc[:, ~train_feats.columns.duplicated()].copy()

    print(
        "train_feats shape:", train_feats.shape, "time(s):", round(time.time() - t0, 1)
    )
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Load train_feats.csv")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)

display(train_feats.head())




## === cell 15
a = 2.948
b = 1.092


class QWKObjective:
    def calc_ders_range(self, approxes, targets, weights):
        preds = np.asarray(approxes, dtype=np.float64) + a
        labels = np.asarray(targets, dtype=np.float64) + a
        preds = np.clip(preds, 1, 6)

        f = 0.5 * np.sum((preds - labels) ** 2)
        g = 0.5 * np.sum((preds - a) ** 2 + b)

        df_ = preds - labels
        dg = preds - a
        grad = (df_ / g - f * dg / (g**2)) * len(labels)
        hess = np.ones_like(grad)

        return list(zip(grad, hess))


class QWKMetric:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, targets, weights):
        preds = np.asarray(approxes[0], dtype=np.float64) + a
        labels = np.asarray(targets, dtype=np.float64) + a
        preds = np.clip(preds, 1, 6).round()
        qwk = cohen_kappa_score(labels, preds, weights="quadratic")
        return qwk, 1.0


qwk_obj = QWKObjective()
quadratic_weighted_kappa = QWKMetric()




## === cell 16
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"
print("n_features:", len(FEATURES))
print("Catboost Version: ", catboost.__version__)




## === cell 17
from scipy import sparse


def _catboost_task_type():
    return "GPU"


def _make_numeric_matrix_fast(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    X = df.reindex(columns=cols)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    X = np.clip(X.to_numpy(dtype=np.float32, copy=False), 0, 10000)
    return X


def catboost_train():
    oof_pred = np.zeros(len(train_feats), dtype=np.float32)

    X_dense_all = _make_numeric_matrix_fast(train_feats, FEATURES)
    y_all = train_feats[TARGET].astype(float).to_numpy()

    X_all = sparse.hstack([sparse.csr_matrix(X_dense_all), train_tfid], format="csr")

    skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
    for i, (train_index, valid_index) in enumerate(skf.split(train_feats, y_all)):
        print("#" * 25)
        print(f"### Fold {i+1}")
        print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
        print("#" * 25)

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=5,
            task_type=_catboost_task_type(),
            loss_function=qwk_obj,
            eval_metric=quadratic_weighted_kappa,
            random_seed=CFG.SEED,
            allow_writing_files=False,
            thread_count=-1,
            bootstrap_type="Bayesian",
            od_type="Iter",
            od_wait=200,
            verbose=100,
        )

        train_pool = Pool(data=X_all[train_index], label=y_all[train_index])
        valid_pool = Pool(data=X_all[valid_index], label=y_all[valid_index])

        try:
            model.fit(train_pool, eval_set=valid_pool, use_best_model=False)
        except Exception as e:
            print("Fit failed, retrying with CPU. Error:", repr(e))
            model = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=5,
                task_type="CPU",
                loss_function=qwk_obj,
                eval_metric=quadratic_weighted_kappa,
                random_seed=CFG.SEED,
                allow_writing_files=False,
                thread_count=-1,
                bootstrap_type="Bayesian",
                od_type="Iter",
                od_wait=200,
                verbose=100,
            )
            model.fit(train_pool, eval_set=valid_pool, use_best_model=False)

        pickle.dump(model, open(f"CAT_v{CFG.VER}_f{i}.pkl", "wb"))

        oof = model.predict(valid_pool)
        oof_pred[valid_index] = oof.astype(np.float32)

        del train_pool, valid_pool, oof, model
        clean_memory()

    cv = cohen_kappa_score(y_all, np.clip(oof_pred, 1, 6).round(), weights="quadratic")
    print("CV Score for Catboost = ", cv)

    lr = LinearRegression()
    lr.fit(oof_pred.reshape(-1, 1), y_all)
    cal_oof = lr.predict(oof_pred.reshape(-1, 1))
    cv_cal = cohen_kappa_score(
        y_all, np.clip(cal_oof, 1, 6).round(), weights="quadratic"
    )
    print("CV Score after linear calibration = ", cv_cal)

    with open(f"CAL_v{CFG.VER}.pkl", "wb") as f:
        pickle.dump(lr, f)




## === cell 18
if CFG.LOAD_MODELS_FROM is None:
    print("Training CatBoost")
    catboost_train()




## === cell 19
try:
    model = pickle.load(open(f"CAT_v{CFG.VER}_f0.pkl", "rb"))

    try:
        fi = model.get_feature_importance()
        df_importance = pd.DataFrame(
            {"feature_idx": np.arange(len(fi)), "importance": fi}
        ).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(12, 6))
        plt.bar(
            x=df_importance.head(30)["feature_idx"].astype(str),
            height=df_importance.head(30)["importance"],
            color="pink",
            edgecolor="black",
        )
        plt.title("Distribution of Feature Importance of Catboost (top indices)")
        plt.xticks(rotation=90)
        plt.show()
    except Exception as e:
        print("Skipping importance plot:", repr(e))
except Exception as e:
    print("Skipping importance plot:", repr(e))




## === cell 20
test_tokens: List[List[str]] = [s.split(" ") for s in test["full_text_clean"].to_list()]
test_tfid = vectorizer.transform(test_tokens)
test_essay_id = df_test["essay_id"].values




## === cell 21
t0 = time.time()
test_feats1 = Paragraph_aggregation(
    Paragraph_Features(test.select(["essay_id", "paragraph"]))
)
test_feats2 = Sentence_aggregation(
    Sentence_Features(test.select(["essay_id", "full_text_clean"]))
)
test_feats3 = Word_aggregation(
    Word_Features(test.select(["essay_id", "full_text_clean"]))
)
test_feats4 = lexical_aggregation(
    Lexical_Features(test.select(["essay_id", "full_text_clean2"]))
)
print("Test feature extraction time(s):", round(time.time() - t0, 1))




## === cell 22
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats4, on="essay_id", how="left")

if test_feats.columns.duplicated().any():
    test_feats = test_feats.loc[:, ~test_feats.columns.duplicated()].copy()

print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())




## === cell 23
from scipy import sparse

preds = []

missing_cols = [c for c in FEATURES if c not in test_feats.columns]
for c in missing_cols:
    test_feats[c] = 0

X_test_dense = _make_numeric_matrix_fast(test_feats, FEATURES)
X_test = sparse.hstack([sparse.csr_matrix(X_test_dense), test_tfid], format="csr")

for i in range(5):
    print(f"Fold {i+1}")
    if CFG.LOAD_MODELS_FROM:
        model = pickle.load(
            open(f"{CFG.LOAD_MODELS_FROM}CAT_v{CFG.VER}_f{i}.pkl", "rb")
        )
    else:
        model = pickle.load(open(f"CAT_v{CFG.VER}_f{i}.pkl", "rb"))
    preds.append(model.predict(X_test))

pred = np.mean(preds, axis=0)

try:
    lr = pickle.load(open(f"CAL_v{CFG.VER}.pkl", "rb"))
    pred = lr.predict(np.asarray(pred).reshape(-1, 1)).reshape(-1)
except Exception as e:
    print("Skipping calibration load:", repr(e))

sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub["score"] = np.clip(pred, 1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
display(sub.head())
print("Saved to submission.csv")
