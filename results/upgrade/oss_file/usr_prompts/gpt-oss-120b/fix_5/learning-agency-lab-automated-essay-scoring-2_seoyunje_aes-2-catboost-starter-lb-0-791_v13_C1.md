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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The script now creates the required features for both training and test data, trains the CatBoost models when no pre‑saved models are provided, and finally writes a proper `submission.csv` with the correct column names. Paths for loading external models or features are set to `None` so the pipeline runs end‑to‑end without missing‑file errors.'
- What this solution (achieved 0.0) has done: 'The fix adds all missing imports, initializes required libraries (e.g., nltk), and ensures helper functions and variables (`Clean`, `seed_everything`, etc.) are defined before they are used. This resolves the NameError issues, lets the feature‑engineering and CatBoost training run, and finally writes a correctly formatted `submission.csv` with the required columns, enabling a valid Kaggle submission and moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'I keep the overall pipeline unchanged and only adjust the CatBoost training hyper‑parameters to give the model a bit more capacity, which should raise the validation QWK and therefore move the Kaggle score closer to the target. The change is limited to increasing the iteration count and early‑stopping patience, a minimal tweak that preserves the core logic.'

# 9. Code solution

## === cell 0
import os
import gc
import ctypes
import random
import re
import string
import pickle

import numpy as np
import pandas as pd
import polars as pl

import nltk

nltk.download("punkt", quiet=True)
from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

from catboost import CatBoostRegressor, Pool
import catboost


class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None  # no external models
    LOAD_FEATURES_FROM = None  # generate features on the fly
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 1
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()




## === cell 2
def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()




## === cell 3
df_train = pd.read_csv(os.path.join(CFG.BASE_PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(CFG.BASE_PATH, "test.csv"))




## === cell 4
train = pl.read_csv(os.path.join(CFG.BASE_PATH, "train.csv")).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.read_csv(os.path.join(CFG.BASE_PATH, "test.csv")).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)




## === cell 5
def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ".", x)
    return x.strip()


def dataPreprocessing2(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()




## === cell 6
paragraph_features = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]


def Paragraph_Features(df):
    df = df.explode("paragraph")
    df = df.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))
    df = df.with_columns(
        pl.col("paragraph").map_elements(lambda x: len(x)).alias("paragraph_len")
    )
    df = df.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("paragraph_word_cnt"),
    )
    return df


def Paragraph_aggregation(df):
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
    res = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    return res.to_pandas()




## === cell 7
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(df):
    df = df.with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(x))
        .str.split(".")
        .alias("sentence")
    )
    df = df.explode("sentence")
    df = df.with_columns(
        pl.col("sentence").map_elements(lambda x: len(x)).alias("sentence_len")
    )
    df = df.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("sentence_word_cnt")
    )
    return df


def Sentence_aggregation(df):
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
    res = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    return res.to_pandas()




## === cell 8
word_features = ["word_len"]


def Word_Features(df):
    df = df.with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(x))
        .str.split(" ")
        .alias("word")
    )
    df = df.explode("word")
    df = df.with_columns(
        pl.col("word").map_elements(lambda x: len(x)).alias("word_len")
    )
    return df


def Word_aggregation(df):
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i)
            .count()
            .alias(f"sentence_{i}_cnt")
            for i in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
        ],
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in word_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in word_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in word_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in word_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in word_features],
    ]
    res = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    return res.to_pandas()




## === cell 9
lexical_features = ["unique_word_text", "num_puncts_text"]


def Lexical_Features(df):
    df = df.with_columns(pl.col("full_text").map_elements(dataPreprocessing2))
    df = df.with_columns(
        pl.col("full_text")
        .map_elements(lambda x: len([c for c in x if c in string.punctuation]))
        .alias("num_puncts_text")
    )
    df = df.with_columns(
        pl.col("full_text")
        .map_elements(lambda x: len(set(word_tokenize(x))))
        .alias("unique_word_text")
    )
    return df


def lexical_aggregation(df):
    aggs = [
        *[
            pl.col("full_text")
            .filter(pl.col("unique_word_text") >= i)
            .count()
            .alias(f"unique_word_{i}_cnt")
            for i in [50, 100, 125, 150, 175, 200]
        ],
        *[
            pl.col("full_text")
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
    res = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    return res.to_pandas()




## === cell 10
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

tmp_train = train.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
train_tfid = vectorizer.fit_transform([i for i in tmp_train["full_text"]])
df_tfid_train = pd.DataFrame(train_tfid.toarray())
df_tfid_train.columns = [f"tfidf_{i}" for i in df_tfid_train.columns]
df_tfid_train["essay_id"] = df_train["essay_id"].values




## === cell 11
train_p1 = Paragraph_Features(train)
train_p1 = Paragraph_aggregation(train_p1)

train_s1 = Sentence_Features(train)
train_s1 = Sentence_aggregation(train_s1)

train_w1 = Word_Features(train)
train_w1 = Word_aggregation(train_w1)

train_l1 = Lexical_Features(train)
train_l1 = lexical_aggregation(train_l1)

train_feats = train_p1.merge(train_s1, on="essay_id", how="left")
train_feats = train_feats.merge(train_w1, on="essay_id", how="left")
train_feats = train_feats.merge(train_l1, on="essay_id", how="left")
train_feats = train_feats.merge(df_tfid_train, on="essay_id", how="left")
train_feats["score"] = df_train["score"].values




## === cell 12
tmp_test = test.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
test_tfid = vectorizer.transform([i for i in tmp_test["full_text"]])
df_tfid_test = pd.DataFrame(test_tfid.toarray())
df_tfid_test.columns = [f"tfidf_{i}" for i in df_tfid_test.columns]
df_tfid_test["essay_id"] = df_test["essay_id"].values

test_p1 = Paragraph_Features(test)
test_p1 = Paragraph_aggregation(test_p1)

test_s1 = Sentence_Features(test)
test_s1 = Sentence_aggregation(test_s1)

test_w1 = Word_Features(test)
test_w1 = Word_aggregation(test_w1)

test_l1 = Lexical_Features(test)
test_l1 = lexical_aggregation(test_l1)

test_feats = test_p1.merge(test_s1, on="essay_id", how="left")
test_feats = test_feats.merge(test_w1, on="essay_id", how="left")
test_feats = test_feats.merge(test_l1, on="essay_id", how="left")
test_feats = test_feats.merge(df_tfid_test, on="essay_id", how="left")




## === cell 13
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [c for c in train_feats.columns if c not in categorical_columns + ["score"]]
TARGET = "score"




## === cell 14
print("CatBoost version:", catboost.__version__)


def catboost_training():
    all_oof = []
    all_true = []
    skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(train_feats, train_feats[TARGET])
    ):
        print("#" * 20, f"Fold {fold+1}", "#" * 20)
        model = CatBoostRegressor(
            iterations=3000,  # increased from 2000
            learning_rate=0.03,
            depth=7,
            task_type="CPU",
            objective="RMSE",
            eval_metric="RMSE",
            random_seed=CFG.SEED,
            loss_function="RMSE",
        )
        train_pool = Pool(
            data=np.clip(train_feats.loc[train_idx, FEATURES].fillna(0), 0, 10000),
            label=train_feats.loc[train_idx, TARGET],
        )
        valid_pool = Pool(
            data=np.clip(train_feats.loc[valid_idx, FEATURES].fillna(0), 0, 10000),
            label=train_feats.loc[valid_idx, TARGET],
        )
        model.fit(
            train_pool,
            verbose=100,
            eval_set=valid_pool,
            early_stopping_rounds=200,  # relaxed from 100 to allow more learning
        )
        pickle.dump(model, open(f"CAT_v{CFG.VER}_f{fold}.pkl", "wb"))
        oof_pred = model.predict(valid_pool)
        all_oof.append(oof_pred)
        all_true.append(train_feats.loc[valid_idx, TARGET].values)
        del model, train_pool, valid_pool, oof_pred
        clean_memory()
    oof_all = np.concatenate(all_oof)
    true_all = np.concatenate(all_true)
    cv_score = cohen_kappa_score(
        true_all, np.clip(oof_all, 1, 6).round(), weights="quadratic"
    )
    print("Overall CV QWK:", cv_score)




## === cell 15
if CFG.LOAD_MODELS_FROM is None:
    catboost_training()
else:
    print("Skipping training – models will be loaded from external path.")




## === cell 16
preds = []
for fold in range(5):
    model_path = f"CAT_v{CFG.VER}_f{fold}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    model = pickle.load(open(model_path, "rb"))
    pred_fold = model.predict(test_feats[FEATURES])
    preds.append(pred_fold)
pred = np.mean(preds, axis=0)




## === cell 17
sub = pd.DataFrame(
    {"essay_id": df_test["essay_id"], TARGET: np.clip(pred, 1, 6).round().astype(int)}
)
sub.to_csv("submission.csv", index=False)
print("Submission saved. Shape:", sub.shape)
