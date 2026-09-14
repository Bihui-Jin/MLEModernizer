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
datasets==4.4.1
geopandas==0.14.4
imbalanced-learn==0.13.0
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3
vega-datasets==0.9.0

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
import gc
import re
import copy
import random
import pickle
import warnings

import numpy as np
import pandas as pd
import polars as pl

import torch  # kept to preserve original imports/compat
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    cohen_kappa_score,
)

from datasets import Dataset  # noqa: F401
from transformers import (  # noqa: F401
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import nltk  # noqa: F401

from scipy import sparse

warnings.filterwarnings("ignore")

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)

try:
    nltk.data.find("corpora/wordnet")
except Exception:
    try:
        nltk.download("wordnet", quiet=True)
    except Exception as e:
        print(f"[WARN] nltk wordnet download skipped due to environment issue: {e}")




## === cell 1
def dataPreprocessing_expr(col: pl.Expr) -> pl.Expr:
    return (
        col.str.to_lowercase()
        .str.replace_all(r"<.*?>", "")
        .str.replace_all(r"@\w+", "")
        .str.replace_all(r"'\d+", "")
        .str.replace_all(r"\d+", "")
        .str.replace_all(r"http\w+", "")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"\.+", ".")
        .str.replace_all(r"\,+", ",")
        .str.strip_chars()
    )


columns = [
    (pl.col("full_text").str.split(by="\n\n").alias("paragraph")),
]
PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
train = pl.read_csv(PATH + "train.csv").with_columns(columns)
test = pl.read_csv(PATH + "test.csv").with_columns(columns)

train.head(1)



## === cell 2
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
}

c_re = re.compile("(%s)" % "|".join(cList.keys()))


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub(r"@\w+", "", x)
    x = re.sub(r"'\d+", "", x)
    x = re.sub(r"\d+", "", x)
    x = re.sub(r"http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x




## === cell 3
paragraph_fea = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]
sentence_fea = ["sentence_len", "sentence_word_cnt"]


def build_engineered_features(df: pl.DataFrame) -> pd.DataFrame:
    base = df.select(["essay_id", "full_text", "paragraph"])

    base_lf = base.lazy().with_columns(
        dataPreprocessing_expr(pl.col("full_text")).alias("full_text_clean")
    )

    p = (
        base_lf.explode("paragraph")
        .with_columns(dataPreprocessing_expr(pl.col("paragraph")).alias("paragraph"))
        .with_columns(
            pl.col("paragraph").str.len_chars().alias("paragraph_len"),
            (pl.col("paragraph").str.count_matches(r"\.") + 1).alias(
                "paragraph_sentence_cnt"
            ),
            (pl.col("paragraph").str.count_matches(r" ") + 1).alias(
                "paragraph_word_cnt"
            ),
        )
    )
    p_aggs = [
        *[
            (pl.col("paragraph_len") >= i).sum().alias(f"paragraph_{i}_cnt")
            for i in [
                50,
                75,
                100,
                125,
                150,
                175,
                200,
                250,
                300,
                350,
                400,
                500,
                600,
                700,
            ]
        ],
        *[
            (pl.col("paragraph_len") <= i).sum().alias(f"paragraph_{i}_cnt")
            for i in [25, 49]
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea],
    ]
    p_df = p.group_by(["essay_id"], maintain_order=True).agg(p_aggs)

    s = (
        base_lf.with_columns(
            pl.col("full_text_clean").str.split(by=".").alias("sentence")
        )
        .explode("sentence")
        .with_columns(pl.col("sentence").str.len_chars().alias("sentence_len"))
        .filter(pl.col("sentence_len") >= 15)
        .with_columns(
            (pl.col("sentence").str.count_matches(r" ") + 1).alias("sentence_word_cnt")
        )
    )
    s_aggs = [
        *[
            (pl.col("sentence_len") >= i).sum().alias(f"sentence_{i}_cnt")
            for i in [15, 50, 100, 150, 200, 250, 300]
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in sentence_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sentence_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in sentence_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in sentence_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in sentence_fea],
    ]
    s_df = s.group_by(["essay_id"], maintain_order=True).agg(s_aggs)

    w = (
        base_lf.with_columns(pl.col("full_text_clean").str.split(by=" ").alias("word"))
        .explode("word")
        .with_columns(pl.col("word").str.len_chars().alias("word_len"))
        .filter(pl.col("word_len") != 0)
    )
    w_aggs = [
        *[
            (pl.col("word_len") >= (i + 1)).sum().alias(f"word_{i+1}_cnt")
            for i in range(15)
        ],
        pl.col("word_len").max().alias("word_len_max"),
        pl.col("word_len").mean().alias("word_len_mean"),
        pl.col("word_len").std().alias("word_len_std"),
        pl.col("word_len").quantile(0.25).alias("word_len_q1"),
        pl.col("word_len").quantile(0.50).alias("word_len_q2"),
        pl.col("word_len").quantile(0.75).alias("word_len_q3"),
    ]
    w_df = w.group_by(["essay_id"], maintain_order=True).agg(w_aggs)

    feats = (
        p_df.join(s_df, on="essay_id", how="left")
        .join(w_df, on="essay_id", how="left")
        .sort("essay_id")
        .collect(streaming=True)
        .to_pandas()
    )
    return feats


train_feats = build_engineered_features(train)
train_feats["score"] = train["score"].to_numpy()

feature_names = list(
    filter(lambda x: x not in ["essay_id", "score"], train_feats.columns)
)
print("Features Number: ", len(feature_names))
train_feats.head(3)




## === cell 4
def Paragraph_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = tmp.explode("paragraph")
    tmp = tmp.with_columns(
        dataPreprocessing_expr(pl.col("paragraph")).alias("paragraph")
    )
    tmp = tmp.with_columns(
        pl.col("paragraph").str.len_chars().alias("paragraph_len"),
        (pl.col("paragraph").str.count_matches(r"\.") + 1).alias(
            "paragraph_sentence_cnt"
        ),
        (pl.col("paragraph").str.count_matches(r" ") + 1).alias("paragraph_word_cnt"),
    )
    return tmp


def Paragraph_Eng(train_tmp: pl.DataFrame) -> pd.DataFrame:
    aggs = [
        *[
            (pl.col("paragraph_len") >= i).sum().alias(f"paragraph_{i}_cnt")
            for i in [
                50,
                75,
                100,
                125,
                150,
                175,
                200,
                250,
                300,
                350,
                400,
                500,
                600,
                700,
            ]
        ],
        *[
            (pl.col("paragraph_len") <= i).sum().alias(f"paragraph_{i}_cnt")
            for i in [25, 49]
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea],
    ]
    df = (
        train_tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    )
    return df.to_pandas()




## === cell 5
def Sentence_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = tmp.with_columns(
        dataPreprocessing_expr(pl.col("full_text")).str.split(by=".").alias("sentence")
    )
    tmp = tmp.explode("sentence")
    tmp = tmp.with_columns(pl.col("sentence").str.len_chars().alias("sentence_len"))
    tmp = tmp.filter(pl.col("sentence_len") >= 15)
    tmp = tmp.with_columns(
        (pl.col("sentence").str.count_matches(r" ") + 1).alias("sentence_word_cnt")
    )
    return tmp


def Sentence_Eng(train_tmp: pl.DataFrame) -> pd.DataFrame:
    aggs = [
        *[
            (pl.col("sentence_len") >= i).sum().alias(f"sentence_{i}_cnt")
            for i in [15, 50, 100, 150, 200, 250, 300]
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in sentence_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sentence_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in sentence_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in sentence_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in sentence_fea],
    ]
    df = (
        train_tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    )
    return df.to_pandas()




## === cell 6
def Word_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = tmp.with_columns(
        dataPreprocessing_expr(pl.col("full_text")).str.split(by=" ").alias("word")
    )
    tmp = tmp.explode("word")
    tmp = tmp.with_columns(pl.col("word").str.len_chars().alias("word_len"))
    tmp = tmp.filter(pl.col("word_len") != 0)
    return tmp


def Word_Eng(train_tmp: pl.DataFrame) -> pd.DataFrame:
    aggs = [
        *[
            (pl.col("word_len") >= (i + 1)).sum().alias(f"word_{i+1}_cnt")
            for i in range(15)
        ],
        pl.col("word_len").max().alias("word_len_max"),
        pl.col("word_len").mean().alias("word_len_mean"),
        pl.col("word_len").std().alias("word_len_std"),
        pl.col("word_len").quantile(0.25).alias("word_len_q1"),
        pl.col("word_len").quantile(0.50).alias("word_len_q2"),
        pl.col("word_len").quantile(0.75).alias("word_len_q3"),
    ]
    df = (
        train_tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    )
    return df.to_pandas()




## === cell 7
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(4, 8),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,
)

train_texts_iter = train.get_column("full_text").to_list().__iter__()
train_tfid = vectorizer.fit_transform(train_texts_iter)

n_tfid = train_tfid.shape[1]
print("TFIDF features:", n_tfid)

gc.collect()



## === cell 8
vectorizer_cnt = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(3, 5),
    min_df=0.10,
    max_df=0.85,
)

train_texts_iter2 = train.get_column("full_text").to_list().__iter__()
train_cnt = vectorizer_cnt.fit_transform(train_texts_iter2)
n_cnt = train_cnt.shape[1]
print("Count features:", n_cnt)

gc.collect()




## === cell 9
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 1 / 2 * np.sum((preds - labels) ** 2)
    g = 1 / 2 * np.sum((preds - a) ** 2 + b)
    df_ = preds - labels
    dg = preds - a
    grad = (df_ / g - f * dg / g**2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess


a = 2.948
b = 1.092



## === cell 10
engineered_feature_names = list(
    filter(lambda x: x not in ["essay_id", "score"], train_feats.columns)
)
X_dense = train_feats[engineered_feature_names].astype(np.float32).values

train_tfid = train_tfid.tocsr()
train_cnt = train_cnt.tocsr()

X = sparse.hstack(
    [sparse.csr_matrix(X_dense), train_tfid, train_cnt],
    format="csr",
    dtype=np.float32,
)

y_split = train_feats["score"].astype(int).values
y = train_feats["score"].astype(np.float32).values - a
oof = train_feats["score"].astype(np.float32).values

print(
    "Final X shape:",
    X.shape,
    " (engineered:",
    X_dense.shape[1],
    "tfidf:",
    n_tfid,
    "cnt:",
    n_cnt,
    ")",
)

del X_dense
gc.collect()



## === cell 11
model_path = "/kaggle/input/llgbm-models/lgbm_models.pkl"

if_train = True
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            models = pickle.load(f)
        if_train = False
        print(f"Loaded {len(models)} models from: {model_path}")
    except Exception as e:
        print(
            f"[WARN] Failed to load models from {model_path}: {e}. Will train models instead."
        )
        if_train = True

if if_train:
    n_splits = 15
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    f1_scores = []
    kappa_scores = []
    models = []
    predictions = []
    callbacks = [
        log_evaluation(period=25),
        early_stopping(stopping_rounds=75, first_metric_only=True),
    ]

    split_dummy = np.zeros(len(y_split), dtype=np.uint8)

    i = 1
    for train_index, test_index in skf.split(split_dummy, y_split):
        print("fold", i)
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold, y_test_fold_int = (
            y[train_index],
            y[test_index],
            y_split[test_index],
        )

        model = lgb.LGBMRegressor(
            objective=qwk_obj,
            metrics="None",
            learning_rate=0.05,
            max_depth=5,
            num_leaves=10,
            colsample_bytree=0.3,
            reg_alpha=0.7,
            reg_lambda=0.1,
            n_estimators=700,
            random_state=42,
            extra_trees=True,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=-1,
        )

        predictor = model.fit(
            X_train_fold,
            y_train_fold,
            eval_names=["valid"],
            eval_set=[(X_test_fold, y_test_fold)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=callbacks,
        )
        models.append(predictor)

        predictions_fold = predictor.predict(X_test_fold)
        predictions_fold = predictions_fold + a
        oof[test_index] = predictions_fold

        predictions_fold = predictions_fold.clip(1, 6).round()
        predictions.append(predictions_fold)

        f1_fold = f1_score(y_test_fold_int, predictions_fold, average="weighted")
        f1_scores.append(f1_fold)

        kappa_fold = cohen_kappa_score(
            y_test_fold_int, predictions_fold, weights="quadratic"
        )
        kappa_scores.append(kappa_fold)

        print(f"F1 score across fold: {f1_fold}")
        print(f"Cohen kappa score across fold: {kappa_fold}")
        i += 1

    mean_f1_score = np.mean(f1_scores)
    mean_kappa_score = np.mean(kappa_scores)

    print("=" * 50)
    print(f"Mean F1 score across {n_splits} folds: {mean_f1_score}")
    print(f"Mean Cohen kappa score across {n_splits} folds: {mean_kappa_score}")
    print("=" * 50)

    with open("lgbm_models.pkl", "wb") as fp:
        pickle.dump(models, fp)
    print("Saved trained models to ./lgbm_models.pkl")



## === cell 12
test_feats = build_engineered_features(test)

missing_in_test = [
    c for c in train_feats.columns if c not in ["score"] and c not in test_feats.columns
]
if missing_in_test:
    for c in missing_in_test:
        if c != "essay_id":
            test_feats[c] = 0.0

engineered_feature_names = list(
    filter(lambda x: x not in ["essay_id", "score"], test_feats.columns)
)

test_texts_iter = test.get_column("full_text").to_list().__iter__()
test_tfid = vectorizer.transform(test_texts_iter).tocsr()

test_texts_iter2 = test.get_column("full_text").to_list().__iter__()
test_cnt = vectorizer_cnt.transform(test_texts_iter2).tocsr()

gc.collect()

X_test_dense = test_feats[engineered_feature_names].astype(np.float32).values
X_test = sparse.hstack(
    [sparse.csr_matrix(X_test_dense), test_tfid, test_cnt],
    format="csr",
    dtype=np.float32,
)

print("Features number (engineered): ", len(engineered_feature_names))
print("Final X_test shape:", X_test.shape)

del X_test_dense
gc.collect()



## === cell 13
probabilities = []
for model in models:
    proba = model.predict(X_test) + a
    probabilities.append(proba)

predictions = np.mean(probabilities, axis=0)
predictions = np.round(np.clip(predictions, 1, 6)).astype(int)

print(predictions[:20], " ... total:", len(predictions))

submission = pd.read_csv(PATH + "sample_submission.csv")

pred_df = pd.DataFrame(
    {"essay_id": test_feats["essay_id"].values, "score": predictions}
)
submission = submission.drop(columns=["score"]).merge(
    pred_df, on="essay_id", how="left"
)

if submission["score"].isna().any():
    fill_val = int(np.round(np.clip(train_feats["score"].mean(), 1, 6)))
    submission["score"] = submission["score"].fillna(fill_val).astype(int)
else:
    submission["score"] = submission["score"].astype(int)

submission.to_csv("submission.csv", index=None)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
