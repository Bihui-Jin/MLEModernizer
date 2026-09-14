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

# 5. Target score

0.807160703763817

# 6. Current score

0.0039

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00514) has done: 'Main bottlenecks are repeated Python-level `map_elements(dataPreprocessing, ...)` over ~140k long essays and redoing similar text cleaning for paragraphs/sentences/words, plus unnecessary dense copies and dual-eval during LightGBM fit. I keep the exact features/model/training semantics, but (1) switch text cleaning to Polars native vectorized string ops that are equivalent to your `dataPreprocessing`, (2) reuse the already-created `clean_text` everywhere (avoid re-cleaning), (3) reduce memory copies when building sparse matrices, and (4) keep LightGBM evaluation behavior but drop the redundant train eval set to cut fit-time overhead. These changes are deterministic and preserve the same algorithm/architecture and metrics, with only negligible floating-point differences.'
- What this solution (achieved -0.02787) has done: 'Your current score (0.00514) is far below the target (0.80716), so we should improve performance rather than tune for speed. The biggest issue is that your TF-IDF/CountVectorizer are configured with `tokenizer=lambda x: x` while you are passing raw strings, which makes scikit-learn treat each essay as an iterable of characters and produces character-level “word” n-grams—this severely breaks the feature signal. I keep your model, objective, engineered features, and training loop the same, and make the minimal fix: use the default word tokenizer (remove the custom tokenizer/preprocessor and `token_pattern=None`) and apply the same cleaning you already compute (`clean_text`) to the vectorizers for train/test. This should move QWK sharply upward toward the target without changing your core approach.'
- What this solution (achieved 0.01681) has done: 'Your current score is far below the target, so we should fix the most likely root cause of the negative QWK without changing your model/features: the TF-IDF and CountVectorizer settings are using *word* analyzer with very long n-grams (4–8 and 3–5) and high `min_df`, which effectively wipes out useful vocabulary and makes the model behave poorly. I keep your exact pipeline (same engineered features, same LightGBM objective/training loop, same rounding-to-1..6 submission), but adjust the vectorizers to standard word n-grams (1–2) and reasonable `min_df/max_df` so text features become informative again. This is a minimal, localized change that should move QWK substantially upward toward the target while preserving the core approach. The rest of the code remains the same and still produces `submission.csv`.'
- What this solution (achieved 0.0039) has done: 'Your current score is far below the target, so we should improve predictive signal without changing the overall pipeline (same engineered features + TFIDF/Count + LightGBM with your custom objective). The biggest likely score-killer remaining is the mismatch between the QWK metric (ordinal 1–6) and how we post-process/regress: simple rounding can be badly calibrated, especially with a custom objective. I keep the model/training the same, but add a tiny, deterministic “rounding thresholds” calibration on the validation split (optimize 5 cutpoints to maximize QWK) and then apply those thresholds to test predictions—this typically boosts QWK substantially while preserving core logic. I also make the train/val split deterministic across all RNGs and ensure the custom metric uses proper integer labels consistently.'
- What this solution (achieved 0.0039) has done: 'Your current score is extremely far below the target, so the priority is to fix likely evaluation-breaking issues without changing the overall pipeline (same engineered features + TFIDF/Count + LightGBM + thresholding). The biggest bug is that your custom LightGBM objective/metric uses a global offset `a` inside the metric and objective in a way that mixes shifted and unshifted targets/predictions, which can make training optimize the wrong thing and collapse QWK. I keep your model and feature extraction identical, but make the objective and eval metric consistently operate in the same “shifted target” space (y is already `score - a`), and make threshold optimization use the correct continuous score predictions. This is a minimal semantic fix expected to move QWK sharply upward toward the target while preserving your core logic and submission format.'

# 9. Code solution

## === cell 0
import copy
from glob import glob
import gc
import os
import re
import random
import warnings

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from lightgbm.callback import log_evaluation, early_stopping

import pickle

from scipy import sparse
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

SEED = 52
random.seed(SEED)
np.random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

print("Imports OK. lightgbm:", lgb.__version__)



## === cell 1
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
    "I'll've": "I will",
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
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


def _polars_clean_expr(col: str) -> pl.Expr:
    e = pl.col(col).fill_null("").cast(pl.Utf8).str.to_lowercase()
    e = e.str.replace_all(r"<.*?>", "")
    e = e.str.replace_all(r"@\w+", "")
    e = e.str.replace_all(r"'\d+", "")
    e = e.str.replace_all(r"\d+", "")
    e = e.str.replace_all(r"http\w+", "")
    e = e.str.replace_all(r"\s+", " ")
    e = e.str.replace_all(r"\.+", ".")
    e = e.str.replace_all(r"\,+", ",")
    e = e.str.strip_chars()
    return e




## === cell 2
columns = [
    pl.col("full_text")
    .fill_null("")
    .cast(pl.Utf8)
    .str.split(by="\n\n")
    .alias("paragraph"),
    pl.col("full_text").fill_null("").cast(pl.Utf8).alias("full_text"),
    _polars_clean_expr("full_text").alias("clean_text"),
]

PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
train = pl.read_csv(PATH + "train.csv").with_columns(columns)
test = pl.read_csv(PATH + "test.csv").with_columns(columns)

train.head(1)




## === cell 3
def _add_clean_text(df: pl.DataFrame) -> pl.DataFrame:
    if "clean_text" in df.columns:
        return df
    return df.with_columns(_polars_clean_expr("full_text").alias("clean_text"))


def Paragraph_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = _add_clean_text(tmp)
    tmp = tmp.explode("paragraph")
    tmp = tmp.with_columns(_polars_clean_expr("paragraph").alias("paragraph"))
    tmp = tmp.with_columns(
        pl.col("paragraph").str.len_chars().alias("paragraph_len"),
        (pl.col("paragraph").str.count_matches(r"\.") + 1).alias(
            "paragraph_sentence_cnt"
        ),
        (pl.col("paragraph").str.count_matches(r" ") + 1).alias("paragraph_word_cnt"),
    )
    return tmp


paragraph_fea = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]


def Paragraph_Eng(train_tmp):
    aggs = [
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_len") >= i)
            .count()
            .alias(f"paragraph_{i}_cnt")
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
            pl.col("paragraph")
            .filter(pl.col("paragraph_len") <= i)
            .count()
            .alias(f"paragraph_{i}_cnt")
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


tmp = Paragraph_Preprocess(train)
train_feats = Paragraph_Eng(tmp)
train_feats["score"] = train["score"].to_list()

feature_names = list(
    filter(lambda x: x not in ["essay_id", "score"], train_feats.columns)
)
print("Features Number: ", len(feature_names))
train_feats.head(3)




## === cell 4
def Sentence_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = _add_clean_text(tmp)
    tmp = tmp.with_columns(pl.col("clean_text").str.split(by=".").alias("sentence"))
    tmp = tmp.explode("sentence")
    tmp = tmp.with_columns(pl.col("sentence").str.len_chars().alias("sentence_len"))
    tmp = tmp.filter(pl.col("sentence_len") >= 15)
    tmp = tmp.with_columns(
        (pl.col("sentence").str.count_matches(r" ") + 1).alias("sentence_word_cnt")
    )
    return tmp


sentence_fea = ["sentence_len", "sentence_word_cnt"]


def Sentence_Eng(train_tmp):
    aggs = [
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") >= i)
            .count()
            .alias(f"sentence_{i}_cnt")
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


tmp = Sentence_Preprocess(train)
train_feats = train_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")

feature_names = list(
    filter(lambda x: x not in ["essay_id", "score"], train_feats.columns)
)
print("Features Number: ", len(feature_names))
train_feats.head(3)




## === cell 5
def Word_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = _add_clean_text(tmp)
    tmp = tmp.with_columns(pl.col("clean_text").str.split(by=" ").alias("word"))
    tmp = tmp.explode("word")
    tmp = tmp.with_columns(pl.col("word").str.len_chars().alias("word_len"))
    tmp = tmp.filter(pl.col("word_len") != 0)
    return tmp


def Word_Eng(train_tmp):
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i + 1)
            .count()
            .alias(f"word_{i+1}_cnt")
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


tmp = Word_Preprocess(train)
train_feats = train_feats.merge(Word_Eng(tmp), on="essay_id", how="left")

feature_names = list(
    filter(lambda x: x not in ["essay_id", "score"], train_feats.columns)
)
print("Features Number: ", len(feature_names))
train_feats.head(3)



## === cell 6
vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90,
    sublinear_tf=True,
)

train_text_list = train["clean_text"].to_list()
train_tfid = vectorizer.fit_transform(train_text_list)
print("TFIDF train shape:", train_tfid.shape)



## === cell 7
vectorizer_cnt = CountVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90,
)

train_cnt = vectorizer_cnt.fit_transform(train_text_list)
print("Count train shape:", train_cnt.shape)



## === cell 8
a = 2.948
b = 1.092


def quadratic_weighted_kappa(y_true, y_pred):
    y_true_score = np.clip(np.round(y_true + a), 1, 6).astype(int)
    y_pred_score = np.clip(np.round(y_pred + a), 1, 6).astype(int)
    qwk = cohen_kappa_score(y_true_score, y_pred_score, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    labels = y_true  # shifted
    preds_score = np.clip(y_pred + a, 1.0, 6.0)
    preds = preds_score - a  # shifted back to align with labels

    f = 0.5 * np.sum((preds - labels) ** 2)
    g = 0.5 * np.sum(
        (preds - 0.0) ** 2 + b
    )  # 0.0 corresponds to score==a in shifted space
    df_ = preds - labels
    dg = preds - 0.0
    grad = (df_ / g - f * dg / (g**2)) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess


def _apply_thresholds(
    preds_continuous: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    t = np.sort(np.asarray(thresholds, dtype=np.float64))
    bins = [-np.inf, t[0], t[1], t[2], t[3], t[4], np.inf]
    out = np.digitize(preds_continuous, bins=bins)  # 1..6
    return np.clip(out, 1, 6).astype(int)


def _optimize_thresholds(y_true_int: np.ndarray, preds_cont: np.ndarray) -> np.ndarray:
    init = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
    bounds = [(1.0, 6.0)] * 5

    def objective(t):
        t = np.asarray(t, dtype=np.float64)
        penalty = 0.0
        diffs = np.diff(t)
        if np.any(diffs <= 1e-3):
            penalty += 10_000.0 * np.sum((1e-3 - diffs[diffs <= 1e-3]) ** 2)
        y_pred = _apply_thresholds(preds_cont, t)
        return -(cohen_kappa_score(y_true_int, y_pred, weights="quadratic")) + penalty

    res = minimize(
        objective,
        init,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 200, "xtol": 1e-4, "ftol": 1e-5},
    )
    t_best = np.sort(res.x.astype(np.float64))
    return t_best




## === cell 9
feature_names_dense = list(
    filter(lambda x: x not in ["essay_id", "score"], train_feats.columns)
)
train_feats[feature_names_dense] = train_feats[feature_names_dense].replace(
    [np.inf, -np.inf], np.nan
)

X_dense = train_feats[feature_names_dense].to_numpy(dtype=np.float32, copy=False)
np.nan_to_num(X_dense, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X = sparse.hstack(
    [
        sparse.csr_matrix(X_dense),
        train_tfid.tocsr(),
        train_cnt.tocsr(),
    ],
    format="csr",
)

y_split = train_feats["score"].astype(int).values
y = train_feats["score"].astype(np.float32).values - a
oof = train_feats["score"].astype(np.float32).values

print("Train matrix:", X.shape, "y:", y.shape)



## === cell 10
if_train = True

if if_train:
    callbacks = [
        log_evaluation(period=25),
        early_stopping(stopping_rounds=75, first_metric_only=True),
    ]

    X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
        X,
        y,
        y_split,
        test_size=0.2,
        random_state=SEED,
        stratify=train_feats["score"],
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
        random_state=SEED,
        extra_trees=True,
        class_weight="balanced",
        verbosity=-1,
        n_jobs=-1,
    )

    predictor = model.fit(
        X_train,
        y_train,
        eval_names=["valid"],
        eval_set=[(X_val, y_val)],
        eval_metric=quadratic_weighted_kappa,
        callbacks=callbacks,
    )

    val_pred_cont = predictor.predict(X_val) + a

    val_pred_round = np.round(np.clip(val_pred_cont, 1, 6)).astype(int)
    f1 = f1_score(y_val_int, val_pred_round, average="weighted")
    kappa = cohen_kappa_score(y_val_int, val_pred_round, weights="quadratic")

    best_thresholds = _optimize_thresholds(
        y_val_int.astype(int), val_pred_cont.astype(np.float64)
    )
    val_pred_thr = _apply_thresholds(val_pred_cont.astype(np.float64), best_thresholds)
    kappa_thr = cohen_kappa_score(y_val_int, val_pred_thr, weights="quadratic")
    f1_thr = f1_score(y_val_int, val_pred_thr, average="weighted")

    cm = confusion_matrix(y_val_int, val_pred_thr, labels=[x for x in range(1, 7)])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
    )
    disp.plot()
    plt.show()

    print(f"Baseline (round) F1: {f1}")
    print(f"Baseline (round) QWK: {kappa}")
    print(f"Thresholds: {best_thresholds}")
    print(f"Thresholded F1: {f1_thr}")
    print(f"Thresholded QWK: {kappa_thr}")

    with open("lgbm_model.pkl", "wb") as fp:
        pickle.dump(model, fp)
    with open("rounding_thresholds.pkl", "wb") as fp:
        pickle.dump(best_thresholds, fp)
else:
    with open("lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    if os.path.exists("rounding_thresholds.pkl"):
        with open("rounding_thresholds.pkl", "rb") as f:
            best_thresholds = pickle.load(f)
    else:
        best_thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)



## === cell 11
tmp = Paragraph_Preprocess(test)
test_feats = Paragraph_Eng(tmp)

tmp = Sentence_Preprocess(test)
test_feats = test_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")

tmp = Word_Preprocess(test)
test_feats = test_feats.merge(Word_Eng(tmp), on="essay_id", how="left")

for col in feature_names_dense:
    if col not in test_feats.columns:
        test_feats[col] = 0.0

test_feats[feature_names_dense] = (
    test_feats[feature_names_dense].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)

test_text_list = test["clean_text"].to_list()
test_tfid = vectorizer.transform(test_text_list)
test_cnt = vectorizer_cnt.transform(test_text_list)

X_test_dense = test_feats[feature_names_dense].to_numpy(dtype=np.float32, copy=False)
np.nan_to_num(X_test_dense, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X_test = sparse.hstack(
    [sparse.csr_matrix(X_test_dense), test_tfid.tocsr(), test_cnt.tocsr()],
    format="csr",
)

print("Test matrix:", X_test.shape, "engineered_dense:", len(feature_names_dense))
test_feats.head(3)



## === cell 12
if "model" not in globals():
    if os.path.exists("lgbm_model.pkl"):
        with open("lgbm_model.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        raise RuntimeError("Model not found. Set if_train=True at least once to train.")

if "best_thresholds" not in globals():
    if os.path.exists("rounding_thresholds.pkl"):
        with open("rounding_thresholds.pkl", "rb") as f:
            best_thresholds = pickle.load(f)
    else:
        best_thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)

pred_cont = model.predict(X_test) + a
predictions = _apply_thresholds(pred_cont.astype(np.float64), best_thresholds)

print(predictions[:20], predictions.min(), predictions.max(), len(predictions))



## === cell 13
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)

pred_df = pd.DataFrame(
    {"essay_id": test_feats["essay_id"].values, "score": predictions}
)
submission = submission.drop(columns=["score"]).merge(
    pred_df, on="essay_id", how="left"
)

if submission["score"].isna().any():
    fill_val = int(np.clip(np.round(train_feats["score"].mean()), 1, 6))
    submission["score"] = submission["score"].fillna(fill_val).astype(int)
else:
    submission["score"] = submission["score"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Saved to submission.csv, rows:", len(submission))
print("Unique scores:", submission["score"].value_counts().sort_index().to_dict())
