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
import copy
import re
import random
import warnings
import gc

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import nltk

import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import scipy.sparse as sp

warnings.filterwarnings("ignore")

try:
    nltk.download("wordnet", quiet=True)
except Exception:
    pass



## === cell 1
PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"

train = pl.read_csv(PATH + "train.csv").with_columns(
    (pl.col("full_text").str.split(by="\n\n")).alias("paragraph")
)
test = pl.read_csv(PATH + "test.csv").with_columns(
    (pl.col("full_text").str.split(by="\n\n")).alias("paragraph")
)



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
    x = re.sub("@\\w+", "", x)
    x = re.sub("'\\d+", "", x)
    x = re.sub("\\d+", "", x)
    x = re.sub("http\\w+", "", x)
    x = re.sub(r"\\s+", " ", x)
    x = re.sub(r"\\.+", ".", x)
    x = re.sub(r"\\,+", ",", x)
    x = x.strip()
    return x




## === cell 3
def Paragraph_Preprocess(tmp):
    tmp = tmp.explode("paragraph")
    tmp = tmp.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))
    tmp = tmp.with_columns(
        pl.col("paragraph").map_elements(lambda x: len(x)).alias("paragraph_len")
    )
    tmp = tmp.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("paragraph_word_cnt"),
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
train_feats["score"] = train["score"]
feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
print("Features Number after paragraph:", len(feature_names))




## === cell 4
def Sentence_Preprocess(tmp):
    tmp = tmp.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(by=".")
        .alias("sentence")
    )
    tmp = tmp.explode("sentence")
    tmp = tmp.with_columns(
        pl.col("sentence").map_elements(lambda x: len(x)).alias("sentence_len")
    )
    tmp = tmp.filter(pl.col("sentence_len") >= 15)
    tmp = tmp.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("sentence_word_cnt")
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
feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
print("Features Number after sentence:", len(feature_names))




## === cell 5
def Word_Preprocess(tmp):
    tmp = tmp.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(by=" ")
        .alias("word")
    )
    tmp = tmp.explode("word")
    tmp = tmp.with_columns(
        pl.col("word").map_elements(lambda x: len(x)).alias("word_len")
    )
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
feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
print("Features Number after word:", len(feature_names))



## === cell 6
raw_train = pd.read_csv(PATH + "train.csv")
raw_test = pd.read_csv(PATH + "test.csv")

raw_train["clean_text"] = raw_train["full_text"].apply(dataPreprocessing)
raw_test["clean_text"] = raw_test["full_text"].apply(dataPreprocessing)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english",
    dtype=np.float32,
    sublinear_tf=True,
)

tfidf_train = vectorizer.fit_transform(raw_train["clean_text"])
tfidf_test = vectorizer.transform(raw_test["clean_text"])

train_other = (
    train_feats.drop(columns=["essay_id", "score"]).fillna(-1).astype(np.float32)
)
X_sparse = sp.hstack([sp.csr_matrix(train_other.values), tfidf_train])

test_other = test_feats.drop(columns=["essay_id"]).fillna(-1).astype(np.float32)
X_test_sparse = sp.hstack([sp.csr_matrix(test_other.values), tfidf_test])

feature_names = None  # not needed for sparse matrix usage
print("Total feature count after TF‑IDF (sparse):", X_sparse.shape[1])




## === cell 7
def quadratic_weighted_kappa(y_true, y_pred):
    y_pred = np.clip(y_pred, 1, 6)
    y_pred = np.round(y_pred).astype(int)
    y_true = y_true.astype(int)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


X = X_sparse
y = train_feats["score"].astype(np.float32).values

if_train = True

if if_train:
    callbacks = [
        lgb.callback.log_evaluation(period=25),
    ]

    X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
        X,
        y,
        train_feats["score"].astype(int).values,
        test_size=0.2,
        random_state=52,
        stratify=train_feats["score"],
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="None",
        learning_rate=0.1,
        max_depth=-1,
        num_leaves=511,
        colsample_bytree=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_child_samples=20,
        n_estimators=5000,
        random_state=42,
        verbosity=-1,
    )

    predictor = model.fit(
        X_train,
        y_train,
        eval_names=["train", "valid"],
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=quadratic_weighted_kappa,
        callbacks=callbacks,
    )

    val_pred = predictor.predict(X_val)
    val_pred = np.clip(val_pred, 1, 6)
    val_pred = np.round(val_pred).astype(int)

    f1 = f1_score(y_val_int, val_pred, average="weighted")
    kappa = cohen_kappa_score(y_val_int, val_pred, weights="quadratic")
    print(f"F1 score: {f1}")
    print(f"Quadratic weighted kappa: {kappa}")

    with open("lgbm_model.pkl", "wb") as fp:
        pickle.dump(model, fp)
else:
    with open("/kaggle/input/lgbm-model/lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)



## === cell 8
proba = model.predict(X_test_sparse)
predictions = np.clip(proba, 1, 6)
predictions = np.round(predictions).astype(int)

print("First 10 predictions:", predictions[:10])



## === cell 9
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission["score"] = predictions
submission.to_csv("submission.csv", index=False)
display(submission.head())
