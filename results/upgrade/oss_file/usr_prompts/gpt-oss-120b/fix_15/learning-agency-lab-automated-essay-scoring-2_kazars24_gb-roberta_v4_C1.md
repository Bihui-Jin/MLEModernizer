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

0.69983

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00131) has done: 'I fixed the import errors that prevented the script from running (corrected LightGBM callback imports and safe NLTK download) and ensured the model variable is defined so the prediction and submission steps work, resulting in a valid `submission.csv` file.'
- What this solution (achieved 0.00138) has done: 'Implemented two key fixes:

1. **Removed unnecessary Transformer‑related imports** that caused a protobuf `AttributeError`, allowing the notebook to run in the current environment.
2. **Adjusted LightGBM hyper‑parameters** (greater depth, more leaves, higher learning rate, more estimators) to boost model capacity and improve the quadratic weighted kappa score while keeping the original training pipeline intact.'
- What this solution (achieved 0.00589) has done: 'I switch to a standard regression objective (removing the offset a and the custom qwk gradient), simplify the QWK metric to work directly on raw predictions, and fill missing feature values so LightGBM can train effectively. These minimal changes keep the overall pipeline and feature engineering intact while substantially improving the validation QWK, moving the score toward the target.'
- What this solution (achieved -0.00379) has done: 'Implemented test feature construction to resolve the NameError and ensure a complete end‑to‑end run. The script now generates paragraph, sentence, and word engineered features for the test set, merges TF‑IDF vectors, and writes a proper `submission.csv` while keeping the original model and evaluation logic unchanged.'
- What this solution (achieved 0.03431) has done: 'I increase the TF‑IDF capacity (more discriminative word features) and boost the LightGBM model size and learning rate so the regressor can capture the essay‑score relationships better. These minimal tweaks keep the same preprocessing, feature‑engineering, and evaluation pipeline while moving the validation QWK toward the target score.'
- What this solution (achieved 0.69983) has done: 'I replaced the deprecated `arr.lengths()` calls with the current Polars `list.len()` API in the paragraph, sentence, and word engineering functions. This removes the AttributeError, allows the feature‑engineered DataFrames to be built, and restores the end‑to‑end pipeline so a valid `submission.csv` with the required columns is finally written.'

# 9. Code solution

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

train = pl.read_csv(PATH + "train.csv")
test = pl.read_csv(PATH + "test.csv")


def dataPreprocessing(x):
    x = x.lower()
    x = re.sub(r"<.*?>", "", x)  # remove HTML
    x = re.sub(r"@\w+", "", x)  # remove mentions
    x = re.sub(r"'\d+", "", x)  # remove apostrophe‑numbers
    x = re.sub(r"\d+", "", x)  # remove numbers
    x = re.sub(r"http\w+", "", x)  # remove URLs
    x = re.sub(r"\s+", " ", x)  # collapse whitespace
    x = re.sub(r"\.+", ".", x)  # collapse periods
    x = re.sub(r"\,+", ",", x)  # collapse commas
    return x.strip()


train = train.with_columns(
    pl.col("full_text").map_elements(dataPreprocessing).alias("clean_text")
)
test = test.with_columns(
    pl.col("full_text").map_elements(dataPreprocessing).alias("clean_text")
)




## === cell 2
def Paragraph_Eng(df_pl):
    tmp = (
        df_pl.with_columns(pl.col("clean_text").str.split("\n\n").alias("paragraph"))
        .explode("paragraph")
        .with_columns(
            pl.col("paragraph").str.len_chars().alias("paragraph_len"),
            pl.col("paragraph")
            .str.split(".")
            .list.len()
            .alias("paragraph_sentence_cnt"),
            pl.col("paragraph").str.split(" ").list.len().alias("paragraph_word_cnt"),
        )
    )
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
    df = tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()


paragraph_fea = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]

train_feats = Paragraph_Eng(train)
test_feats = Paragraph_Eng(test)
print(
    "Features Number after paragraph:",
    len([c for c in train_feats.columns if c not in ["essay_id", "score"]]),
)




## === cell 3
def Sentence_Eng(df_pl):
    tmp = (
        df_pl.with_columns(pl.col("clean_text").str.split(".").alias("sentence"))
        .explode("sentence")
        .with_columns(pl.col("sentence").str.len_chars().alias("sentence_len"))
        .filter(pl.col("sentence_len") >= 15)
        .with_columns(
            pl.col("sentence").str.split(" ").list.len().alias("sentence_word_cnt")
        )
    )
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
    df = tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()


sentence_fea = ["sentence_len", "sentence_word_cnt"]

train_feats = train_feats.merge(Sentence_Eng(train), on="essay_id", how="left")
test_feats = test_feats.merge(Sentence_Eng(test), on="essay_id", how="left")
print(
    "Features Number after sentence:",
    len([c for c in train_feats.columns if c not in ["essay_id", "score"]]),
)




## === cell 4
def Word_Eng(df_pl):
    tmp = (
        df_pl.with_columns(pl.col("clean_text").str.split(" ").alias("word"))
        .explode("word")
        .with_columns(pl.col("word").str.len_chars().alias("word_len"))
        .filter(pl.col("word_len") != 0)
    )
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
    df = tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()


train_feats = train_feats.merge(Word_Eng(train), on="essay_id", how="left")
test_feats = test_feats.merge(Word_Eng(test), on="essay_id", how="left")
print(
    "Features Number after word:",
    len([c for c in train_feats.columns if c not in ["essay_id", "score"]]),
)




## === cell 5
raw_train = pd.read_csv(PATH + "train.csv")
raw_test = pd.read_csv(PATH + "test.csv")

raw_train["clean_text"] = raw_train["full_text"].apply(dataPreprocessing)
raw_test["clean_text"] = raw_test["full_text"].apply(dataPreprocessing)

train_feats = train_feats.merge(
    raw_train[["essay_id", "score"]], on="essay_id", how="left"
)

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
    train_feats.drop(columns=["essay_id", "score"], errors="ignore")
    .fillna(-1)
    .astype(np.float32)
)
X_sparse = sp.hstack([sp.csr_matrix(train_other.values), tfidf_train])

test_other = (
    test_feats.drop(columns=["essay_id"], errors="ignore").fillna(-1).astype(np.float32)
)
X_test_sparse = sp.hstack([sp.csr_matrix(test_other.values), tfidf_test])

print("Total feature count after TF‑IDF (sparse):", X_sparse.shape[1])




## === cell 6
def quadratic_weighted_kappa(y_true, y_pred):
    y_pred = np.clip(y_pred, 1, 6)
    y_pred = np.round(y_pred).astype(int)
    y_true = y_true.astype(int)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


X = X_sparse
y = raw_train["score"].astype(np.float32).values

if_train = True

if if_train:
    callbacks = [
        lgb.callback.log_evaluation(period=25),
        lgb.callback.early_stopping(stopping_rounds=200, verbose=False),
    ]

    X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
        X,
        y,
        raw_train["score"].astype(int).values,
        test_size=0.2,
        random_state=52,
        stratify=raw_train["score"],
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




## === cell 7
proba = model.predict(X_test_sparse)
predictions = np.clip(proba, 1, 6)
predictions = np.round(predictions).astype(int)

print("First 10 predictions:", predictions[:10])




## === cell 8
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission["score"] = predictions
submission.to_csv("submission.csv", index=False)
display(submission.head())
