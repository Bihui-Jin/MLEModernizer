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
import numpy as np
import pandas as pd
import polars as pl
import re
import warnings

warnings.filterwarnings("ignore")
import gc
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from lightgbm import log_evaluation, early_stopping
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



## === cell 1
PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
columns = [pl.col("full_text").str.split(by="\n\n").alias("paragraph")]
train = pl.read_csv(PATH + "train.csv").with_columns(columns)
test = pl.read_csv(PATH + "test.csv").with_columns(columns)



## === cell 2
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
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
train_feats.head(3)




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
train_feats.head(3)




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
train_feats.head(3)



## === cell 6
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
train_tfid = vectorizer.fit_transform([i for i in train["full_text"]])
df_tfid = pd.DataFrame(train_tfid.toarray())
df_tfid.columns = [f"tfid_{i}" for i in range(df_tfid.shape[1])]
df_tfid["essay_id"] = train_feats["essay_id"].values
train_feats = train_feats.merge(df_tfid, on="essay_id", how="left")
feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
print("Features Number after TF‑IDF:", len(feature_names))
train_feats.head(3)



## === cell 7
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
train_cnt = vectorizer_cnt.fit_transform([i for i in train["full_text"]])
df_cnt = pd.DataFrame(train_cnt.toarray())
df_cnt.columns = [f"tfid_cnt_{i}" for i in range(df_cnt.shape[1])]
df_cnt["essay_id"] = train_feats["essay_id"].values
train_feats = train_feats.merge(df_cnt, on="essay_id", how="left")
feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
print("Features Number after CountVectorizer:", len(feature_names))
train_feats.head(3)




## === cell 8
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 0.5 * np.sum((preds - labels) ** 2)
    g = 0.5 * np.sum((preds - a) ** 2 + b)
    df = preds - labels
    dg = preds - a
    grad = (df / g - f * dg / (g**2)) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess


a = 2.948
b = 1.092

X = train_feats[feature_names].astype(np.float32).values
y_int = train_feats["score"].astype(int).values  # integer labels for evaluation
y = train_feats["score"].astype(np.float32).values - a  # shifted target for regressor
oof = np.zeros_like(y_int, dtype=float)

if_train = True  # <-- train models in‑notebook

if if_train:
    n_splits = 5  # modest number of folds for speed
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    models = []
    f1_scores = []
    kappa_scores = []

    callbacks = [
        log_evaluation(period=25),
        early_stopping(stopping_rounds=30, first_metric_only=True),
    ]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_int), start=1):
        print(f"Fold {fold}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        y_val_int = y_int[val_idx]

        model = lgb.LGBMRegressor(
            objective=qwk_obj,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            colsample_bytree=0.7,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_estimators=500,
            random_state=42,
            verbosity=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_names=["train", "valid"],
            eval_metric=quadratic_weighted_kappa,
            callbacks=callbacks,
        )
        models.append(model)

        val_pred = model.predict(X_val) + a
        val_pred = np.clip(val_pred, 1, 6).round()
        oof[val_idx] = val_pred

        f1 = f1_score(y_val_int, val_pred, average="weighted")
        kappa = cohen_kappa_score(y_val_int, val_pred, weights="quadratic")
        f1_scores.append(f1)
        kappa_scores.append(kappa)

        print(f"  Fold F1: {f1:.4f}, QWK: {kappa:.4f}")

    print("--- Overall ---")
    print(f"Mean F1: {np.mean(f1_scores):.4f}")
    print(f"Mean QWK: {np.mean(kappa_scores):.4f}")
else:
    models = []



## === cell 9
tmp = Paragraph_Preprocess(test)
test_feats = Paragraph_Eng(tmp)
tmp = Sentence_Preprocess(test)
test_feats = test_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")
tmp = Word_Preprocess(test)
test_feats = test_feats.merge(Word_Eng(tmp), on="essay_id", how="left")

test_tfid = vectorizer.transform([i for i in test["full_text"]])
df_tfid = pd.DataFrame(test_tfid.toarray())
df_tfid.columns = [f"tfid_{i}" for i in range(df_tfid.shape[1])]
df_tfid["essay_id"] = test_feats["essay_id"].values
test_feats = test_feats.merge(df_tfid, on="essay_id", how="left")

test_cnt = vectorizer_cnt.transform([i for i in test["full_text"]])
df_cnt = pd.DataFrame(test_cnt.toarray())
df_cnt.columns = [f"tfid_cnt_{i}" for i in range(df_cnt.shape[1])]
df_cnt["essay_id"] = test_feats["essay_id"].values
test_feats = test_feats.merge(df_cnt, on="essay_id", how="left")

feature_names_test = [c for c in test_feats.columns if c not in ["essay_id", "score"]]
print("Test features count:", len(feature_names_test))
test_feats.head(3)



## === cell 10
probabilities = []
for model in models:
    pred = model.predict(test_feats[feature_names_test]) + a
    probabilities.append(pred)

if len(probabilities) == 0:
    mean_score = int(round(train_feats["score"].mean()))
    predictions = np.full(test_feats.shape[0], mean_score, dtype=int)
else:
    predictions = np.mean(probabilities, axis=0)
    if np.isscalar(predictions):
        predictions = np.full(test_feats.shape[0], predictions)

predictions = np.clip(predictions, 1, 6).round().astype(int)

print("Sample predictions:", predictions[:10])



## === cell 11
submission = pd.read_csv(PATH + "sample_submission.csv")
submission["score"] = predictions
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
display(submission.head())
