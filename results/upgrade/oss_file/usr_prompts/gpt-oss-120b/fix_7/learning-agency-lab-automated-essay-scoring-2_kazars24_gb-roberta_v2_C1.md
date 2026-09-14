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

0.8155382193257055

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I cleaned up the imports that caused a protobuf‑related crash, switched the script to actually train the LightGBM models (instead of trying to load a non‑existent pickle), and limited the cross‑validation to a modest number of folds so the notebook finishes quickly while still using the original feature engineering and evaluation logic. The rest of the pipeline remains unchanged, and a valid `submission.csv` is written at the end.'

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, polars as pl, re, warnings, gc, scipy.sparse as sp
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, f1_score
from lightgbm import log_evaluation, early_stopping
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

warnings.filterwarnings("ignore")



## === cell 1
PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
train = pl.read_csv(PATH + "train.csv")
test = pl.read_csv(PATH + "test.csv")



## === cell 2
html_re = re.compile(r"<.*?>")


def dataPreprocessing(x: str) -> str:
    x = x.lower()
    x = html_re.sub("", x)
    x = re.sub("@\\w+", "", x)
    x = re.sub("'\\d+", "", x)
    x = re.sub("\\d+", "", x)
    x = re.sub("http\\w+", "", x)
    x = re.sub("\\s+", " ", x)
    x = re.sub("\\.+", ".", x)
    x = re.sub(",+", ",", x)
    return x.strip()


train = train.with_columns(
    pl.col("full_text").map_elements(dataPreprocessing).alias("clean_text")
)
test = test.with_columns(
    pl.col("full_text").map_elements(dataPreprocessing).alias("clean_text")
)


def Paragraph_Preprocess(df):
    df = df.with_columns(
        pl.col("clean_text").str.split("\n\n").alias("paragraph")
    ).explode("paragraph")
    df = df.with_columns(
        pl.col("paragraph").str.len().alias("paragraph_len"),
        pl.col("paragraph")
        .str.split(".")
        .arr.lengths()
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph").str.split(" ").arr.lengths().alias("paragraph_word_cnt"),
    )
    return df


paragraph_fea = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]


def Paragraph_Eng(df):
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
    return (
        df.group_by("essay_id", maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .to_pandas()
    )


tmp = Paragraph_Preprocess(train)
train_feats = Paragraph_Eng(tmp)
train_feats["score"] = train["score"].to_numpy()
feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
print("Features Number after paragraph:", len(feature_names))
print(train_feats.head(3))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1780132852.py in <cell line: 0>()
     86 
     87 
---> 88 tmp = Paragraph_Preprocess(train)
     89 train_feats = Paragraph_Eng(tmp)
     90 train_feats["score"] = train["score"].to_numpy()

/tmp/ipykernel_11/1780132852.py in Paragraph_Preprocess(df)
     28     ).explode("paragraph")
     29     df = df.with_columns(
---> 30         pl.col("paragraph").str.len().alias("paragraph_len"),
     31         pl.col("paragraph")
     32         .str.split(".")

AttributeError: 'ExprStringNameSpace' object has no attribute 'len'

## === cell 3
def Sentence_Preprocess(df):
    df = df.with_columns(pl.col("clean_text").str.split(".").alias("sentence")).explode(
        "sentence"
    )
    df = df.with_columns(pl.col("sentence").str.len().alias("sentence_len"))
    df = df.filter(pl.col("sentence_len") >= 15)
    df = df.with_columns(
        pl.col("sentence").str.split(" ").arr.lengths().alias("sentence_word_cnt")
    )
    return df


sentence_fea = ["sentence_len", "sentence_word_cnt"]


def Sentence_Eng(df):
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
    return (
        df.group_by("essay_id", maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .to_pandas()
    )


tmp = Sentence_Preprocess(train)
train_feats = train_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")
feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
print("Features Number after sentence:", len(feature_names))
print(train_feats.head(3))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3871193215.py in <cell line: 0>()
     37 
     38 
---> 39 tmp = Sentence_Preprocess(train)
     40 train_feats = train_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")
     41 feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]

/tmp/ipykernel_11/3871193215.py in Sentence_Preprocess(df)
      3         "sentence"
      4     )
----> 5     df = df.with_columns(pl.col("sentence").str.len().alias("sentence_len"))
      6     df = df.filter(pl.col("sentence_len") >= 15)
      7     df = df.with_columns(

AttributeError: 'ExprStringNameSpace' object has no attribute 'len'

## === cell 4
def Word_Preprocess(df):
    df = df.with_columns(pl.col("clean_text").str.split(" ").alias("word")).explode(
        "word"
    )
    df = df.with_columns(pl.col("word").str.len().alias("word_len"))
    return df.filter(pl.col("word_len") != 0)


def Word_Eng(df):
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
    return (
        df.group_by("essay_id", maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .to_pandas()
    )


tmp = Word_Preprocess(train)
train_feats = train_feats.merge(Word_Eng(tmp), on="essay_id", how="left")
feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
print("Features Number after word:", len(feature_names))
print(train_feats.head(3))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2470858144.py in <cell line: 0>()
     31 
     32 
---> 33 tmp = Word_Preprocess(train)
     34 train_feats = train_feats.merge(Word_Eng(tmp), on="essay_id", how="left")
     35 feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]

/tmp/ipykernel_11/2470858144.py in Word_Preprocess(df)
      3         "word"
      4     )
----> 5     df = df.with_columns(pl.col("word").str.len().alias("word_len"))
      6     return df.filter(pl.col("word_len") != 0)
      7 

AttributeError: 'ExprStringNameSpace' object has no attribute 'len'

## === cell 5
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
train_tfid = vectorizer.fit_transform(train["full_text"].to_list())

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
train_cnt = vectorizer_cnt.fit_transform(train["full_text"].to_list())

X_dense = train_feats[feature_names].astype(np.float32).values
X = sp.hstack([X_dense, train_tfid, train_cnt]).tocsr()
y_int = train_feats["score"].astype(int).values
y = train_feats["score"].astype(np.float32).values - 2.948  # shifted target
print("Total feature count (dense + sparse):", X.shape[1])




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1562088871.py in <cell line: 0>()
     24 train_cnt = vectorizer_cnt.fit_transform(train["full_text"].to_list())
     25 
---> 26 X_dense = train_feats[feature_names].astype(np.float32).values
     27 X = sp.hstack([X_dense, train_tfid, train_cnt]).tocsr()
     28 y_int = train_feats["score"].astype(int).values

NameError: name 'train_feats' is not defined

## === cell 6
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + 2.948
    y_pred = (y_pred + 2.948).clip(1, 6).round()
    return "QWK", cohen_kappa_score(y_true, y_pred, weights="quadratic"), True


def qwk_obj(y_true, y_pred):
    a = 2.948
    f = 0.5 * np.sum((y_pred + a - (y_true + a)) ** 2)
    g = 0.5 * np.sum((y_pred + a - a) ** 2 + 1.092)
    df = (y_pred + a) - (y_true + a)
    dg = (y_pred + a) - a
    grad = (df / g - f * dg / (g**2)) * len(y_true)
    hess = np.ones(len(y_true))
    return grad, hess


oof = np.zeros_like(y_int, dtype=float)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

models = []
f1_scores = []
kappa_scores = []

callbacks = [
    log_evaluation(period=25),
    early_stopping(stopping_rounds=30, first_metric_only=True),
]

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_int), start=1):
    print(f"Fold {fold}")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
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

    val_pred = model.predict(X_val) + 2.948
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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3977296327.py in <cell line: 0>()
     16 
     17 
---> 18 oof = np.zeros_like(y_int, dtype=float)
     19 
     20 n_splits = 5

NameError: name 'y_int' is not defined

## === cell 7
tmp = Paragraph_Preprocess(test)
test_feats = Paragraph_Eng(tmp)
tmp = Sentence_Preprocess(test)
test_feats = test_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")
tmp = Word_Preprocess(test)
test_feats = test_feats.merge(Word_Eng(tmp), on="essay_id", how="left")

test_tfid = vectorizer.transform(test["full_text"].to_list())
test_cnt = vectorizer_cnt.transform(test["full_text"].to_list())

X_test_dense = (
    test_feats[[c for c in test_feats.columns if c not in ["essay_id", "score"]]]
    .astype(np.float32)
    .values
)
X_test = sp.hstack([X_test_dense, test_tfid, test_cnt]).tocsr()
print("Test features count:", X_test.shape[1])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3428226552.py in <cell line: 0>()
----> 1 tmp = Paragraph_Preprocess(test)
      2 test_feats = Paragraph_Eng(tmp)
      3 tmp = Sentence_Preprocess(test)
      4 test_feats = test_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")
      5 tmp = Word_Preprocess(test)

/tmp/ipykernel_11/1780132852.py in Paragraph_Preprocess(df)
     28     ).explode("paragraph")
     29     df = df.with_columns(
---> 30         pl.col("paragraph").str.len().alias("paragraph_len"),
     31         pl.col("paragraph")
     32         .str.split(".")

AttributeError: 'ExprStringNameSpace' object has no attribute 'len'

## === cell 8
probabilities = []
for model in models:
    pred = model.predict(X_test) + 2.948
    probabilities.append(pred)

if probabilities:
    predictions = np.mean(probabilities, axis=0)
else:
    mean_score = int(round(train_feats["score"].mean()))
    predictions = np.full(X_test.shape[0], mean_score, dtype=int)

predictions = np.clip(predictions, 1, 6).round().astype(int)
print("Sample predictions:", predictions[:10])



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/469967069.py in <cell line: 0>()
      1 probabilities = []
----> 2 for model in models:
      3     pred = model.predict(X_test) + 2.948
      4     probabilities.append(pred)
      5 

NameError: name 'models' is not defined

## === cell 9
submission = pd.read_csv(PATH + "sample_submission.csv")
submission["score"] = predictions
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
display(submission.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3918797640.py in <cell line: 0>()
      1 submission = pd.read_csv(PATH + "sample_submission.csv")
----> 2 submission["score"] = predictions
      3 submission.to_csv("submission.csv", index=False)
      4 print("Submission written to submission.csv")
      5 display(submission.head())

NameError: name 'predictions' is not defined
