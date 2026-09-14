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

0.7995683527601928

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, ctypes, random, time, re
from tqdm import tqdm
import pickle

import pandas as pd, numpy as np
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostRegressor, Pool
import catboost

import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # GPU flag (unused for CPU)




## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None
    LOAD_FEATURES_FROM = None
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 2
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()




## === cell 3
def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()



## === cell 4
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values("essay_id")
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values("essay_id")



## === cell 5
train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)




## === cell 6
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
    x = re.sub(r"[^\w\s.,;:\"'?!]", "", x)
    x = re.sub(r"paragraph", "", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    return x.strip()




## === cell 7
def count_misspelled_words(text):
    return 0  # placeholder – not used in final logic




## === cell 8
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]


def Paragraph_Features(df):
    df = df.explode("paragraph")
    df = df.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))
    df = df.with_columns(pl.col("paragraph").map_elements(len).alias("paragraph_len"))
    df = df.with_columns(pl.lit(0).alias("paragraph_misspelled_cnt"))
    df = df.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: x.count(","))
        .alias("paragraph_comma_cnt")
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
            for i in [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_len") <= i)
            .count()
            .alias(f"paragraph_{i}_cnt_v2")
            for i in [100, 200]
        ],
        pl.col("paragraph")
        .filter((pl.col("paragraph_len") <= 300) & (pl.col("paragraph_len") > 100))
        .count()
        .alias("short_paragraph_cnt"),
        pl.col("paragraph")
        .filter((pl.col("paragraph_len") <= 500) & (pl.col("paragraph_len") > 300))
        .count()
        .alias("mid_paragraph_cnt"),
        pl.col("paragraph")
        .filter((pl.col("paragraph_len") <= 700) & (pl.col("paragraph_len") > 500))
        .count()
        .alias("long_paragraph_cnt"),
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_sentence_cnt") >= i)
            .count()
            .alias(f"paragraph_sentence_{i}_cnt")
            for i in [2, 4, 6, 8, 10]
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_word_cnt") >= i)
            .count()
            .alias(f"paragraph_word_{i}_cnt")
            for i in [20, 40, 60, 90, 120]
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_comma_cnt") >= i)
            .count()
            .alias(f"paragraph_comma_{i}_cnt")
            for i in [1, 2, 3, 4, 5]
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_misspelled_cnt") >= i)
            .count()
            .alias(f"paragraph_misspelled_{i}_cnt")
            for i in [4, 8, 12, 16]
        ],
        pl.col("paragraph").count().alias("paragraph_cnt"),
    ]
    for feat in paragraph_features:
        aggs += [
            pl.col(feat).max().alias(f"{feat}_max"),
            pl.col(feat).mean().alias(f"{feat}_mean"),
            pl.col(feat).min().alias(f"{feat}_min"),
            pl.col(feat).std().alias(f"{feat}_std"),
            pl.col(feat).sum().alias(f"{feat}_sum"),
            pl.col(feat).quantile(0.25).alias(f"{feat}_q1"),
            pl.col(feat).quantile(0.75).alias(f"{feat}_q3"),
        ]
    df = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()




## === cell 9
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(df):
    df = df.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(".")
        .alias("sentence")
    )
    df = df.explode("sentence")
    df = df.with_columns(pl.col("sentence").str.lengths().alias("sentence_len"))
    df = df.filter(pl.col("sentence_len") > 3)
    df = df.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.replace(" ", "")))
        .alias("only_sentence_len")
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
            for i in [40, 60, 70, 80, 100, 120, 140]
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") <= i)
            .count()
            .alias(f"sentence_{i}_cnt_v2")
            for i in [10, 20, 30]
        ],
        pl.col("sentence")
        .filter((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 40))
        .count()
        .alias("short_sentence_cnt"),
        pl.col("sentence")
        .filter((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
        .count()
        .alias("mid_sentence_cnt"),
        pl.col("sentence")
        .filter((pl.col("sentence_len") <= 140) & (pl.col("sentence_len") > 100))
        .count()
        .alias("long_sentence_cnt"),
        *[
            pl.col("sentence")
            .filter(pl.col("only_sentence_len") >= i)
            .count()
            .alias(f"only_sentence_{i}_cnt")
            for i in [40, 60, 80, 100, 120]
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_word_cnt") >= i)
            .count()
            .alias(f"sentence_word_{i}_cnt")
            for i in [10, 15, 20, 25]
        ],
        pl.col("sentence").count().alias("sentence_cnt"),
    ]
    for feat in sentence_features:
        aggs += [
            pl.col(feat).max().alias(f"{feat}_max"),
            pl.col(feat).mean().alias(f"{feat}_mean"),
            pl.col(feat).min().alias(f"{feat}_min"),
            pl.col(feat).std().alias(f"{feat}_std"),
            pl.col(feat).sum().alias(f"{feat}_sum"),
            pl.col(feat).quantile(0.25).alias(f"{feat}_q1"),
            pl.col(feat).quantile(0.75).alias(f"{feat}_q3"),
        ]
    df = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    df = df.with_columns(
        (pl.col("sentence_cnt") / pl.col("sentence_cnt")).alias("dummy")
    )
    df = df.with_columns(
        *[
            (pl.col(f"sentence_{i}_cnt") / pl.col("sentence_cnt")).alias(
                f"sentence_{i}_cnt_ratio"
            )
            for i in [40, 60, 70, 80, 100, 120, 140]
        ],
        (pl.col("short_sentence_cnt") / pl.col("sentence_cnt")).alias(
            "short_sentence_cnt_ratio"
        ),
        (pl.col("mid_sentence_cnt") / pl.col("sentence_cnt")).alias(
            "mid_sentence_cnt_ratio"
        ),
        (pl.col("long_sentence_cnt") / pl.col("sentence_cnt")).alias(
            "long_sentence_cnt_ratio"
        ),
    ).sort("essay_id")
    return df.to_pandas()




## === cell 10
word_features = ["word_len"]


def Word_Features(df):
    df = df.with_columns(
        pl.col("full_text").map_elements(dataPreprocessing).str.split(" ").alias("word")
    )
    df = df.explode("word")
    df = df.with_columns(pl.col("word").str.len().alias("word_len"))
    df = df.filter(pl.col("word_len") > 0)
    return df


def Word_aggregation(df):
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i)
            .count()
            .alias(f"word_{i}_cnt")
            for i in [3, 4, 5, 6, 7, 8, 10]
        ],
        *[
            pl.col("word")
            .filter(pl.col("word_len") <= i)
            .count()
            .alias(f"word_{i}_cnt_v2")
            for i in [1, 2, 3]
        ],
        pl.col("word").count().alias("word_cnt"),
    ]
    for feat in word_features:
        aggs += [
            pl.col(feat).max().alias(f"{feat}_max"),
            pl.col(feat).mean().alias(f"{feat}_mean"),
            pl.col(feat).min().alias(f"{feat}_min"),
            pl.col(feat).std().alias(f"{feat}_std"),
            pl.col(feat).sum().alias(f"{feat}_sum"),
            pl.col(feat).quantile(0.25).alias(f"{feat}_q1"),
            pl.col(feat).quantile(0.75).alias(f"{feat}_q3"),
        ]
    df = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    df = df.with_columns(
        *[
            (pl.col(f"word_{i}_cnt") / pl.col("word_cnt")).alias(f"word_{i}_cnt_ratio")
            for i in [3, 4, 5, 6, 7, 8, 10]
        ],
        *[
            (pl.col(f"word_{i}_cnt_v2") / pl.col("word_cnt")).alias(
                f"word_{i}_cnt_v2_ratio"
            )
            for i in [1, 2, 3]
        ],
    ).sort("essay_id")
    return df.to_pandas()




## === cell 11
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 4),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,
)

train_a = train.with_columns(pl.col("full_text").map_elements(dataPreprocessing))
train_tfid = vectorizer.fit_transform([i for i in train_a["full_text"]])
df_tfid = pd.DataFrame(
    train_tfid.toarray(), columns=[f"tfidf_{i}" for i in range(train_tfid.shape[1])]
)
df_tfid["essay_id"] = df_train["essay_id"].values



## === cell 12
vectorizer_cnt = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(2, 4),
    min_df=0.10,
    max_df=0.85,
)

train_b = train.with_columns(pl.col("full_text").map_elements(dataPreprocessing))
train_cnt = vectorizer_cnt.fit_transform([i for i in train_b["full_text"]])
df_cnt = pd.DataFrame(
    train_cnt.toarray(), columns=[f"cnt_{i}" for i in range(train_cnt.shape[1])]
)
df_cnt["essay_id"] = df_train["essay_id"].values



## === cell 13
if CFG.LOAD_FEATURES_FROM and os.path.exists(CFG.LOAD_FEATURES_FROM):
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)
else:
    pf = Paragraph_Features(train)
    pf = Paragraph_aggregation(pf)
    sf = Sentence_Features(train)
    sf = Sentence_aggregation(sf)
    wf = Word_Features(train)
    wf = Word_aggregation(wf)
    train_feats = pf.merge(sf, on="essay_id", how="left")
    train_feats = train_feats.merge(wf, on="essay_id", how="left")
    train_feats = train_feats.merge(df_tfid, on="essay_id", how="left")
    train_feats = train_feats.merge(df_cnt, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/4000879788.py in <cell line: 0>()
      4     pf = Paragraph_Features(train)
      5     pf = Paragraph_aggregation(pf)
----> 6     sf = Sentence_Features(train)
      7     sf = Sentence_aggregation(sf)
      8     wf = Word_Features(train)

/tmp/ipykernel_55/826531372.py in Sentence_Features(df)
     11     df = df.explode("sentence")
     12     # Use the correct Polars method for string length
---> 13     df = df.with_columns(pl.col("sentence").str.lengths().alias("sentence_len"))
     14     df = df.filter(pl.col("sentence_len") > 3)
     15     df = df.with_columns(

AttributeError: 'ExprStringNameSpace' object has no attribute 'lengths'

## === cell 14
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [c for c in train_feats.columns if c not in categorical_columns + ["score"]]
TARGET = "score"




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3953984253.py in <cell line: 0>()
----> 1 categorical_columns = train_feats.select_dtypes(
      2     include=["object", "category"]
      3 ).columns.tolist()
      4 FEATURES = [c for c in train_feats.columns if c not in categorical_columns + ["score"]]
      5 TARGET = "score"

NameError: name 'train_feats' is not defined

## === cell 15
def train_catboost():
    all_oof = []
    all_true = []
    skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)
    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(train_feats, train_feats[TARGET])
    ):
        print(f"Fold {fold+1}")
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=5,
            subsample=0.8,
            l2_leaf_reg=1,
            task_type="CPU",
            devices="0",
            objective="RMSE",
            eval_metric="RMSE",
            random_state=CFG.SEED,
            verbose=0,
        )
        train_pool = Pool(
            data=np.clip(train_feats.loc[tr_idx, FEATURES].fillna(0), 0, 10000),
            label=train_feats.loc[tr_idx, TARGET],
        )
        valid_pool = Pool(
            data=np.clip(train_feats.loc[val_idx, FEATURES].fillna(0), 0, 10000),
            label=train_feats.loc[val_idx, TARGET],
        )
        model.fit(
            train_pool, eval_set=valid_pool, early_stopping_rounds=75, verbose=100
        )
        pickle.dump(model, open(f"CAT_v{CFG.VER}_f{fold}.pkl", "wb"))
        oof = model.predict(valid_pool)
        all_oof.append(oof)
        all_true.append(train_feats.loc[val_idx, TARGET].values)
        del model, train_pool, valid_pool, oof
        clean_memory()
    oof_all = np.concatenate(all_oof)
    true_all = np.concatenate(all_true)
    cv = cohen_kappa_score(
        true_all, np.clip(oof_all, 1, 6).round(), weights="quadratic"
    )
    print("Overall CV QWK:", cv)




## === cell 16
if not os.path.exists(f"CAT_v{CFG.VER}_f0.pkl"):
    train_catboost()
else:
    print("Models already exist – skipping training.")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3788825951.py in <cell line: 0>()
      1 if not os.path.exists(f"CAT_v{CFG.VER}_f0.pkl"):
----> 2     train_catboost()
      3 else:
      4     print("Models already exist – skipping training.")
      5 

/tmp/ipykernel_55/3169039594.py in train_catboost()
      4     skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)
      5     for fold, (tr_idx, val_idx) in enumerate(
----> 6         skf.split(train_feats, train_feats[TARGET])
      7     ):
      8         print(f"Fold {fold+1}")

NameError: name 'train_feats' is not defined

## === cell 17
model = pickle.load(open(f"CAT_v{CFG.VER}_f0.pkl", "rb"))
df_importance = pd.DataFrame(
    {"features_name": FEATURES, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_importance.head(30), x="importance", y="features_name", palette="rocket"
)
plt.title("Top 30 Feature Importances")
plt.show()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4030708921.py in <cell line: 0>()
----> 1 model = pickle.load(open(f"CAT_v{CFG.VER}_f0.pkl", "rb"))
      2 df_importance = pd.DataFrame(
      3     {"features_name": FEATURES, "importance": model.feature_importances_}
      4 ).sort_values("importance", ascending=False)
      5 plt.figure(figsize=(12, 6))

FileNotFoundError: [Errno 2] No such file or directory: 'CAT_v1_f0.pkl'

## === cell 18
test_a = test.with_columns(pl.col("full_text").map_elements(dataPreprocessing))
test_tfid = vectorizer.transform([i for i in test_a["full_text"]])
df_test_tfid = pd.DataFrame(
    test_tfid.toarray(), columns=[f"tfidf_{i}" for i in range(test_tfid.shape[1])]
)
df_test_tfid["essay_id"] = df_test["essay_id"].values

test_b = test.with_columns(pl.col("full_text").map_elements(dataPreprocessing))
test_cnt = vectorizer_cnt.transform([i for i in test_b["full_text"]])
df_test_cnt = pd.DataFrame(
    test_cnt.toarray(), columns=[f"cnt_{i}" for i in range(test_cnt.shape[1])]
)
df_test_cnt["essay_id"] = df_test["essay_id"].values



## === cell 19
test_pf = Paragraph_Features(test)
test_pf = Paragraph_aggregation(test_pf)
test_sf = Sentence_Features(test)
test_sf = Sentence_aggregation(test_sf)
test_wf = Word_Features(test)
test_wf = Word_aggregation(test_wf)

test_feats = test_pf.merge(test_sf, on="essay_id", how="left")
test_feats = test_feats.merge(test_wf, on="essay_id", how="left")
test_feats = test_feats.merge(df_test_tfid, on="essay_id", how="left")
test_feats = test_feats.merge(df_test_cnt, on="essay_id", how="left")
print("Test feature shape:", test_feats.shape)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/348806747.py in <cell line: 0>()
      1 test_pf = Paragraph_Features(test)
      2 test_pf = Paragraph_aggregation(test_pf)
----> 3 test_sf = Sentence_Features(test)
      4 test_sf = Sentence_aggregation(test_sf)
      5 test_wf = Word_Features(test)

/tmp/ipykernel_55/826531372.py in Sentence_Features(df)
     11     df = df.explode("sentence")
     12     # Use the correct Polars method for string length
---> 13     df = df.with_columns(pl.col("sentence").str.lengths().alias("sentence_len"))
     14     df = df.filter(pl.col("sentence_len") > 3)
     15     df = df.with_columns(

AttributeError: 'ExprStringNameSpace' object has no attribute 'lengths'

## === cell 20
preds = []
test_categorical = test_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
test_FEATURES = [c for c in test_feats.columns if c not in test_categorical]

for fold in range(10):
    model_path = f"CAT_v{CFG.VER}_f{fold}.pkl"
    if os.path.exists(model_path):
        model = pickle.load(open(model_path, "rb"))
        pred = model.predict(test_feats[test_FEATURES])
        preds.append(pred)
    else:
        print(f"Warning: model file {model_path} not found.")

pred = np.mean(preds, axis=0)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1471645202.py in <cell line: 0>()
      1 preds = []
----> 2 test_categorical = test_feats.select_dtypes(
      3     include=["object", "category"]
      4 ).columns.tolist()
      5 test_FEATURES = [c for c in test_feats.columns if c not in test_categorical]

NameError: name 'test_feats' is not defined

## === cell 21
submission = pd.DataFrame(
    {"essay_id": df_test["essay_id"], TARGET: pred.clip(1, 6).round().astype(int)}
)
submission.to_csv("submission.csv", index=False)
print("Submission saved. Shape:", submission.shape)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2502951147.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"essay_id": df_test["essay_id"], TARGET: pred.clip(1, 6).round().astype(int)}
      3 )
      4 submission.to_csv("submission.csv", index=False)
      5 print("Submission saved. Shape:", submission.shape)

NameError: name 'TARGET' is not defined
