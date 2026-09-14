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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

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

0.8037614614128175

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the data source used for the TF‑IDF and CountVectorizer pipelines (they were mistakenly using the Polars frames instead of the pandas DataFrames), and I adjust the final rounding to use `np.rint` for a more neutral rounding method. These minimal changes ensure the script runs end‑to‑end, creates a proper `submission.csv`, and aligns the predictions more closely with the target quadratic weighted kappa.'
- What this solution (achieved 0.0) has done: 'I make a tiny adjustment to the XGBoost training hyper‑parameters so the model converges a bit earlier (learning_rate = 0.05 and early_stopping_rounds = 100). This small change keeps the core logic unchanged, still produces a valid `submission.csv`, and is expected to move the quadratic weighted kappa toward the target score without risking over‑fitting or large computational overhead.'

# 9. Code solution

## === cell 0
import os
import re
import gc
import random
import pickle
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb

OOF_BIAS = 0.0


def seed_everything(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clean_memory():
    gc.collect()


class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None
    LOAD_FEATURES_FROM = None
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"


seed_everything(CFG.SEED)
clean_memory()




## === cell 1
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
print("Shape of Train: ", df_train.shape)
display(df_train.head())

df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
print("Shape of Test: ", df_test.shape)
display(df_test.head())




## === cell 2
train = pl.read_csv(os.path.join(CFG.BASE_PATH, "train.csv")).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.read_csv(os.path.join(CFG.BASE_PATH, "test.csv")).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)




## === cell 3
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
    x = x.strip()
    return x




## === cell 4
paragraph_features = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]


def Paragraph_Features(x):
    x = x.explode("paragraph")
    print("Paragraph Preprocessing")
    x = x.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))
    print("Calculate the length of each paragraph")
    x = x.with_columns(
        pl.col("paragraph").map_elements(lambda s: len(s)).alias("paragraph_len")
    )
    print("Calculate the number of sentences and words in each paragraph")
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda s: len(s.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda s: len(s.split(" ")))
        .alias("paragraph_word_cnt"),
    )
    return x


def Paragraph_aggregation(x):
    print("Aggregation")
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
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_sentence_cnt") >= i)
            .count()
            .alias(f"paragraph_sentence_{i}_cnt")
            for i in [2, 4, 6, 8, 10]
        ],
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 300) & (pl.col("paragraph_len") > 100))
            .count()
            .alias("short_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 500) & (pl.col("paragraph_len") > 300))
            .count()
            .alias("mid_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 700) & (pl.col("paragraph_len") > 500))
            .count()
            .alias("long_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 4)
                & (pl.col("paragraph_sentence_cnt") > 2)
            )
            .count()
            .alias("short_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 8)
                & (pl.col("paragraph_sentence_cnt") > 4)
            )
            .count()
            .alias("mid_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 10)
                & (pl.col("paragraph_sentence_cnt") > 8)
            )
            .count()
            .alias("long_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_word_cnt") >= i)
            .count()
            .alias(f"paragraph_word_{i}_cnt")
            for i in [30, 60, 90, 120]
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 60)
                & (pl.col("paragraph_word_cnt") > 30)
            )
            .count()
            .alias("short_paragraph_word_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 90)
                & (pl.col("paragraph_word_cnt") > 60)
            )
            .count()
            .alias("mid_paragraph_word_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 120)
                & (pl.col("paragraph_word_cnt") > 90)
            )
            .count()
            .alias("long_paragraph_word_cnt")
        ],
        *[pl.col("paragraph").count().alias("paragraph_cnt")],
    ]
    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()




## === cell 5
sentence_features = [
    "sentence_len",
    "sentence_word_cnt",
    "sentence_len_space_ratio",
    "sentence_word_space_ratio",
]


def Sentence_Features(x):
    print("Preprocess full_text and split into sentences")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(lambda s: dataPreprocessing(s))
        .str.split(".")
        .alias("sentence")
    )
    x = x.explode("sentence")
    print("Calculate sentence statistics")
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: s.count(" "))
        .alias("sentence_space_cnt")
    )
    x = x.filter(pl.col("sentence_space_cnt") > 0)
    x = x.with_columns(
        pl.col("sentence").map_elements(lambda s: len(s)).alias("sentence_len")
    )
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(
        (pl.col("sentence_len") / pl.col("sentence_space_cnt")).alias(
            "sentence_len_space_ratio"
        )
    )
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s.split(" ")))
        .alias("sentence_word_cnt")
    )
    x = x.with_columns(
        (pl.col("sentence_word_cnt") / pl.col("sentence_space_cnt")).alias(
            "sentence_word_space_ratio"
        )
    )
    return x


def Sentence_aggregation(x):
    print("Aggregation")
    aggs = [
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") >= i)
            .count()
            .alias(f"sentence_{i}_cnt")
            for i in [30, 40, 50, 60, 70, 80, 100, 150]
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") <= i)
            .count()
            .alias(f"sentence_{i}_cnt_v2")
            for i in [10, 20, 30]
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 50) & (pl.col("sentence_len") > 30))
            .count()
            .alias("short_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 50))
            .count()
            .alias("mid_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
            .count()
            .alias("long_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_word_cnt") >= i)
            .count()
            .alias(f"sentence_word_{i}_cnt")
            for i in [5, 10, 15, 20]
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 10) & (pl.col("sentence_word_cnt") > 5)
            )
            .count()
            .alias("short_sentence_word_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 15) & (pl.col("sentence_word_cnt") > 10)
            )
            .count()
            .alias("mid_sentence_word_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15)
            )
            .count()
            .alias("long_sentence_word_cnt")
        ],
        *[pl.col("sentence").count().alias("sentence_cnt")],
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in sentence_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in sentence_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in sentence_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in sentence_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in sentence_features],
    ]
    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()




## === cell 6
word_features = ["word_len"]


def Word_Features(x):
    print("Preprocess full_text and split into words")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(lambda s: dataPreprocessing(s))
        .str.split(".")
        .alias("sentence")
    )
    x = x.explode("sentence")
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: dataPreprocessing(s))
        .str.split(" ")
        .alias("word")
    )
    x = x.explode("word")
    print("Calculate word length")
    x = x.with_columns(pl.col("word").map_elements(lambda w: len(w)).alias("word_len"))
    x = x.filter(pl.col("word_len") > 0)
    return x


def Word_aggregation(x):
    print("Aggregation")
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i)
            .count()
            .alias(f"word_{i}_cnt")
            for i in [3, 4, 5, 6, 8, 10, 15]
        ],
        *[
            pl.col("word")
            .filter(pl.col("word_len") <= i)
            .count()
            .alias(f"word_{i}_cnt_v2")
            for i in [2, 3]
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 4) & (pl.col("word_len") > 2))
            .count()
            .alias("short_word_cnt")
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 7) & (pl.col("word_len") > 4))
            .count()
            .alias("mid_word_cnt")
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 10) & (pl.col("word_len") > 7))
            .count()
            .alias("long_word_cnt")
        ],
        *[pl.col("word").count().alias("word_cnt")],
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in word_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in word_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in word_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in word_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in word_features],
        *[pl.col(feat).quantile(0.25).alias(f"{feat}_q1") for feat in word_features],
        *[pl.col(feat).quantile(0.75).alias(f"{feat}_q3") for feat in word_features],
    ]
    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.with_columns(
        *[
            (pl.col(f"word_{i}_cnt") / pl.col("word_2_cnt_v2")).alias(
                f"word_2_{i}_cnt_ratio"
            )
            for i in [3, 4, 5, 6, 8, 10, 15]
        ],
        *[
            (pl.col(f"word_{i}_cnt") / pl.col("word_3_cnt_v2")).alias(
                f"word_3_{i}_cnt_ratio"
            )
            for i in [3, 4, 5, 6, 8, 10, 15]
        ],
        *[
            (pl.col("short_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"short_word_ratio_{i}"
            )
            for i in [2, 3]
        ],
        *[
            (pl.col("mid_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"mid_word_ratio_{i}"
            )
            for i in [2, 3]
        ],
        *[
            (pl.col("long_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"long_word_ratio_{i}"
            )
            for i in [2, 3]
        ],
    ).sort("essay_id")
    return df.to_pandas()




## === cell 7
vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 3),
    min_df=0.001,
    max_df=0.98,
    sublinear_tf=True,
    max_features=1000,
)

train_tfid = vectorizer.fit_transform(df_train["full_text"].astype(str))
dense_matrix = train_tfid.astype(np.float32).toarray()
df_tfid = pd.DataFrame(dense_matrix)
df_tfid.columns = [f"tfidf_{i}" for i in range(df_tfid.shape[1])]
df_tfid["essay_id"] = df_train["essay_id"]




## === cell 8
vectorizer_cnt = CountVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=0.001,
    max_df=0.95,
    max_features=1000,
)

train_cnt = vectorizer_cnt.fit_transform(df_train["full_text"].astype(str))
dense_matrix2 = train_cnt.astype(np.float32).toarray()
df_cnt = pd.DataFrame(dense_matrix2)
df_cnt.columns = [f"cnt_{i}" for i in range(df_cnt.shape[1])]
df_cnt["essay_id"] = df_train["essay_id"]




## === cell 9
if CFG.LOAD_FEATURES_FROM is None:
    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train)
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train)
    train_feats3 = Word_aggregation(train_feats3)

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats = train_feats.merge(df_tfid, on="essay_id", how="left")
    train_feats = train_feats.merge(df_cnt, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values
else:
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)




## === cell 10
if CFG.LOAD_FEATURES_FROM is None:
    print("Saving computed train features")
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Loading pre‑computed train features")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)

print("Train features shape:", train_feats.shape)
display(train_feats.head())




## === cell 11
print("XGBoost version:", xgb.__version__)




## === cell 12
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"




## === cell 13
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")




## === cell 14
def xgboost_train():
    global OOF_BIAS  # allow updating the global bias variable
    all_oof = []
    all_true = []
    skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(
        skf.split(train_feats, train_feats[TARGET])
    ):
        print("#" * 25)
        print(f"Fold {i+1}")
        print(f"Train size: {len(train_idx)}, Valid size: {len(valid_idx)}")
        print("#" * 25)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=0.05,
            max_depth=8,  # reduced from 12
            min_child_weight=1,  # relaxed from 3
            subsample=0.9,  # slightly increased
            colsample_bytree=0.9,  # slightly increased
            n_estimators=500,
            random_state=CFG.SEED,
            verbosity=0,
        )

        train_x = np.clip(train_feats.loc[train_idx, FEATURES].fillna(0), 0, 10000)
        train_y = train_feats.loc[train_idx, TARGET]

        valid_x = np.clip(train_feats.loc[valid_idx, FEATURES].fillna(0), 0, 10000)
        valid_y = train_feats.loc[valid_idx, TARGET]

        model.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            early_stopping_rounds=100,
            verbose=50,
        )

        pickle.dump(model, open(f"XGB_v{CFG.VER}_f{i}.pkl", "wb"))

        oof = model.predict(valid_x)
        all_oof.append(oof)
        all_true.append(valid_y.values)

        del train_x, train_y, valid_x, valid_y, oof, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    cv_score = cohen_kappa_score(
        all_true, np.rint(np.clip(all_oof, 1, 6)), weights="quadratic"
    )
    print("CV Quadratic Weighted Kappa:", cv_score)

    OOF_BIAS = float(train_feats[TARGET].mean() - all_oof.mean())
    print("Derived OOF bias for test adjustment:", OOF_BIAS)

    bias_path = f"OOF_bias_v{CFG.VER}.pkl"
    with open(bias_path, "wb") as f:
        pickle.dump(OOF_BIAS, f)
    print(f"Saved OOF bias to {bias_path}")




## === cell 15
if CFG.LOAD_MODELS_FROM is None:
    print("Training XBoost models")
    xgboost_train()
else:
    print("Skipping training – loading existing models")
    bias_path = f"OOF_bias_v{CFG.VER}.pkl"
    if os.path.exists(bias_path):
        OOF_BIAS = pickle.load(open(bias_path, "rb"))
        print(f"Loaded OOF bias from {bias_path}: {OOF_BIAS}")
    else:
        print("No saved OOF bias found – using default bias of 0.0")




## === cell 16
if os.path.exists(f"XGB_v{CFG.VER}_f0.pkl"):
    model = pickle.load(open(f"XGB_v{CFG.VER}_f0.pkl", "rb"))
    df_importance = pd.DataFrame(
        {"features_name": FEATURES, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(
        df_importance["features_name"].head(30), df_importance["importance"].head(30)
    )
    plt.title("Top 30 Feature Importances (XGBoost)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
else:
    print("Model file not found – skipping importance plot")




## === cell 17
test_tfid = vectorizer.transform(df_test["full_text"].astype(str))
dense_matrix_test = test_tfid.astype(np.float32).toarray()
df_test_tfid = pd.DataFrame(dense_matrix_test)
df_test_tfid.columns = [f"tfidf_{i}" for i in range(df_test_tfid.shape[1])]
df_test_tfid["essay_id"] = df_test["essay_id"]




## === cell 18
test_cnt = vectorizer_cnt.transform(df_test["full_text"].astype(str))
dense_matrix_test_cnt = test_cnt.astype(np.float32).toarray()
df_test_cnt = pd.DataFrame(dense_matrix_test_cnt)
df_test_cnt.columns = [f"cnt_{i}" for i in range(df_test_cnt.shape[1])]
df_test_cnt["essay_id"] = df_test["essay_id"]




## === cell 19
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)




## === cell 20
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
test_feats = test_feats.merge(df_test_tfid, on="essay_id", how="left")
test_feats = test_feats.merge(df_test_cnt, on="essay_id", how="left")
print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())




## === cell 21
preds = []
categorical_columns_test = test_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES_TEST = [c for c in test_feats.columns if c not in categorical_columns_test]

for i in range(5):
    print(f"Predicting with Fold {i+1}")
    model_path = f"XGB_v{CFG.VER}_f{i}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    model = pickle.load(open(model_path, "rb"))
    pred = model.predict(test_feats[FEATURES_TEST])
    preds.append(pred)

pred = np.mean(preds, axis=0)

pred = pred + OOF_BIAS

sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
sub[TARGET] = np.clip(np.rint(pred), 1, 6).astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv with shape", sub.shape)
display(sub.head())
