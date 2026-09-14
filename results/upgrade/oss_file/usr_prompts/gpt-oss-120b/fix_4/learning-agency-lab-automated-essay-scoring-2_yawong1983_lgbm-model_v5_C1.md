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
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.8034006002900264

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I convert the Polars dataframes to pandas only when extracting the raw text for TF‑IDF, and switch the script to train the LightGBM models instead of trying to load non‑existent checkpoints. These minimal changes fix the attribute errors, allow the TF‑IDF vectorizer to be fitted, and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.01713) has done: 'I adjust the TF‑IDF vectorizer to keep more informative n‑grams, slightly enlarge the LightGBM tree capacity, and make sure any missing feature values in the test set are filled before prediction. These small tweaks preserve the overall pipeline while allowing the model to produce meaningful scores instead of the current zero‑score output, moving the evaluation metric closer to the target.'

# 9. Code solution

## === cell 0
import re
import copy
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgbm
from tqdm.auto import tqdm, trange
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import cohen_kappa_score, accuracy_score

import warnings

warnings.filterwarnings("ignore")



## === cell 1
path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"


def load_db(name):
    columns = [(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))]

    df = pl.read_csv(path + name + ".csv").with_columns(columns)
    print(f"< {name} DataFrame Info >")
    print(df.head(1))
    return df




## === cell 2
df_train = load_db("train")
df_test = load_db("test")




## === cell 3
def Data_clearning(x):
    x = x.lower()  # lowercast
    x = re.compile(r"<.*?>").sub(r"", x)  # remove html
    x = re.sub("@\w+", "", x)  # starting @ word remove
    x = re.sub("'\d+", "", x)  # number
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)  # url
    x = re.sub(r"\s+", " ", x)  # single space
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


def get_feature(x):
    x = x.explode("paragraph")  # expand dataframe by paragrahp
    x = x.with_columns(pl.col("paragraph").map_elements(Data_clearning))
    x = x.with_columns(
        pl.col("paragraph").map_elements(lambda x: len(x)).alias("par_len")
    )  # paragraph length
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(".")))
        .alias("par_sentence_cnt")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("par_word_cnt")
    )
    return x




## === cell 4
tmp = get_feature(df_train)
add_f = ["par_len", "par_sentence_cnt", "par_word_cnt"]
tmp = tmp.to_pandas()



## === cell 5
print(tmp.describe())



## === cell 6
import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(3, 1, figsize=(13, 15))

for i, col in enumerate(add_f):
    sns.histplot(data=tmp, x=col, bins=100, kde=True, ax=ax[i])
    ax[i].set_title(f"Distribution: {col}")

plt.show()




## === cell 7
def add_satis(df, add_f):  # make additional features
    aggs = [
        *[
            pl.col("paragraph")
            .filter(pl.col(add_f[0]) >= i)
            .count()
            .alias(f"par_{i}_cnt")
            for i in np.arange(0, 701, 25)
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in add_f],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in add_f],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in add_f],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in add_f],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in add_f],
    ]

    df = df.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df


tmp = get_feature(df_train)
train_df2 = add_satis(tmp, add_f)
train_df2["score"] = df_train["score"]  # score col add

print("Feature cnt: ", train_df2.shape[1])
train_df2.head(3)




## === cell 8
def Sentence_process(df):
    df = df.with_columns(
        pl.col("full_text").map_elements(Data_clearning).str.split(by=".").alias("sen")
    )
    df = df.explode("sen")  # exploding df by sentence
    df = df.with_columns(pl.col("sen").map_elements(lambda x: len(x)).alias("sen_len"))

    sns.histplot(x="sen_len", data=df, stat="percent", bins=100)
    plt.show()

    df = df.filter(pl.col("sen_len") >= 15)
    df = df.with_columns(
        pl.col("sen").map_elements(lambda x: len(x.split(" "))).alias("sen_word_cnt")
    )

    return df


sen_fea = ["sen_len", "sen_word_cnt"]


def Sentence_addf(df, sen_fea):
    _ = [15]
    _.extend(np.arange(50, 301, 50))
    aggs = [
        *[
            pl.col("sen").filter(pl.col("sen_len") >= i).count().alias(f"sen_{i}_cnt")
            for i in _
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in sen_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sen_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in sen_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in sen_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in sen_fea],
    ]
    df = df.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df


tmp = Sentence_process(df_train)

train_df = train_df2.merge(Sentence_addf(tmp, sen_fea), on="essay_id", how="left")

print("train_df's feature number: ", train_df.shape[1])
train_df.head(3)




## === cell 9
def Word_process(df):
    df = df.with_columns(
        pl.col("full_text").map_elements(Data_clearning).str.split(by=" ").alias("word")
    )
    df = df.explode("word")  # exploding df by sentence
    df = df.with_columns(
        pl.col("word").map_elements(lambda x: len(x)).alias("word_len")
    )

    sns.histplot(x="word_len", data=df, stat="percent", bins=100)
    plt.show()

    df = df.filter(pl.col("word_len") != 0)

    return df


def Word_addf(df):
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i + 1)
            .count()
            .alias(f"word_{i}_cnt")
            for i in np.arange(15)
        ],
        *[pl.col("word_len").max().alias("word_len_max")],
        *[pl.col("word_len").mean().alias("word_len_mean")],
        *[pl.col("word_len").min().alias("word_len_min")],
        *[pl.col("word_len").std().alias("word_len_std")],
        *[pl.col("word_len").quantile(0.25).alias("word_len_q1")],
        *[pl.col("word_len").quantile(0.5).alias("word_len_q2")],
        *[pl.col("word_len").quantile(0.75).alias("word_len_q3")],
    ]
    df = df.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df


tmp = Word_process(df_train)

train_df = train_df.merge(Word_addf(tmp), on="essay_id", how="left")

print("train_df's feature number: ", train_df.shape[1])
train_df.head(3)



## === cell 10
train_text = df_train.to_pandas()["full_text"].apply(Data_clearning)
test_text = df_test.to_pandas()["full_text"].apply(Data_clearning)




## === cell 11
def quard_w_kappa(y_true, y_pred):
    pred = y_pred.clip(1, 6).round()
    true = y_true
    qwk = cohen_kappa_score(true, pred, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    labels = y_true
    preds = y_pred
    df = preds - labels
    dg = preds
    f = 1 / 2 * np.sum((df) ** 2)  # loss sum
    g = 1 / 2 * np.sum((dg) ** 2 + b)
    grad = (df / g - f * df / g**2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess


a = 2.94972423  # retained for compatibility (not used now)
b = 1.092



## === cell 12
LOAD = False  # Train models from scratch instead of loading missing files
models = []
fold_num = 5


def get_score(ms, df, best_itr):
    oof = []
    for i in range(len(ms)):
        prediction = ms[i].predict(
            df.drop(columns=["score", "essay_id"]), num_iteration=best_itr[i]
        )
        _ = df[["essay_id", "score"]].copy()
        _["pred"] = prediction  # no offset addition
        oof.append(_)
    df_oof = pd.concat(oof)

    for k in range(fold_num):
        start = len(df) * k
        end = len(df) * (k + 1)
        print(f"model_{k}'s score")
        acc = accuracy_score(
            df_oof["score"][start:end],
            df_oof["pred"][start:end].clip(1, 6).round(),
        )
        kappa = cohen_kappa_score(
            df_oof["score"][start:end],
            df_oof["pred"][start:end].clip(1, 6).round(),
            weights="quadratic",
        )
        print("acc: ", acc)
        print("kappa: ", kappa)
        print("=" * 30)
    return df_oof


if LOAD:
    path = "/kaggle/input/mymodel1/"
    best_itr = [639, 649, 581, 386, 444]
    for i in range(fold_num):
        models.append(lgbm.Booster(model_file=f"{path}fold_{i}.txt"))
    df_oof = get_score(models, train_df, best_itr)
else:
    oof = []
    best_itr = []
    x = train_df.drop(columns=["score", "essay_id"])
    y = train_df["score"].values  # use raw scores
    kfold = StratifiedKFold(n_splits=fold_num, random_state=0, shuffle=True)
    callbacks = [
        log_evaluation(period=25),
        early_stopping(stopping_rounds=75, first_metric_only=True),
    ]

    for fold_id, (train_idx, val_idx) in tqdm(
        enumerate(kfold.split(x.copy(), y.copy().astype(str)))
    ):
        model = lgbm.LGBMRegressor(
            objective="regression",  # standard regression objective
            metrics="None",
            learning_rate=0.08,
            max_depth=7,
            num_leaves=31,
            colsample_bytree=0.5,
            reg_alpha=0.1,
            reg_lambda=0.8,
            n_estimators=1024,
            random_state=0,
            verbosity=-1,
        )

        x_train = x.iloc[train_idx]
        y_train = y[train_idx]  # raw scores
        x_val = x.iloc[val_idx]
        y_val = y[val_idx]

        print(f'{fold_id+1}_Fold Training {"="*50}')

        lgbm_model = model.fit(
            x_train,
            y_train,
            eval_names=["train", "val"],
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_metric=quard_w_kappa,
            callbacks=callbacks,
        )

        best_itr.append(lgbm_model.best_iteration_)
        print(lgbm_model.best_iteration_)

        models.append(lgbm_model.booster_)
        lgbm_model.booster_.save_model(f"fold_{fold_id}.txt")
    df_oof = get_score(models, train_df, best_itr)



## === cell 13
tmp = get_feature(df_test)
test_feats = add_satis(tmp, add_f)
tmp = Sentence_process(df_test)
test_feats = test_feats.merge(Sentence_addf(tmp, sen_fea), on="essay_id", how="left")
tmp = Word_process(df_test)
test_feats = test_feats.merge(Word_addf(tmp), on="essay_id", how="left")
test_tfid = vectorizer.transform([i for i in test_text])
dense_matrix = test_tfid.toarray()
df = pd.DataFrame(dense_matrix)
tfid_columns = [f"tfid_{i}" for i in range(len(df.columns))]
df.columns = tfid_columns
df["essay_id"] = test_feats["essay_id"]
test_feats = test_feats.merge(df, on="essay_id", how="left")
feature_names = list(
    filter(lambda x: x not in ["essay_id", "score"], test_feats.columns)
)
print("Features number: ", len(feature_names))
test_feats.head(3)

test_feats[feature_names] = test_feats[feature_names].fillna(0)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3704756501.py in <cell line: 0>()
      5 tmp = Word_process(df_test)
      6 test_feats = test_feats.merge(Word_addf(tmp), on="essay_id", how="left")
----> 7 test_tfid = vectorizer.transform([i for i in test_text])
      8 dense_matrix = test_tfid.toarray()
      9 df = pd.DataFrame(dense_matrix)

NameError: name 'vectorizer' is not defined

## === cell 14
prediction = test_feats[["essay_id"]].copy()
prediction["score"] = 0
pred_test = models[0].predict(test_feats[feature_names])  # no offset
for i in range(1, 5):
    other_p_test = models[i].predict(test_feats[feature_names])
    pred_test = np.add(pred_test, other_p_test)
pred_test = pred_test / 5
print(pred_test)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/650158381.py in <cell line: 0>()
      1 prediction = test_feats[["essay_id"]].copy()
      2 prediction["score"] = 0
----> 3 pred_test = models[0].predict(test_feats[feature_names])  # no offset
      4 for i in range(1, 5):
      5     other_p_test = models[i].predict(test_feats[feature_names])

NameError: name 'feature_names' is not defined

## === cell 15
pred_test = pred_test.clip(1, 6).round().astype(int)
prediction["score"] = pred_test
prediction.to_csv("submission.csv", index=False)
prediction.head(3)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2016835793.py in <cell line: 0>()
----> 1 pred_test = pred_test.clip(1, 6).round().astype(int)
      2 prediction["score"] = pred_test
      3 prediction.to_csv("submission.csv", index=False)
      4 prediction.head(3)

NameError: name 'pred_test' is not defined
