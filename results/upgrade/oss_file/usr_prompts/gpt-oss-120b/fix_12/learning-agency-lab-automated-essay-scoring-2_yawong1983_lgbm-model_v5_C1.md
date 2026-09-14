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
- What this solution (achieved 0.0) has done: 'I added a TF‑IDF vectorizer, fitted it on the cleaned training essays, and merged the resulting sparse features into both the training and test feature tables. This defines the missing `vectorizer` variable, supplies matching TF‑IDF columns for the model, and restores the `feature_names` variable so predictions and the final CSV can be generated successfully.'
- What this solution (achieved 0.0) has done: 'Implemented modest upgrades to boost predictive power while keeping the original workflow intact.  
- Expanded TF‑IDF vocabulary to 15000 features for richer text representation.  
- Increased LightGBM capacity (more trees and leaves) and slightly raised feature‑fraction to capture additional patterns.  
- Adjusted early‑stopping callbacks to stay compatible. These tweaks are expected to raise the quadratic weighted‑kappa toward the target without overhauling the core pipeline.'
- What this solution (achieved -0.00104) has done: 'I keep the overall pipeline unchanged but slightly strengthen the LightGBM models so they can achieve a higher quadratic weighted‑kappa, moving the score toward the target. The changes are limited to the model hyper‑parameters and early‑stopping patience (still using the same regressor and metric), which should improve learning without altering the core logic.'
- What this solution (achieved -0.00104) has done: 'I fix the out‑of‑fold evaluation: instead of predicting on the whole training set for every fold (which gave meaningless scores), I store each model’s predictions only on its validation split, then compute the overall quadratic weighted‑kappa and accuracy after training. This corrects the metric calculation and moves the score toward the target while keeping the original model and feature pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'I switch the LightGBM model from a regressor to a multiclass classifier (keeping the same features) and adapt the custom QWK metric to work with probability outputs. This lets the model directly predict the 1‑6 score classes, which should raise the quadratic weighted‑kappa toward the target. I also adjust the training‑/prediction‑logic to handle zero‑based class labels and to average class probabilities across folds before taking the final arg‑max.'

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
from scipy import sparse  # new import for sparse matrix handling

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
    x = x.lower()
    x = re.compile(r"<.*?>").sub(r"", x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


def get_feature(x):
    x = x.explode("paragraph")
    x = x.with_columns(pl.col("paragraph").map_elements(Data_clearning))
    x = x.with_columns(
        pl.col("paragraph").map_elements(lambda x: len(x)).alias("par_len")
    )
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
pass



## === cell 5
pass




## === cell 6
def add_satis(df, add_f):
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
add_f = ["par_len", "par_sentence_cnt", "par_word_cnt"]
train_df2 = add_satis(tmp, add_f)
train_df2["score"] = df_train["score"]
print("Feature cnt: ", train_df2.shape[1])
train_df2.head(3)




## === cell 7
def Sentence_process(df):
    df = df.with_columns(
        pl.col("full_text").map_elements(Data_clearning).str.split(by=".").alias("sen")
    )
    df = df.explode("sen")
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




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2295510596.py in <cell line: 0>()
     36 
     37 
---> 38 tmp = Sentence_process(df_train)
     39 train_df = train_df2.merge(Sentence_addf(tmp, sen_fea), on="essay_id", how="left")
     40 print("train_df's feature number: ", train_df.shape[1])

/tmp/ipykernel_11/2295510596.py in Sentence_process(df)
      5     df = df.explode("sen")
      6     df = df.with_columns(pl.col("sen").map_elements(lambda x: len(x)).alias("sen_len"))
----> 7     sns.histplot(x="sen_len", data=df, stat="percent", bins=100)
      8     plt.show()
      9     df = df.filter(pl.col("sen_len") >= 15)

NameError: name 'sns' is not defined

## === cell 8
def Word_process(df):
    df = df.with_columns(
        pl.col("full_text").map_elements(Data_clearning).str.split(by=" ").alias("word")
    )
    df = df.explode("word")
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



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/279256182.py in <cell line: 0>()
     35 
     36 
---> 37 tmp = Word_process(df_train)
     38 train_df = train_df.merge(Word_addf(tmp), on="essay_id", how="left")
     39 print("train_df's feature number: ", train_df.shape[1])

/tmp/ipykernel_11/279256182.py in Word_process(df)
      7         pl.col("word").map_elements(lambda x: len(x)).alias("word_len")
      8     )
----> 9     sns.histplot(x="word_len", data=df, stat="percent", bins=100)
     10     plt.show()
     11     df = df.filter(pl.col("word_len") != 0)

NameError: name 'sns' is not defined

## === cell 9
train_text = df_train.to_pandas()["full_text"].apply(Data_clearning)
test_text = df_test.to_pandas()["full_text"].apply(Data_clearning)

vectorizer = TfidfVectorizer(
    max_features=15000, ngram_range=(1, 2), stop_words="english"
)

train_tfid = vectorizer.fit_transform(train_text)  # shape: (n_train, 15000)

train_numeric = train_df.copy()  # includes essay_id, score, engineered columns
print("TF‑IDF and engineered features prepared (sparse).")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1320386688.py in <cell line: 0>()
      8 train_tfid = vectorizer.fit_transform(train_text)  # shape: (n_train, 15000)
      9 
---> 10 train_numeric = train_df.copy()  # includes essay_id, score, engineered columns
     11 print("TF‑IDF and engineered features prepared (sparse).")
     12 

NameError: name 'train_df' is not defined

## === cell 10
LOAD = False
models = []
fold_num = 5

if LOAD:
    path = "/kaggle/input/mymodel1/"
    for i in range(fold_num):
        models.append(lgbm.Booster(model_file=f"{path}fold_{i}.txt"))
else:
    y = train_numeric["score"].values
    y_zero = y - 1  # 0‑based labels for classifier

    numeric_cols = [c for c in train_numeric.columns if c not in ["score", "essay_id"]]
    X_numeric = sparse.csr_matrix(train_numeric[numeric_cols].astype(np.float32).values)

    X = sparse.hstack([X_numeric, train_tfid]).tocsr()

    kfold = StratifiedKFold(n_splits=fold_num, random_state=0, shuffle=True)
    callbacks = [
        log_evaluation(period=25),
        early_stopping(stopping_rounds=150, first_metric_only=True),
    ]

    oof_pred = np.zeros(len(train_numeric), dtype=int)

    for fold_id, (train_idx, val_idx) in tqdm(
        enumerate(kfold.split(X, y_zero)),
        total=fold_num,
        desc="Training folds",
    ):
        model = lgbm.LGBMClassifier(
            objective="multiclass",
            num_class=6,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=511,
            colsample_bytree=0.5,
            feature_fraction=0.8,
            reg_alpha=0.1,
            reg_lambda=0.8,
            n_estimators=8000,
            n_jobs=-1,  # <-- enable full CPU parallelism
            random_state=0,
            verbosity=-1,
        )
        x_train = X[train_idx]
        y_train = y_zero[train_idx]
        x_val = X[val_idx]
        y_val = y_zero[val_idx]

        print(f'{fold_id+1}_Fold Training {"="*50}')

        lgbm_model = model.fit(
            x_train,
            y_train,
            eval_names=["train", "val"],
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=callbacks,
        )

        print("Best iteration:", lgbm_model.best_iteration_)

        oof_pred[val_idx] = lgbm_model.predict(x_val) + 1

        models.append(lgbm_model.booster_)
        lgbm_model.booster_.save_model(f"fold_{fold_id}.txt")

    overall_qwk = cohen_kappa_score(y, oof_pred, weights="quadratic")
    overall_acc = accuracy_score(y, oof_pred)
    print("\nOverall OOF Accuracy :", overall_acc)
    print("Overall OOF Quadratic Weighted Kappa :", overall_qwk)

    tfidf_cols = [f"tfid_{i}" for i in range(train_tfid.shape[1])]
    feature_names = numeric_cols + tfidf_cols



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/797878071.py in <cell line: 0>()
      8         models.append(lgbm.Booster(model_file=f"{path}fold_{i}.txt"))
      9 else:
---> 10     y = train_numeric["score"].values
     11     y_zero = y - 1  # 0‑based labels for classifier
     12 

NameError: name 'train_numeric' is not defined

## === cell 11
tmp = get_feature(df_test)
test_feats = add_satis(tmp, add_f)
tmp = Sentence_process(df_test)
test_feats = test_feats.merge(Sentence_addf(tmp, sen_fea), on="essay_id", how="left")
tmp = Word_process(df_test)
test_feats = test_feats.merge(Word_addf(tmp), on="essay_id", how="left")

test_tfid = vectorizer.transform(test_text)

numeric_test_cols = [c for c in test_feats.columns if c != "essay_id"]
X_test_numeric = sparse.csr_matrix(
    test_feats[numeric_test_cols].astype(np.float32).values
)

X_test = sparse.hstack([X_test_numeric, test_tfid]).tocsr()
print("Features number: ", X_test.shape[1])



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3116049249.py in <cell line: 0>()
      1 tmp = get_feature(df_test)
      2 test_feats = add_satis(tmp, add_f)
----> 3 tmp = Sentence_process(df_test)
      4 test_feats = test_feats.merge(Sentence_addf(tmp, sen_fea), on="essay_id", how="left")
      5 tmp = Word_process(df_test)

/tmp/ipykernel_11/2295510596.py in Sentence_process(df)
      5     df = df.explode("sen")
      6     df = df.with_columns(pl.col("sen").map_elements(lambda x: len(x)).alias("sen_len"))
----> 7     sns.histplot(x="sen_len", data=df, stat="percent", bins=100)
      8     plt.show()
      9     df = df.filter(pl.col("sen_len") >= 15)

NameError: name 'sns' is not defined

## === cell 12
prediction = test_feats[["essay_id"]].copy()
prob_sum = np.zeros((X_test.shape[0], 6))

for booster in models:
    prob = booster.predict(X_test)
    prob_sum += prob

avg_prob = prob_sum / len(models)
prediction["score"] = np.argmax(avg_prob, axis=1) + 1  # convert to 1‑6
prediction.to_csv("submission.csv", index=False)
prediction.head(3)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3271160305.py in <cell line: 0>()
      1 prediction = test_feats[["essay_id"]].copy()
----> 2 prob_sum = np.zeros((X_test.shape[0], 6))
      3 
      4 for booster in models:
      5     prob = booster.predict(X_test)

NameError: name 'X_test' is not defined
