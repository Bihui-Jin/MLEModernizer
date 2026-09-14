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

0.8047773778856417

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import copy
import re
import random
import warnings
import gc
import pickle

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")




## === cell 1
PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
train = pl.read_csv(PATH + "train.csv")
test = pl.read_csv(PATH + "test.csv")

columns = [pl.col("full_text").str.split(by="\n\n").alias("paragraph")]
train = train.with_columns(columns)
test = test.with_columns(columns)




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
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "must've": "must have",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "who'd": "who would",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "y'all": "you all",
    "you'd": "you had",
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
print("Features Number after Paragraph:", len(feature_names))
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
print("Features Number after Sentence:", len(feature_names))
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
print("Features Number after Word:", len(feature_names))
train_feats.head(3)




## === cell 6
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
df_tfid["essay_id"] = train_feats["essay_id"].reset_index(drop=True)
train_feats = train_feats.merge(df_tfid, on="essay_id", how="left")

feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
print("Features Number after TFIDF:", len(feature_names))
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
df_cnt["essay_id"] = train_feats["essay_id"].reset_index(drop=True)
train_feats = train_feats.merge(df_cnt, on="essay_id", how="left")

feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
print("Features Number after CountVectorizer:", len(feature_names))
train_feats.head(3)




## === cell 8
a = 0.0


def quadratic_weighted_kappa(y_pred, dataset):
    """
    LightGBM custom evaluation function (sklearn API).
    Computes Quadratic Weighted Kappa between integer true scores and
    rounded predictions clipped to the valid range [1, 6].
    Returns (name, value, higher_is_better).
    """
    y_true = dataset.get_label()
    y_pred_rounded = np.rint(y_pred).clip(1, 6)
    kappa = cohen_kappa_score(
        y_true.astype(int), y_pred_rounded.astype(int), weights="quadratic"
    )
    return "QWK", kappa, True


X = train_feats[feature_names].astype(np.float32).values
y = train_feats["score"].astype(np.float32).values
y_int = train_feats["score"].astype(int).values

if_train = True  # force training in this run

if if_train:
    callbacks = [
        log_evaluation(period=25),
        early_stopping(stopping_rounds=75, first_metric_only=True),
    ]
    X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
        X,
        y,
        y_int,
        test_size=0.2,
        random_state=52,
        stratify=train_feats["score"],
    )
    model = lgb.LGBMRegressor(
        objective="regression",
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
    )
    predictor = model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "valid"],
        eval_metric=quadratic_weighted_kappa,
        callbacks=callbacks,
    )
    with open("lgbm_model.pkl", "wb") as fp:
        pickle.dump(predictor, fp)
else:
    with open("lgbm_model.pkl", "rb") as fp:
        predictor = pickle.load(fp)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2399420836.py in <cell line: 0>()
     51         verbosity=-1,
     52     )
---> 53     predictor = model.fit(
     54         X_train,
     55         y_train,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, eval_set, eval_names, eval_sample_weight, eval_init_score, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1396     ) -> "LGBMRegressor":
   1397         """Docstring is inherited from the LGBMModel."""
-> 1398         super().fit(
   1399             X,
   1400             y,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_class_weight, eval_init_score, eval_group, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1047         callbacks.append(record_evaluation(evals_result))
   1048 
-> 1049         self._Booster = train(
   1050             params=params,
   1051             train_set=train_set,

/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)
    326         if valid_sets is not None:
    327             if is_valid_contain_train:
--> 328                 evaluation_result_list.extend(booster.eval_train(feval))
    329             evaluation_result_list.extend(booster.eval_valid(feval))
    330         try:

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in eval_train(self, feval)
   4406             List with (train_dataset_name, eval_name, eval_result, is_higher_better) tuples.
   4407         """
-> 4408         return self.__inner_eval(self._train_data_name, 0, feval)
   4409 
   4410     def eval_valid(

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __inner_eval(self, data_name, data_idx, feval)
   5204                 if eval_function is None:
   5205                     continue
-> 5206                 feval_ret = eval_function(self.__inner_predict(data_idx), cur_data)
   5207                 if isinstance(feval_ret, list):
   5208                     for eval_name, val, is_higher_better in feval_ret:

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in __call__(self, preds, dataset)
    304         argc = len(signature(self.func).parameters)
    305         if argc == 2:
--> 306             return self.func(labels, preds)  # type: ignore[call-arg]
    307 
    308         weight = _get_weight_from_constructed_dataset(dataset)

/tmp/ipykernel_55/2399420836.py in quadratic_weighted_kappa(y_pred, dataset)
     10     Returns (name, value, higher_is_better).
     11     """
---> 12     y_true = dataset.get_label()
     13     y_pred_rounded = np.rint(y_pred).clip(1, 6)
     14     kappa = cohen_kappa_score(

AttributeError: 'numpy.ndarray' object has no attribute 'get_label'

## === cell 9
tmp = Paragraph_Preprocess(test)
test_feats = Paragraph_Eng(tmp)
tmp = Sentence_Preprocess(test)
test_feats = test_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")
tmp = Word_Preprocess(test)
test_feats = test_feats.merge(Word_Eng(tmp), on="essay_id", how="left")

test_tfid = vectorizer.transform([i for i in test["full_text"]])
df_test_tfid = pd.DataFrame(test_tfid.toarray())
df_test_tfid.columns = [f"tfid_{i}" for i in range(df_test_tfid.shape[1])]
df_test_tfid["essay_id"] = test_feats["essay_id"].reset_index(drop=True)
test_feats = test_feats.merge(df_test_tfid, on="essay_id", how="left")

test_cnt = vectorizer_cnt.transform([i for i in test["full_text"]])
df_test_cnt = pd.DataFrame(test_cnt.toarray())
df_test_cnt.columns = [f"tfid_cnt_{i}" for i in range(df_test_cnt.shape[1])]
df_test_cnt["essay_id"] = test_feats["essay_id"].reset_index(drop=True)
test_feats = test_feats.merge(df_test_cnt, on="essay_id", how="left")

feature_names_test = [c for c in test_feats.columns if c not in ["essay_id", "score"]]
print("Test Features number:", len(feature_names_test))
test_feats.head(3)




## === cell 10
predictions_gb = predictor.predict(test_feats[feature_names_test])
predictions_gb = np.rint(predictions_gb.clip(1, 6)).astype(int)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1590818523.py in <cell line: 0>()
----> 1 predictions_gb = predictor.predict(test_feats[feature_names_test])
      2 predictions_gb = np.rint(predictions_gb.clip(1, 6)).astype(int)
      3 
      4 

NameError: name 'predictor' is not defined

## === cell 11
submission = pd.read_csv(PATH + "sample_submission.csv")
submission["score"] = predictions_gb
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
display(submission.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1897899649.py in <cell line: 0>()
      1 submission = pd.read_csv(PATH + "sample_submission.csv")
----> 2 submission["score"] = predictions_gb
      3 submission.to_csv("submission.csv", index=False)
      4 print("Submission saved to submission.csv")
      5 display(submission.head())

NameError: name 'predictions_gb' is not defined
