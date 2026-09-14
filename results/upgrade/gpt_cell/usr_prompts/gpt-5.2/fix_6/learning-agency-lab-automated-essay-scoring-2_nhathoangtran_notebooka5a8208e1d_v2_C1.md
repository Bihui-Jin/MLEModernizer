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

catboost==1.2.8
cudf-polars-cu12==25.6.0
geopandas==0.14.4
imbalanced-learn==0.13.0
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

# 5. Code solution

## === cell 0
import os
import re
import polars as pl
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import VotingRegressor
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from collections import Counter
from sklearn.metrics import confusion_matrix

try:
    from imblearn.over_sampling import BorderlineSMOTE
except ModuleNotFoundError:
    BorderlineSMOTE = None

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from scipy import sparse
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)




## === cell 1
class FeatureEngineering:
    def __init__(self):
        self.columns = [(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))]
        self.train_dataset = pl.read_csv(
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
        ).with_columns(self.columns)
        self.test_dataset = pl.read_csv(
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
        ).with_columns(self.columns)

        self.sentence_fea = ["sentence_len", "sentence_word_cnt"]
        self.paragraph_fea = [
            "paragraph_len",
            "paragraph_sentence_cnt",
            "paragraph_word_cnt",
        ]

        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(2, 3),
            min_df=0.05,
            max_df=0.9,
            sublinear_tf=True,
        )

        self._re_html = r"<.*?>"
        self._re_at = r"@\w+"
        self._re_quote_digit = r"'\d+"
        self._re_digits = r"\d+"
        self._re_http = r"http\w+"
        self._re_space = r"\s+"
        self._re_periods = r"\.+"
        self._re_commas = r"\,+"

    def _preprocess_expr(self, col: str) -> pl.Expr:
        return (
            pl.col(col)
            .str.to_lowercase()
            .str.replace_all(self._re_html, "")
            .str.replace_all(self._re_at, "")
            .str.replace_all(self._re_quote_digit, "")
            .str.replace_all(self._re_digits, "")
            .str.replace_all(self._re_http, "")
            .str.replace_all(self._re_space, " ")
            .str.replace_all(self._re_periods, ".")
            .str.replace_all(self._re_commas, ",")
            .str.strip_chars()
        )

    def Paragraph_Preprocess(self, tmp: pl.DataFrame) -> pl.DataFrame:
        tmp = tmp.explode("paragraph").with_columns(
            self._preprocess_expr("paragraph").alias("paragraph")
        )
        tmp = tmp.with_columns(
            pl.col("paragraph").str.len_chars().alias("paragraph_len")
        )
        tmp = tmp.filter(pl.col("paragraph_len") >= 25)
        tmp = tmp.with_columns(
            pl.col("paragraph")
            .str.split(".")
            .list.len()
            .alias("paragraph_sentence_cnt"),
            pl.col("paragraph").str.split(" ").list.len().alias("paragraph_word_cnt"),
        )
        return tmp

    def Paragraph_Eng(self, train_tmp: pl.DataFrame) -> pd.DataFrame:
        aggs = [
            *[
                pl.col("paragraph")
                .filter(pl.col("paragraph_len") >= i)
                .count()
                .alias(f"paragraph_{i}_cnt")
                for i in [25, 100, 200, 300, 400, 500, 600, 700]
            ],
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in self.paragraph_fea],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in self.paragraph_fea],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in self.paragraph_fea],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in self.paragraph_fea],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in self.paragraph_fea],
        ]
        df = (
            train_tmp.group_by(["essay_id"], maintain_order=True)
            .agg(aggs)
            .sort("essay_id")
        )
        df = df.to_pandas()
        print("done Paragraph_Eng +", len(df.columns), "features")
        return df

    def Sentence_Preprocess(self, tmp: pl.DataFrame) -> pl.DataFrame:
        tmp = tmp.with_columns(
            self._preprocess_expr("full_text").str.split(by=".").alias("sentence")
        ).explode("sentence")
        tmp = tmp.with_columns(pl.col("sentence").str.len_chars().alias("sentence_len"))
        tmp = tmp.filter(pl.col("sentence_len") >= 15)
        tmp = tmp.with_columns(
            pl.col("sentence").str.split(" ").list.len().alias("sentence_word_cnt")
        )
        return tmp

    def Sentence_Eng(self, train_tmp: pl.DataFrame) -> pd.DataFrame:
        aggs = [
            *[
                pl.col("sentence")
                .filter(pl.col("sentence_len") >= i)
                .count()
                .alias(f"sentence_{i}_cnt")
                for i in [15, 50, 100, 150, 200, 250, 300]
            ],
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in self.sentence_fea],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in self.sentence_fea],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in self.sentence_fea],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in self.sentence_fea],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in self.sentence_fea],
        ]
        df = (
            train_tmp.group_by(["essay_id"], maintain_order=True)
            .agg(aggs)
            .sort("essay_id")
        )
        df = df.to_pandas()
        print("done Sentence_Eng +", len(df.columns), "features")
        return df

    def Word_Preprocess(self, tmp: pl.DataFrame) -> pl.DataFrame:
        tmp = tmp.with_columns(
            self._preprocess_expr("full_text").str.split(by=" ").alias("word")
        ).explode("word")
        tmp = tmp.with_columns(pl.col("word").str.len_chars().alias("word_len"))
        tmp = tmp.filter(pl.col("word_len") != 0)
        return tmp

    def Word_Eng(self, train_tmp: pl.DataFrame) -> pd.DataFrame:
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
            train_tmp.group_by(["essay_id"], maintain_order=True)
            .agg(aggs)
            .sort("essay_id")
        )
        df = df.to_pandas()
        print("done Word_Eng +", len(df.columns), "features")
        return df

    def _tfidf_fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def _tfidf_transform(self, texts):
        return self.vectorizer.transform(texts)

    def process(self):
        tmp = self.Paragraph_Preprocess(self.train_dataset)
        train_feats = self.Paragraph_Eng(tmp)
        train_feats["score"] = self.train_dataset["score"].to_numpy()

        tmp = self.Sentence_Preprocess(self.train_dataset)
        train_feats = train_feats.merge(
            self.Sentence_Eng(tmp), on="essay_id", how="left"
        )

        tmp = self.Word_Preprocess(self.train_dataset)
        train_feats = train_feats.merge(self.Word_Eng(tmp), on="essay_id", how="left")

        train_texts = self.train_dataset["full_text"].to_list()
        X_tfid = self._tfidf_fit_transform(train_texts)

        essay_ids = train_feats["essay_id"].to_numpy()
        print("feature_num (dense): ", len(train_feats.columns) - 2)
        print("tfidf_dim: ", X_tfid.shape[1])
        return train_feats, X_tfid, essay_ids

    def process_test(self):
        temp = self.Paragraph_Preprocess(self.test_dataset)
        test_feats = self.Paragraph_Eng(temp)

        temp = self.Sentence_Preprocess(self.test_dataset)
        test_feats = test_feats.merge(
            self.Sentence_Eng(temp), on="essay_id", how="left"
        )

        temp = self.Word_Preprocess(self.test_dataset)
        test_feats = test_feats.merge(self.Word_Eng(temp), on="essay_id", how="left")

        test_texts = self.test_dataset["full_text"].to_list()
        X_tfid = self._tfidf_transform(test_texts)

        essay_ids = test_feats["essay_id"].to_numpy()
        print("feature_num (dense): ", len(test_feats.columns) - 1)
        print("tfidf_dim: ", X_tfid.shape[1])
        return test_feats, X_tfid, essay_ids




## === cell 2
class LGBM:
    def __init__(self):
        self.data_train = pl.read_csv(
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
        )
        self.data_test = pl.read_csv(
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
        )
        self.num_models = 5
        self.acc_metrics = []
        self.cohen_metrics = []

        self.a = 2.948
        self.b = 1.092

        self.lgb_parameters = {
            "metrics": "None",
            "objective": self.qwk_obj,
            "learning_rate": 0.1,
            "max_depth": 5,
            "num_leaves": 15,
            "colsample_bytree": 0.5,
            "min_data_in_leaf": 100,
            "reg_alpha": 0.8,
            "n_estimators": 256,
            "verbosity": -1,
            "device": "gpu",
        }

        self.model = VotingRegressor(
            estimators=[
                (
                    f"lgb_{i}",
                    lgb.LGBMRegressor(**self.lgb_parameters, random_state=i + 40),
                )
                for i in range(self.num_models)
            ],
            n_jobs=-1,
        )

    def quadratic_weighted_kappa(self, y_true, y_pred):
        y_true = y_true + self.a
        y_pred = (y_pred + self.a).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return "QWK", qwk, True

    def qwk_obj(self, y_true, y_pred):
        labels = y_true + self.a
        preds = y_pred + self.a
        preds = preds.clip(1, 6)
        f = 1 / 2 * np.sum((preds - labels) ** 2)
        g = 1 / 2 * np.sum((preds - self.a) ** 2 + self.b)
        df = preds - labels
        dg = preds - self.a
        grad = (df / g - f * dg / g**2) * len(labels)
        hess = np.ones(len(labels))
        return grad, hess

    def fit(
        self,
        df_dense: pd.DataFrame,
        X_tfid: sparse.csr_matrix,
        training_fold=[0, 1, 2, 3, 4],
    ):
        feature_names = [c for c in df_dense.columns if c not in ["essay_id", "score"]]
        X_dense = df_dense[feature_names].to_numpy(dtype=np.float32, copy=False)
        y = df_dense["score"].to_numpy(copy=False)

        counter = Counter(y)
        print(counter)

        if BorderlineSMOTE is not None:
            oversample = BorderlineSMOTE(
                sampling_strategy={1: 2000, 6: 500, 5: 2000}, k_neighbors=5
            )
            X_res, y_res = oversample.fit_resample(X_dense, y)
            counter = Counter(y_res)
            print("distribution after oversample:", counter)
        else:
            X_res, y_res = X_dense, y
            print("BorderlineSMOTE unavailable; skipping oversampling.")

        if BorderlineSMOTE is not None:
            print(
                "TFIDF+SMOTE would require synthesizing TFIDF features; skipping oversampling to preserve correctness."
            )
            X_res, y_res = X_dense, y

        X_all = sparse.hstack([sparse.csr_matrix(X_res), X_tfid], format="csr")
        splits = StratifiedKFold(n_splits=5).split(X_res, y_res)

        for fold in training_fold:
            trn_idx, val_idx = next(idx for i, idx in enumerate(splits) if i == fold)

            X_train = X_all[trn_idx]
            Y_train = y_res[trn_idx] - self.a
            X_val = X_all[val_idx]
            Y_val = y_res[val_idx] - self.a

            print("\nFold_{} Training ================================\n".format(fold))
            self.model.fit(X_train, Y_train)

            pred_val = self.model.predict(X_val)
            cohen_score = self.quadratic_weighted_kappa(Y_val, pred_val)
            accuracy = accuracy_score(
                Y_val + self.a, (pred_val + self.a).clip(1, 6).round()
            )
            self.acc_metrics.append(accuracy)
            self.cohen_metrics.append(cohen_score[1])
            print(f"Accuracy fold {fold}: {accuracy:.4f}")
            print(f"Cohen score fold {fold}: {cohen_score[1]:.4f}")

        average_accuracy = np.mean(self.acc_metrics)
        average_cohen = np.mean(self.cohen_metrics)
        print(f"Average Accuracy all fold: {average_accuracy:.4f}")
        print(f"Average Cohen all fold: {average_cohen:.4f}")

    def predict(self, df_dense: pd.DataFrame, X_tfid: sparse.csr_matrix):
        feature_names = [c for c in df_dense.columns if c != "essay_id"]
        X_dense = df_dense[feature_names].to_numpy(dtype=np.float32, copy=False)
        X_all = sparse.hstack([sparse.csr_matrix(X_dense), X_tfid], format="csr")
        predictions = (self.model.predict(X_all) + self.a).clip(1, 6).round()
        return predictions

    def submit(self, df_dense: pd.DataFrame, X_tfid: sparse.csr_matrix):
        preds = self.predict(df_dense, X_tfid)
        return self.data_test.select("essay_id").with_columns(score=preds)




## === cell 3
FE = FeatureEngineering()
train_feature, X_train_tfid, train_ids = FE.process()
test_feature, X_test_tfid, test_ids = FE.process_test()

train_feature = train_feature.sort_values("essay_id").reset_index(drop=True)
test_feature = test_feature.sort_values("essay_id").reset_index(drop=True)




## === cell 4
model = LGBM()
model.fit(df_dense=train_feature, X_tfid=X_train_tfid, training_fold=[0])

submission = model.submit(test_feature, X_test_tfid)
submission.write_csv("submission.csv")
print(submission.head(5))
