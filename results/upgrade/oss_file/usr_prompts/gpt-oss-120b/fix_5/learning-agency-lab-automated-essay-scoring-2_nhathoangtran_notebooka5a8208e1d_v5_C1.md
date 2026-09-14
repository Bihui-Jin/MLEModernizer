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

# 5. Target score

0.80934

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.01576) has done: 'Implemented fixes to resolve import errors, removed unavailable SMOTE oversampling, added missing Counter import, corrected prediction handling, and revised the submission generation to produce a proper CSV file.'
- What this solution (achieved 0.00228) has done: 'I simplify the LightGBM wrapper to use a standard regression objective (removing the custom QWK objective and the unnecessary shift `a`). This lets the model learn directly on the original scores (1‑6) and then round/clamp predictions, which is a straightforward way to raise the QWK from the near‑zero baseline toward the target. I also replace the VotingRegressor with a single LGBMRegressor for clarity and keep the rest of the pipeline unchanged, ensuring a valid `submission.csv` is still produced.'

# 9. Code solution

## === cell 0
import re
import polars as pl
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm
from collections import Counter




## === cell 1
class FeatureEngineering:
    def __init__(self):
        self.train_dataset = pl.read_csv(
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
        ).with_columns(
            pl.col("full_text").map_elements(self.dataPreprocessing).alias("clean_text")
        )
        self.test_dataset = pl.read_csv(
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
        ).with_columns(
            pl.col("full_text").map_elements(self.dataPreprocessing).alias("clean_text")
        )
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

    def removeHTML(self, x):
        html = re.compile(r"<.*?>")
        return html.sub(r"", x)

    def dataPreprocessing(self, x):
        x = x.lower()
        x = self.removeHTML(x)
        x = re.sub("@\w+", "", x)
        x = re.sub("'\d+", "", x)
        x = re.sub("\d+", "", x)
        x = re.sub("http\w+", "", x)
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"\.+", ".", x)
        x = re.sub(r"\,+", ",", x)
        x = x.strip()
        return x

    def Paragraph_Preprocess(self, tmp):
        tmp = tmp.with_columns(
            pl.col("clean_text").str.split(by="\n\n").alias("paragraph")
        ).explode("paragraph")
        tmp = tmp.with_columns(pl.col("paragraph").str.lengths().alias("paragraph_len"))
        tmp = tmp.filter(pl.col("paragraph_len") >= 25)
        tmp = tmp.with_columns(
            pl.col("paragraph")
            .str.split(".")
            .list.len()
            .alias("paragraph_sentence_cnt"),
            pl.col("paragraph").str.split(" ").list.len().alias("paragraph_word_cnt"),
        )
        return tmp

    def Paragraph_Eng(self, train_tmp):
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
        print("done Paragraph_Eng +", len(df.columns), "features")
        return df

    def Sentence_Preprocess(self, tmp):
        tmp = tmp.with_columns(
            pl.col("clean_text").str.split(by=".").alias("sentence")
        ).explode("sentence")
        tmp = tmp.with_columns(pl.col("sentence").str.lengths().alias("sentence_len"))
        tmp = tmp.filter(pl.col("sentence_len") >= 15)
        tmp = tmp.with_columns(
            pl.col("sentence").str.split(" ").list.len().alias("sentence_word_cnt")
        )
        return tmp

    def Sentence_Eng(self, train_tmp):
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
        print("done Sentence_Eng +", len(df.columns), "features")
        return df

    def Word_Preprocess(self, tmp):
        tmp = tmp.with_columns(
            pl.col("clean_text").str.split(by=" ").alias("word")
        ).explode("word")
        tmp = tmp.with_columns(pl.col("word").str.lengths().alias("word_len"))
        tmp = tmp.filter(pl.col("word_len") != 0)
        return tmp

    def Word_Eng(self, train_tmp):
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
        print("done Word_Eng +", len(df.columns), "features")
        return df

    def process(self):
        tmp = self.Paragraph_Preprocess(self.train_dataset)
        train_feats = self.Paragraph_Eng(tmp)
        train_feats = train_feats.with_columns(
            pl.Series("score", self.train_dataset["score"])
        )

        tmp = self.Sentence_Preprocess(self.train_dataset)
        train_feats = train_feats.join(
            self.Sentence_Eng(tmp), on="essay_id", how="left"
        )

        tmp = self.Word_Preprocess(self.train_dataset)
        train_feats = train_feats.join(self.Word_Eng(tmp), on="essay_id", how="left")

        train_tfid = self.vectorizer.fit_transform(
            [i for i in self.train_dataset["full_text"]]
        )
        dense_matrix = train_tfid.toarray()
        tfidf_df = pd.DataFrame(dense_matrix)
        tfidf_df.columns = [f"tfid_{i}" for i in range(tfidf_df.shape[1])]
        tfidf_df["essay_id"] = train_feats["essay_id"].to_pandas()
        train_feats = train_feats.join(
            pl.from_pandas(tfidf_df), on="essay_id", how="left"
        )
        print("feature_num:", len(train_feats.columns) - 2)
        return train_feats.to_pandas()

    def process_test(self):
        tmp = self.Paragraph_Preprocess(self.test_dataset)
        test_feats = self.Paragraph_Eng(tmp)

        tmp = self.Sentence_Preprocess(self.test_dataset)
        test_feats = test_feats.join(self.Sentence_Eng(tmp), on="essay_id", how="left")

        tmp = self.Word_Preprocess(self.test_dataset)
        test_feats = test_feats.join(self.Word_Eng(tmp), on="essay_id", how="left")

        test_tfid = self.vectorizer.transform(
            [i for i in self.test_dataset["full_text"]]
        )
        dense_matrix = test_tfid.toarray()
        tfidf_df = pd.DataFrame(dense_matrix)
        tfidf_df.columns = [f"tfid_{i}" for i in range(tfidf_df.shape[1])]
        tfidf_df["essay_id"] = test_feats["essay_id"].to_pandas()
        test_feats = test_feats.join(
            pl.from_pandas(tfidf_df), on="essay_id", how="left"
        )
        print("feature_num:", len(test_feats.columns) - 2)
        return test_feats.to_pandas()




## === cell 2
class LGBM:
    def __init__(self):
        self.lgb_parameters = {
            "objective": "regression",
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 127,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "n_estimators": 2000,
            "verbosity": -1,
            "device": "cpu",
        }
        self.model = lgb.LGBMRegressor(**self.lgb_parameters, random_state=42)

    def fit(self, df):
        feature_names = [c for c in df.columns if c not in ["essay_id", "score"]]
        X = df[feature_names]
        y = df["score"]
        self.model.fit(X, y)

    def predict(self, df):
        feature_names = [c for c in df.columns if c != "essay_id"]
        preds = self.model.predict(df[feature_names])
        preds = np.clip(np.round(preds), 1, 6).astype(int)
        return preds

    def submit(self, df):
        preds = self.predict(df)
        submission = pd.DataFrame({"essay_id": df["essay_id"], "score": preds})
        return submission




## === cell 3
FE = FeatureEngineering()
train_feature = FE.process()
test_feature = FE.process_test()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1304445936.py in <cell line: 0>()
      1 FE = FeatureEngineering()
----> 2 train_feature = FE.process()
      3 test_feature = FE.process_test()
      4 
      5 

/tmp/ipykernel_55/1063119667.py in process(self)
    159     def process(self):
    160         # ----- paragraph -----
--> 161         tmp = self.Paragraph_Preprocess(self.train_dataset)
    162         train_feats = self.Paragraph_Eng(tmp)
    163         train_feats = train_feats.with_columns(

/tmp/ipykernel_55/1063119667.py in Paragraph_Preprocess(self, tmp)
     54             pl.col("clean_text").str.split(by="\n\n").alias("paragraph")
     55         ).explode("paragraph")
---> 56         tmp = tmp.with_columns(pl.col("paragraph").str.lengths().alias("paragraph_len"))
     57         tmp = tmp.filter(pl.col("paragraph_len") >= 25)
     58         tmp = tmp.with_columns(

AttributeError: 'ExprStringNameSpace' object has no attribute 'lengths'

## === cell 4
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_pred_accum = np.zeros(len(test_feature), dtype=float)

for fold, (tr_idx, _) in enumerate(skf.split(train_feature, train_feature["score"])):
    print(f"Training fold {fold + 1}")
    model = LGBM()
    model.fit(df=train_feature.iloc[tr_idx])
    test_pred_accum += model.predict(test_feature)

final_preds = np.clip(np.round(test_pred_accum / skf.n_splits), 1, 6).astype(int)
submission = pd.DataFrame({"essay_id": test_feature["essay_id"], "score": final_preds})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1435773486.py in <cell line: 0>()
      1 skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
----> 2 test_pred_accum = np.zeros(len(test_feature), dtype=float)
      3 
      4 for fold, (tr_idx, _) in enumerate(skf.split(train_feature, train_feature["score"])):
      5     print(f"Training fold {fold + 1}")

NameError: name 'test_feature' is not defined
