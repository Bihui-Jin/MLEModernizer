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

0.80809

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from collections import Counter
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sn




## === cell 1
class FeatureEngineering:
    def __init__(self):
        self.train_dataset = pd.read_csv(
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
        )
        self.test_dataset = pd.read_csv(
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
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
        self.html_regex = re.compile(r"<.*?>")
        self.at_user_regex = re.compile(r"@\w+")
        self.digit_regex = re.compile(r"\d+")
        self.http_regex = re.compile(r"http\w+")
        self.space_regex = re.compile(r"\s+")
        self.dot_regex = re.compile(r"\.+")
        self.comma_regex = re.compile(r"\,+")

    def removeHTML(self, x):
        return self.html_regex.sub("", x)

    def dataPreprocessing(self, x):
        x = x.lower()
        x = self.removeHTML(x)
        x = self.at_user_regex.sub("", x)
        x = re.sub(r"'\d+", "", x)
        x = self.digit_regex.sub("", x)
        x = self.http_regex.sub("", x)
        x = self.space_regex.sub(" ", x)
        x = self.dot_regex.sub(".", x)
        x = self.comma_regex.sub(",", x)
        return x.strip()

    def paragraph_stats(self, text):
        paragraphs = [p for p in text.split("\n\n") if len(p) >= 25]
        plen = [len(p) for p in paragraphs]
        p_sent_cnt = [p.count(".") for p in paragraphs]
        p_word_cnt = [len(p.split()) for p in paragraphs]

        res = {}
        for i in [25, 100, 200, 300, 400, 500, 600, 700]:
            res[f"paragraph_{i}_cnt"] = sum(1 for l in plen if l >= i)
        for fea, lst in zip(self.paragraph_fea, [plen, p_sent_cnt, p_word_cnt]):
            if lst:
                res[f"{fea}_max"] = max(lst)
                res[f"{fea}_mean"] = np.mean(lst)
                res[f"{fea}_min"] = min(lst)
                res[f"{fea}_first"] = lst[0]
                res[f"{fea}_last"] = lst[-1]
            else:
                res[f"{fea}_max"] = 0
                res[f"{fea}_mean"] = 0
                res[f"{fea}_min"] = 0
                res[f"{fea}_first"] = 0
                res[f"{fea}_last"] = 0
        return pd.Series(res)

    def sentence_stats(self, text):
        sentences = [s for s in text.split(".") if len(s) >= 15]
        slen = [len(s) for s in sentences]
        s_word_cnt = [len(s.split()) for s in sentences]

        res = {}
        for i in [15, 50, 100, 150, 200, 250, 300]:
            res[f"sentence_{i}_cnt"] = sum(1 for l in slen if l >= i)
        for fea, lst in zip(self.sentence_fea, [slen, s_word_cnt]):
            if lst:
                res[f"{fea}_max"] = max(lst)
                res[f"{fea}_mean"] = np.mean(lst)
                res[f"{fea}_min"] = min(lst)
                res[f"{fea}_first"] = lst[0]
                res[f"{fea}_last"] = lst[-1]
            else:
                res[f"{fea}_max"] = 0
                res[f"{fea}_mean"] = 0
                res[f"{fea}_min"] = 0
                res[f"{fea}_first"] = 0
                res[f"{fea}_last"] = 0
        return pd.Series(res)

    def word_stats(self, text):
        words = [w for w in text.split() if len(w) > 0]
        wlen = [len(w) for w in words]

        res = {}
        for i in range(1, 16):
            res[f"word_{i}_cnt"] = sum(1 for l in wlen if l >= i)
        if wlen:
            res["word_len_max"] = max(wlen)
            res["word_len_mean"] = np.mean(wlen)
            res["word_len_std"] = np.std(wlen, ddof=0)
            res["word_len_q1"] = np.quantile(wlen, 0.25)
            res["word_len_q2"] = np.quantile(wlen, 0.50)
            res["word_len_q3"] = np.quantile(wlen, 0.75)
        else:
            res["word_len_max"] = 0
            res["word_len_mean"] = 0
            res["word_len_std"] = 0
            res["word_len_q1"] = 0
            res["word_len_q2"] = 0
            res["word_len_q3"] = 0
        return pd.Series(res)

    def process(self):
        self.train_dataset["clean_text"] = self.train_dataset["full_text"].apply(
            self.dataPreprocessing
        )

        para_feat = self.train_dataset["clean_text"].apply(self.paragraph_stats)
        train_feats = pd.concat(
            [self.train_dataset[["essay_id", "score"]], para_feat], axis=1
        )

        sent_feat = self.train_dataset["clean_text"].apply(self.sentence_stats)
        train_feats = train_feats.merge(
            sent_feat, left_on="essay_id", right_index=True, how="left"
        )

        word_feat = self.train_dataset["clean_text"].apply(self.word_stats)
        train_feats = train_feats.merge(
            word_feat, left_on="essay_id", right_index=True, how="left"
        )

        train_tfid = self.vectorizer.fit_transform(self.train_dataset["full_text"])
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(train_tfid)
        tfidf_df.columns = [f"tfid_{i}" for i in range(tfidf_df.shape[1])]
        tfidf_df["essay_id"] = self.train_dataset["essay_id"]
        train_feats = train_feats.merge(tfidf_df, on="essay_id", how="left")

        print("feature_num: ", train_feats.shape[1] - 2)
        return train_feats

    def process_test(self):
        self.test_dataset["clean_text"] = self.test_dataset["full_text"].apply(
            self.dataPreprocessing
        )

        para_feat = self.test_dataset["clean_text"].apply(self.paragraph_stats)
        test_feats = pd.concat([self.test_dataset[["essay_id"]], para_feat], axis=1)

        sent_feat = self.test_dataset["clean_text"].apply(self.sentence_stats)
        test_feats = test_feats.merge(
            sent_feat, left_on="essay_id", right_index=True, how="left"
        )

        word_feat = self.test_dataset["clean_text"].apply(self.word_stats)
        test_feats = test_feats.merge(
            word_feat, left_on="essay_id", right_index=True, how="left"
        )

        test_tfid = self.vectorizer.transform(self.test_dataset["full_text"])
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(test_tfid)
        tfidf_df.columns = [f"tfid_{i}" for i in range(tfidf_df.shape[1])]
        tfidf_df["essay_id"] = self.test_dataset["essay_id"]
        test_feats = test_feats.merge(tfidf_df, on="essay_id", how="left")

        print("feature_num: ", test_feats.shape[1] - 1)
        return test_feats




## === cell 2
class LGBM:
    def __init__(self):
        self.data_train = pd.read_csv(
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
        )
        self.data_test = pd.read_csv(
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

    def fit(self, df, training_fold=[0, 1, 2, 3, 4]):
        feature_names = [c for c in df.columns if c not in ["essay_id", "score"]]
        x = df[feature_names]
        y = df["score"].values

        counter = Counter(y)
        print(counter)

        splits = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y)
        )

        for fold in training_fold:
            trn_idx, val_idx = splits[fold]

            X_train = x.iloc[trn_idx]
            Y_train = y[trn_idx] - self.a
            X_val = x.iloc[val_idx]
            Y_val = y[val_idx] - self.a
            print("\nFold_{} Training ================================\n".format(fold))
            self.model.fit(
                X_train,
                Y_train,
            )
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

        cf_matrix = confusion_matrix(
            Y_val + self.a, (pred_val + self.a).clip(1, 6).round()
        )
        df_cm = pd.DataFrame(
            cf_matrix, index=[i for i in "123456"], columns=[i for i in "123456"]
        )
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)

    def predict(self, df):
        feature_names = [c for c in df.columns if c != "essay_id"]
        predictions = self.model.predict(df[feature_names] + self.a).clip(1, 6).round()
        return predictions

    def submit(self, df):
        feature_names = [c for c in df.columns if c != "essay_id"]
        return self.data_test[["essay_id"]].assign(
            score=(self.model.predict(df[feature_names]) + self.a).clip(1, 6).round()
        )




## === cell 3
FE = FeatureEngineering()
train_feature = FE.process()
test_feature = FE.process_test()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3792444245.py in <cell line: 0>()
      1 FE = FeatureEngineering()
----> 2 train_feature = FE.process()
      3 test_feature = FE.process_test()
      4 

/tmp/ipykernel_55/2518377111.py in process(self)
    140         # sentence features
    141         sent_feat = self.train_dataset["clean_text"].apply(self.sentence_stats)
--> 142         train_feats = train_feats.merge(
    143             sent_feat, left_on="essay_id", right_index=True, how="left"
    144         )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    805         # validate the merge keys dtypes. We may need to coerce
    806         # to avoid incompatible dtypes
--> 807         self._maybe_coerce_merge_keys()
    808 
    809         # If argument passed to validate,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _maybe_coerce_merge_keys(self)
   1506                     inferred_right in string_types and inferred_left not in string_types
   1507                 ):
-> 1508                     raise ValueError(msg)
   1509 
   1510             # datetimelikes must match exactly

ValueError: You are trying to merge on object and int64 columns for key 'essay_id'. If you wish to proceed you should use pd.concat

## === cell 4
model = LGBM()
model.fit(df=train_feature, training_fold=[0])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1596969949.py in <cell line: 0>()
      1 model = LGBM()
----> 2 model.fit(df=train_feature, training_fold=[0])
      3 

NameError: name 'train_feature' is not defined

## === cell 5
submission = model.submit(test_feature)
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2878154155.py in <cell line: 0>()
----> 1 submission = model.submit(test_feature)
      2 submission.to_csv("submission.csv", index=False)

NameError: name 'test_feature' is not defined
