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
import os, re, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from collections import Counter
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
from joblib import Parallel, delayed
from scipy import sparse

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)




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

    def combined_stats(self, text):
        res = {}

        paragraphs = [p for p in text.split("\n\n") if len(p) >= 25]
        plen = [len(p) for p in paragraphs]
        p_sent_cnt = [p.count(".") for p in paragraphs]
        p_word_cnt = [len(p.split()) for p in paragraphs]

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
                for suf in ["max", "mean", "min", "first", "last"]:
                    res[f"{fea}_{suf}"] = 0

        sentences = [s for s in text.split(".") if len(s) >= 15]
        slen = [len(s) for s in sentences]
        s_word_cnt = [len(s.split()) for s in sentences]

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
                for suf in ["max", "mean", "min", "first", "last"]:
                    res[f"{fea}_{suf}"] = 0

        words = [w for w in text.split() if w]
        wlen = [len(w) for w in words]

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
            for key in [
                "word_len_max",
                "word_len_mean",
                "word_len_std",
                "word_len_q1",
                "word_len_q2",
                "word_len_q3",
            ]:
                res[key] = 0

        return pd.Series(res)

    def _apply_combined_stats_parallel(self, texts):
        """Parallelize combined_stats using all available CPUs."""
        max_jobs = max(1, os.cpu_count())
        results = Parallel(n_jobs=max_jobs, backend="loky")(
            delayed(self.combined_stats)(txt) for txt in texts
        )
        return pd.DataFrame(results)

    def _process(self, df, is_train=True):
        df["clean_text"] = df["full_text"].apply(self.dataPreprocessing)
        stats_feat = self._apply_combined_stats_parallel(df["clean_text"])
        if is_train:
            stats_feat = pd.concat(
                [df[["essay_id", "score"]].reset_index(drop=True), stats_feat], axis=1
            )
        else:
            stats_feat = pd.concat(
                [df[["essay_id"]].reset_index(drop=True), stats_feat], axis=1
            )

        tfidf_mat = (
            self.vectorizer.fit_transform(df["full_text"])
            if is_train
            else self.vectorizer.transform(df["full_text"])
        )
        return stats_feat, tfidf_mat

    def process(self):
        return self._process(self.train_dataset, is_train=True)

    def process_test(self):
        return self._process(self.test_dataset, is_train=False)




## === cell 2
FE = FeatureEngineering()
train_stats, train_tfidf = FE.process()
test_stats, test_tfidf = FE.process_test()




## === cell 3
class LGBM:
    def __init__(self):
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
            "device": "cpu",
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
        f = 0.5 * np.sum((preds - labels) ** 2)
        g = 0.5 * np.sum((preds - self.a) ** 2 + self.b)
        df = preds - labels
        dg = preds - self.a
        grad = (df / g - f * dg / g**2) * len(labels)
        hess = np.ones(len(labels))
        return grad, hess

    def _build_feature_matrix(self, stats_df, tfidf_mat):
        stat_features = stats_df.drop(columns=["essay_id", "score"], errors="ignore")
        stat_csr = sparse.csr_matrix(stat_features.values)
        return sparse.hstack([stat_csr, tfidf_mat]).tocsr()

    def fit(self, stats_df, tfidf_mat, training_fold=[0, 1, 2, 3, 4]):
        X = self._build_feature_matrix(stats_df, tfidf_mat)
        y = stats_df["score"].values

        counter = Counter(y)
        print(counter)

        splits = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y)
        )

        for fold in training_fold:
            trn_idx, val_idx = splits[fold]

            X_train = X[trn_idx]
            Y_train = y[trn_idx] - self.a
            X_val = X[val_idx]
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

    def predict(self, stats_df, tfidf_mat):
        X = self._build_feature_matrix(stats_df, tfidf_mat)
        predictions = self.model.predict(X)
        return (predictions + self.a).clip(1, 6).round()

    def submit(self, stats_df, tfidf_mat):
        predictions = self.predict(stats_df, tfidf_mat)
        return self.data_test[["essay_id"]].assign(score=predictions)




## === cell 4
model = LGBM()
model.fit(stats_df=train_stats, tfidf_mat=train_tfidf, training_fold=[0])




## === cell 5
submission = model.submit(stats_df=test_stats, tfidf_mat=test_tfidf)
submission.to_csv("submission.csv", index=False)
