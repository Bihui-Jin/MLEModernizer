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

0.72596

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.70592) has done: 'I remove the `imblearn` dependency because it is incompatible with the installed scikit-learn version in this environment, and it is the first import-time crash preventing any run. Then I fix the missing `Counter` import and make the training actually fit the model used for submission by fitting per-fold LightGBM models (instead of repeatedly fitting a single `VotingRegressor`), and average their predictions at inference. Finally, I correct a logic bug where `+self.a` was incorrectly added to features during prediction, ensure feature alignment between train/test, and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.71014) has done: 'Your current score (0.70592) is below the target (0.80809), so we should improve performance with minimal changes that preserve the same feature pipeline and LightGBM training loop. The biggest gain with minimal risk is to train on all 5 CV folds (instead of only fold 0), since you already average predictions across fold-models at inference but currently only have one fold trained. I also fix an important LightGBM parameter typo (`metrics` → `metric`) so the custom objective setup behaves as intended (this doesn’t change the core approach, just makes it correctly configured). Everything else (feature engineering, objective, rounding/clipping, submission schema/paths) stays the same.'
- What this solution (achieved 0.71014) has done: 'We’re below the target (0.71014 vs 0.80809), so the safest way to move score upward without changing the model/feature logic is to fix a few correctness issues that directly hurt QWK. I (1) compute QWK on the same 1–6 integer scale for both y_true and y_pred (your current metric adds `self.a` twice to y_true), (2) ensure stratification uses integer labels (rounding can prevent subtle binning issues), and (3) align train/test feature columns by explicit reindexing to the training feature list to prevent any accidental ordering drift. These are minimal changes that preserve your feature extraction, LightGBM setup, training loop, and prediction semantics (still clipping+rounding to 1–6), but should improve the learned signal and validation behavior.'
- What this solution (achieved 0.72596) has done: 'We’re substantially below the target (0.71014 vs 0.80809), so the lowest-risk way to move QWK upward without changing the model/feature logic is to fix score-calibration issues that QWK is very sensitive to. I (1) correct the custom `quadratic_weighted_kappa` so it evaluates on the true 1–6 scale (it currently adds `self.a` twice to `y_true`), and (2) learn an optimal post-hoc linear calibration `(a,b)` on out-of-fold predictions (same rounding/clipping semantics) and then apply it consistently at inference; this preserves the same model training loop and objective, but improves the discretization step that drives QWK. I also ensure the test feature columns exactly match the training feature list used by the model to avoid any silent column-order issues. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score (0.72596) is below the target (0.80809), so we should push QWK upward with the smallest changes that keep the same feature pipeline and LightGBM training loop. The biggest low-risk gain here is to make the custom QWK objective consistent with the label shift used in training (your `qwk_obj` currently double-shifts labels by adding `self.a` to already-shifted `y_true`). I also make LightGBM aware of this custom objective by passing it via `objective` at `fit()` (rather than inside params where it can be ignored/treated inconsistently), while keeping the same model/parameters. Everything else (features, folds, calibration grid search, rounding/clipping, and submission format/path) stays the same.'
- What this solution (achieved 0.72596) has done: 'I fix the LightGBM runtime error by attaching the custom objective correctly (via the estimator’s `objective` parameter, not `fit()`), which restores end-to-end execution. I also ensure the custom QWK evaluation callback is consistent with your shifted-label training setup by converting both labels and predictions back to the 1–6 score scale inside the metric. These changes are minimal, keep the same feature engineering, CV training loop, ensembling, and calibration logic, and produce a valid `submission.csv`. The adjustments should also improve the score versus the current broken run (0.0) by actually training with the intended objective/metric coherently.'
- What this solution (achieved 0.72596) has done: 'Your current score (0.72596) is below the target (0.80809), so we should move QWK upward with the smallest changes that keep your feature pipeline, LightGBM training loop, and custom objective intact. The main issue is that your learned calibration (cal_a_/cal_b_) is computed after CV but never applied to the fold validation metrics (and also not used when you compute the fold-wise QWK/accuracy during training), so the model is effectively evaluated and tuned with inconsistent calibration. I (1) store out-of-fold calibrated predictions and report/verify an overall OOF QWK (sanity check that calibration actually helps), and (2) apply the *final learned calibration* consistently for both fold metrics and final test prediction (same rounding/clipping semantics), without changing the model/feature logic. This is a minimal, metric-aligned correction that typically improves QWK because QWK is very sensitive to the final discretization thresholds.'
- What this solution (achieved 0.72596) has done: 'We’re below the target (0.72596 vs 0.80809), so the smallest reliable way to move QWK upward without changing your feature pipeline or model family is to fix one correctness issue in the custom objective: it currently adds `self.a` to predictions even though the model already predicts the shifted target, which distorts gradients and hurts fit. I make the objective operate consistently in the shifted space (no double shift), while keeping the exact same training loop, LightGBM parameters, CV setup, and the same post-hoc linear calibration and rounding/clipping to 1–6. I also update the fold metric call to pass the correct shifted labels to the metric (already mostly correct) and keep the submission generation unchanged.'

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from collections import Counter
from sklearn.metrics import confusion_matrix




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
        tmp = tmp.explode("paragraph")
        tmp = tmp.with_columns(pl.col("paragraph").map_elements(self.dataPreprocessing))
        tmp = tmp.with_columns(
            pl.col("paragraph").map_elements(lambda x: len(x)).alias("paragraph_len")
        )
        tmp = tmp.filter(pl.col("paragraph_len") >= 25)
        tmp = tmp.with_columns(
            pl.col("paragraph")
            .map_elements(lambda x: len(x.split(".")))
            .alias("paragraph_sentence_cnt"),
            pl.col("paragraph")
            .map_elements(lambda x: len(x.split(" ")))
            .alias("paragraph_word_cnt"),
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
        df = df.to_pandas()
        print("done Paragraph_Eng +", len(df.columns), "features")
        return df

    def Sentence_Preprocess(self, tmp):
        tmp = tmp.with_columns(
            pl.col("full_text")
            .map_elements(self.dataPreprocessing)
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
        df = df.to_pandas()
        print("done Sentence_Eng +", len(df.columns), "features")
        return df

    def Word_Preprocess(self, tmp):
        tmp = tmp.with_columns(
            pl.col("full_text")
            .map_elements(self.dataPreprocessing)
            .str.split(by=" ")
            .alias("word")
        )
        tmp = tmp.explode("word")
        tmp = tmp.with_columns(
            pl.col("word").map_elements(lambda x: len(x)).alias("word_len")
        )
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
        df = df.to_pandas()
        print("done Word_Eng +", len(df.columns), "features")
        return df

    def process(self):
        tmp = self.Paragraph_Preprocess(self.train_dataset)
        train_feats = self.Paragraph_Eng(tmp)

        score_df = self.train_dataset.select(["essay_id", "score"]).to_pandas()
        train_feats = train_feats.merge(score_df, on="essay_id", how="left")

        tmp = self.Sentence_Preprocess(self.train_dataset)
        train_feats = train_feats.merge(
            self.Sentence_Eng(tmp), on="essay_id", how="left"
        )

        tmp = self.Word_Preprocess(self.train_dataset)
        train_feats = train_feats.merge(self.Word_Eng(tmp), on="essay_id", how="left")

        train_tfid = self.vectorizer.fit_transform(
            [i for i in self.train_dataset["full_text"]]
        )
        dense_matrix = train_tfid.toarray()
        df = pd.DataFrame(dense_matrix)
        tfid_columns = [f"tfid_{i}" for i in range(len(df.columns))]
        df.columns = tfid_columns
        df["essay_id"] = train_feats["essay_id"].values
        train_feats = train_feats.merge(df, on="essay_id", how="left")

        print("feature_num: ", len(train_feats.columns) - 2)
        return train_feats

    def process_test(self):
        temp = self.Paragraph_Preprocess(self.test_dataset)
        test_feats = self.Paragraph_Eng(temp)

        temp = self.Sentence_Preprocess(self.test_dataset)
        test_feats = test_feats.merge(
            self.Sentence_Eng(temp), on="essay_id", how="left"
        )

        temp = self.Word_Preprocess(self.test_dataset)
        test_feats = test_feats.merge(self.Word_Eng(temp), on="essay_id", how="left")

        test_tfid = self.vectorizer.transform(
            [i for i in self.test_dataset["full_text"]]
        )
        dense_matrix = test_tfid.toarray()
        df = pd.DataFrame(dense_matrix)
        tfid_columns = [f"tfid_{i}" for i in range(len(df.columns))]
        df.columns = tfid_columns
        df["essay_id"] = test_feats["essay_id"].values
        test_feats = test_feats.merge(df, on="essay_id", how="left")

        print("feature_num: ", len(test_feats.columns) - 1)
        return test_feats




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

        self.cal_a_ = 2.948
        self.cal_b_ = 1.092

        self.a = 2.948
        self.b = 1.092

        self.lgb_parameters = {
            "metric": "None",
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

        self.models_ = []
        self.feature_names_ = None
        self.oof_pred_raw_ = None
        self.oof_pred_cal_ = None

    def quadratic_weighted_kappa(self, y_true_shifted, y_pred_score):
        y_true_score = y_true_shifted + self.a
        y_true_int = np.clip(np.rint(y_true_score), 1, 6).astype(int)
        y_pred_int = np.clip(np.rint(y_pred_score), 1, 6).astype(int)
        qwk = cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")
        return "QWK", qwk, True

    def qwk_obj(self, y_true, y_pred):
        labels_shift = y_true
        preds_shift = y_pred

        preds_score = (preds_shift + self.a).clip(1, 6)
        preds_shift = preds_score - self.a

        f = 0.5 * np.sum((preds_shift - labels_shift) ** 2)
        g = 0.5 * np.sum(
            (preds_shift - 0.0) ** 2 + self.b
        )  # 0.0 corresponds to score=a in shifted space

        df = preds_shift - labels_shift
        dg = preds_shift - 0.0
        grad = (df / g - f * dg / (g**2)) * len(labels_shift)
        hess = np.ones(len(labels_shift))
        return grad, hess

    def _prepare_xy(self, df):
        feature_names = [c for c in df.columns if c not in ["essay_id", "score"]]
        X = df[feature_names].copy()
        y = df["score"].values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X, y, feature_names

    def _predict_raw_from_fold_models(self, fold_models, X):
        return np.mean([m.predict(X) for m in fold_models], axis=0)

    def _apply_calibration(self, pred_raw):
        return self.cal_a_ + self.cal_b_ * pred_raw

    def _fit_calibration_from_oof(self, y_true_score, oof_pred_raw):
        y_true_int = np.clip(np.rint(y_true_score), 1, 6).astype(int)

        a_grid = np.linspace(2.5, 3.4, 46)
        b_grid = np.linspace(0.85, 1.25, 41)

        best = (-1.0, self.cal_a_, self.cal_b_)
        for a in a_grid:
            for b in b_grid:
                pred = a + b * oof_pred_raw
                pred_int = np.clip(np.rint(pred), 1, 6).astype(int)
                qwk = cohen_kappa_score(y_true_int, pred_int, weights="quadratic")
                if qwk > best[0]:
                    best = (qwk, float(a), float(b))

        self.cal_a_, self.cal_b_ = best[1], best[2]
        print(
            f"OOF calibration selected: cal_a_={self.cal_a_:.4f}, cal_b_={self.cal_b_:.4f}, OOF_QWK={best[0]:.5f}"
        )

    def fit(self, df, training_fold=[0, 1, 2, 3, 4]):
        X, y, self.feature_names_ = self._prepare_xy(df)

        counter = Counter(y)
        print("train distribution:", counter)

        y_strat = np.clip(np.rint(y), 1, 6).astype(int)
        splits = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y_strat)
        )

        self.models_ = []
        self.acc_metrics = []
        self.cohen_metrics = []

        oof_pred_raw = np.full(shape=len(y), fill_value=np.nan, dtype=np.float64)

        for fold in training_fold:
            trn_idx, val_idx = splits[fold]
            X_train = X.iloc[trn_idx]
            Y_train = y[trn_idx] - self.a
            X_val = X.iloc[val_idx]
            Y_val = y[val_idx] - self.a

            print("\nFold_{} Training ================================\n".format(fold))

            fold_models = []
            for i in range(self.num_models):
                m = lgb.LGBMRegressor(
                    **self.lgb_parameters, objective=self.qwk_obj, random_state=i + 40
                )
                m.fit(X_train, Y_train)
                fold_models.append(m)

            self.models_.append(fold_models)

            pred_val_raw = self._predict_raw_from_fold_models(fold_models, X_val)
            oof_pred_raw[val_idx] = pred_val_raw

            pred_val_score = self._apply_calibration(pred_val_raw)

            cohen_score = self.quadratic_weighted_kappa(Y_val, pred_val_score)
            accuracy = accuracy_score(
                np.clip(np.rint(Y_val + self.a), 1, 6).astype(int),
                np.clip(np.rint(pred_val_score), 1, 6).astype(int),
            )
            self.acc_metrics.append(accuracy)
            self.cohen_metrics.append(cohen_score[1])

            print(f"Accuracy fold {fold}: {accuracy:.4f}")
            print(f"Cohen score fold {fold}: {cohen_score[1]:.4f}")

        if np.any(np.isnan(oof_pred_raw)):
            print(
                "Warning: missing OOF predictions for some rows; skipping calibration fit and using default cal_a_/cal_b_."
            )
        else:
            self._fit_calibration_from_oof(y_true_score=y, oof_pred_raw=oof_pred_raw)

        self.oof_pred_raw_ = oof_pred_raw.copy()
        self.oof_pred_cal_ = self._apply_calibration(oof_pred_raw)

        if not np.any(np.isnan(self.oof_pred_raw_)):
            y_true_int = np.clip(np.rint(y), 1, 6).astype(int)
            y_oof_int = np.clip(np.rint(self.oof_pred_cal_), 1, 6).astype(int)
            oof_qwk = cohen_kappa_score(y_true_int, y_oof_int, weights="quadratic")
            print(f"Overall OOF QWK (with final calibration): {oof_qwk:.5f}")

        print(f"Average Accuracy all fold: {np.mean(self.acc_metrics):.4f}")
        print(f"Average Cohen all fold: {np.mean(self.cohen_metrics):.4f}")

        cf_matrix = confusion_matrix(
            np.clip(np.rint(Y_val + self.a), 1, 6).astype(int),
            np.clip(np.rint(self._apply_calibration(pred_val_raw)), 1, 6).astype(int),
            labels=[1, 2, 3, 4, 5, 6],
        )
        df_cm = pd.DataFrame(
            cf_matrix, index=[i for i in "123456"], columns=[i for i in "123456"]
        )
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)

    def predict(self, df):
        feature_names = [c for c in df.columns if c != "essay_id"]
        if self.feature_names_ is not None:
            feature_names = self.feature_names_
            X = df.reindex(columns=feature_names, fill_value=0).copy()
        else:
            X = df[feature_names].copy()

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        preds_all = []
        for fold_models in self.models_:
            preds_fold_raw = self._predict_raw_from_fold_models(fold_models, X)
            preds_all.append(preds_fold_raw)

        pred_raw = np.mean(preds_all, axis=0)

        pred_score = self._apply_calibration(pred_raw)
        return np.clip(np.rint(pred_score), 1, 6).astype(int)

    def submit(self, df):
        pred = self.predict(df)
        sub = pd.DataFrame({"essay_id": df["essay_id"].values, "score": pred})
        return sub




## === cell 3
FE = FeatureEngineering()
train_feature = FE.process()
test_feature = FE.process_test()

train_feature_names = [
    c for c in train_feature.columns if c not in ["essay_id", "score"]
]
test_feature = test_feature.reindex(
    columns=["essay_id"] + train_feature_names, fill_value=0
)



## === cell 4
model = LGBM()
model.fit(df=train_feature, training_fold=[0, 1, 2, 3, 4])



## === cell 5
submission = model.submit(test_feature)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
