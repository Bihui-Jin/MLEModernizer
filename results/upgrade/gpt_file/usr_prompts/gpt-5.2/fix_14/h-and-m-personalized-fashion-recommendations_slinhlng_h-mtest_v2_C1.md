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
The training data is the purchase history of customers across time. The task is to predict what articles each customer will purchase in the 7-day period immediately after the training data ends.

## Metric
Mean Average Precision @ 12 (MAP@12):

$$
\text{MAP@12}=\frac{1}{U} \sum_{u=1}^U \frac{1}{\min (m, 12)} \sum_{k=1}^{\min (n, 12)} P(k) \times \text{rel}(k)
$$

where $U$ is the number of customers, $P(k)$ is the precision at cutoff $k, n$ is the number predictions per customer, $m$ is the number of ground truth values per customer, and $\text{rel}(k)$ is an indicator function equaling 1 if the item at rank $k$ is a relevant (correct) label, zero otherwise.

You must make predictions for all `customer_id` values found in the sample submission. All customers who made purchases during the test period are scored, regardless of whether they had purchase history in the training data.

## Submission Format
For each `customer_id` observed in the training data, you may predict up to 12 labels for the `article_id`, which is the predicted items a customer will buy in the next 7-day period after the training time period. The file should contain a header and have the following format:

```
customer_id,prediction
00000dba,0706016001 0706016002 0372860001 ...
0000423b,0706016001 0706016002 0372860001 ...
...
```

## Dataset
- **images/** - a folder of images corresponding to each `article_id`; images are placed in subfolders starting with the first three digits of the `article_id`; note, not all `article_id` values have a corresponding image.
- **articles.csv** - detailed metadata for each `article_id` available for purchase
- **customers.csv** - metadata for each `customer_id` in dataset
- **sample_submission.csv** - a sample submission file in the correct format
- **transactions_train.csv** - the training data, consisting of the purchases each customer for each date, as well as additional information. Duplicate rows correspond to multiple purchases of the same item. Your task is to predict the `article_id`s each customer will purchase during the 7-day period immediately after the training data period.

# 2. Python version

3.14

# 3. Installed packages

cuml-cu12==25.2.1
geopandas==0.14.4
libcuml-cu12==25.2.1
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            articles.csv (105543 lines)
            articles.csv.zip (4.4 MB)
            customers.csv (1371981 lines)
            customers.csv.zip (102.4 MB)
            description.md (74 lines)
            images.zip (30.0 GB)
            sample_submission.csv (1371981 lines)
            sample_submission.csv.zip (53.3 MB)
            transactions_train.csv (31521961 lines)
            transactions_train.csv.zip (604.1 MB)
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
            images/
                010/
                    0108775015.jpg (154.6 kB)
                    0108775044.jpg (106.7 kB)
                    ... and 1 other files
                011/
                    0110065001.jpg (148.6 kB)
                    0110065002.jpg (85.7 kB)
                    ... and 18 other files
                ... and 84 other folders
        input/
            articles.csv (105543 lines)
            articles.csv.zip (4.4 MB)
            customers.csv (1371981 lines)
            customers.csv.zip (102.4 MB)
            description.md (74 lines)
            images.zip (30.0 GB)
            sample_submission.csv (1371981 lines)
            sample_submission.csv.zip (53.3 MB)
            transactions_train.csv (31521961 lines)
            transactions_train.csv.zip (604.1 MB)
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
            images/
                010/
                    0108775015.jpg (154.6 kB)
                    0108775044.jpg (106.7 kB)
                    ... and 1 other files
                011/
                    0110065001.jpg (148.6 kB)
                    0110065002.jpg (85.7 kB)
                    ... and 18 other files
                ... and 84 other folders
        working/
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
```

-> data/articles.csv has 105542 rows and 25 columns.
The columns are: article_id, product_code, prod_name, product_type_no, product_type_name, product_group_name, graphical_appearance_no, graphical_appearance_name, colour_group_code, colour_group_name, perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name, department_no... and 10 more columns

-> data/customers.csv has 1371980 rows and 7 columns.
The columns are: customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code

-> data/h-and-m-personalized-fashion-recommendations/articles.csv has 105542 rows and 25 columns.
The columns are: article_id, product_code, prod_name, product_type_no, product_type_name, product_group_name, graphical_appearance_no, graphical_appearance_name, colour_group_code, colour_group_name, perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name, department_no... and 10 more columns

-> data/h-and-m-personalized-fashion-recommendations/customers.csv has 1371980 rows and 7 columns.
The columns are: customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code

-> data/h-and-m-personalized-fashion-recommendations/sample_submission.csv has 1371980 rows and 2 columns.
The columns are: customer_id, prediction

-> data/h-and-m-personalized-fashion-recommendations/transactions_train.csv has 31521960 rows and 5 columns.
The columns are: t_dat, customer_id, article_id, price, sales_channel_id

-> data/sample_submission.csv has 1371980 rows and 2 columns.
The columns are: customer_id, prediction

-> data/transactions_train.csv has 31521960 rows and 5 columns.
The columns are: t_dat, customer_id, article_id, price, sales_channel_id

-> (stopped after 10 files for performance)

# 5. Target score

0.028964833733073

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the `InvalidIndexError` by ensuring the `article_pairs_df` mapping index is unique before calling `Series.map`, since `create_pairs` produces multiple rows per `article_id`. I keep the existing candidate-generation and scoring logic intact, only changing the pair-candidate construction to use a de-duplicated mapping (top-weight pair per article) and avoiding index-alignment mistakes. I also adjust the pair-candidate assignment to preserve row alignment (don’t `.dropna()` mid-assignment) so it cannot crash. With these fixes, the CV run and full prediction run complete and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an invalid submission format (typically `article_id` not being 10-digit zero-padded strings), because MAP@12 be zero if none of the predicted IDs match the ground truth string format. I make the smallest change that preserves your candidate generation and popularity scoring: force `article_id` to be read as a string (not int), and ensure all internal candidate `article_id` values are consistently 10-digit strings before grouping/formatting. I also make the pair-generation and popularity counting operate on the string ids (no semantic change—just consistent typing), so the final `prediction` field matches Kaggle’s expected IDs. This should move the score upward toward your target without changing the modeling approach.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 public score is almost certainly due to a submission mismatch where `article_id` formatting/types don’t exactly match Kaggle’s expected 10-digit strings at the final `prediction` string stage (even a small drift like `123` vs `0000000123` yields zero hits). I make a minimal, score-directed fix by enforcing `_aid10()` normalization at the last possible moment: before scoring (`art_pop` index + candidate `article_id`) and when building the final per-customer prediction strings. I also ensure `value_counts()` popularity is computed on already-normalized 10-digit ids, and that the submission `prediction` field is guaranteed to be space-separated 10-digit ids with no “nan/None” tokens. This preserves your exact candidate-generation and popularity-scoring logic, but removes the main cause of MAP@12 collapsing to 0.0.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with a silent submission mismatch rather than model quality: even if you build candidates correctly, Kaggle score zero if `article_id` tokens don’t exactly match the expected 10-digit strings at the very last step. I make a minimal, score-directed fix by enforcing 10-digit normalization on every prediction token right before writing `submission.csv`, and by guaranteeing exactly up to 12 space-separated tokens per customer (no `nan/None`, no empty tokens). I also ensure `customer_id` alignment is preserved by building the submission strictly from `sample_submission.csv` order and using a normalized prediction series indexed by `customer_id`. Core candidate generation and popularity scoring remain unchanged; this only fixes output correctness so MAP@12 can move up toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with predictions not matching any ground-truth tokens due to the “week” logic: `day_week_numbers_fixed()` currently labels the last week of training as week 1 (not ~104), so your `label_week = max_week` ends up training on almost the entire history but then predicting the “next week” from a misaligned week index and pair-weeks like 96–104 that don’t exist. I make the smallest semantic fix by restoring a stable, Kaggle-standard week numbering anchored to the known last training date (2020-09-22) so that `week_number` matches the expected ~1..104 scale and your existing CV/sub parameters operate as intended. I also adjust the pair-week generation to derive from the computed `t["week_number"].max()` instead of hard-coding 96–104, keeping the same `pairs_per_item` and pair logic but ensuring pairs are actually available for the intended weeks. These changes preserve your candidate generation and popularity scoring, but should move the public MAP@12 upward toward your target by making the temporal split and pair candidates consistent.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the submission’s `customer_id` values don’t match Kaggle’s expected 8-hex format: the sample file uses the full 64-bit hex string, so mapping with truncated/altered IDs yields all-empty predictions and MAP@12 = 0. I make the smallest score-directed fix by enforcing correct `customer_id` parsing as a non-truncated string (not pandas `string` with potential NA coercion), and I build the submission predictions by *position* (reindex) rather than `map` to avoid any silent key-mismatch. I also add a hard sanity-check that fails fast if too many predictions are empty, so you don’t upload a 0-scoring file again. Core candidate generation and popularity scoring stay identical; this only fixes ID alignment/output correctness so the score can move up toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with a “valid CSV but effectively empty/invalid predictions” issue, and the current pipeline is very likely producing almost all-empty predictions due to an incorrect `feature_label_split` for the real submission case (there is no “label week” in test, so using `label_week=max_week` makes `features_df` exclude the last training week and often collapses your candidate generation). I make the smallest semantic fix for submission only: when `cv=False`, build `features_df` from the last `feature_periods` weeks ending at `max_week` (inclusive) and ignore `label_df`, while keeping CV behavior unchanged. I also ensure pair-candidates for submission use the most recent available pairs week (the last week in `week_number_pairs`) rather than `label_week-1`, again only when `cv=False`, so pair candidates aren’t silently empty. These changes keep your architecture/candidate logic/popularity scoring the same, but restore non-empty candidate sets so MAP@12 can move up toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely because the submission contains mostly empty predictions (or predictions not aligned to the sample customers), which yields MAP@12≈0 even if the code “runs.” I make a minimal, score-directed change to guarantee every customer gets 12 valid 10-digit `article_id` tokens by adding a safe fallback: fill missing/empty per-customer predictions with the global top-12 popular articles from the most recent `pa_num_weeks` window. This keeps your core candidate generation and popularity scoring intact, but prevents empty rows and key-mismatch from collapsing the public score to 0.0. I also add a small sanity check ensuring the fallback list is non-empty and properly formatted before writing `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your pipeline likely still scores ~0 because you truncate `customer_id` to the last 8 characters (`_cid`), which breaks alignment with `sample_submission` and the ground truth customer keys; even with a fallback, you end up assigning predictions to the wrong customers. I make the smallest score-directed change: keep `customer_id` as the full original string everywhere (no truncation), so candidate generation, grouping, and submission mapping align correctly. I also keep your existing article_id 10-digit normalization and fallback logic, only adjusting places where `_cid` was applied so the code runs end-to-end and writes a valid `submission.csv`. This should move the public MAP@12 upward toward your target without changing the model/candidate/popularity core logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is still most consistent with “no matches at all,” and the most likely remaining cause is that the `week_number` logic is off by one for the last training date, so your “recent week” popularity/candidates are drawn from the wrong (often empty/irrelevant) window. I make the smallest semantic fix: anchor `week_number` to the true last transaction date in the provided `transactions_train.csv` (not a hard-coded constant), ensuring the last week becomes `max_week` consistently. This preserves your exact candidate-generation and popularity scoring logic, but makes the time windows (recent weeks, pairs week, label week) align with the dataset, which should move MAP@12 upward toward your target. I also add a tiny sanity check that the computed `week_number.max()` matches the intended label/sub weeks to avoid silently generating near-random predictions again.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with *systematic label mismatch* rather than model quality: the code forces `week_number` into a hard 1..104 scale (`104 - weeks_ago`), but the actual last-transaction date in your CSV is not guaranteed to align to “week 104”, so your “recent weeks” windows and pair weeks can be shifted, making candidates effectively unrelated to the true target week. I make the smallest semantic fix: compute `week_number` directly from the dataset’s true last date so that the most recent week is always `max_week`, and then use that `max_week` consistently (CV keeps using 104 only if it exists). This preserves your candidate generation and popularity scoring logic, but aligns time windows to the data so predictions have a chance to match the held-out/test week. I also add a tiny safety tweak so `pairs_week` never points to a week for which pairs weren’t built (preventing silent empty pair-candidates).'

# 9. Code solution

## === cell 0
import os
import sys
import gc
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


DATA_ROOTS = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data/h-and-m-personalized-fashion-recommendations",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(fname: str) -> str:
    for root in DATA_ROOTS:
        path = os.path.join(root, fname)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {fname} in any of {DATA_ROOTS}")


ARTICLES_PATH = _find_file("articles.csv")
CUSTOMERS_PATH = _find_file("customers.csv")
TRANSACTIONS_PATH = _find_file("transactions_train.csv")
SAMPLE_SUB_PATH = _find_file("sample_submission.csv")

print("Using paths:")
print(" articles:", ARTICLES_PATH)
print(" customers:", CUSTOMERS_PATH)
print(" transactions:", TRANSACTIONS_PATH)
print(" sample_sub:", SAMPLE_SUB_PATH)


def _aid10(x) -> str:
    if pd.isna(x):
        return np.nan
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    s = s.strip()
    if s.isdigit():
        return s.zfill(10)
    return s


def _clean_pred_str(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    toks = str(s).split()
    out = []
    seen = set()
    for t in toks:
        tt = _aid10(t)
        if not isinstance(tt, str) or tt.lower() in ("nan", "none"):
            continue
        if tt not in seen:
            seen.add(tt)
            out.append(tt)
        if len(out) >= 12:
            break
    return " ".join(out)


def _finalize_pred_series(pred_ser: pd.Series) -> pd.Series:
    if pred_ser is None:
        return pd.Series(dtype="object")
    if not isinstance(pred_ser, pd.Series):
        pred_ser = pd.Series(pred_ser)
    pred_ser = pred_ser.copy()
    pred_ser = pred_ser.map(_clean_pred_str)
    return pred_ser


def _cid(x) -> str:
    if pd.isna(x):
        return np.nan
    return str(x).strip()




## === cell 1
class _CudfCompat:
    DataFrame = pd.DataFrame
    Series = pd.Series

    @staticmethod
    def concat(objs, **kwargs):
        return pd.concat(objs, **kwargs)

    @staticmethod
    def to_datetime(x):
        return pd.to_datetime(x)


cudf = _CudfCompat()




## === cell 2
class h_io:
    @staticmethod
    def load_data(files):
        out = []
        for f in files:
            path = _find_file(f)
            if f == "transactions_train.csv":
                df = pd.read_csv(
                    path,
                    dtype={"customer_id": "object", "article_id": "object"},
                )
            elif f == "customers.csv":
                df = pd.read_csv(
                    path, dtype={"customer_id": "object", "postal_code": "object"}
                )
            elif f == "articles.csv":
                df = pd.read_csv(path, dtype={"article_id": "object"})
            else:
                df = pd.read_csv(path)
            out.append(df)
        return out


class h_fe:
    @staticmethod
    def reduce_customer_id_memory(c, t_list):
        return "index_to_id_dummy.pkl"

    @staticmethod
    def day_numbers(dates):
        d = pd.to_datetime(dates)
        base = d.min()
        return (d - base).dt.days.astype("int32")

    @staticmethod
    def day_week_numbers(dates):
        d = pd.to_datetime(dates)
        base = d.min()
        days = (d - base).dt.days + 1
        extra = int(days.max() % 7)
        adjusted = (days - extra).astype("int32")
        weeks = -((-adjusted) // 7)
        return weeks.astype("int16")

    @staticmethod
    def how_many_ago(x):
        s = pd.Series(x)
        return (s.max() - s).astype("int32")


class h_cv:
    @staticmethod
    def feature_label_split(t, label_week, feature_periods):
        t = t.copy()
        max_week = int(t["week_number"].max())
        label_week = min(label_week, max_week)
        feat_start = max(1, label_week - feature_periods)
        features_df = t[
            (t["week_number"] >= feat_start) & (t["week_number"] <= label_week - 1)
        ]
        label_df = t[t["week_number"] == label_week]
        return features_df, label_df

    @staticmethod
    def ground_truth(label_df):
        gt = (
            label_df.groupby("customer_id")["article_id"]
            .apply(lambda s: " ".join([_aid10(x) for x in pd.unique(s)[:12]]))
            .reset_index(name="prediction")
        )
        gt["prediction"] = gt["prediction"].map(_clean_pred_str)
        return gt

    @staticmethod
    def comp_average_precision(true_labels: pd.Series, predictions: pd.Series, k=12):
        def apk(actual_list, pred_list, k=12):
            if len(pred_list) > k:
                pred_list = pred_list[:k]
            if len(actual_list) == 0:
                return 0.0
            score = 0.0
            hits = 0
            used = set()
            for i, p in enumerate(pred_list):
                if p in used:
                    continue
                used.add(p)
                if p in actual_list:
                    hits += 1
                    score += hits / (i + 1)
            return score / min(len(actual_list), k)

        idx = true_labels.index.intersection(predictions.index)
        if len(idx) == 0:
            return 0.0
        vals = []
        for cid in idx:
            a = _clean_pred_str(true_labels.loc[cid]).split()
            p = _clean_pred_str(predictions.loc[cid]).split()
            vals.append(apk(a, p, k=k))
        return float(np.mean(vals))

    @staticmethod
    def report_candidates(cand_df, ground_truth_candidates):
        merged = cand_df.merge(
            ground_truth_candidates, on=["customer_id", "article_id"], how="inner"
        )
        recall = len(merged) / max(1, len(ground_truth_candidates))
        print(f"[CV] Candidate recall (pair-level): {recall:.4f}")


class h_sub:
    @staticmethod
    def create_sub(customer_ids, predictions: pd.Series, index_to_id_dict_path=None):
        df = pd.DataFrame(
            {"customer_id": pd.Series(customer_ids, dtype="object").map(_cid)}
        )
        pred_map = predictions
        if isinstance(pred_map, pd.DataFrame) and "prediction" in pred_map.columns:
            pred_map = pred_map.set_index("customer_id")["prediction"]
        if not isinstance(pred_map, pd.Series):
            pred_map = pd.Series(pred_map)

        df["prediction"] = (
            df["customer_id"].map(pred_map).fillna("").map(_clean_pred_str)
        )
        return df


class h_pairs:
    @staticmethod
    def create_pairs(t, week_number, pairs_per_item=5, verbose=False):
        week = t[t["week_number"] == week_number]
        if len(week) == 0:
            return pd.DataFrame(
                {
                    "article_id": pd.Series(dtype="object"),
                    "pair": pd.Series(dtype="object"),
                }
            )

        baskets = week.groupby("customer_id")["article_id"].apply(
            lambda x: pd.unique(x)
        )
        from collections import Counter

        cnt = Counter()
        for arts in baskets:
            arts = list(map(_aid10, arts))
            arts = [x for x in arts if isinstance(x, str)]
            arts.sort()
            for i in range(len(arts)):
                ai = arts[i]
                for j in range(i + 1, len(arts)):
                    aj = arts[j]
                    cnt[(ai, aj)] += 1
                    cnt[(aj, ai)] += 1

        if not cnt:
            return pd.DataFrame(
                {
                    "article_id": pd.Series(dtype="object"),
                    "pair": pd.Series(dtype="object"),
                }
            )

        df = pd.DataFrame(
            [(k[0], k[1], v) for k, v in cnt.items()],
            columns=["article_id", "pair", "w"],
        )
        df = df.sort_values(["article_id", "w"], ascending=[True, False])
        df = (
            df.groupby("article_id")
            .head(pairs_per_item)[["article_id", "pair"]]
            .reset_index(drop=True)
        )
        return df


class h_can:
    @staticmethod
    def create_recent_customer_candidates(features_df, ca_num_weeks, customers=None):
        max_week = int(features_df["week_number"].max()) if len(features_df) else 0
        start_week = max(1, max_week - ca_num_weeks + 1)
        df = features_df[features_df["week_number"] >= start_week]
        if customers is not None:
            df = df[df["customer_id"].isin(customers)]
        df = df.sort_values(["customer_id", "t_dat"], ascending=[True, False])
        cand = (
            df.groupby("customer_id")["article_id"]
            .apply(lambda s: pd.unique(s)[:60])
            .explode()
            .reset_index()
        )
        cand["article_id"] = cand["article_id"].map(_aid10).astype("object")
        cand["customer_id"] = cand["customer_id"].map(_cid).astype("object")
        return cand.dropna(), {}

    @staticmethod
    def create_last_customer_weeks_and_pairs(
        features_df, article_pairs_df, clw_num_weeks, clw_num_pair_weeks, customers=None
    ):
        max_week = int(features_df["week_number"].max()) if len(features_df) else 0
        start_week = max(1, max_week - clw_num_weeks + 1)
        df = features_df[features_df["week_number"] >= start_week]
        if customers is not None:
            df = df[df["customer_id"].isin(customers)]

        df = df.sort_values(
            ["customer_id", "week_number", "t_dat"], ascending=[True, False, False]
        )
        last_items = (
            df.groupby("customer_id")["article_id"]
            .apply(lambda s: pd.unique(s)[:12])
            .explode()
            .reset_index()
        )
        last_items["article_id"] = last_items["article_id"].map(_aid10).astype("object")
        last_items["customer_id"] = last_items["customer_id"].map(_cid).astype("object")

        if (
            article_pairs_df is None
            or len(article_pairs_df) == 0
            or len(last_items) == 0
        ):
            pair_cand = last_items.iloc[0:0].copy()
        else:
            ap = article_pairs_df[["article_id", "pair"]].dropna().copy()
            ap["article_id"] = ap["article_id"].map(_aid10).astype("object")
            ap["pair"] = ap["pair"].map(_aid10).astype("object")
            ap = ap.drop_duplicates(subset=["article_id"], keep="first")

            pair_map = ap.set_index("article_id")["pair"]
            pair_cand = last_items.copy()
            mapped = pair_cand["article_id"].map(pair_map)
            pair_cand["article_id"] = mapped
            pair_cand = pair_cand.dropna()
            pair_cand["article_id"] = pair_cand["article_id"].astype("object")

        return last_items.dropna(), pair_cand.dropna(), {}, {}

    @staticmethod
    def create_popular_article_cand(
        features_df,
        c,
        a,
        pa_num_weeks,
        hier_col,
        num_candidates,
        num_articles,
        customers=None,
    ):
        max_week = int(features_df["week_number"].max()) if len(features_df) else 0
        start_week = max(1, max_week - pa_num_weeks + 1)
        df = features_df[features_df["week_number"] >= start_week]

        pop = (
            df["article_id"]
            .map(_aid10)
            .astype("object")
            .value_counts()
            .head(num_articles)
            .index.astype("object")
            .tolist()
        )

        if customers is None:
            customers = pd.unique(c["customer_id"])
        customers = pd.Series(customers).map(_cid).astype("object").to_numpy()

        out = pd.DataFrame(
            {
                "customer_id": np.repeat(customers, len(pop)),
                "article_id": pop * len(customers),
            }
        )
        out["article_id"] = out["article_id"].astype("object")
        out["customer_id"] = out["customer_id"].astype("object")
        return out, {"popular_articles": pop}

    @staticmethod
    def create_age_bucket_candidates(
        features_df, c, num_age_buckets, articles, customers=None
    ):
        empty = pd.DataFrame(
            {
                "customer_id": pd.Series(dtype="object"),
                "article_id": pd.Series(dtype="object"),
            }
        )
        return empty, {}, {}

    @staticmethod
    def filter_candidates(cand, t, **kwargs):
        return cand.drop_duplicates(["customer_id", "article_id"]).reset_index(
            drop=True
        )

    @staticmethod
    def add_features_to_candidates(cand, features_db, c, a):
        return cand.copy()


class h_modeling:
    @staticmethod
    def create_predictions(ids_df, preds):
        tmp = ids_df.copy()
        tmp["score"] = np.asarray(preds)

        tmp["customer_id"] = tmp["customer_id"].map(_cid).astype("object")
        tmp["article_id"] = tmp["article_id"].map(_aid10).astype("object")

        tmp = tmp.sort_values(["customer_id", "score"], ascending=[True, False])
        pred = tmp.groupby("customer_id")["article_id"].apply(
            lambda s: " ".join([_aid10(x) for x in s.head(12)])
        )
        pred.name = "prediction"
        return pred.map(_clean_pred_str)

    @staticmethod
    def run_all_cvs(t, c, a, cand_features_func, scoring_func, cv_weeks, **cv_params):
        results = {}
        for wk in cv_weeks:
            params = dict(cv_params)
            params["label_week"] = wk
            cand_df, label_df = cand_features_func(t, c, a, **params)

            features_df, _ = h_cv.feature_label_split(t, wk, params["feature_periods"])

            art_pop = (
                features_df["article_id"].map(_aid10).astype("object").value_counts()
            )

            score = (
                cand_df["article_id"]
                .map(_aid10)
                .astype("object")
                .map(art_pop)
                .fillna(0)
                .astype("float32")
                .to_numpy()
            )

            ids_df = cand_df[["customer_id", "article_id"]]
            pred_ser = h_modeling.create_predictions(ids_df, score)
            true_labels = h_cv.ground_truth(label_df).set_index("customer_id")[
                "prediction"
            ]
            cv_score = round(h_cv.comp_average_precision(true_labels, pred_ser), 5)
            results[wk] = cv_score
            print(f"CV week {wk}: MAP@12 = {cv_score}")
        return results

    @staticmethod
    def full_sub_train_run(t, c, a, cand_features_func, scoring_func, **sub_params):
        return None

    @staticmethod
    def full_sub_predict_run(t, c, a, cand_features_func, **sub_params):
        sample = pd.read_csv(
            SAMPLE_SUB_PATH, usecols=["customer_id"], dtype={"customer_id": "object"}
        )
        sample["customer_id"] = sample["customer_id"].map(_cid).astype("object")
        customers = sample["customer_id"].values

        cand_df, _ = cand_features_func(t, c, a, customer_batch=customers, **sub_params)

        features_df, _ = h_cv.feature_label_split(
            t, sub_params["label_week"], sub_params["feature_periods"]
        )

        art_pop = features_df["article_id"].map(_aid10).astype("object").value_counts()

        scores = (
            cand_df["article_id"]
            .map(_aid10)
            .astype("object")
            .map(art_pop)
            .fillna(0)
            .astype("float32")
            .to_numpy()
        )
        pred_ser = h_modeling.create_predictions(
            cand_df[["customer_id", "article_id"]], scores
        )

        pred_ser = (
            pred_ser.reindex(sample["customer_id"]).fillna("").map(_clean_pred_str)
        )
        pred_ser.index = sample["customer_id"]
        return pred_ser




## === cell 3
_LAST_DATE_ANCHOR = {"value": None}


def day_week_numbers_fixed(dates):
    d = pd.to_datetime(dates)
    if _LAST_DATE_ANCHOR["value"] is None:
        _LAST_DATE_ANCHOR["value"] = d.max().normalize()
    last_date = pd.Timestamp(_LAST_DATE_ANCHOR["value"]).normalize()

    weeks_ago = ((last_date - d.dt.normalize()).dt.days // 7).astype("int32")
    week_number = (weeks_ago.max() - weeks_ago + 1).astype("int16")
    week_number = week_number.clip(lower=1)
    return week_number


h_fe.day_week_numbers = day_week_numbers_fixed



## === cell 4
c, t, a = h_io.load_data(
    files=["customers.csv", "transactions_train.csv", "articles.csv"]
)

index_to_id_dict_path = h_fe.reduce_customer_id_memory(c, [t])

t["article_id"] = t["article_id"].map(_aid10).astype("object")
a["article_id"] = a["article_id"].map(_aid10).astype("object")

t["customer_id"] = t["customer_id"].map(_cid).astype("object")
c["customer_id"] = c["customer_id"].map(_cid).astype("object")

t["t_dat"] = pd.to_datetime(t["t_dat"])
t["week_number"] = h_fe.day_week_numbers(t["t_dat"])

_anchor = _LAST_DATE_ANCHOR["value"]
if _anchor is not None:
    last_week = int(
        t.loc[t["t_dat"].dt.normalize() == pd.Timestamp(_anchor), "week_number"].max()
    )
    print(
        "week_number at last transaction date:",
        last_week,
        "anchor_last_date:",
        str(_anchor)[:10],
    )

t["t_dat"] = h_fe.day_numbers(t["t_dat"])

print(c.shape, t.shape, a.shape)
print("week range:", int(t["week_number"].min()), int(t["week_number"].max()))
print("example customer_ids (transactions):", t["customer_id"].head(3).tolist())



## === cell 5
pairs_per_item = 5
week_number_pairs = {}

max_wk = int(t["week_number"].max())
start_wk = max(1, max_wk - 8)  # inclusive window of 9 weeks
for week_number in list(range(start_wk, max_wk + 1)):
    print(f"Creating pairs for week number {week_number}")
    week_number_pairs[week_number] = h_pairs.create_pairs(
        t, week_number, pairs_per_item, verbose=False
    )




## === cell 6
def create_candidates_with_features_df(t, c, a, customer_batch=None, **kwargs):
    if kwargs.get("cv", False):
        features_df, label_df = h_cv.feature_label_split(
            t, kwargs["label_week"], kwargs["feature_periods"]
        )
        desired_pairs_week = kwargs["label_week"] - 1
        if desired_pairs_week in week_number_pairs:
            pairs_week = desired_pairs_week
        else:
            pairs_week = (
                max(week_number_pairs.keys()) if len(week_number_pairs) else None
            )
    else:
        max_week = int(t["week_number"].max())
        feat_start = max(1, max_week - int(kwargs["feature_periods"]) + 1)
        features_df = t[
            (t["week_number"] >= feat_start) & (t["week_number"] <= max_week)
        ]
        label_df = t.iloc[0:0].copy()
        pairs_week = max(week_number_pairs.keys()) if len(week_number_pairs) else None

    features_df = features_df.copy()
    features_df["t_dat"] = h_fe.how_many_ago(features_df["t_dat"])
    features_df["week_number"] = h_fe.how_many_ago(features_df["week_number"])

    article_pairs_df = (
        week_number_pairs.get(pairs_week, None) if pairs_week is not None else None
    )

    if len(label_df) > 0:
        customers = pd.unique(label_df["customer_id"])
    elif customer_batch is not None:
        customers = customer_batch
    else:
        customers = None

    recent_customer_cand, _ = h_can.create_recent_customer_candidates(
        features_df, kwargs["ca_num_weeks"], customers=customers
    )

    (cust_last_week_cand, cust_last_week_pair_cand, _, _) = (
        h_can.create_last_customer_weeks_and_pairs(
            features_df,
            article_pairs_df,
            kwargs["clw_num_weeks"],
            kwargs["clw_num_pair_weeks"],
            customers=customers,
        )
    )

    pop_cand, _ = h_can.create_popular_article_cand(
        features_df,
        c,
        a,
        kwargs["pa_num_weeks"],
        kwargs["hier_col"],
        num_candidates=kwargs["num_recent_candidates"],
        num_articles=kwargs["num_recent_articles"],
        customers=customers if customers is not None else pd.unique(c["customer_id"]),
    )

    age_bucket_can, _, _ = h_can.create_age_bucket_candidates(
        features_df,
        c,
        kwargs["num_age_buckets"],
        articles=kwargs["num_recent_articles"],
        customers=customers,
    )

    cand = pd.concat(
        [
            recent_customer_cand,
            cust_last_week_cand,
            cust_last_week_pair_cand,
            age_bucket_can,
            pop_cand,
        ],
        ignore_index=True,
    )
    cand["customer_id"] = cand["customer_id"].map(_cid).astype("object")
    cand["article_id"] = cand["article_id"].map(_aid10).astype("object")

    cand = cand.drop_duplicates(["customer_id", "article_id"])
    cand = cand.sort_values(["customer_id", "article_id"]).reset_index(drop=True)

    cand = h_can.filter_candidates(cand, t, **kwargs)

    if customers is not None:
        cand = cand[
            cand["customer_id"].isin(
                pd.Series(customers).map(_cid).astype("object").to_numpy()
            )
        ]

    if kwargs.get("cv", False):
        ground_truth_candidates = label_df[
            ["customer_id", "article_id"]
        ].drop_duplicates()
        h_cv.report_candidates(cand, ground_truth_candidates)

    cand_with_f_df = h_can.add_features_to_candidates(cand, {}, c, a)

    for article_col in kwargs.get("article_columns", []):
        if article_col in a.columns:
            art_col_map = a.set_index("article_id")[article_col]
            cand_with_f_df[article_col] = cand_with_f_df["article_id"].map(art_col_map)

    if kwargs.get("selected_features", None) is not None:
        cand_with_f_df = cand_with_f_df[
            ["customer_id", "article_id"] + kwargs["selected_features"]
        ]

    assert len(cand) == len(
        cand_with_f_df
    ), "Duplicates introduced in candidate feature join"
    return cand_with_f_df, label_df




## === cell 7
def calculate_model_score(ids_df, preds, truth_df):
    predictions = h_modeling.create_predictions(ids_df, preds)
    true_labels = h_cv.ground_truth(truth_df).set_index("customer_id")["prediction"]
    score = round(h_cv.comp_average_precision(true_labels, predictions), 5)
    return score




## === cell 8
cv_params = {
    "cv": True,
    "feature_periods": 105,
    "label_week": 104,
    "index_to_id_dict_path": index_to_id_dict_path,
    "pairs_file_version": "_v3_5_ex",
    "num_recent_candidates": 36,
    "num_recent_articles": 12,
    "hier_col": "department_no",
    "ca_num_weeks": 3,
    "clw_num_weeks": 12,
    "clw_num_pair_weeks": 2,
    "pa_num_weeks": 1,
    "num_age_buckets": 4,
    "filter_recent_art_weeks": 1,
    "filter_num_articles": None,
    "lag_days": [1, 3, 14, 30],
    "article_columns": ["index_code"],
    "hier_cols": [
        "department_no",
        "section_no",
        "index_group_no",
        "index_code",
        "product_type_no",
        "product_group_name",
    ],
    "selected_features": None,
    "lgbm_params": {"n_estimators": 200, "num_leaves": 20},
    "log_evaluation": 10,
    "early_stopping": 20,
    "eval_at": 12,
    "save_model": True,
    "num_concats": 5,
}
sub_params = {
    "cv": False,
    "feature_periods": 105,
    "label_week": int(t["week_number"].max()),
    "index_to_id_dict_path": index_to_id_dict_path,
    "pairs_file_version": "_v3_5_ex",
    "num_recent_candidates": 60,
    "num_recent_articles": 12,
    "hier_col": "department_no",
    "ca_num_weeks": 3,
    "clw_num_weeks": 12,
    "clw_num_pair_weeks": 2,
    "pa_num_weeks": 1,
    "num_age_buckets": 4,
    "filter_recent_art_weeks": 1,
    "filter_num_articles": None,
    "lag_days": [1, 3, 14, 30],
    "article_columns": ["index_code"],
    "hier_cols": [
        "department_no",
        "section_no",
        "index_group_no",
        "index_code",
        "product_type_no",
        "product_group_name",
    ],
    "selected_features": None,
    "lgbm_params": {"n_estimators": 150, "num_leaves": 20},
    "log_evaluation": 10,
    "eval_at": 12,
    "prediction_models": ["model_104", "model_105"],
    "save_model": True,
    "num_concats": 5,
}

cand_features_func = create_candidates_with_features_df
scoring_func = calculate_model_score



## === cell 9
gc.collect()
cv_weeks = (
    [104] if int(t["week_number"].max()) >= 104 else [int(t["week_number"].max())]
)
results = h_modeling.run_all_cvs(
    t, c, a, cand_features_func, scoring_func, cv_weeks=cv_weeks, **cv_params
)
print("CV results:", results)



## === cell 10
gc.collect()
h_modeling.full_sub_train_run(t, c, a, cand_features_func, scoring_func, **sub_params)
predictions = h_modeling.full_sub_predict_run(t, c, a, cand_features_func, **sub_params)

predictions = _finalize_pred_series(predictions)
predictions.index = pd.Index(predictions.index).map(_cid).astype("object")

sample = pd.read_csv(SAMPLE_SUB_PATH, dtype={"customer_id": "object"})
sample["customer_id"] = sample["customer_id"].map(_cid).astype("object")
sub = sample[["customer_id"]].copy()

overlap = sub["customer_id"].isin(predictions.index).mean()
print("Prediction index overlap with sample customers:", round(float(overlap), 6))
if overlap < 0.95:
    raise RuntimeError(
        f"Low overlap between generated predictions and sample_submission customer_id keys ({overlap:.3f}). "
        "This would cause near-constant fallback predictions and a near-zero Kaggle score."
    )

max_week = int(t["week_number"].max())
pa_w = int(sub_params.get("pa_num_weeks", 1))
start_week = max(1, max_week - pa_w + 1)
t_recent = t[(t["week_number"] >= start_week) & (t["week_number"] <= max_week)]

global_top12 = (
    t_recent["article_id"].map(_aid10).astype("object").value_counts().head(12).index
).tolist()
global_top12 = [_aid10(x) for x in global_top12 if isinstance(_aid10(x), str)]
global_top12 = [
    x for x in global_top12 if isinstance(x, str) and len(x) == 10 and x.isdigit()
]
global_fallback_str = " ".join(global_top12[:12])

if len(global_top12) == 0:
    raise RuntimeError(
        "Global fallback popularity list is empty; cannot build non-empty submission."
    )

base_pred = predictions.reindex(sub["customer_id"]).fillna("").map(_clean_pred_str)
base_pred = base_pred.where(base_pred.str.len() > 0, other=global_fallback_str)

sub["prediction"] = base_pred.to_numpy()

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(sub.head())
print("submission shape:", sub.shape)
print("wrote:", out_path)

first_nonempty = sub.loc[sub["prediction"].str.len() > 0, "prediction"].head(1).tolist()
if first_nonempty:
    toks = first_nonempty[0].split()
    ok10 = all((len(x) == 10 and x.isdigit()) for x in toks)
    print(
        "Sanity check: first non-empty prediction token count:",
        len(toks),
        "all_10_digit:",
        ok10,
    )
else:
    print("Sanity check: all predictions empty (this would score 0.0).")

empty_rate = float((sub["prediction"].str.len() == 0).mean())
print("Empty prediction rate:", round(empty_rate, 6))
if empty_rate > 0.99:
    raise RuntimeError(
        "More than 99% empty predictions -> submission will score ~0. "
        "Customer_id alignment is still broken; inspect sample vs predictions indices."
    )
