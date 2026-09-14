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

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the `InvalidIndexError` by ensuring the `article_pairs_df` mapping index is unique before calling `Series.map`, since `create_pairs` produces multiple rows per `article_id`. I keep the existing candidate-generation and scoring logic intact, only changing the pair-candidate construction to use a de-duplicated mapping (top-weight pair per article) and avoiding index-alignment mistakes. I also adjust the pair-candidate assignment to preserve row alignment (don’t `.dropna()` mid-assignment) so it cannot crash. With these fixes, the CV run and full prediction run complete and write a valid `submission.csv`.'

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
                    path, dtype={"customer_id": "string", "article_id": "int32"}
                )
            elif f == "customers.csv":
                df = pd.read_csv(
                    path, dtype={"customer_id": "string", "postal_code": "string"}
                )
            elif f == "articles.csv":
                df = pd.read_csv(path)
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
            .apply(lambda s: " ".join([f"{int(x):010d}" for x in pd.unique(s)[:12]]))
            .reset_index(name="prediction")
        )
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
            a = str(true_labels.loc[cid]).split()
            p = str(predictions.loc[cid]).split()
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
        df = pd.DataFrame({"customer_id": pd.Series(customer_ids, dtype="string")})
        pred_map = predictions
        if isinstance(pred_map, pd.DataFrame) and "prediction" in pred_map.columns:
            pred_map = pred_map.set_index("customer_id")["prediction"]
        if not isinstance(pred_map, pd.Series):
            pred_map = pd.Series(pred_map)

        df["prediction"] = df["customer_id"].map(pred_map).fillna("")
        return df


class h_pairs:
    @staticmethod
    def create_pairs(t, week_number, pairs_per_item=5, verbose=False):
        week = t[t["week_number"] == week_number]
        if len(week) == 0:
            return pd.DataFrame(
                {
                    "article_id": pd.Series(dtype="int32"),
                    "pair": pd.Series(dtype="int32"),
                }
            )

        baskets = week.groupby("customer_id")["article_id"].apply(
            lambda x: pd.unique(x)
        )
        from collections import Counter

        cnt = Counter()
        for arts in baskets:
            arts = list(map(int, arts))
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
                    "article_id": pd.Series(dtype="int32"),
                    "pair": pd.Series(dtype="int32"),
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
        cand["article_id"] = cand["article_id"].astype("int32")
        return cand.dropna(), {}  # second return mimics features blob

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
        last_items["article_id"] = last_items["article_id"].astype("int32")

        if (
            article_pairs_df is None
            or len(article_pairs_df) == 0
            or len(last_items) == 0
        ):
            pair_cand = last_items.iloc[0:0].copy()
        else:
            ap = article_pairs_df[["article_id", "pair"]].dropna().copy()
            ap["article_id"] = ap["article_id"].astype("int32", copy=False)
            ap["pair"] = ap["pair"].astype("int32", copy=False)
            ap = ap.drop_duplicates(subset=["article_id"], keep="first")

            pair_map = ap.set_index("article_id")["pair"]
            pair_cand = last_items.copy()
            mapped = pair_cand["article_id"].map(pair_map)
            pair_cand["article_id"] = mapped
            pair_cand = pair_cand.dropna()
            pair_cand["article_id"] = pair_cand["article_id"].astype("int32")

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
            .value_counts()
            .head(num_articles)
            .index.astype("int32")
            .tolist()
        )

        if customers is None:
            customers = pd.unique(c["customer_id"])
        out = pd.DataFrame(
            {
                "customer_id": np.repeat(customers, len(pop)),
                "article_id": pop * len(customers),
            }
        )
        return out, {"popular_articles": pop}

    @staticmethod
    def create_age_bucket_candidates(
        features_df, c, num_age_buckets, articles, customers=None
    ):
        empty = pd.DataFrame(
            {
                "customer_id": pd.Series(dtype="string"),
                "article_id": pd.Series(dtype="int32"),
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
        tmp = tmp.sort_values(["customer_id", "score"], ascending=[True, False])
        pred = tmp.groupby("customer_id")["article_id"].apply(
            lambda s: " ".join([f"{int(x):010d}" for x in s.head(12)])
        )
        pred.name = "prediction"
        return pred

    @staticmethod
    def run_all_cvs(t, c, a, cand_features_func, scoring_func, cv_weeks, **cv_params):
        results = {}
        for wk in cv_weeks:
            params = dict(cv_params)
            params["label_week"] = wk
            cand_df, label_df = cand_features_func(t, c, a, **params)

            features_df, _ = h_cv.feature_label_split(t, wk, params["feature_periods"])
            art_pop = features_df["article_id"].value_counts()
            score = (
                cand_df["article_id"]
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
            SAMPLE_SUB_PATH, usecols=["customer_id"], dtype={"customer_id": "string"}
        )
        customers = sample["customer_id"].values

        cand_df, _ = cand_features_func(t, c, a, customer_batch=customers, **sub_params)

        features_df, _ = h_cv.feature_label_split(
            t, sub_params["label_week"], sub_params["feature_periods"]
        )
        art_pop = features_df["article_id"].value_counts()

        scores = (
            cand_df["article_id"].map(art_pop).fillna(0).astype("float32").to_numpy()
        )
        pred_ser = h_modeling.create_predictions(
            cand_df[["customer_id", "article_id"]], scores
        )

        pred_ser = pred_ser.reindex(sample["customer_id"]).fillna("")
        return pred_ser




## === cell 3
def day_week_numbers_fixed(dates):
    pd_dates = pd.to_datetime(dates)
    unique_dates = pd.Series(pd_dates.unique())
    numbered_days = unique_dates - unique_dates.min() + timedelta(1)
    numbered_days = numbered_days.dt.days
    extra_days = int(numbered_days.max() % 7)
    adjusted_days = (numbered_days - extra_days).astype("int32")
    day_weeks = -((-adjusted_days) // 7)
    day_weeks_map = pd.Series(day_weeks.values, index=unique_dates.values)
    return pd_dates.map(day_weeks_map).astype("int16")


h_fe.day_week_numbers = day_week_numbers_fixed



## === cell 4
c, t, a = h_io.load_data(
    files=["customers.csv", "transactions_train.csv", "articles.csv"]
)

index_to_id_dict_path = h_fe.reduce_customer_id_memory(c, [t])
t["t_dat"] = pd.to_datetime(t["t_dat"])
t["week_number"] = h_fe.day_week_numbers(t["t_dat"])
t["t_dat"] = h_fe.day_numbers(t["t_dat"])

t["customer_id"] = t["customer_id"].astype("string")
c["customer_id"] = c["customer_id"].astype("string")
t["article_id"] = t["article_id"].astype("int32")

print(c.shape, t.shape, a.shape)
print("week range:", int(t["week_number"].min()), int(t["week_number"].max()))



## === cell 5
pairs_per_item = 5
week_number_pairs = {}
for week_number in [96, 97, 98, 99, 100, 101, 102, 103, 104]:
    print(f"Creating pairs for week number {week_number}")
    week_number_pairs[week_number] = h_pairs.create_pairs(
        t, week_number, pairs_per_item, verbose=False
    )




## === cell 6
def create_candidates_with_features_df(t, c, a, customer_batch=None, **kwargs):
    features_df, label_df = h_cv.feature_label_split(
        t, kwargs["label_week"], kwargs["feature_periods"]
    )

    features_df = features_df.copy()
    features_df["t_dat"] = h_fe.how_many_ago(features_df["t_dat"])
    features_df["week_number"] = h_fe.how_many_ago(features_df["week_number"])

    article_pairs_df = week_number_pairs.get(kwargs["label_week"] - 1, None)

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
    cand = cand.drop_duplicates(["customer_id", "article_id"])
    cand = cand.sort_values(["customer_id", "article_id"]).reset_index(drop=True)

    cand = h_can.filter_candidates(cand, t, **kwargs)

    if customers is not None:
        cand = cand[cand["customer_id"].isin(customers)]

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

sample = pd.read_csv(SAMPLE_SUB_PATH, dtype={"customer_id": "string"})
sub = sample[["customer_id"]].copy()
sub["prediction"] = sub["customer_id"].map(predictions).fillna("")

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(sub.head())
print("submission shape:", sub.shape)
print("wrote:", out_path)
