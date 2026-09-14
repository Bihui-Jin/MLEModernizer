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

3.10

# 3. Installed packages

cudf-cu12==25.2.2
cudf-polars-cu12==25.6.0
dask-cudf-cu12==25.2.2
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
libcudf-cu12==25.2.2
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
pylibcudf-cu12==25.2.2
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

0.02246

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the smallest fixes needed to ensure your pipeline reliably produces a valid `submission.csv` and also nudges MAP@12 upward without changing the overall approach (cluster customers, compute per-cluster recency-weighted scores, fallback to general/top). The key scoring improvement is to compute the “general_pred” (cluster fallback list) from the same *recent window* you already used earlier (Sep 1–21), rather than from the whole history, because next-week purchases are much more correlated with recent trends. I also cap per-customer predictions to exactly 12 and add a deterministic, safe fallback for clusters missing from `df100` (so no cluster produces weak/empty outputs due to missing keys). All changes are localized inside the submission-building cells and keep your model logic intact.'
- What this solution (achieved 0.0) has done: 'I make the smallest changes needed to ensure your submission is always valid and to avoid a common MAP@12 “silent killer”: many customers never appear in any per-cluster file because you only write predictions for customers present in `dfCustomers` (train customers), not for all customers in `sample_submission` (which includes unseen customers). Concretely, I write one extra per-cluster file for “unknown cluster” customers (those not found in `dfCustomers`), using your existing global-recent fallback list so every `customer_id` gets 12 items. I also adjust the final merge to explicitly include that unknown-cluster file and keep the rest of your cluster-based scoring logic unchanged. This should move the score up from “not yielded/likely invalid or 0.0” toward a sensible baseline while preserving your core approach.'

# 9. Code solution

## === cell 0
import sys, warnings, time, os, copy, gc, re, random, pickle

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

from collections import Counter
from datetime import datetime, timedelta

import cudf

from sklearn.cluster import KMeans
from sklearn import preprocessing

from pandas import json_normalize




## === cell 1
class Clustering_HandM:
    def customers_preprocessing(self, customers, dropcol=["postal_code"], **kwargs):
        customers = customers.drop(dropcol, axis=1)
        customers_col = list(customers.columns)

        if "fashion_news_frequency" in customers_col:
            customers["fashion_news_frequency"] = customers[
                "fashion_news_frequency"
            ].replace("NONE", "None")
            customers["fashion_news_frequency"] = customers[
                "fashion_news_frequency"
            ].replace({np.nan: 0, "None": 0, "Monthly": 1, "Regularly": 2})

        if "club_member_status" in customers_col:
            customers["club_member_status"] = customers["club_member_status"].replace(
                {np.nan: 0, "PRE-CREATE": 1, "ACTIVE": 2, "LEFT CLUB": -1}
            )

        if "age" in customers_col:
            customers["age"] = customers["age"].fillna(-1)

        if "FN" in customers_col:
            customers["FN"] = customers["FN"].fillna(0)

        if "Active" in customers_col:
            customers["Active"] = customers["Active"].fillna(0)

            print(f"###NULL DESCRIPTION###\n{customers.isnull().sum()}")

        return customers

    def clustering(
        self, df, predcol, usecol, normmethod="StandardScaler", clusters=12, DEBUG=False
    ):
        X = np.array(df[usecol])

        if normmethod == "StandardScaler":
            nm = preprocessing.StandardScaler()
            X = nm.fit_transform(X)
        elif normmethod == "minMax":
            nm = preprocessing.MinMaxScaler()
            X = nm.fit_transform(X)
        print(f"NormarlizationMethod:{normmethod}")

        km = KMeans(n_clusters=clusters, random_state=2022)
        km.fit(X)
        print("Distortion: %.2f" % km.inertia_)
        pred = km.labels_
        df_pred = pd.DataFrame(pred, columns=["pred"])
        df_pred = pd.concat([df, df_pred], axis=1)

        df_norm = pd.DataFrame(X, columns=usecol)
        print(df_norm.describe())

        if DEBUG:
            df_norm = pd.concat([df[predcol], df_norm], axis=1)
            return df_pred, df_norm
        else:
            return df_pred




## === cell 2
DEBUG = False

PATH_INPUT = r"../input/h-and-m-personalized-fashion-recommendations/"

if not os.path.exists(PATH_INPUT):
    PATH_INPUT = r"/kaggle/input/h-and-m-personalized-fashion-recommendations/"

print("Using PATH_INPUT:", PATH_INPUT)



## === cell 3
customers = pd.read_csv(PATH_INPUT + "customers.csv")

clst = Clustering_HandM()
customers = clst.customers_preprocessing(customers)

usecol = ["club_member_status", "fashion_news_frequency", "age", "FN", "Active"]
predcol = ["customer_id"]

dfCustomers = clst.clustering(customers, predcol=predcol, usecol=usecol)
dfCustomers.head()



## === cell 4
listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
dfCustomers["age_bins"] = pd.cut(dfCustomers["age"], listBin)
dfCustomers["age_bins"].value_counts(dropna=False).head()



## === cell 5
_ = pd.crosstab(dfCustomers["pred"], dfCustomers["age_bins"])
dfCustomers = dfCustomers.drop(["age_bins"], axis=1)



## === cell 6
dfTransactions = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"article_id": "int32", "t_dat": "string", "customer_id": "string"},
)
dfTransactions["t_dat"] = cudf.to_datetime(dfTransactions["t_dat"])
dfTransactions = dfTransactions.set_index("t_dat")

dfTransactions = dfTransactions.sort_index()
dfTransactions.head()



## === cell 7
start_dt = np.datetime64("2020-09-01")
end_dt = np.datetime64("2020-09-21")
idx = dfTransactions.index
dfRecent = dfTransactions[(idx >= start_dt) & (idx <= end_dt)]
dfRecent.head()



## === cell 8
dfRecent_pd = dfRecent.to_pandas()
dfRecent_pd = dfRecent_pd.merge(
    dfCustomers[["customer_id", "pred"]], on="customer_id", how="inner"
)
dfRecent_pd.head()



## === cell 9
dfRecentAgg = (
    dfRecent_pd.groupby(["pred", "article_id"])["customer_id"]
    .count()
    .reset_index()
    .rename(columns={"customer_id": "counts"})
)
listUniBins_recent = dfRecentAgg["pred"].unique().tolist()

dict100 = {}
for uniBin in listUniBins_recent:
    dfTemp = dfRecentAgg[dfRecentAgg["pred"] == uniBin].sort_values(
        by="counts", ascending=False
    )
    dict100[uniBin] = dfTemp.head(100)["article_id"].values.tolist()

df100 = pd.DataFrame([dict100]).T.rename(columns={0: "top100"})
df100.head()



## === cell 10
if "df100" in globals() and df100.shape[0] > 1:
    dfOverlap = df100.copy()
    for index in dfOverlap.index:
        dfOverlap[index] = [
            len(set(dfOverlap.at[index, "top100"]) & set(dfOverlap.at[x, "top100"]))
            / 100
            for x in dfOverlap.index
        ]
    dfOverlap = dfOverlap.drop(columns="top100")
    plt.figure(figsize=(10, 6))
    sns.heatmap(dfOverlap, annot=False, cbar=False)
    plt.show()



## === cell 11
N = 12
listUniBins = sorted(dfCustomers["pred"].dropna().unique().tolist())
print("Num clusters in dfCustomers:", len(listUniBins))
print("Clusters:", listUniBins[:20], "..." if len(listUniBins) > 20 else "")




## === cell 12
def _to_pandas_timestamp(x):
    if isinstance(x, pd.Timestamp):
        return x
    try:
        return pd.Timestamp(x)
    except Exception:
        return pd.to_datetime(x)


def _fmt_article10(a) -> str:
    try:
        return f"{int(a):010d}"
    except Exception:
        s = str(a)
        return s.zfill(10)


def _split_pred_tokens(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if s == "":
        return []
    return [x for x in s.split() if x != ""]


def _merge_preds_to12(primary_pred: str, fallback_pred: str, n: int = 12) -> str:
    out = []
    seen = set()
    for token in _split_pred_tokens(primary_pred) + _split_pred_tokens(fallback_pred):
        t = str(token).strip()
        if t == "":
            continue
        if not t.isdigit():
            continue
        t = t.zfill(10)
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= n:
            break
    return " ".join(out)


def _ensure_exactly_12(pred: str, pad_with: str, n: int = 12) -> str:
    merged = _merge_preds_to12(pred, pad_with, n=n)
    toks = _split_pred_tokens(merged)
    if len(toks) < n:
        merged = _merge_preds_to12(merged, pad_with, n=n)
    toks = _split_pred_tokens(merged)[:n]
    return " ".join([t.zfill(10) for t in toks])


def _cudf_value_counts_df(s: cudf.Series, key_name: str):
    vc = s.value_counts()
    if isinstance(vc, cudf.Series):
        df = vc.reset_index()
        cols = list(df.columns)
        if len(cols) != 2:
            df = df.iloc[:, :2]
            cols = list(df.columns)
        df = df.rename(columns={cols[0]: key_name, cols[1]: "cnt"})
        return df
    else:
        df = vc.reset_index()
        cols = list(df.columns)
        if key_name not in cols:
            df = df.rename(columns={cols[0]: key_name})
        if "cnt" not in cols:
            df = df.rename(columns={cols[1]: "cnt"})
        return df[[key_name, "cnt"]]


df_all = dfTransactions.reset_index()
if "t_dat" not in df_all.columns:
    if "index" in df_all.columns:
        df_all = df_all.rename(columns={"index": "t_dat"})
    else:
        raise RuntimeError(
            f"df_all reset_index() did not produce t_dat/index. Columns: {list(df_all.columns)}"
        )

df_recent_all = dfRecent.reset_index()
if "t_dat" not in df_recent_all.columns:
    if "index" in df_recent_all.columns:
        df_recent_all = df_recent_all.rename(columns={"index": "t_dat"})
    else:
        raise RuntimeError(
            f"df_recent_all reset_index() did not produce t_dat/index. Columns: {list(df_recent_all.columns)}"
        )

sub_all = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
numCustomers = int(sub_all.shape[0])

dfCustomers_cu = cudf.from_pandas(dfCustomers[["customer_id", "pred", "age"]])
dfCustomers_cu["customer_id"] = dfCustomers_cu["customer_id"].astype("string")

top_global_all = (
    _cudf_value_counts_df(df_all["article_id"], "article_id")
    .sort_values("cnt", ascending=False)
    .head(12)["article_id"]
    .to_pandas()
    .tolist()
)
top_global_all_str = " ".join([_fmt_article10(a) for a in top_global_all])

top_global_recent = (
    _cudf_value_counts_df(df_recent_all["article_id"], "article_id")
    .sort_values("cnt", ascending=False)
    .head(12)["article_id"]
    .to_pandas()
    .tolist()
)
top_global_recent_str = " ".join([_fmt_article10(a) for a in top_global_recent])

top_global_str = top_global_recent_str

for uniBin in listUniBins:
    fname = f"submission_{uniBin}.csv"
    if os.path.exists(fname):
        try:
            os.remove(fname)
        except Exception:
            pass

UNK_BIN = -999
unk_fname = f"submission_{UNK_BIN}.csv"
if os.path.exists(unk_fname):
    try:
        os.remove(unk_fname)
    except Exception:
        pass

for uniBin in listUniBins:
    dfCustomersTemp = dfCustomers_cu[dfCustomers_cu["pred"] == uniBin][
        ["customer_id", "age"]
    ]

    if "df100" in globals() and uniBin in list(df100.index):
        cluster_top100 = df100.loc[uniBin, "top100"]
        cluster_top12 = " ".join([_fmt_article10(a) for a in cluster_top100[:12]])
    else:
        cluster_top12 = top_global_recent_str
    cluster_top12 = _ensure_exactly_12(cluster_top12, top_global_recent_str, n=12)

    if int(dfCustomersTemp.shape[0]) == 0:
        empty_sub = pd.DataFrame(
            {"customer_id": pd.Series(dtype=str), "prediction": pd.Series(dtype=str)}
        )
        empty_sub.to_csv(f"submission_{uniBin}.csv", index=False)
        print(
            f"Cluster {uniBin}: no customers. Wrote empty per-cluster file to avoid overwrites."
        )
        print("-" * 50)
        continue

    cust_ids = dfCustomersTemp[["customer_id"]].drop_duplicates()
    df = df_all.merge(cust_ids, on="customer_id", how="inner")
    print(f"The shape of scope transaction for cluster {uniBin} is {df.shape}.")

    if int(df.shape[0]) == 0:
        sub = sub_all.merge(dfCustomersTemp, on="customer_id", how="inner").to_pandas()
        sub["prediction"] = _ensure_exactly_12("", cluster_top12, n=12)
        sub = sub[["customer_id", "prediction"]]
        sub.to_csv(f"submission_{uniBin}.csv", index=False)
        print(
            f"Cluster {uniBin}: no transactions. Wrote fallback submission with shape {sub.shape}."
        )
        print("-" * 50)
        continue

    df["customer_id"] = df["customer_id"].str[-16:].str.hex_to_int().astype("int64")
    last_ts = df["t_dat"].max()
    last_ts_pd = _to_pandas_timestamp(last_ts)

    tmp = df[["t_dat"]].copy().to_pandas()
    tmp["dow"] = tmp["t_dat"].dt.dayofweek
    tmp["ldbw"] = tmp["t_dat"] - pd.TimedeltaIndex(tmp["dow"] - 1, unit="D")
    tmp.loc[tmp["dow"] >= 2, "ldbw"] = tmp.loc[
        tmp["dow"] >= 2, "ldbw"
    ] + pd.TimedeltaIndex(np.ones(len(tmp.loc[tmp["dow"] >= 2])) * 7, unit="D")
    df["ldbw"] = tmp["ldbw"].values

    weekly_sales = (
        df.drop("customer_id", axis=1)
        .groupby(["ldbw", "article_id"])
        .count()
        .reset_index()
    )
    weekly_sales = weekly_sales.rename(columns={"t_dat": "count"})

    df = df.merge(weekly_sales, on=["ldbw", "article_id"], how="left")

    last_ldbw = df["ldbw"].max()

    weekly_sales = weekly_sales.reset_index().set_index("article_id")
    df = df.merge(
        weekly_sales.loc[weekly_sales["ldbw"] == last_ldbw, ["count"]],
        on="article_id",
        suffixes=("", "_targ"),
    )
    df["count_targ"].fillna(0, inplace=True)
    del weekly_sales

    df["quotient"] = df["count_targ"] / df["count"]

    df_recent_cluster = df_recent_all.merge(
        dfCustomersTemp[["customer_id"]], on="customer_id", how="inner"
    )
    if int(df_recent_cluster.shape[0]) > 0:
        recent_pop = (
            _cudf_value_counts_df(df_recent_cluster["article_id"], "article_id")
            .sort_values("cnt", ascending=False)
            .head(100)["article_id"]
            .to_pandas()
            .tolist()
        )
        cluster_recent_top12 = " ".join([_fmt_article10(a) for a in recent_pop[:12]])
    else:
        cluster_recent_top12 = cluster_top12

    target_sales = (
        df.drop("customer_id", axis=1).groupby("article_id")["quotient"].sum()
    )

    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()

    general_pred_str = _merge_preds_to12(
        " ".join([_fmt_article10(article_id) for article_id in general_pred]),
        _merge_preds_to12(cluster_recent_top12, cluster_top12, n=12),
        n=12,
    )
    general_pred_str = _ensure_exactly_12(general_pred_str, top_global_recent_str, n=12)

    del target_sales

    tmp = df.copy().to_pandas()
    tmp["x"] = ((last_ts_pd - tmp["t_dat"]) / np.timedelta64(1, "D")).astype(int)
    tmp["dummy_1"] = 1
    tmp["x"] = tmp[["x", "dummy_1"]].max(axis=1)

    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    tmp["y"] = a / np.sqrt(tmp["x"]) + b * np.exp(-c * tmp["x"]) - d
    tmp["dummy_0"] = 0
    tmp["y"] = tmp[["y", "dummy_0"]].max(axis=1)
    tmp["value"] = tmp["quotient"] * tmp["y"]

    tmp = tmp.groupby(["customer_id", "article_id"]).agg({"value": "sum"}).reset_index()
    tmp = tmp.loc[tmp["value"] > 0]
    tmp["rank"] = tmp.groupby("customer_id")["value"].rank("dense", ascending=False)
    tmp = tmp.loc[tmp["rank"] <= 12]

    purchase_df = tmp.sort_values(
        ["customer_id", "value"], ascending=False
    ).reset_index(drop=True)

    purchase_df["article_id"] = purchase_df["article_id"].map(_fmt_article10)
    purchase_df = (
        purchase_df.groupby("customer_id")["article_id"].apply(list).reset_index()
    )
    purchase_df["prediction"] = purchase_df["article_id"].map(
        lambda xs: " ".join(xs[:12])
    )
    purchase_df = purchase_df.drop(columns=["article_id"])
    purchase_df = cudf.DataFrame(purchase_df)

    sub = sub_all.merge(dfCustomersTemp, on="customer_id", how="inner")
    sub["customer_id2"] = sub["customer_id"].str[-16:].str.hex_to_int().astype("int64")

    sub = sub.merge(
        purchase_df,
        left_on="customer_id2",
        right_on="customer_id",
        how="left",
        suffixes=("", "_ignored"),
    )

    sub = sub.to_pandas()

    sub["prediction"] = sub["prediction"].fillna("").map(lambda s: str(s).strip())
    sub["prediction"] = sub["prediction"].map(
        lambda s: s if s != "" else general_pred_str
    )
    sub["prediction"] = sub["prediction"].map(
        lambda s: _merge_preds_to12(
            s,
            _merge_preds_to12(general_pred_str, cluster_recent_top12, n=12),
            n=12,
        )
    )
    sub["prediction"] = sub["prediction"].map(
        lambda s: _ensure_exactly_12(s, top_global_recent_str, n=12)
    )

    sub = sub[["customer_id", "prediction"]]
    sub.to_csv(f"submission_{uniBin}.csv", index=False)
    print(f"Saved prediction for cluster {uniBin}. The shape is {sub.shape}.")
    print("-" * 50)

    del df, tmp, purchase_df, sub
    gc.collect()

dfCustomers_ids = cudf.from_pandas(dfCustomers[["customer_id"]])
dfCustomers_ids["customer_id"] = dfCustomers_ids["customer_id"].astype("string")

dfCustomers_marker = dfCustomers_ids.copy()
dfCustomers_marker["__in_train__"] = 1
unknown_merge = sub_all.merge(dfCustomers_marker, on="customer_id", how="left")
unknown_sub = unknown_merge[unknown_merge["__in_train__"].isnull()][
    ["customer_id"]
].to_pandas()

unknown_sub["prediction"] = _ensure_exactly_12("", top_global_recent_str, n=12)
unknown_sub.to_csv(unk_fname, index=False)
print(f"Wrote unknown-cluster file {unk_fname} with shape {unknown_sub.shape}.")

print("Finished per-cluster files.\n" + "=" * 50)




## === cell 13
def _compute_top_global_recent_str_fallback(PATH_INPUT_local: str) -> str:
    df_tmp = cudf.read_csv(
        PATH_INPUT_local + "transactions_train.csv",
        usecols=["t_dat", "article_id"],
        dtype={"article_id": "int32", "t_dat": "string"},
    )
    df_tmp["t_dat"] = cudf.to_datetime(df_tmp["t_dat"])
    df_tmp = df_tmp.set_index("t_dat").sort_index()
    idx = df_tmp.index
    start_dt = np.datetime64("2020-09-01")
    end_dt = np.datetime64("2020-09-21")
    df_tmp_recent = df_tmp[(idx >= start_dt) & (idx <= end_dt)].reset_index()
    if "t_dat" not in df_tmp_recent.columns and "index" in df_tmp_recent.columns:
        df_tmp_recent = df_tmp_recent.rename(columns={"index": "t_dat"})

    top = (
        _cudf_value_counts_df(df_tmp_recent["article_id"], "article_id")
        .sort_values("cnt", ascending=False)
        .head(12)["article_id"]
        .to_pandas()
        .tolist()
    )
    return " ".join([_fmt_article10(a) for a in top])


if "top_global_str" not in globals():
    top_global_str = _compute_top_global_recent_str_fallback(PATH_INPUT)
    print("top_global_str was missing; recomputed from recent window.")

top_global_str = _ensure_exactly_12(top_global_str, top_global_str, n=12)

dfSub_list = []
missing_bins = []
for i, uniBin in enumerate(listUniBins):
    fname = f"submission_{uniBin}.csv"
    if not os.path.exists(fname):
        missing_bins.append(uniBin)
        continue
    dfTemp = pd.read_csv(fname)
    dfSub_list.append(dfTemp)

UNK_BIN = -999
unk_fname = f"submission_{UNK_BIN}.csv"
if os.path.exists(unk_fname):
    dfSub_list.append(pd.read_csv(unk_fname))
else:
    print(
        "WARNING: unknown-cluster file missing; will rely on global fallback in final merge."
    )

sample_sub = pd.read_csv(PATH_INPUT + "sample_submission.csv", usecols=["customer_id"])

if len(dfSub_list) == 0:
    dfSub = sample_sub.copy()
    dfSub["prediction"] = _ensure_exactly_12("", top_global_str, n=12)
else:
    dfSub = pd.concat(dfSub_list, axis=0, ignore_index=True)
    dfSub = dfSub.drop_duplicates(subset=["customer_id"], keep="first")
    dfSub = sample_sub.merge(dfSub, on="customer_id", how="left")
    dfSub["prediction"] = dfSub["prediction"].fillna("")

dfSub["prediction"] = dfSub["prediction"].map(
    lambda s: _ensure_exactly_12(s, top_global_str, n=12)
)

assert (
    dfSub.shape[0] == sample_sub.shape[0]
), f"Row count mismatch: {dfSub.shape[0]} vs {sample_sub.shape[0]}"
assert dfSub["customer_id"].isna().sum() == 0
assert dfSub["prediction"].isna().sum() == 0

dfSub.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", dfSub.shape)
if len(missing_bins) > 0:
    print(
        "WARNING: Missing per-cluster files for bins (filled by global fallback at final merge):",
        missing_bins,
    )



## === cell 14
dfCheck = pd.read_csv("./submission.csv")
print(dfCheck.head())
print(dfCheck.columns.tolist())
print("Rows:", dfCheck.shape[0], "Unique customers:", dfCheck["customer_id"].nunique())

pred_series = dfCheck["prediction"].astype(str).str.strip()
token_counts = pred_series.map(lambda s: len(s.split()))
bad_len = int((token_counts != 12).sum())


def _is_good_row(s: str) -> bool:
    toks = s.split()
    if len(toks) != 12:
        return False
    for tok in toks:
        if (not tok.isdigit()) or (len(tok) != 10):
            return False
    return True


sample_n = min(50000, len(dfCheck))
bad_fmt = 0
for s in pred_series.head(sample_n).tolist():
    if not _is_good_row(s):
        bad_fmt += 1

empty_preds = int((pred_series == "").sum())
print("Sanity check: rows with token_count != 12:", bad_len)
print(f"Sanity check: bad-format rows in first {sample_n}:", bad_fmt)
print("Empty prediction rows:", empty_preds)

assert empty_preds == 0, "Found empty predictions; submission would likely score 0.0."
assert (
    bad_len == 0
), "Found rows not having exactly 12 predictions; fix before submitting."
