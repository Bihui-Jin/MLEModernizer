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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Diagnosis: Cell 12 crashes because `last_ts = df_cluster["t_dat"].max()` can return a `numpy.datetime64` (after mixed cuDF/Pandas conversions), which does not have `.to_pandas()`. The code then calls `pd.Timestamp(last_ts.to_pandas())`, triggering the AttributeError. We normalize `last_ts` into a pandas `Timestamp` safely regardless of whether it is a cuDF scalar, numpy datetime64, python datetime, or already a pandas Timestamp, without changing any downstream logic.

Patch summary: In cell 12, replace the direct `pd.Timestamp(last_ts.to_pandas())` usage with a small conversion block that produces `last_ts_pd: pd.Timestamp` via type checks and `pd.to_datetime`, then use `last_ts_pd` in the `tmp["x"]` computation. This is minimal and preserves all computations and outputs.

Updated cells:'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a submission that is either invalid or effectively empty/constant for many customers; the quickest legitimate improvement is to ensure every `customer_id` gets exactly 12 **unique** article IDs, and that the “global fallback” is computed robustly (your current fallback can silently break because `df_last7.groupby().size().reset_index(...).sort_values(...)` is not a cuDF pattern). I keep your clustering/weighting logic intact, but fix the global fallback to a reliable cuDF computation and make the final post-processing deduplicate and pad using the global top-12 list (rather than repeating the customer’s own list). These are minimal, metric-aligned changes that should move MAP@12 upward from 0.0 without changing the model logic. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an invalid/near-empty submission (often caused by missing/NaN predictions, wrong formatting, or miscomputed fallbacks). I keep your clustering + per-bin quotient/recency weighting logic intact, but make two minimal “metric-aligned” fixes: (1) compute the global fallback top-12 robustly in cuDF (your current `groupby().size().reset_index()` pattern is pandas-like and can misbehave in cuDF), and (2) ensure every customer gets exactly 12 **unique** article_ids by deduplicating and padding using that global top-12 list deterministically. These changes should move MAP@12 upward from 0.0 toward your 0.02246 target without changing the modeling approach. The script still run end-to-end and always write a valid `submission.csv` with the correct columns/row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a formatting/type issue that makes most predicted `article_id`s invalid for scoring (the leading `"0"` prefix can create IDs that don’t match true labels) and/or from cluster IDs not matching because `pred` became float (e.g., `0.0`) during merges. To move the score up toward your target with minimal logic changes, I (1) keep `article_id` as an 10-digit zero-padded string instead of `"0"+str(id)` (so IDs match Kaggle’s ground truth format), and (2) force the cluster label `pred` to integer consistently before using it to filter customers, preventing silent empty clusters. Everything else (your clustering, weekly quotient, recency weighting, ranking, and fallback behavior) is preserved, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with predictions that don’t match the ground-truth `article_id` format (H&M expects **10-digit zero-padded strings**), which can happen if `article_id` is treated as an integer and later serialized without padding or with accidental formatting drift. I make the `article_id` representation consistent end-to-end by explicitly reading `article_id` as string (preserving leading zeros) and only converting to int where required for grouping, while keeping your clustering + quotient/recency weighting logic unchanged. I also make the cluster-customer merge type-stable by forcing `customer_id` dtype consistency across pandas/cudf merges to prevent silent empty joins that can lead to constant fallbacks. These are minimal, metric-aligned fixes aimed at moving the score upward toward 0.02246 without changing the modeling approach.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most likely due to a type mismatch causing the clustering merge to silently drop most customers (so many customers end up with empty/NaN predictions that later collapse to a weak constant fallback). I keep your clustering + quotient/recency weighting logic intact, but make `customer_id` dtype consistent before the merge in cell 9 so clusters populate correctly. I also make the global last-7-days popularity fallback robust to cuDF API differences (using `value_counts`) to ensure padding always works and every row has 12 valid 10-digit `article_id`s. These are minimal, metric-aligned fixes expected to move MAP@12 up toward your 0.02246 target without changing the modeling approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with a “valid-looking” submission where the predicted article IDs rarely match the ground truth due to ID formatting drift. I keep your clustering + quotient/recency ranking exactly as-is, but enforce a single invariant end-to-end: every `article_id` written to `prediction` is a 10-digit zero-padded string derived from the raw `article_id` string, and the integer version is used only for grouping/scoring computations. Concretely, I (1) build a robust mapping from `article_id_int -> article_id_str10` from the transactions file itself, (2) use that mapping instead of `aid_str(int)` everywhere predictions are materialized, and (3) keep your “ensure 12 unique + global fallback” logic unchanged except that it now pads with correctly formatted IDs. This is a minimal change that should move the score upward toward your target without altering the model logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with predictions not being aligned to the correct `customer_id2` join key (because `customer_id2` is built from the **last 16 chars** which are not guaranteed to be valid hex for H&M customer IDs), causing most per-customer predictions to be missing and collapsing to a weak fallback. To move the score upward toward the 0.02246 target with minimal logic change, I keep your clustering + quotient/recency ranking intact but rebuild `customer_id2` deterministically by factorizing the `customer_id` strings using a single shared mapping for both transactions and submission customers. I also fix one pandas aggregation bug where `groupby(...).agg({"prediction": sum})` can behave inconsistently on strings by switching to `''.join`, preserving semantics but making the concatenation reliable. Everything else (bins, scoring, weighting, dedupe/pad-to-12, global fallback) remains the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with predictions not being scored because many `article_id`s in `prediction` don’t exactly match the ground-truth format (H&M expects **10-digit strings with leading zeros preserved**). I keep your clustering + quotient/recency ranking exactly the same, but make the `article_id_int -> article_id_str10` mapping *complete* (built from `articles.csv` plus transactions) so every predicted int is rendered into a valid 10-digit ID. I also ensure the “global fallback” and all padding use that same canonical formatting, and add a small safety conversion for `last_ts_all` so the last-7-days cutoff doesn’t silently break. These are minimal, metric-aligned fixes intended to move score upward toward 0.02246 without changing your modeling logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is likely coming from a key mismatch: the per-customer ranking is computed on `customer_id2` for **all** customers (submission + transaction customers), but in each cluster you filter `sub_cluster_pd` only by `customer_id` and then left-join predictions on `customer_id2`, which can silently miss due to dtype drift and cluster membership not being carried onto the submission rows. I minimally add the cluster label (`pred`) onto the submission dataframe using the same `cust2id` mapping, then build each cluster’s submission slice by filtering on `pred` directly (not by `customer_id` inner-join), keeping your scoring/ranking logic intact. This should materially increase MAP@12 from 0.0 toward your target by ensuring non-empty, correctly aligned predictions per customer. I also keep your existing 10-digit `article_id` formatting and 12-unique padding logic unchanged.'

# 9. Code solution

## === cell 0
import sys, warnings, time, os, copy, gc, re, random, pickle

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from tqdm import tqdm

tqdm.pandas()

from datetime import datetime, timedelta
from collections import Counter

import cudf
from sklearn.cluster import KMeans
from sklearn import preprocessing




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



## === cell 3
customers = pd.read_csv(PATH_INPUT + "customers.csv")

clst = Clustering_HandM()
customers = clst.customers_preprocessing(customers)

usecol = ["club_member_status", "fashion_news_frequency", "age", "FN", "Active"]
predcol = ["customer_id"]

dfCustomers = clst.clustering(customers, predcol=predcol, usecol=usecol)



## === cell 4
listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
dfCustomers["age_bins"] = pd.cut(dfCustomers["age"], listBin)



## === cell 5
pd.crosstab(dfCustomers["pred"], dfCustomers["age_bins"])



## === cell 6
dfCustomers = dfCustomers.drop(["age_bins"], axis=1)



## === cell 7
dfTransactions = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"article_id": "string", "t_dat": "string", "customer_id": "string"},
)
dfTransactions["t_dat"] = cudf.to_datetime(dfTransactions["t_dat"])

dfTransactions["article_id_int"] = dfTransactions["article_id"].astype("int32")

dfTransactions = dfTransactions.sort_values("t_dat")

dfTransactions.head()



## === cell 8
start = cudf.to_datetime("2020-09-01")
end = cudf.to_datetime("2020-09-21")
dfRecent = dfTransactions[
    (dfTransactions["t_dat"] >= start) & (dfTransactions["t_dat"] <= end)
][["t_dat", "customer_id", "article_id_int"]]
dfRecent.head()



## === cell 9
dfRecent_pd = dfRecent.to_pandas()
dfRecent_pd["customer_id"] = dfRecent_pd["customer_id"].astype("string")
dfCustomers["customer_id"] = dfCustomers["customer_id"].astype("string")

dfRecent_pd = dfRecent_pd.merge(
    dfCustomers[["customer_id", "pred"]], on="customer_id", how="inner"
)
dfRecent_pd.head()



## === cell 10
dfRecent_pd = (
    dfRecent_pd.groupby(["pred", "article_id_int"])
    .count()
    .reset_index()
    .rename(columns={"customer_id": "counts"})
)
listUniBins_recent = dfRecent_pd["pred"].unique().tolist()

dict100 = {}
for uniBin in listUniBins_recent:
    dfTemp = dfRecent_pd[dfRecent_pd["pred"] == uniBin]
    dfTemp = dfTemp.sort_values(by="counts", ascending=False)
    dict100[uniBin] = dfTemp.head(100)["article_id_int"].values.tolist()

df100 = pd.DataFrame([dict100]).T.rename(columns={0: "top100"})



## === cell 11
for index in df100.index:
    df100[index] = [
        len(set(df100.at[index, "top100"]) & set(df100.at[x, "top100"])) / 100
        for x in df100.index
    ]

df100 = df100.drop(columns="top100")
plt.figure(figsize=(10, 6))
sns.heatmap(df100, annot=True, cbar=False)



## === cell 12
N = 12

dfCustomers["pred"] = dfCustomers["pred"].astype("int32")
listUniBins = sorted(dfCustomers["pred"].unique().tolist())


def _zfill10_from_any(x) -> str:
    s = str(x)
    s = re.sub(r"\D", "", s)
    return s.zfill(10)[-10:]


articles_pd = pd.read_csv(
    PATH_INPUT + "articles.csv", usecols=["article_id"], dtype={"article_id": "string"}
)
articles_pd["article_id_str"] = articles_pd["article_id"].map(_zfill10_from_any)
articles_pd["article_id_int"] = articles_pd["article_id_str"].astype(np.int32)

aid_map_tx_cu = (
    dfTransactions[["article_id_int", "article_id"]]
    .drop_duplicates()
    .rename(columns={"article_id": "article_id_str"})
)
aid_map_tx_pd = aid_map_tx_cu.to_pandas()
aid_map_tx_pd["article_id_str"] = aid_map_tx_pd["article_id_str"].map(_zfill10_from_any)

aid_map_pd = pd.concat(
    [
        articles_pd[["article_id_int", "article_id_str"]],
        aid_map_tx_pd[["article_id_int", "article_id_str"]],
    ],
    axis=0,
    ignore_index=True,
).drop_duplicates("article_id_int", keep="first")

AID_INT2STR = dict(
    zip(aid_map_pd["article_id_int"].astype(np.int32), aid_map_pd["article_id_str"])
)


def aid_str_from_int(x: int) -> str:
    try:
        return AID_INT2STR.get(np.int32(x), f"{int(x):010d}")
    except Exception:
        return _zfill10_from_any(x)


def compute_ldbw_from_pandas(ts: pd.Series) -> pd.Series:
    dow = ts.dt.dayofweek
    ldbw = ts - pd.TimedeltaIndex(dow - 1, unit="D")
    mask = dow >= 2
    ldbw.loc[mask] = ldbw.loc[mask] + pd.TimedeltaIndex(
        np.ones(mask.sum()) * 7, unit="D"
    )
    return ldbw


sub_all = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
numCustomers = sub_all.shape[0]
sub_all_pd = sub_all.to_pandas()
sub_all_pd["customer_id"] = sub_all_pd["customer_id"].astype("string")

dfCustomers["customer_id"] = dfCustomers["customer_id"].astype("string")

tx_cust_pd = dfTransactions[["customer_id"]].drop_duplicates().to_pandas()
tx_cust_pd["customer_id"] = tx_cust_pd["customer_id"].astype("string")

all_cust = pd.Index(
    pd.concat([sub_all_pd["customer_id"], tx_cust_pd["customer_id"]], axis=0)
    .astype("string")
    .unique()
)
cust2id = pd.Series(np.arange(len(all_cust), dtype=np.int64), index=all_cust)

sub_all_pd["customer_id2"] = (
    cust2id.reindex(sub_all_pd["customer_id"]).astype("int64").values
)

sub_all_pd = sub_all_pd.merge(
    dfCustomers[["customer_id", "pred"]], on="customer_id", how="left"
)
sub_all_pd["pred"] = sub_all_pd["pred"].fillna(-1).astype("int32")

dfTransactions = dfTransactions.merge(
    cudf.from_pandas(
        pd.DataFrame(
            {"customer_id": all_cust.astype("string"), "customer_id2": cust2id.values}
        )
    ),
    on="customer_id",
    how="left",
)

dfCustomers_pd = dfCustomers[["customer_id", "pred", "age"]].copy()
dfCustomers_pd["customer_id"] = dfCustomers_pd["customer_id"].astype("string")
dfCustomers_pd["customer_id2"] = (
    cust2id.reindex(dfCustomers_pd["customer_id"]).astype("int64").values
)
dfCustomers_cu = cudf.from_pandas(dfCustomers_pd)

pred_parts = []

for uniBin in listUniBins:
    t0 = time.time()

    dfCustomersTemp_cu = dfCustomers_cu[dfCustomers_cu["pred"] == np.int32(uniBin)][
        ["customer_id", "customer_id2", "age"]
    ]

    df_cluster = dfTransactions.merge(
        dfCustomersTemp_cu[["customer_id"]], on="customer_id", how="inner"
    )
    print(f"The shape of scope transaction for {uniBin} is {df_cluster.shape}. \n")

    last_ts = df_cluster["t_dat"].max()

    tmp_pd = df_cluster[["t_dat"]].to_pandas()
    tmp_pd["ldbw"] = compute_ldbw_from_pandas(tmp_pd["t_dat"])
    df_cluster["ldbw"] = tmp_pd["ldbw"].values

    weekly_sales = (
        df_cluster[["ldbw", "article_id_int", "t_dat"]]
        .groupby(["ldbw", "article_id_int"])
        .count()
        .reset_index()
    )
    weekly_sales = weekly_sales.rename(columns={"t_dat": "count"})
    df_cluster = df_cluster.merge(
        weekly_sales, on=["ldbw", "article_id_int"], how="left"
    )

    last_ldbw = df_cluster["ldbw"].max()
    weekly_sales_idx = weekly_sales.set_index("article_id_int")
    df_cluster = df_cluster.merge(
        weekly_sales_idx.loc[weekly_sales_idx["ldbw"] == last_ldbw, ["count"]],
        on="article_id_int",
        suffixes=("", "_targ"),
    )
    df_cluster["count_targ"] = df_cluster["count_targ"].fillna(0)

    df_cluster["quotient"] = df_cluster["count_targ"] / df_cluster["count"]

    target_sales = (
        df_cluster[["article_id_int", "quotient"]]
        .groupby("article_id_int")["quotient"]
        .sum()
    )
    general_pred_int = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = [aid_str_from_int(article_id) for article_id in general_pred_int]
    general_pred_str = " ".join(general_pred)

    tmp = df_cluster[
        ["t_dat", "customer_id2", "article_id_int", "quotient"]
    ].to_pandas()

    try:
        last_ts_pd = pd.Timestamp(last_ts.to_pandas())
    except AttributeError:
        last_ts_pd = pd.to_datetime(last_ts)

    tmp["x"] = ((last_ts_pd - tmp["t_dat"]) / np.timedelta64(1, "D")).astype(int)
    tmp["dummy_1"] = 1
    tmp["x"] = tmp[["x", "dummy_1"]].max(axis=1)

    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    tmp["y"] = a / np.sqrt(tmp["x"]) + b * np.exp(-c * tmp["x"]) - d
    tmp["dummy_0"] = 0
    tmp["y"] = tmp[["y", "dummy_0"]].max(axis=1)

    tmp["value"] = tmp["quotient"] * tmp["y"]
    tmp = (
        tmp.groupby(["customer_id2", "article_id_int"])
        .agg({"value": "sum"})
        .reset_index()
    )
    tmp = tmp.loc[tmp["value"] > 0]
    tmp["rank"] = tmp.groupby("customer_id2")["value"].rank("dense", ascending=False)
    tmp = tmp.loc[tmp["rank"] <= 12]

    purchase_df = tmp.sort_values(
        ["customer_id2", "value"], ascending=False
    ).reset_index(drop=True)

    purchase_df["prediction"] = (
        purchase_df["article_id_int"].map(aid_str_from_int) + " "
    )

    purchase_df = (
        purchase_df.groupby("customer_id2")["prediction"].apply("".join).reset_index()
    )
    purchase_df["prediction"] = purchase_df["prediction"].str.strip()

    sub_cluster_pd = sub_all_pd.loc[
        sub_all_pd["pred"] == np.int32(uniBin), ["customer_id", "customer_id2"]
    ].copy()
    sub_cluster_pd = sub_cluster_pd.merge(purchase_df, on="customer_id2", how="left")
    sub_cluster_pd["prediction"] = sub_cluster_pd["prediction"].fillna(general_pred_str)

    gen_list = general_pred  # already length N

    def finalize_pred(pred_str: str) -> str:
        items = pred_str.split() if isinstance(pred_str, str) and len(pred_str) else []
        seen = set()
        out = []
        for it in items:
            it2 = _zfill10_from_any(it)
            if it2 not in seen:
                out.append(it2)
                seen.add(it2)
            if len(out) == 12:
                return " ".join(out)
        for it in gen_list:
            if it not in seen:
                out.append(it)
                seen.add(it)
            if len(out) == 12:
                break
        if len(out) < 12:
            out = out + gen_list[: (12 - len(out))]
            out = out[:12]
        return " ".join(out[:12])

    sub_cluster_pd["prediction"] = sub_cluster_pd["prediction"].map(finalize_pred)
    pred_parts.append(sub_cluster_pd[["customer_id", "prediction"]])

    print(
        f"Finished bin {uniBin} in {time.time()-t0:.1f}s. Rows: {sub_cluster_pd.shape[0]}"
    )
    print("-" * 50)

print("Finished.\n")
print("=" * 50)



## === cell 13
dfPredAll = pd.concat(pred_parts, axis=0)
dfPredAll = dfPredAll.drop_duplicates("customer_id", keep="last")

sub_final = sub_all_pd[["customer_id"]].merge(dfPredAll, on="customer_id", how="left")

last_ts_all = dfTransactions["t_dat"].max()
try:
    last_ts_all_pd = pd.Timestamp(last_ts_all.to_pandas())
except AttributeError:
    last_ts_all_pd = pd.to_datetime(last_ts_all)
cutoff_pd = last_ts_all_pd - pd.Timedelta(days=7)

df_last7 = dfTransactions[dfTransactions["t_dat"] >= np.datetime64(cutoff_pd)][
    ["article_id_int"]
]

pop_vc = df_last7["article_id_int"].value_counts().reset_index()
pop_vc.columns = ["article_id_int", "cnt"]
pop_vc = pop_vc.sort_values("cnt", ascending=False)
global_top_int = pop_vc.head(12)["article_id_int"].to_pandas().tolist()
global_top = [aid_str_from_int(x) for x in global_top_int]
global_top_str = " ".join(global_top)

sub_final["prediction"] = sub_final["prediction"].fillna(global_top_str)


def ensure12_unique(s: str, pad_list=global_top) -> str:
    items = s.split() if isinstance(s, str) else []
    seen = set()
    out = []
    for it in items:
        it2 = _zfill10_from_any(it)
        if it2 not in seen:
            out.append(it2)
            seen.add(it2)
        if len(out) == 12:
            return " ".join(out)
    for it in pad_list:
        if it not in seen:
            out.append(it)
            seen.add(it)
        if len(out) == 12:
            break
    while len(out) < 12:
        for it in pad_list:
            out.append(it)
            if len(out) == 12:
                break
    return " ".join(out[:12])


sub_final["prediction"] = sub_final["prediction"].map(ensure12_unique)

assert (
    sub_final.shape[0] == numCustomers
), f"Row count mismatch: {sub_final.shape[0]} vs {numCustomers}"
sub_final.to_csv("submission.csv", index=False)
print("Saved submission.csv.")



## === cell 14
dfCheck = pd.read_csv("./submission.csv")
print(dfCheck.head())
print(dfCheck.shape)
print(dfCheck.columns.tolist())
print("Example prediction length (tokens):", len(dfCheck.loc[0, "prediction"].split()))
print("Example unique tokens:", len(set(dfCheck.loc[0, "prediction"].split())))
print(
    "Example token format:",
    dfCheck.loc[0, "prediction"].split()[0],
    "len=",
    len(dfCheck.loc[0, "prediction"].split()[0]),
)
