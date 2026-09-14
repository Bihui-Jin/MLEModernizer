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

0.02244

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime errors (the deprecated `json_normalize` import and the cuDF datetime slicing on a non-monotonic index) so the notebook runs end-to-end. Then I fix the core submission logic bug: your per-age-bin loop currently *drops* customers not in that bin (inner merge), so the final concatenated submission is missing many `customer_id`s, causing the “superset” error. The minimal correction is to build one final submission initialized from `sample_submission.csv` and fill predictions for customers that have age info and appear in each bin (left-join), while also handling missing-age customers with a fallback. This keeps your existing scoring logic (weekly quotient + recency weighting + general fallback) but makes the output valid and complete.'
- What this solution (achieved 0.0) has done: 'I fix the cuDF datetime slice error by avoiding value-based `.loc` slicing on a non-monotonic DatetimeIndex and instead filtering with boolean masks, which preserves the intent of the “recent” window without changing the modeling logic. Then I fix the age-bin masking bug that crashes in the per-bin loop: `age_bins` is an interval/categorical type and cannot be compared to `str(bin_unique)` inside cuDF, so I create a stable string key for bins (both in pandas and cuDF) and compare strings consistently. Finally, I keep the existing “fill per bin then fallback to global popular last-7-days” logic, ensuring every customer gets a non-empty prediction so the score moves up from 0.0 and the submission passes the format/superset checks.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a “valid CSV but effectively empty/garbled predictions” failure mode for MAP@12, and in this code the main culprit is prediction string construction and truncation: `str[:131]` can cut article_id tokens mid-way, producing invalid IDs that won’t match any ground-truth article_id, and concatenation can yield duplicate/partial tokens. I keep your per-age-bin modeling exactly the same, but change post-processing to always output exactly up to 12 *valid* 10-digit `article_id` tokens per customer (deduped, preserved order), without any character-based truncation. I also ensure that article_id formatting is robust (`zfill(10)` rather than `"0"+str(...)`) so IDs are always correct length, and apply the same token-safe logic to both bin-level and global fallback predictions. These are minimal changes that should move the score up toward your target without changing the underlying ranking logic.'
- What this solution (achieved 0.0) has done: 'I fix the runtime error in the per-bin loop by ensuring `purchase_df` stays a cuDF DataFrame until we explicitly convert, so `purchase_df["article_id"].to_pandas()` is called on a cuDF Series (not a pandas Series). I also make the customer-id join consistent by converting `tmp["customer_id"]` to the same int64 encoding used elsewhere before grouping/ranking, which prevents silent mismatches that can lead to mostly-empty per-customer predictions (and thus a 0.0 score). Finally, I keep your existing bin-wise model + global-pop fallback logic unchanged, only making the minimal type/merge corrections needed so it runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import sys
import warnings

warnings.filterwarnings("ignore")
import time
import os
import copy
import gc
import re
import random
import pickle
import cudf

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 10000)

import seaborn as sns

sns.set()

from pandas import json_normalize
from pprint import pprint
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()
from collections import Counter
from datetime import datetime, timedelta



## === cell 1
DEBUG = False
PATH_INPUT = r"../input/h-and-m-personalized-fashion-recommendations/"



## === cell 2
COUNT = 1
ORDER = 0
N = 12




## === cell 3
def display_df(df, head=3):
    print(f"SHAPE: {df.shape}\n")
    display(df.head(head))




## === cell 4
def pre_increment(name, local={}):
    if name in local:
        local[name] += 1
        return local[name]
    globals()[name] += 1
    return globals()[name]




## === cell 5
def info_df(df, count, order):
    if count:
        try:
            name = [x for x in globals() if globals()[x] is df][0]
        except IndexError:
            name = ""
        order = pre_increment("order")
        print("=" * 30)
        print(f"{order} INFO_DF {name}:\n")
        display_df(df)




## === cell 6
def format_article_id(a) -> str:
    s = str(a)
    s = re.sub(r"\D", "", s)  # keep digits only, defensive
    return s.zfill(10)


def finalize_pred_tokens(primary_tokens, fallback_tokens, n=12):
    out = []
    seen = set()
    for tok in list(primary_tokens) + list(fallback_tokens):
        if tok is None:
            continue
        t = str(tok).strip()
        if not t:
            continue
        t = format_article_id(t)
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= n:
            break
    return " ".join(out)


def prediction_str_to_tokens(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [t for t in s.split() if t.strip()]




## === cell 7
articles_df = cudf.read_csv(
    PATH_INPUT + "articles.csv",
    usecols=["article_id", "product_group_name", "perceived_colour_master_name"],
)
display_df(articles_df)



## === cell 8
customers_df = cudf.read_csv(
    PATH_INPUT + "customers.csv", usecols=["customer_id", "age"]
)
display_df(customers_df)



## === cell 9
customers_df = customers_df.to_pandas()
bin_list = [-1, 19, 29, 39, 49, 59, 69, 119]
customers_df["age_bins"] = pd.cut(customers_df["age"], bin_list)
customers_df["age_bin_str"] = customers_df["age_bins"].astype(
    str
)  # includes "nan" for missing
display_df(customers_df)



## === cell 10
age_missing = customers_df[customers_df["age_bins"].isnull()].shape[0]
age_missing



## === cell 11
transactions_df = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
)
transactions_df["t_dat"] = cudf.to_datetime(transactions_df["t_dat"])

transactions_df = transactions_df.sort_values("t_dat")
transactions_df = transactions_df.set_index("t_dat")
display_df(transactions_df)



## === cell 12
start = np.datetime64("2020-09-01")
end = np.datetime64("2020-09-21")
idx = transactions_df.index
recent_df = transactions_df[(idx >= start) & (idx <= end)]
display_df(recent_df)



## === cell 13
recent_df = recent_df.to_pandas()
recent_df = recent_df.merge(
    customers_df[["customer_id", "age_bins"]], on="customer_id", how="inner"
)
display_df(recent_df)



## === cell 14
recent_df = (
    recent_df.groupby(["age_bins", "article_id"])
    .count()
    .reset_index()
    .rename(columns={"customer_id": "counts"})
)
display_df(recent_df)



## === cell 15
bins_unique_list = recent_df["age_bins"].unique().tolist()
bins_unique_list



## === cell 16
if len(bins_unique_list) > 0:
    bins_unique_list[0]



## === cell 17
ages_article_id = {}
count = COUNT
order = ORDER
for bins_unique in bins_unique_list:
    temp_df = recent_df[recent_df["age_bins"] == bins_unique]
    info_df(temp_df, count, order)

    temp_df = temp_df.sort_values(by="counts", ascending=False)
    info_df(temp_df, count, order)

    ages_article_id[bins_unique] = temp_df.head(100)["article_id"].values.tolist()

    count = 0



## === cell 18
from itertools import islice

for key, value in islice(ages_article_id.items(), 3):
    print(f"{key} {len(value)}")



## === cell 19
topcustcnt_byage_df = pd.DataFrame([ages_article_id])
display_df(topcustcnt_byage_df)



## === cell 20
topcustcnt_byage_df = pd.DataFrame([ages_article_id]).T.rename(columns={0: "top_100"})
display_df(topcustcnt_byage_df)



## === cell 21
for i in topcustcnt_byage_df.index:
    topcustcnt_byage_df[i] = [
        len(
            set(topcustcnt_byage_df.at[i, "top_100"])
            & set(topcustcnt_byage_df.at[j, "top_100"])
        )
        / 100
        for j in topcustcnt_byage_df.index
    ]
display_df(topcustcnt_byage_df)



## === cell 22
topcustcnt_byage_df = topcustcnt_byage_df.drop(columns="top_100")
display_df(topcustcnt_byage_df, head=10)



## === cell 23
if topcustcnt_byage_df.shape[0] > 0 and topcustcnt_byage_df.shape[1] > 0:
    plt.figure(figsize=(10, 6))
    sns.heatmap(topcustcnt_byage_df, cmap="winter", annot=True, cbar=False)



## === cell 24
bins_unique_list = customers_df["age_bin_str"].unique().tolist()
bins_unique_list



## === cell 25
sub_all = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
num_customers = sub_all.shape[0]
sub_all["prediction"] = cudf.Series([""] * num_customers, dtype="str")

sub_all["customer_id_2"] = (
    sub_all["customer_id"].str[-16:].str.hex_to_int().astype("int64")
)

cust_age_cudf = cudf.from_pandas(customers_df[["customer_id", "age", "age_bin_str"]])
sub_all = sub_all.merge(
    cust_age_cudf[["customer_id", "age", "age_bin_str"]], on="customer_id", how="left"
)
info_df(sub_all, COUNT, ORDER)



## === cell 26
count = COUNT
order = ORDER

for bin_unique in bins_unique_list:
    df = cudf.read_csv(
        PATH_INPUT + "transactions_train.csv",
        usecols=["t_dat", "customer_id", "article_id"],
        dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
    )
    info_df(df, count, order)

    if str(bin_unique) == "nan":
        temp_customers_pd = customers_df[customers_df["age_bin_str"] == "nan"][
            ["customer_id", "age"]
        ]
        bin_mask = sub_all["age_bin_str"].isnull() | (sub_all["age_bin_str"] == "nan")
    else:
        temp_customers_pd = customers_df[
            customers_df["age_bin_str"] == str(bin_unique)
        ][["customer_id", "age"]]
        bin_mask = sub_all["age_bin_str"] == str(bin_unique)

    temp_customers_df = cudf.from_pandas(temp_customers_pd)
    info_df(temp_customers_df, count, order)

    df = df.merge(
        temp_customers_df[["customer_id", "age"]], on="customer_id", how="inner"
    )
    info_df(df, count, order)

    print(f"TRANSACTION SHAPE FOR SCOPE {bin_unique}: {df.shape}\n")

    if df.shape[0] == 0:
        print(f"NO TRANSACTIONS FOR {bin_unique} -> SKIP BIN MODEL")
        print("-" * 50)
        count = 0
        continue

    df["customer_id"] = df["customer_id"].str[-16:].str.hex_to_int().astype("int64")
    df["t_dat"] = cudf.to_datetime(df["t_dat"])

    last_date_train = df["t_dat"].max()
    if count:
        print("=" * 30 + f"\n4 LAST_DATE_TRAIN:\n{last_date_train}")

    tmp = (
        df[
            [
                "t_dat",
                "customer_id",
                "article_id",
                "count" if "count" in df.columns else "article_id",
            ]
        ]
        .copy()
        .to_pandas()
    )
    tmp = df.copy().to_pandas()
    tmp["customer_id"] = tmp["customer_id"].astype("int64")

    tmp["weekday"] = tmp["t_dat"].dt.dayofweek
    info_df(tmp, count, order)

    tmp["t_dat_shiftday"] = tmp["t_dat"] - pd.TimedeltaIndex(
        tmp["weekday"] - 1, unit="D"
    )
    info_df(tmp, count, order)

    tmp.loc[tmp["weekday"] >= 2, "t_dat_shiftday"] = tmp.loc[
        tmp["weekday"] >= 2, "t_dat_shiftday"
    ] + pd.TimedeltaIndex(np.ones(len(tmp.loc[tmp["weekday"] >= 2])) * 7, unit="D")
    info_df(tmp, count, order)

    df["t_dat_shiftday"] = tmp["t_dat_shiftday"].values
    info_df(df, count, order)

    weekly_sales = (
        df.drop("customer_id", axis=1)
        .groupby(["t_dat_shiftday", "article_id"])
        .count()
        .reset_index()
    )
    info_df(weekly_sales, count, order)

    weekly_sales = weekly_sales.rename(columns={"t_dat": "count"})
    info_df(weekly_sales, count, order)

    df = df.merge(weekly_sales, on=["t_dat_shiftday", "article_id"], how="left")
    info_df(df, count, order)

    weekly_sales = weekly_sales.reset_index().set_index("article_id")
    info_df(weekly_sales, count, order)

    df = df.merge(
        weekly_sales.loc[weekly_sales["t_dat_shiftday"] == last_date_train, ["count"]],
        on="article_id",
        suffixes=("", "_targ"),
    )
    info_df(df, count, order)

    df["count_targ"].fillna(0, inplace=True)
    del weekly_sales

    df["quotient"] = df["count_targ"] / df["count"]
    info_df(df, count, order)

    target_sales = (
        df.drop("customer_id", axis=1).groupby("article_id")["quotient"].sum()
    )
    info_df(target_sales, count, order)

    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = [format_article_id(article_id) for article_id in general_pred]
    general_pred_str = " ".join(general_pred)
    del target_sales

    info_df(tmp, count, order)

    tmp["x"] = ((last_date_train - tmp["t_dat"]) / np.timedelta64(1, "D")).astype(int)
    info_df(tmp, count, order)

    tmp["dummy_1"] = 1
    tmp["x"] = tmp[["x", "dummy_1"]].max(axis=1)
    info_df(tmp, count, order)

    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    tmp["y"] = a / np.sqrt(tmp["x"]) + b * np.exp(-c * tmp["x"]) - d
    info_df(tmp, count, order)

    tmp["dummy_0"] = 0
    tmp["y"] = tmp[["y", "dummy_0"]].max(axis=1)
    tmp["value"] = tmp["quotient"] * tmp["y"]
    info_df(tmp, count, order)

    tmp = tmp.groupby(["customer_id", "article_id"]).agg({"value": "sum"})
    info_df(tmp, count, order)

    tmp = tmp.reset_index()
    tmp = tmp.loc[tmp["value"] > 0]
    info_df(tmp, count, order)

    tmp["rank"] = tmp.groupby("customer_id")["value"].rank("dense", ascending=False)
    info_df(tmp, count, order)

    tmp = tmp.loc[tmp["rank"] <= 12]
    info_df(tmp, count, order)

    purchase_df = tmp.sort_values(
        ["customer_id", "value"], ascending=False
    ).reset_index(drop=True)
    info_df(purchase_df, count, order)

    purchase_gdf = cudf.from_pandas(purchase_df)
    purchase_gdf["pred_tok"] = (
        purchase_gdf["article_id"].to_pandas().map(format_article_id).values
    )
    info_df(purchase_gdf, count, order)

    purchase_df = (
        purchase_gdf[["customer_id", "pred_tok"]]
        .groupby("customer_id")
        .agg({"pred_tok": list})
        .reset_index()
    )
    info_df(purchase_df, count, order)

    sub_bin = sub_all.loc[bin_mask, ["customer_id_2"]].merge(
        purchase_df, left_on="customer_id_2", right_on="customer_id", how="left"
    )
    sub_bin = sub_bin.to_pandas()

    fallback_tokens = prediction_str_to_tokens(general_pred_str)
    sub_bin["prediction"] = sub_bin["pred_tok"].apply(
        lambda lst: finalize_pred_tokens(
            lst if isinstance(lst, list) else [], fallback_tokens, n=N
        )
    )
    sub_bin = sub_bin[["prediction"]]

    sub_all_pd = sub_all.to_pandas()
    sub_all_pd.loc[np.array(bin_mask.to_pandas()), "prediction"] = sub_bin[
        "prediction"
    ].values
    sub_all = cudf.from_pandas(sub_all_pd)

    print(f"FILLED PREDICTION FOR BIN {bin_unique}, ROWS: {int(bin_mask.sum())}")
    print("-" * 50)

    count = 0

print("FINISHED BIN MODELS")
print("=" * 50)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/217464687.py in <cell line: 0>()
     45 
     46     tmp = (
---> 47         df[
     48             [
     49                 "t_dat",

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/dataframe.py in __getitem__(self, arg)
   1382                 return self._apply_boolean_mask(BooleanMask(mask, len(self)))
   1383             else:
-> 1384                 return self._get_columns_by_label(mask)
   1385         elif isinstance(arg, DataFrame):
   1386             return self.where(arg)

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/frame.py in _get_columns_by_label(self, labels)
    404         Akin to cudf.DataFrame(...).loc[:, labels]
    405         """
--> 406         return self._from_data_like_self(self._data.select_by_label(labels))
    407 
    408     @property

/usr/local/lib/python3.11/dist-packages/cudf/core/column_accessor.py in select_by_label(self, key)
    406             return self._select_by_label_slice(key)
    407         elif pd.api.types.is_list_like(key) and not isinstance(key, tuple):
--> 408             return self._select_by_label_list_like(tuple(key))
    409         else:
    410             if isinstance(key, tuple):

/usr/local/lib/python3.11/dist-packages/cudf/core/column_accessor.py in _select_by_label_list_like(self, key)
    559             data = {k: self._grouped_data[k] for k in key}
    560             if len(data) != len(key):
--> 561                 raise ValueError(
    562                     "Selecting duplicate column labels is not supported."
    563                 )

ValueError: Selecting duplicate column labels is not supported.

## === cell 27
tx = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "article_id"],
    dtype={"t_dat": "string", "article_id": "int32"},
)
tx["t_dat"] = cudf.to_datetime(tx["t_dat"])
max_date = tx["t_dat"].max()
tx = tx[tx["t_dat"] >= (max_date - np.timedelta64(7, "D"))]
pop = (
    tx.groupby("article_id")
    .size()
    .sort_values(ascending=False)
    .head(N)
    .index.to_pandas()
    .tolist()
)

global_pop_tokens = [format_article_id(a) for a in pop]
global_pop_str = " ".join(global_pop_tokens)

sub_all_pd = sub_all.to_pandas()

sub_all_pd["prediction"] = sub_all_pd["prediction"].replace("", np.nan).fillna("")
sub_all_pd["prediction"] = sub_all_pd["prediction"].apply(
    lambda s: finalize_pred_tokens(prediction_str_to_tokens(s), global_pop_tokens, n=N)
)

sub_all_pd = sub_all_pd[["customer_id", "prediction"]]
sub_all_pd.to_csv("submission.csv", index=False)

print(sub_all_pd.shape)
print(sub_all_pd.head())



## === cell 28
check = pd.read_csv(PATH_INPUT + "sample_submission.csv", usecols=["customer_id"])
sub_check = pd.read_csv("submission.csv")

assert (
    sub_check.shape[0] == check.shape[0]
), f"Row count mismatch: {sub_check.shape[0]} vs {check.shape[0]}"
assert set(check["customer_id"]).issubset(
    set(sub_check["customer_id"])
), "Submission customer_id must be a superset of sample_submission customer_id"
assert sub_check["prediction"].isna().sum() == 0, "prediction contains NaN"
assert (
    sub_check["prediction"].str.len() > 0
).all(), "Some prediction strings are empty"
print("submission.csv looks valid:", sub_check.shape)
