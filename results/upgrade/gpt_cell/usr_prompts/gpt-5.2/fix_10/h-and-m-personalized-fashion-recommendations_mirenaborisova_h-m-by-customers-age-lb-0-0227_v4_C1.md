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

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'You’re currently not yielding a Kaggle score because the notebook doesn’t reliably create a single valid submission: it writes one CSV per age bin, then concatenates them, but each per-bin file contains only customers in that bin (via an inner-merge) so concatenation produces duplicates and misses many customers. I keep your scoring logic intact but change the submission construction to start from the full sample submission and left-join age-bin predictions onto it, ensuring exactly one row per customer in the right order. I also remove the length-based string truncation and instead enforce “max 12 unique article_ids” deterministically, which better matches MAP@12 formatting without changing the underlying ranking logic. Finally, I write a single `submission.csv` (required by Kaggle) plus your previous debug output file for inspection.'
- What this solution (achieved 0.0) has done: 'Diagnosis: Cell 26 crashes because `df_all` does not contain the column `age_bins_key` at the moment the per-bin loop starts. That can happen if the preceding merge in cell 25 did not create/populate `age_bins_key` as expected (e.g., schema mismatch or merge behavior differences), so indexing `df_all["age_bins_key"]` raises `KeyError`.  
Patch summary: In cell 26, add a minimal guard that ensures `df_all` has an `age_bins_key` column before it is used; if it’s missing, create it as an all-null string column so the existing “nan” bin logic still works deterministically. This preserves the rest of the per-bin computation unchanged.  
Updated cells: Only cell 26 is modified.  
Compatibility notes for cell k+1: The output objects (`bin_pred_parts`, per-bin `purchase_df` schema) remain identical; the patch only guarantees the required `age_bins_key` selector exists, so cell 27 can proceed without interface changes.  
Assumptions: If `age_bins_key` truly cannot be derived due to missing upstream data, falling back to all-null route all transactions into the `nan` bin (matching existing handling) rather than crashing.'

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

try:
    from pandas import json_normalize
except ImportError:  # pragma: no cover
    from pandas.io.json import json_normalize

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
articles_df = cudf.read_csv(
    PATH_INPUT + "articles.csv",
    usecols=["article_id", "product_group_name", "perceived_colour_master_name"],
)
display_df(articles_df)



## === cell 7
customers_df = cudf.read_csv(
    PATH_INPUT + "customers.csv", usecols=["customer_id", "age"]
)
display_df(customers_df)



## === cell 8
customers_df = customers_df.to_pandas()
bin_list = [-1, 19, 29, 39, 49, 59, 69, 119]
customers_df["age_bins"] = pd.cut(customers_df["age"], bin_list)

display_df(customers_df)



## === cell 9
age_missing = customers_df[customers_df["age_bins"].isnull()].shape[0]
age_missing



## === cell 10
transactions_df = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
)
transactions_df["t_dat"] = cudf.to_datetime(transactions_df["t_dat"])
transactions_df.set_index("t_dat", inplace=True)

display_df(transactions_df)



## === cell 11
transactions_df = transactions_df.sort_index()

start = np.datetime64("2020-09-01")
end = np.datetime64("2020-09-21T23:59:59.999999999")

recent_df = transactions_df[
    (transactions_df.index >= start) & (transactions_df.index <= end)
]

display_df(recent_df)



## === cell 12
recent_df = recent_df.to_pandas()
recent_df = recent_df.merge(
    customers_df[["customer_id", "age_bins"]], on="customer_id", how="inner"
)

display_df(recent_df)



## === cell 13
recent_df = (
    recent_df.groupby(["age_bins", "article_id"])
    .count()
    .reset_index()
    .rename(columns={"customer_id": "counts"})
)

display_df(recent_df)



## === cell 14
bins_unique_list = recent_df["age_bins"].unique().tolist()
bins_unique_list



## === cell 15
bins_unique_list[0]



## === cell 16
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



## === cell 17
from itertools import islice

for key, value in islice(ages_article_id.items(), 3):
    print(f"{key} {len(value)}")



## === cell 18
topcustcnt_byage_df = pd.DataFrame([ages_article_id])

display_df(topcustcnt_byage_df)



## === cell 19
topcustcnt_byage_df = pd.DataFrame([ages_article_id]).T.rename(columns={0: "top_100"})

display_df(topcustcnt_byage_df)



## === cell 20
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



## === cell 21
topcustcnt_byage_df = topcustcnt_byage_df.drop(columns="top_100")

display_df(topcustcnt_byage_df, head=10)



## === cell 22
plt.figure(figsize=(10, 6))
sns.heatmap(topcustcnt_byage_df, cmap="winter", annot=True, cbar=False)



## === cell 23
bins_unique_list



## === cell 24
bins_unique_list = customers_df["age_bins"].unique().tolist()
bins_unique_list




## === cell 25
def _fmt_article_id(a) -> str:
    return str(int(a)).zfill(10)


sub_full = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
num_customers = sub_full.shape[0]
display_df(sub_full)

cust_agebins_cudf = cudf.from_pandas(customers_df[["customer_id", "age_bins", "age"]])
sub_full = sub_full.merge(
    cust_agebins_cudf[["customer_id", "age_bins", "age"]], on="customer_id", how="left"
)

sub_full["customer_id_2"] = (
    sub_full["customer_id"].str[-16:].str.hex_to_int().astype("int64")
)
display_df(sub_full)

df_all = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
)
df_all["t_dat"] = cudf.to_datetime(df_all["t_dat"])
df_all["customer_id"] = df_all["customer_id"].str[-16:].str.hex_to_int().astype("int64")

cust_map = cudf.from_pandas(
    customers_df[["customer_id", "age_bins", "age"]].assign(
        customer_id_2=lambda x: x["customer_id"].str[-16:].apply(lambda s: int(s, 16))
    )
)
cust_map["customer_id_2"] = cust_map["customer_id_2"].astype("int64")

cust_map["age_bins_key"] = cudf.from_pandas(
    cust_map["age_bins"].to_pandas().astype(str)
)

df_all = df_all.merge(
    cust_map[["customer_id_2", "age_bins_key"]].rename(
        columns={"customer_id_2": "customer_id"}
    ),
    on="customer_id",
    how="left",
)

last_date_train = df_all["t_dat"].max()

tmp_dates = df_all[["t_dat"]].to_pandas()
tmp_dates["weekday"] = tmp_dates["t_dat"].dt.dayofweek
tmp_dates["t_dat_shiftday"] = tmp_dates["t_dat"] - pd.TimedeltaIndex(
    tmp_dates["weekday"] - 1, unit="D"
)
tmp_dates.loc[tmp_dates["weekday"] >= 2, "t_dat_shiftday"] = tmp_dates.loc[
    tmp_dates["weekday"] >= 2, "t_dat_shiftday"
] + pd.TimedeltaIndex(
    np.ones(len(tmp_dates.loc[tmp_dates["weekday"] >= 2])) * 7, unit="D"
)

df_all["t_dat_shiftday"] = cudf.from_pandas(tmp_dates["t_dat_shiftday"])

weekly_sales = (
    df_all.drop("customer_id", axis=1)
    .groupby(["t_dat_shiftday", "article_id"])
    .count()
    .reset_index()
    .rename(columns={"t_dat": "count"})
)

df_all = df_all.merge(weekly_sales, on=["t_dat_shiftday", "article_id"], how="left")

weekly_sales_idx = weekly_sales.reset_index().set_index("article_id")

df_all = df_all.merge(
    weekly_sales_idx.loc[
        weekly_sales_idx["t_dat_shiftday"] == last_date_train, ["count"]
    ],
    on="article_id",
    suffixes=("", "_targ"),
)
df_all["count_targ"].fillna(0, inplace=True)
df_all["quotient"] = df_all["count_targ"] / df_all["count"]

bin_pred_parts = []


## === cell 26
if "age_bins_key" not in df_all.columns:
    df_all["age_bins_key"] = cudf.Series([None] * len(df_all), dtype="str")

count = COUNT
order = ORDER

for bin_unique in bins_unique_list:
    bin_key = str(bin_unique)

    if bin_key == "nan":
        df_bin = df_all[df_all["age_bins_key"].isnull()]
    else:
        df_bin = df_all[df_all["age_bins_key"] == bin_key]

    if count:
        print("=" * 30)
        print(f"TRANSACTION SHAPE FOR SCOPE {bin_unique}: {df_bin.shape}\n")

    if df_bin.shape[0] == 0:
        empty = cudf.DataFrame(
            {
                "customer_id": cudf.Series([], dtype="int64"),
                "prediction": cudf.Series([], dtype="str"),
                "age_bins": cudf.Series([], dtype="str"),
                "general_pred_str": cudf.Series([], dtype="str"),
            }
        )
        bin_pred_parts.append(empty)
        count = 0
        continue

    target_sales = (
        df_bin.drop("customer_id", axis=1).groupby("article_id")["quotient"].sum()
    )
    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = [_fmt_article_id(article_id) for article_id in general_pred]
    general_pred_str = " ".join(general_pred)

    tmp = df_bin[["t_dat", "customer_id", "article_id", "quotient"]].copy().to_pandas()

    tmp["x"] = ((last_date_train - tmp["t_dat"]) / np.timedelta64(1, "D")).astype(int)
    tmp["dummy_1"] = 1
    tmp["x"] = tmp[["x", "dummy_1"]].max(axis=1)

    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    tmp["y"] = a / np.sqrt(tmp["x"]) + b * np.exp(-c * tmp["x"]) - d

    tmp["dummy_0"] = 0
    tmp["y"] = tmp[["y", "dummy_0"]].max(axis=1)
    tmp["value"] = tmp["quotient"] * tmp["y"]

    tmp = tmp.groupby(["customer_id", "article_id"]).agg({"value": "sum"})
    tmp = tmp.reset_index()
    tmp = tmp.loc[tmp["value"] > 0]

    tmp["rank"] = tmp.groupby("customer_id")["value"].rank("dense", ascending=False)
    tmp = tmp.loc[tmp["rank"] <= 12]

    purchase_df = tmp.sort_values(
        ["customer_id", "value"], ascending=False
    ).reset_index(drop=True)
    purchase_df["prediction"] = (
        purchase_df["article_id"].map(_fmt_article_id).astype("string") + " "
    )

    purchase_df = (
        purchase_df.groupby("customer_id").agg({"prediction": sum}).reset_index()
    )

    purchase_df = cudf.DataFrame(purchase_df)
    purchase_df["age_bins"] = bin_key
    purchase_df["general_pred_str"] = general_pred_str

    bin_pred_parts.append(
        purchase_df[["customer_id", "prediction", "age_bins", "general_pred_str"]]
    )

    purchase_df.to_pandas().to_csv(f"purchase_predictions_{bin_key}.csv", index=False)

    print(
        f"BUILT PREDICTIONS FOR {bin_unique}, unique customers: {purchase_df.shape[0]}\n"
    )
    print("-" * 50)

    count = 0

print("FINISHED BIN MODELS")
print("=" * 50)


## === cell 27
if len(bin_pred_parts) == 0:
    raise RuntimeError("No per-bin predictions were built; cannot create submission.")

pred_all = cudf.concat(bin_pred_parts, axis=0)

sub_full_pd = sub_full.to_pandas()
sub_full_pd["age_bins_key"] = sub_full_pd["age_bins"].astype(str)
sub_full_keyed = cudf.from_pandas(
    sub_full_pd[["customer_id", "customer_id_2", "age_bins_key"]]
)

pred_all["age_bins_key"] = pred_all["age_bins"].astype("str")

sub_joined = sub_full_keyed.merge(
    pred_all,
    left_on=["customer_id_2", "age_bins_key"],
    right_on=["customer_id", "age_bins_key"],
    how="left",
    suffixes=("", "_pred"),
)

sub_joined = sub_joined.to_pandas()
display_df(sub_joined)




## === cell 28
def _dedup_and_cap12(pred_str: str) -> str:
    if (
        pred_str is None
        or (isinstance(pred_str, float) and np.isnan(pred_str))
        or pred_str == ""
    ):
        return ""
    toks = pred_str.strip().split()
    out = []
    seen = set()
    for t in toks:
        tt = str(t).strip()
        if tt == "":
            continue
        if tt.isdigit():
            tt = tt.zfill(10)
        if tt not in seen:
            seen.add(tt)
            out.append(tt)
        if len(out) >= 12:
            break
    return " ".join(out)


tx = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["article_id"],
    dtype={"article_id": "int32"},
)
global_top = tx["article_id"].value_counts().head(N).to_pandas().index.tolist()
global_top = [_fmt_article_id(a) for a in global_top]
global_fallback = " ".join(global_top)

sub_joined["general_pred_str"] = sub_joined["general_pred_str"].fillna(global_fallback)
sub_joined["prediction"] = sub_joined["prediction"].fillna("")

sub_joined["prediction"] = (
    sub_joined["prediction"].astype(str).str.strip()
    + " "
    + sub_joined["general_pred_str"].astype(str).str.strip()
).str.strip()

sub_joined["prediction"] = sub_joined["prediction"].map(_dedup_and_cap12)

final_sub = sub_joined[["customer_id", "prediction"]].copy()
assert (
    final_sub.shape[0] == num_customers
), f"Final rows mismatch {final_sub.shape[0]} vs {num_customers}"
display_df(final_sub)

final_sub.to_csv("submission.csv", index=False)
final_sub.to_csv("by_cust_age__.csv", index=False)

print("WROTE submission.csv and by_cust_age__.csv")
print(final_sub.head())



## === cell 29
check_df = cudf.read_csv("./by_cust_age__.csv")
display_df(check_df)
