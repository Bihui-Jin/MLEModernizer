# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
end = np.datetime64("2020-09-21")
recent_df = transactions_df[
    (transactions_df.index >= start) & (transactions_df.index <= end)
]

display_df(recent_df)



## === cell 12
recent_df = recent_df.to_pandas()
recent_df = recent_df.merge(
    customers_df[["customer_id", "age_bins"]], on="customer_id", how="left"
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
bins_unique_list = list(customers_df["age_bins"].cat.categories)
if customers_df["age_bins"].isnull().any():
    bins_unique_list = bins_unique_list + [np.nan]
bins_unique_list



## === cell 25
sub_base = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)

final_sub = sub_base[["customer_id"]].to_pandas()

all_cust_ids = final_sub["customer_id"].astype(str)
final_sub["customer_id_2"] = pd.Categorical(all_cust_ids).codes.astype("int64")

num_customers = final_sub.shape[0]
info_df(sub_base, 1, 0)

final_sub["prediction"] = pd.Series([None] * len(final_sub), dtype="object")


def aid_to_token(aid):
    try:
        return str(int(aid)).zfill(10)
    except Exception:
        s = str(aid).strip()
        if s.isdigit():
            return s.zfill(10)
        return s


def normalize_pred_string(pred_str, fallback_str):
    if (
        pred_str is None
        or (isinstance(pred_str, float) and np.isnan(pred_str))
        or str(pred_str).strip() == ""
    ):
        pred_str = fallback_str

    tokens = [
        aid_to_token(t) for t in str(pred_str).strip().split() if str(t).strip() != ""
    ]
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
        if len(uniq) >= 12:
            break

    if len(uniq) < 12:
        fb = [
            aid_to_token(t)
            for t in str(fallback_str).strip().split()
            if str(t).strip() != ""
        ]
        for t in fb:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
            if len(uniq) >= 12:
                break

    return " ".join(uniq[:12])




## === cell 26
global_top = (
    recent_df.groupby("article_id")["counts"]
    .sum()
    .sort_values(ascending=False)
    .head(N)
    .index.tolist()
)
global_top = [aid_to_token(aid) for aid in global_top]
global_fallback = " ".join(global_top)

df_all = transactions_df.reset_index()[["t_dat", "customer_id", "article_id"]].copy()

cust_code_map = dict(zip(all_cust_ids.values, final_sub["customer_id_2"].values))
df_all["customer_id_2"] = (
    df_all["customer_id"].to_pandas().map(cust_code_map).fillna(-1).astype("int64")
)
df_all = cudf.from_pandas(df_all)

info_df(df_all, COUNT, ORDER)

count = COUNT
order = ORDER

for bin_unique in bins_unique_list:
    if pd.isna(bin_unique):
        temp_customers_df = customers_df[customers_df["age_bins"].isnull()]
    else:
        temp_customers_df = customers_df[customers_df["age_bins"] == bin_unique]

    temp_customers_df = temp_customers_df.drop(["age_bins"], axis=1)
    temp_customers_cu = cudf.from_pandas(temp_customers_df)

    info_df(temp_customers_cu, count, order)

    bin_cust = temp_customers_cu[["customer_id"]].copy()
    bin_cust["customer_id_2"] = (
        bin_cust["customer_id"]
        .to_pandas()
        .map(cust_code_map)
        .fillna(-1)
        .astype("int64")
    )
    bin_cust = cudf.from_pandas(bin_cust)

    bin_cust_ids = bin_cust["customer_id_2"].to_pandas().values
    mask_bin = final_sub["customer_id_2"].isin(bin_cust_ids)

    if int(mask_bin.sum()) == 0:
        print(f"SKIP BIN {bin_unique}: customers_in_bin=0")
        print("-" * 50)
        count = 0
        continue

    df = df_all[df_all["customer_id_2"].isin(bin_cust["customer_id_2"])].copy()
    info_df(df, count, order)

    print(f"TRANSACTION SHAPE FOR SCOPE {bin_unique}: {df.shape}\n")

    if df.shape[0] == 0:
        filled_bin_pred = pd.Series([global_fallback] * int(mask_bin.sum())).apply(
            lambda s: normalize_pred_string(s, global_fallback)
        )
        final_sub.loc[mask_bin, "prediction"] = filled_bin_pred.values
        print(
            f"FILLED BIN {bin_unique} (no transactions): customers_in_bin={int(mask_bin.sum())}, used=global_fallback"
        )
        print("-" * 50)
        count = 0
        continue

    last_date_train = df["t_dat"].max()
    if count:
        print("=" * 30 + f"\n4 LAST_DATE_TRAIN:\n{last_date_train}")

    tmp = df[["t_dat"]].copy().to_pandas()
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

    last_shiftday = pd.to_datetime(tmp["t_dat_shiftday"].max())
    weekly_sales = weekly_sales.reset_index().set_index("article_id")
    info_df(weekly_sales, count, order)

    targ_slice = weekly_sales.loc[
        weekly_sales["t_dat_shiftday"] == last_shiftday, ["count"]
    ]
    if targ_slice.shape[0] == 0:
        df["count_targ"] = 0
    else:
        df = df.merge(
            targ_slice,
            on="article_id",
            suffixes=("", "_targ"),
        )
        df["count_targ"].fillna(0, inplace=True)

    del weekly_sales

    df["quotient"] = df["count_targ"] / df["count"]
    info_df(df, count, order)

    target_sales = (
        df.drop("customer_id", axis=1).groupby("article_id")["quotient"].sum()
    )
    info_df(target_sales, count, order)

    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = [aid_to_token(article_id) for article_id in general_pred]
    general_pred_str = " ".join(general_pred)

    if general_pred_str.strip() == "" or len(general_pred_str.split()) < N:
        general_pred_str = global_fallback

    del target_sales

    tmp = df.copy().to_pandas()
    info_df(tmp, count, order)

    if "customer_id_2" not in tmp.columns:
        tmp["customer_id_2"] = (
            tmp["customer_id"].astype(str).map(cust_code_map).fillna(-1).astype("int64")
        )

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

    tmp = tmp.groupby(["customer_id_2", "article_id"]).agg({"value": "sum"})
    info_df(tmp, count, order)

    tmp = tmp.reset_index()
    tmp = tmp.loc[tmp["value"] > 0]
    info_df(tmp, count, order)

    tmp["rank"] = tmp.groupby("customer_id_2")["value"].rank("dense", ascending=False)
    info_df(tmp, count, order)

    tmp = tmp.loc[tmp["rank"] <= 12]
    info_df(tmp, count, order)

    purchase_df = tmp.sort_values(
        ["customer_id_2", "value"], ascending=False
    ).reset_index(drop=True)
    info_df(purchase_df, count, order)

    purchase_df["prediction"] = purchase_df["article_id"].apply(aid_to_token)
    info_df(purchase_df, count, order)

    purchase_pd_tokens = purchase_df[["customer_id_2", "prediction"]]
    if hasattr(purchase_pd_tokens, "to_pandas"):
        purchase_pd_tokens = purchase_pd_tokens.to_pandas()

    purchase_pd = (
        purchase_pd_tokens.groupby("customer_id_2")["prediction"]
        .apply(lambda s: " ".join(pd.unique(s.astype(str)).tolist()))
        .reset_index()
    )
    info_df(purchase_pd, count, order)

    bin_frame = final_sub.loc[mask_bin, ["customer_id_2"]].merge(
        purchase_pd, on="customer_id_2", how="left"
    )
    filled_bin_pred = (
        bin_frame["prediction"]
        .fillna(general_pred_str)
        .apply(lambda s: normalize_pred_string(s, general_pred_str))
    )
    final_sub.loc[mask_bin, "prediction"] = filled_bin_pred.values

    print(
        f"FILLED BIN {bin_unique}: customers_in_bin={int(mask_bin.sum())}, general_pred_len={len(general_pred_str.split())}"
    )
    print("-" * 50)

    count = 0

print("FINISHED")
print("=" * 50)



## --- ERROR in cell 26, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2849223604.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     17[0m     [0mdf_all[0m[0;34m[[0m[0;34m"customer_id"[0m[0;34m][0m[0;34m.[0m[0mto_pandas[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mcust_code_map[0m[0;34m)[0m[0;34m.[0m[0mfillna[0m[0;34m([0m[0;34m-[0m[0;36m1[0m[0;34m)[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m"int64"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m )
[0;32m---> 19[0;31m [0mdf_all[0m [0;34m=[0m [0mcudf[0m[0;34m.[0m[0mfrom_pandas[0m[0;34m([0m[0mdf_all[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m [0minfo_df[0m[0;34m([0m[0mdf_all[0m[0;34m,[0m [0mCOUNT[0m[0;34m,[0m [0mORDER[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m     49[0m                     )
[1;32m     50[0m                 )
[0;32m---> 51[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     52[0m [0;34m[0m[0m
[1;32m     53[0m     [0;32mreturn[0m [0mwrapper[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/cudf/core/dataframe.py[0m in [0;36mfrom_pandas[0;34m(obj, nan_as_null)[0m
[1;32m   8442[0m         [0;32mreturn[0m [0mcudf[0m[0;34m.[0m[0mIntervalDtype[0m[0;34m.[0m[0mfrom_pandas[0m[0;34m([0m[0mobj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   8443[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 8444[0;31m         raise TypeError(
[0m[1;32m   8445[0m             [0;34mf"from_pandas unsupported for object of type {type(obj).__name__}"[0m[0;34m[0m[0;34m[0m[0m
[1;32m   8446[0m         )

[0;31mTypeError[0m: from_pandas unsupported for object of type DataFrame

## === cell 27
final_sub["prediction"] = final_sub["prediction"].apply(
    lambda s: normalize_pred_string(s, global_fallback)
)

assert (
    final_sub.shape[0] == num_customers
), f"Final rows mismatch {final_sub.shape[0]} vs {num_customers}"
assert final_sub["customer_id"].is_unique, "Duplicate customer_id in final submission"

final_out = final_sub[["customer_id", "prediction"]]
final_out.to_csv("submission.csv", index=False)

print(final_out.head())
print("WROTE: submission.csv")
