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

bin_pred_parts = []

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
        temp_customers_df = customers_df[customers_df["age_bins"].isnull()]
    else:
        temp_customers_df = customers_df[customers_df["age_bins"] == bin_unique]

    temp_customers_df = temp_customers_df.drop(["age_bins"], axis=1)
    temp_customers_df = cudf.from_pandas(temp_customers_df)
    info_df(temp_customers_df, count, order)

    df = df.merge(
        temp_customers_df[["customer_id", "age"]], on="customer_id", how="inner"
    )
    info_df(df, count, order)

    print(f"TRANSACTION SHAPE FOR SCOPE {bin_unique}: {df.shape}\n")

    df["customer_id"] = df["customer_id"].str[-16:].str.hex_to_int().astype("int64")
    df["t_dat"] = cudf.to_datetime(df["t_dat"])

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
    general_pred = [_fmt_article_id(article_id) for article_id in general_pred]
    general_pred_str = " ".join(general_pred)

    del target_sales

    tmp = df.copy().to_pandas()
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

    purchase_df["prediction"] = purchase_df["article_id"].map(_fmt_article_id) + " "
    info_df(purchase_df, count, order)

    purchase_df = (
        purchase_df.groupby("customer_id").agg({"prediction": sum}).reset_index()
    )
    info_df(purchase_df, count, order)

    purchase_df = cudf.DataFrame(purchase_df)
    purchase_df["age_bins"] = str(bin_unique)
    purchase_df["general_pred_str"] = general_pred_str

    bin_pred_parts.append(
        purchase_df[["customer_id", "prediction", "age_bins", "general_pred_str"]]
    )

    purchase_df.to_pandas().to_csv(
        f"purchase_predictions_{str(bin_unique)}.csv", index=False
    )

    print(
        f"BUILT PREDICTIONS FOR {bin_unique}, unique customers: {purchase_df.shape[0]}\n"
    )
    print("-" * 50)

    count = 0

print("FINISHED BIN MODELS")
print("=" * 50)



## --- ERROR in cell 25, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mUFuncTypeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4034813548.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    158[0m [0;34m[0m[0m
[1;32m    159[0m     [0;31m# Change (score-critical, minimal): ensure prediction tokens are valid 10-digit article ids.[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 160[0;31m     [0mpurchase_df[0m[0;34m[[0m[0;34m"prediction"[0m[0;34m][0m [0;34m=[0m [0mpurchase_df[0m[0;34m[[0m[0;34m"article_id"[0m[0;34m][0m[0;34m.[0m[0mmap[0m[0;34m([0m[0m_fmt_article_id[0m[0;34m)[0m [0;34m+[0m [0;34m" "[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    161[0m     [0minfo_df[0m[0;34m([0m[0mpurchase_df[0m[0;34m,[0m [0mcount[0m[0;34m,[0m [0morder[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    162[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py[0m in [0;36mnew_method[0;34m(self, other)[0m
[1;32m     74[0m         [0mother[0m [0;34m=[0m [0mitem_from_zerodim[0m[0;34m([0m[0mother[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     75[0m [0;34m[0m[0m
[0;32m---> 76[0;31m         [0;32mreturn[0m [0mmethod[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mother[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     77[0m [0;34m[0m[0m
[1;32m     78[0m     [0;32mreturn[0m [0mnew_method[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py[0m in [0;36m__add__[0;34m(self, other)[0m
[1;32m    184[0m         [0mmoose[0m     [0;36m3.0[0m     [0mNaN[0m[0;34m[0m[0;34m[0m[0m
[1;32m    185[0m         """
[0;32m--> 186[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_arith_method[0m[0;34m([0m[0mother[0m[0;34m,[0m [0moperator[0m[0;34m.[0m[0madd[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    187[0m [0;34m[0m[0m
[1;32m    188[0m     [0;34m@[0m[0munpack_zerodim_and_defer[0m[0;34m([0m[0;34m"__radd__"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/series.py[0m in [0;36m_arith_method[0;34m(self, other, op)[0m
[1;32m   6133[0m     [0;32mdef[0m [0m_arith_method[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mother[0m[0;34m,[0m [0mop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6134[0m         [0mself[0m[0;34m,[0m [0mother[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_align_for_op[0m[0;34m([0m[0mother[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6135[0;31m         [0;32mreturn[0m [0mbase[0m[0;34m.[0m[0mIndexOpsMixin[0m[0;34m.[0m[0m_arith_method[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mother[0m[0;34m,[0m [0mop[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6136[0m [0;34m[0m[0m
[1;32m   6137[0m     [0;32mdef[0m [0m_align_for_op[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mright[0m[0;34m,[0m [0malign_asobject[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/base.py[0m in [0;36m_arith_method[0;34m(self, other, op)[0m
[1;32m   1380[0m [0;34m[0m[0m
[1;32m   1381[0m         [0;32mwith[0m [0mnp[0m[0;34m.[0m[0merrstate[0m[0;34m([0m[0mall[0m[0;34m=[0m[0;34m"ignore"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1382[0;31m             [0mresult[0m [0;34m=[0m [0mops[0m[0;34m.[0m[0marithmetic_op[0m[0;34m([0m[0mlvalues[0m[0;34m,[0m [0mrvalues[0m[0;34m,[0m [0mop[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1383[0m [0;34m[0m[0m
[1;32m   1384[0m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_construct_result[0m[0;34m([0m[0mresult[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0mres_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py[0m in [0;36marithmetic_op[0;34m(left, right, op)[0m
[1;32m    281[0m         [0;31m# error: Argument 1 to "_na_arithmetic_op" has incompatible type[0m[0;34m[0m[0;34m[0m[0m
[1;32m    282[0m         [0;31m# "Union[ExtensionArray, ndarray[Any, Any]]"; expected "ndarray[Any, Any]"[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 283[0;31m         [0mres_values[0m [0;34m=[0m [0m_na_arithmetic_op[0m[0;34m([0m[0mleft[0m[0;34m,[0m [0mright[0m[0;34m,[0m [0mop[0m[0;34m)[0m  [0;31m# type: ignore[arg-type][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    284[0m [0;34m[0m[0m
[1;32m    285[0m     [0;32mreturn[0m [0mres_values[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py[0m in [0;36m_na_arithmetic_op[0;34m(left, right, op, is_cmp)[0m
[1;32m    216[0m [0;34m[0m[0m
[1;32m    217[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 218[0;31m         [0mresult[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mleft[0m[0;34m,[0m [0mright[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    219[0m     [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    220[0m         if not is_cmp and (

[0;31mUFuncTypeError[0m: ufunc 'add' did not contain a loop with signature matching types (dtype('int32'), dtype('<U1')) -> None

## === cell 26
if len(bin_pred_parts) == 0:
    raise RuntimeError("No per-bin predictions were built; cannot create submission.")

pred_all = cudf.concat(bin_pred_parts, axis=0)

sub_full_pd = sub_full.to_pandas()
sub_full_pd["age_bins_key"] = sub_full_pd["age_bins"].astype(str)
sub_full = cudf.from_pandas(
    sub_full_pd[["customer_id", "customer_id_2", "age_bins_key"]]
)

pred_all["age_bins_key"] = pred_all["age_bins"].astype("str")

sub_joined = sub_full.merge(
    pred_all,
    left_on=["customer_id_2", "age_bins_key"],
    right_on=["customer_id", "age_bins_key"],
    how="left",
    suffixes=("", "_pred"),
)

sub_joined = sub_joined.to_pandas()
display_df(sub_joined)
