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
import sys, warnings, time, os, copy, gc, re, random, pickle, cudf

warnings.filterwarnings("ignore")
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
def display_df(df, head=3):
    print(f"The shape of df is {df.shape}.\n")
    display(df.head(head))




## === cell 3
dfArticles = cudf.read_csv(
    PATH_INPUT + "articles.csv",
    usecols=["article_id", "product_group_name", "perceived_colour_master_name"],
)
display_df(dfArticles, head=3)



## === cell 4
dfCustomers = cudf.read_csv(
    PATH_INPUT + "customers.csv", usecols=["customer_id", "age"]
)
display_df(dfCustomers, head=3)



## === cell 5
dfCustomers = dfCustomers.to_pandas()
listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
dfCustomers["age_bins"] = pd.cut(dfCustomers["age"], listBin)
display_df(dfCustomers, head=3)



## === cell 6
x = dfCustomers[dfCustomers["age_bins"].isnull()].shape[0]
print(f"{x} customer_id do not have age information.\n")



## === cell 7
dfTransactions = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"article_id": "int32", "t_dat": "string", "customer_id": "string"},
)
dfTransactions["t_dat"] = cudf.to_datetime(dfTransactions["t_dat"])
dfTransactions.set_index("t_dat", inplace=True)
display_df(dfTransactions, head=3)



## === cell 8
dfTransactions = dfTransactions.sort_index()

start = cudf.to_datetime("2020-09-01")
end = cudf.to_datetime("2020-09-21")
dfRecent = dfTransactions[
    (dfTransactions.index >= start) & (dfTransactions.index <= end)
]

display_df(dfRecent, head=3)



## === cell 9
dfRecent = dfRecent.to_pandas()
dfRecent = dfRecent.merge(
    dfCustomers[["customer_id", "age_bins"]], on="customer_id", how="inner"
)
display_df(dfRecent, head=3)



## === cell 10
dfRecent = (
    dfRecent.groupby(["age_bins", "article_id"])
    .count()
    .reset_index()
    .rename(columns={"customer_id": "counts"})
)

listUniBins = dfRecent["age_bins"].unique().tolist()

dict100 = {}
for uniBin in listUniBins:
    dfTemp = dfRecent[dfRecent["age_bins"] == uniBin]
    dfTemp = dfTemp.sort_values(by="counts", ascending=False)
    dict100[uniBin] = dfTemp.head(100)["article_id"].values.tolist()

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
listUniBins = dfCustomers["age_bins"].unique().tolist()



## === cell 13
sub_all = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
sub_all["customer_id2"] = (
    sub_all["customer_id"].str[-16:].str.hex_to_int().astype("int64")
)
sub_all["prediction"] = cudf.Series([""] * sub_all.shape[0])

numCustomers = sub_all.shape[0]
print(f"Loaded sample_submission with {numCustomers} customers.\n")



## === cell 14
for uniBin in listUniBins:
    df = cudf.read_csv(
        PATH_INPUT + "transactions_train.csv",
        usecols=["t_dat", "customer_id", "article_id"],
        dtype={"article_id": "int32", "t_dat": "string", "customer_id": "string"},
    )
    if str(uniBin) == "nan":
        dfCustomersTemp_pd = dfCustomers[dfCustomers["age_bins"].isnull()].copy()
    else:
        dfCustomersTemp_pd = dfCustomers[dfCustomers["age_bins"] == uniBin].copy()

    dfCustomersTemp = cudf.from_pandas(dfCustomersTemp_pd[["customer_id", "age"]])

    df = df.merge(
        dfCustomersTemp[["customer_id", "age"]], on="customer_id", how="inner"
    )
    print(f"The shape of scope transaction for {uniBin} is {df.shape}. \n")

    if df.shape[0] == 0:
        print(
            f"No transactions for {uniBin}; skipping model for this bin.\n" + "-" * 50
        )
        continue

    df["customer_id"] = df["customer_id"].str[-16:].str.hex_to_int().astype("int64")
    df["t_dat"] = cudf.to_datetime(df["t_dat"])
    last_ts = df["t_dat"].max()

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

    target_sales = (
        df.drop("customer_id", axis=1).groupby("article_id")["quotient"].sum()
    )
    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = ["0" + str(article_id) for article_id in general_pred]
    general_pred_str = " ".join(general_pred)
    del target_sales

    tmp = df.copy().to_pandas()
    tmp["x"] = ((last_ts - tmp["t_dat"]) / np.timedelta64(1, "D")).astype(int)
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
    purchase_df["prediction"] = "0" + purchase_df["article_id"].astype(str) + " "
    purchase_df = (
        purchase_df.groupby("customer_id").agg({"prediction": sum}).reset_index()
    )
    purchase_df["prediction"] = purchase_df["prediction"].str.strip()
    purchase_df = cudf.DataFrame(purchase_df)

    cust_ids_in_bin = cudf.from_pandas(dfCustomersTemp_pd[["customer_id"]])
    cust_ids_in_bin["customer_id2"] = (
        cust_ids_in_bin["customer_id"].str[-16:].str.hex_to_int().astype("int64")
    )

    sub_bin = cust_ids_in_bin.merge(
        purchase_df,
        left_on="customer_id2",
        right_on="customer_id",
        how="left",
        suffixes=("", "_ignored"),
    )[["customer_id2", "prediction"]]

    sub_bin = sub_bin.to_pandas()
    sub_bin["prediction"] = sub_bin["prediction"].fillna(general_pred_str)
    sub_bin["prediction"] = (sub_bin["prediction"] + " " + general_pred_str).str.strip()
    sub_bin["prediction"] = sub_bin["prediction"].str[
        :131
    ]  # keep your truncation behavior
    sub_bin = cudf.from_pandas(sub_bin)

    sub_all = sub_all.merge(
        sub_bin, on="customer_id2", how="left", suffixes=("", "_new")
    )
    sub_all["prediction"] = cudf.Series(
        np.where(
            (sub_all["prediction_new"].isnull()) | (sub_all["prediction_new"] == ""),
            sub_all["prediction"],
            sub_all["prediction_new"],
        )
    )
    sub_all = sub_all.drop(columns=["prediction_new"])

    print(
        f"Updated prediction for {uniBin}. Customers in bin: {sub_bin.shape[0]}.\n"
        + "-" * 50
    )

print("Finished.\n")
print("=" * 50)



## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/777023448.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    128[0m     )
[1;32m    129[0m     sub_all["prediction"] = cudf.Series(
[0;32m--> 130[0;31m         np.where(
[0m[1;32m    131[0m             [0;34m([0m[0msub_all[0m[0;34m[[0m[0;34m"prediction_new"[0m[0;34m][0m[0;34m.[0m[0misnull[0m[0;34m([0m[0;34m)[0m[0;34m)[0m [0;34m|[0m [0;34m([0m[0msub_all[0m[0;34m[[0m[0;34m"prediction_new"[0m[0;34m][0m [0;34m==[0m [0;34m""[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    132[0m             [0msub_all[0m[0;34m[[0m[0;34m"prediction"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: no implementation found for 'numpy.where' on types that implement __array_function__: [<class 'cudf.core.series.Series'>]

## === cell 15
if (sub_all["prediction"] == "").any():
    df_all = cudf.read_csv(
        PATH_INPUT + "transactions_train.csv",
        usecols=["t_dat", "article_id"],
        dtype={"article_id": "int32", "t_dat": "string"},
    )
    df_all["t_dat"] = cudf.to_datetime(df_all["t_dat"])
    last_ts_all = df_all["t_dat"].max()

    tmp_all = df_all[["t_dat"]].copy().to_pandas()
    tmp_all["dow"] = tmp_all["t_dat"].dt.dayofweek
    tmp_all["ldbw"] = tmp_all["t_dat"] - pd.TimedeltaIndex(tmp_all["dow"] - 1, unit="D")
    tmp_all.loc[tmp_all["dow"] >= 2, "ldbw"] = tmp_all.loc[
        tmp_all["dow"] >= 2, "ldbw"
    ] + pd.TimedeltaIndex(np.ones(len(tmp_all.loc[tmp_all["dow"] >= 2])) * 7, unit="D")
    df_all["ldbw"] = tmp_all["ldbw"].values

    weekly_sales_all = (
        df_all.groupby(["ldbw", "article_id"])
        .count()
        .reset_index()
        .rename(columns={"t_dat": "count"})
    )
    last_ldbw_all = df_all["ldbw"].max()

    df_all = df_all.merge(weekly_sales_all, on=["ldbw", "article_id"], how="left")
    weekly_sales_all = weekly_sales_all.reset_index().set_index("article_id")
    df_all = df_all.merge(
        weekly_sales_all.loc[weekly_sales_all["ldbw"] == last_ldbw_all, ["count"]],
        on="article_id",
        suffixes=("", "_targ"),
    )
    df_all["count_targ"].fillna(0, inplace=True)

    df_all["quotient"] = df_all["count_targ"] / df_all["count"]
    target_sales_all = df_all.groupby("article_id")["quotient"].sum()
    general_pred_all = target_sales_all.nlargest(N).index.to_pandas().tolist()
    general_pred_all = ["0" + str(article_id) for article_id in general_pred_all]
    general_pred_all_str = " ".join(general_pred_all)

    sub_all["prediction"] = cudf.Series(
        np.where(
            (sub_all["prediction"] == "") | (sub_all["prediction"].isnull()),
            general_pred_all_str,
            sub_all["prediction"],
        )
    )

sub_out = sub_all[["customer_id", "prediction"]]
sub_out.to_csv("submission.csv", index=False)
print("Saved submission.csv.")
