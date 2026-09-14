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

0.02249

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I (1) fix the pandas import error by using the supported `pd.json_normalize`/fallback, (2) fix the cuDF datetime slicing KeyError by sorting the datetime index before label-based slicing, and (3) fix the invalid submission issue by ensuring we generate predictions for every `customer_id` in `sample_submission.csv` exactly once (including customers with missing ages) and then left-merge per-bin predictions back onto the full sample list. I keep your scoring logic intact, but make the post-processing safe: fill missing predictions, trim to 12 items, and guarantee the final CSV has the required two columns and correct customer coverage. These changes are execution-unblocking and submission-correctness focused; they should also prevent silent row loss that was hurting score/validity.'

# 9. Code solution

## === cell 0
import sys, warnings, time, os, copy, gc, re, random, pickle

warnings.filterwarnings("ignore")

import cudf
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

try:
    from pandas import json_normalize  # noqa: F401
except Exception:
    json_normalize = pd.json_normalize  # noqa: F401

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
dfTransactions = dfTransactions.set_index("t_dat")
dfTransactions = dfTransactions.sort_index()
display_df(dfTransactions, head=3)



## === cell 8
start_dt = np.datetime64("2020-09-01")
end_dt = np.datetime64("2020-09-21")
dfRecent = dfTransactions[
    (dfTransactions.index >= start_dt) & (dfTransactions.index <= end_dt)
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

listUniBins_recent = dfRecent["age_bins"].unique().tolist()

dict100 = {}
for uniBin in listUniBins_recent:
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
def _fmt_article_id(a) -> str:
    try:
        return f"{int(a):010d}"
    except Exception:
        s = str(a)
        return s.zfill(10) if s.isdigit() else s


def _trim_to_12(pred_str: str) -> str:
    if pred_str is None or (isinstance(pred_str, float) and np.isnan(pred_str)):
        return ""
    toks = str(pred_str).split()
    return " ".join(toks[:12])


def _as_pd_timestamp(x):
    return pd.Timestamp(x)


def _bin_key(uni_bin) -> str:
    if uni_bin is None:
        return "AGE_NA"
    s = str(uni_bin)
    if s.lower() == "nan":
        return "AGE_NA"
    m = re.findall(r"-?\d+", s)
    if len(m) >= 2:
        return f"AGE_{m[0]}_{m[1]}"
    if len(m) == 1:
        return f"AGE_{m[0]}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_") or "AGE_UNKNOWN"


df_all_tx = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"article_id": "int32", "t_dat": "string", "customer_id": "string"},
)
df_all_tx["t_dat"] = cudf.to_datetime(df_all_tx["t_dat"])

sub_all = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
numCustomers = int(sub_all.shape[0])

dfCustomers_bins = dfCustomers[["customer_id", "age", "age_bins"]].copy()

bins_to_run = list(listUniBins)
bins_to_run.append(np.nan)

written_files = []

for uniBin in bins_to_run:
    df = df_all_tx  # reuse in-memory, then merge-filter to scoped customers

    if (uniBin is np.nan) or (str(uniBin).lower() == "nan"):
        dfCustomersTemp_pd = dfCustomers_bins[
            dfCustomers_bins["age_bins"].isnull()
        ].copy()
    else:
        dfCustomersTemp_pd = dfCustomers_bins[
            dfCustomers_bins["age_bins"] == uniBin
        ].copy()

    file_key = _bin_key(uniBin)
    out_file = f"submission_{file_key}.csv"

    if dfCustomersTemp_pd.shape[0] == 0:
        pd.DataFrame({"customer_id": [], "prediction": []}).to_csv(
            out_file, index=False
        )
        written_files.append(out_file)
        print(f"No customers for bin {uniBin}; wrote empty {out_file}\n" + "-" * 50)
        continue

    dfCustomersTemp_cu = cudf.from_pandas(dfCustomersTemp_pd[["customer_id", "age"]])

    df_scoped = df.merge(
        dfCustomersTemp_cu[["customer_id", "age"]], on="customer_id", how="inner"
    )
    print(f"The shape of scope transaction for {uniBin} is {df_scoped.shape}. \n")

    df_scoped["customer_id_int"] = (
        df_scoped["customer_id"].str[-16:].str.hex_to_int().astype("int64")
    )
    last_ts = df_scoped["t_dat"].max()
    last_ts_pd = _as_pd_timestamp(last_ts)

    tmp = df_scoped[["t_dat"]].copy().to_pandas()
    tmp["dow"] = tmp["t_dat"].dt.dayofweek
    tmp["ldbw"] = tmp["t_dat"] - pd.TimedeltaIndex(tmp["dow"] - 1, unit="D")
    tmp.loc[tmp["dow"] >= 2, "ldbw"] = tmp.loc[
        tmp["dow"] >= 2, "ldbw"
    ] + pd.TimedeltaIndex(np.ones(len(tmp.loc[tmp["dow"] >= 2])) * 7, unit="D")
    df_scoped["ldbw"] = tmp["ldbw"].values

    weekly_sales = (
        df_scoped.drop("customer_id_int", axis=1)
        .drop("customer_id", axis=1)
        .groupby(["ldbw", "article_id"])
        .count()
        .reset_index()
    )
    weekly_sales = weekly_sales.rename(columns={"t_dat": "count"})
    df_scoped = df_scoped.merge(weekly_sales, on=["ldbw", "article_id"], how="left")

    weekly_sales_pd = weekly_sales.to_pandas()
    last_ldbw = weekly_sales_pd["ldbw"].max()
    weekly_sales_last = weekly_sales_pd.loc[
        weekly_sales_pd["ldbw"] == last_ldbw, ["article_id", "count"]
    ]
    weekly_sales_last_cu = cudf.from_pandas(weekly_sales_last)

    df_scoped = df_scoped.merge(
        weekly_sales_last_cu,
        on="article_id",
        how="left",
        suffixes=("", "_targ"),
    )
    if "count_targ" not in df_scoped.columns:
        if "count_y" in df_scoped.columns:
            df_scoped = df_scoped.rename(columns={"count_y": "count_targ"})
        else:
            df_scoped["count_targ"] = 0
    df_scoped["count_targ"] = df_scoped["count_targ"].fillna(0)

    df_scoped["quotient"] = df_scoped["count_targ"] / df_scoped["count"]

    target_sales = (
        df_scoped.drop("customer_id_int", axis=1)
        .drop("customer_id", axis=1)
        .groupby("article_id")["quotient"]
        .sum()
    )
    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = [_fmt_article_id(article_id) for article_id in general_pred]
    general_pred_str = " ".join(general_pred)
    del target_sales

    tmp = df_scoped.copy().to_pandas()
    tmp["x"] = ((last_ts_pd - tmp["t_dat"]) / np.timedelta64(1, "D")).astype(int)
    tmp["dummy_1"] = 1
    tmp["x"] = tmp[["x", "dummy_1"]].max(axis=1)

    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    tmp["y"] = a / np.sqrt(tmp["x"]) + b * np.exp(-c * tmp["x"]) - d
    tmp["dummy_0"] = 0
    tmp["y"] = tmp[["y", "dummy_0"]].max(axis=1)
    tmp["value"] = tmp["quotient"] * tmp["y"]

    tmp = (
        tmp.groupby(["customer_id_int", "article_id"])
        .agg({"value": "sum"})
        .reset_index()
    )
    tmp = tmp.loc[tmp["value"] > 0]
    tmp["rank"] = tmp.groupby("customer_id_int")["value"].rank("dense", ascending=False)
    tmp = tmp.loc[tmp["rank"] <= 12]

    purchase_df = tmp.sort_values(
        ["customer_id_int", "value"], ascending=False
    ).reset_index(drop=True)
    purchase_df["prediction"] = purchase_df["article_id"].map(_fmt_article_id) + " "
    purchase_df = (
        purchase_df.groupby("customer_id_int").agg({"prediction": sum}).reset_index()
    )
    purchase_df["prediction"] = purchase_df["prediction"].str.strip()
    purchase_df = cudf.DataFrame(purchase_df)

    sub_bin = sub_all.merge(
        dfCustomersTemp_cu[["customer_id", "age"]], on="customer_id", how="inner"
    )
    sub_bin["customer_id_int"] = (
        sub_bin["customer_id"].str[-16:].str.hex_to_int().astype("int64")
    )

    sub_bin = sub_bin.merge(purchase_df, on="customer_id_int", how="left")

    sub_bin_pd = sub_bin[["customer_id", "prediction"]].to_pandas()
    sub_bin_pd["prediction"] = sub_bin_pd["prediction"].fillna("")
    sub_bin_pd["prediction"] = (
        sub_bin_pd["prediction"].astype(str) + " " + general_pred_str
    ).str.strip()
    sub_bin_pd["prediction"] = sub_bin_pd["prediction"].map(_trim_to_12)

    sub_bin_pd.to_csv(out_file, index=False)
    written_files.append(out_file)
    print(
        f"Saved prediction for {uniBin} as {out_file}. The shape is {sub_bin_pd.shape}. \n"
    )
    print("-" * 50)

print("Finished.\n")
print("=" * 50)



## === cell 14
sub_full = pd.read_csv(PATH_INPUT + "sample_submission.csv", usecols=["customer_id"])
sub_full["prediction"] = ""  # will be filled from bins

for f in written_files:
    if not os.path.exists(f):
        continue
    dfTemp = pd.read_csv(f)
    if dfTemp.shape[0] == 0:
        continue
    sub_full = sub_full.merge(
        dfTemp, on="customer_id", how="left", suffixes=("", "_bin")
    )
    sub_full["prediction"] = sub_full["prediction"].mask(
        sub_full["prediction"].eq(""), sub_full["prediction_bin"].fillna("")
    )
    sub_full = sub_full.drop(columns=["prediction_bin"])

mask_empty = sub_full["prediction"].eq("")
if mask_empty.any():
    df_all = cudf.read_csv(
        PATH_INPUT + "transactions_train.csv",
        usecols=["article_id"],
        dtype={"article_id": "int32"},
    )
    top_global = df_all["article_id"].value_counts().head(12).index.to_pandas().tolist()
    top_global = " ".join([_fmt_article_id(a) for a in top_global])
    sub_full.loc[mask_empty, "prediction"] = top_global

sub_full["prediction"] = sub_full["prediction"].map(_trim_to_12)

assert (
    sub_full.shape[0] == numCustomers
), f"Row count mismatch: {sub_full.shape[0]} vs {numCustomers}"
assert (
    sub_full["customer_id"].nunique() == numCustomers
), "Duplicate customer_id detected in final submission."

sub_full.to_csv("submission.csv", index=False)
print("Saved submission.csv.")



## === cell 15
dfCheck = cudf.read_csv("./submission.csv")
display_df(dfCheck, head=3)
print(dfCheck.shape)
print(dfCheck.columns)
