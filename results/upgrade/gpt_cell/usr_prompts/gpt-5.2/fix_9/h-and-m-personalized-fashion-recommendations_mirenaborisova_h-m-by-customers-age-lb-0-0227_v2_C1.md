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

- What this solution (achieved 0.0) has done: 'Your pipeline currently doesn’t reliably yield a valid single submission because it writes one CSV per age bin, then concatenates them (creating duplicate `customer_id`s) and asserts only on row count, not uniqueness or alignment to `sample_submission`. To move the score toward the target (and to actually get a scorable submission), I keep your scoring logic intact but change the assembly step to fill predictions directly into one `sample_submission`-indexed table, only for customers in each bin, and write exactly one final `submission.csv`. I also ensure predictions are truncated by *number of items* (top 12) rather than by string length, which can silently cut an `article_id` and invalidate ranks, hurting MAP@12. Finally, I avoid repeatedly re-reading the huge transactions file inside the loop (same data, same logic), which helps the code finish within the time limit without changing the model logic.'
- What this solution (achieved 0.0) has done: 'The crash happens because `df_base["customer_id"]` was converted to an `int64` (hex_to_int) in cell 25, but inside the loop in cell 26 the `temp_customers_df["customer_id"]` remains a string (original customer_id) so `cudf.merge` tries to coerce the join keys to a common dtype and fails with a string-to-float conversion error. The minimal fix is to create a matching numeric join key (`customer_id_2`) in `temp_customers_df` using the same conversion as cell 25, and then merge `df_base` on that key. This keeps the logic identical (restricting transactions to customers in the bin) while making join key dtypes compatible. No other logic, ranking, or prediction formatting is changed.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an invalid/ineffective submission caused by silent key misalignment in the per-bin fill step: `sub_bin` merges `purchase_df` using `customer_id_2` vs `customer_id` without ensuring both sides share the same dtype/key, which can yield mostly-null predictions and collapse MAP@12. I keep your model/scoring logic intact, but make the join keys explicit and consistent by carrying `customer_id_2` through `purchase_df` (computed from the string `customer_id`) and joining on that numeric key. I also ensure `customer_id` stays as the sample submission’s string id in the final output (only `customer_id_2` is numeric), preserving the required submission schema. These are minimal, execution-safe changes that should move the score upward toward the target by actually populating personalized predictions for each bin.'
- What this solution (achieved 0.0) has done: 'The crash happens because after merging `sub_bin` with `purchase_df`, cuDF ends up with a renamed prediction column (typically `prediction_x`/`prediction_y`) due to the existing placeholder `prediction` column already present in `sub_bin`. Then selecting `sub_bin[["customer_id","prediction"]]` raises `KeyError: 'prediction'`. In cell 26, keep the same merge logic but explicitly normalize the post-merge column names so there is always a `prediction` column (preferring the newly merged prediction) and drop any temporary prediction columns. This is a minimal, localized fix and preserves downstream behavior in cell 27 which expects `sub_all` to have a `prediction` column.'
- What this solution (achieved 0.0) has done: 'We make two minimal, score-relevant fixes that preserve your scoring logic: (1) ensure the per-bin filling step actually matches customers by joining on `customer_id_2` (your current code merges on string `customer_id` vs numeric `customer_id_2`, which can silently leave most predictions empty and yield ~0 MAP), and (2) make the “general fallback” list consistent by computing `last_date_train` once globally (so each bin uses the correct final training week rather than its own truncated subset). These changes keep the same model/weighting/ranking logic, but fix key alignment so personalized predictions populate the submission. The output remains a single valid `submission.csv` aligned to `sample_submission` with exactly 12 IDs per row.'

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

start_dt = cudf.to_datetime("2020-09-01")
end_dt = cudf.to_datetime("2020-09-21")
recent_df = transactions_df[
    (transactions_df.index >= start_dt) & (transactions_df.index <= end_dt)
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
sub_all = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
num_customers = sub_all.shape[0]
sub_all["prediction"] = ""  # placeholder to be filled
info_df(sub_all, COUNT, ORDER)

df_base = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
)

df_base["t_dat"] = cudf.to_datetime(df_base["t_dat"])

global_last_date_train = df_base["t_dat"].max()

df_base["customer_id"] = (
    df_base["customer_id"].str[-16:].str.hex_to_int().astype("int64")
)

sub_all["customer_id_2"] = (
    sub_all["customer_id"].str[-16:].str.hex_to_int().astype("int64")
)



## === cell 26
count = COUNT
order = ORDER

for bin_unique in bins_unique_list:
    if str(bin_unique) == "nan":
        temp_customers_pd = customers_df[customers_df["age_bins"].isnull()][
            ["customer_id", "age"]
        ]
    else:
        temp_customers_pd = customers_df[customers_df["age_bins"] == bin_unique][
            ["customer_id", "age"]
        ]

    temp_customers_df = cudf.from_pandas(temp_customers_pd)

    temp_customers_df["customer_id_2"] = (
        temp_customers_df["customer_id"].str[-16:].str.hex_to_int().astype("int64")
    )

    info_df(temp_customers_df, count, order)

    df = df_base.merge(
        temp_customers_df[["customer_id_2"]],
        left_on="customer_id",
        right_on="customer_id_2",
        how="inner",
    ).drop(columns=["customer_id_2"])
    info_df(df, count, order)

    print(f"TRANSACTION SHAPE FOR SCOPE {bin_unique}: {df.shape}\n")

    last_date_train = global_last_date_train
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
    general_pred = ["0" + str(article_id) for article_id in general_pred]
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

    purchase_df["article_str"] = "0" + purchase_df["article_id"].astype("str")
    purchase_df = (
        purchase_df.groupby("customer_id").agg({"article_str": list}).reset_index()
    )
    purchase_df["prediction"] = purchase_df["article_str"].apply(
        lambda x: " ".join(x[:12])
    )

    purchase_df = purchase_df.rename(columns={"customer_id": "customer_id_2"})
    purchase_df = purchase_df[["customer_id_2", "prediction"]]
    purchase_df = cudf.DataFrame(purchase_df)
    info_df(purchase_df, count, order)

    sub_bin = sub_all.merge(
        temp_customers_df[["customer_id_2"]],
        on="customer_id_2",
        how="inner",
    )
    info_df(sub_bin, count, order)

    sub_bin = sub_bin.merge(
        purchase_df,
        on="customer_id_2",
        how="left",
    )

    if "prediction" not in sub_bin.columns:
        if "prediction_y" in sub_bin.columns:
            sub_bin["prediction"] = sub_bin["prediction_y"]
        elif "prediction_x" in sub_bin.columns:
            sub_bin["prediction"] = sub_bin["prediction_x"]
    for _col in ("prediction_x", "prediction_y"):
        if _col in sub_bin.columns:
            sub_bin = sub_bin.drop(columns=[_col])

    info_df(sub_bin, count, order)

    sub_bin_pd = sub_bin[["customer_id", "prediction"]].to_pandas()
    sub_bin_pd["prediction"] = sub_bin_pd["prediction"].fillna("")
    sub_bin_pd["prediction"] = (
        sub_bin_pd["prediction"].str.strip() + " " + general_pred_str
    ).str.strip()

    sub_bin_pd["prediction"] = sub_bin_pd["prediction"].apply(
        lambda s: " ".join(str(s).split()[:12])
    )
    info_df(sub_bin_pd, count, order)

    sub_bin_filled = cudf.from_pandas(sub_bin_pd)
    sub_all = sub_all.merge(
        sub_bin_filled, on="customer_id", how="left", suffixes=("", "_new")
    )
    sub_all["prediction"] = sub_all["prediction_new"].fillna(sub_all["prediction"])
    sub_all = sub_all.drop(columns=["prediction_new"])

    print(f"FILLED PREDICTION FOR {bin_unique}, BIN_CUSTOMERS: {sub_bin_pd.shape[0]}\n")
    print("-" * 50)

    count = 0

print("FINISHED")
print("=" * 50)



## === cell 27
try:
    global_top = (
        recent_df.groupby("article_id")["counts"]
        .sum()
        .sort_values(ascending=False)
        .head(N)
        .index.tolist()
        if "counts" in recent_df.columns
        else recent_df["article_id"].value_counts().head(N).index.tolist()
    )
    global_top = ["0" + str(a) for a in global_top]
except Exception:
    tmp_global = df_base[["article_id"]].to_pandas()
    global_top = tmp_global["article_id"].value_counts().head(N).index.tolist()
    global_top = ["0" + str(a) for a in global_top]

global_pred_str = " ".join(global_top)

sub_all_pd = sub_all[["customer_id", "prediction"]].to_pandas()
sub_all_pd["prediction"] = sub_all_pd["prediction"].fillna("").astype(str)
sub_all_pd.loc[sub_all_pd["prediction"].str.strip() == "", "prediction"] = (
    global_pred_str
)
sub_all_pd["prediction"] = sub_all_pd["prediction"].apply(
    lambda s: " ".join(str(s).split()[:12])
)

assert (
    sub_all_pd.shape[0] == num_customers
), f"Row mismatch: {sub_all_pd.shape[0]} vs {num_customers}"
assert (
    sub_all_pd["customer_id"].nunique() == num_customers
), "Duplicate customer_id found in final submission."

sub_all_pd.to_csv("submission.csv", index=False)
display_df(sub_all_pd)



## === cell 28
check_df = cudf.read_csv("./submission.csv")
display_df(check_df)
print(
    "submission.csv saved, rows:",
    check_df.shape[0],
    "unique customers:",
    check_df["customer_id"].nunique(),
)
