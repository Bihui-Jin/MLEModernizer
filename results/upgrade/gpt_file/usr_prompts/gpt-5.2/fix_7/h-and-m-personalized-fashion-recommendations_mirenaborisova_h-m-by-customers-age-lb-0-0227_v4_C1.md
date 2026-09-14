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

- What this solution (achieved 0.0) has done: 'I fix the import error by using the supported `pandas.json_normalize` API so the notebook starts correctly. I also fix the cuDF datetime slicing bug by sorting the datetime index (cuDF requires a monotonic DatetimeIndex for label slicing), which restores the early exploratory cells without changing the later modeling logic. Finally, I fix the invalid submission by ensuring we generate predictions for the full `sample_submission` customer list: instead of inner-merging by age-bin and concatenating partial submissions (which drops customers), we build one full submission and fill each customer’s prediction from their age-bin model if available, otherwise fallback to a global default—keeping the core per-bin training/prediction logic intact while guaranteeing a valid `.csv` for Kaggle.'
- What this solution (achieved 0.0) has done: 'I fix the two runtime blockers: (1) the cuDF DatetimeIndex partial slicing in the exploratory cell by filtering with boolean masks (cuDF is strict about label slicing), and (2) the `to_pandas()` crash in the main training/prediction loop by ensuring we only call `to_pandas()` on cuDF objects and keeping the intermediate `purchase_df` as cuDF until conversion. I also ensure we always generate predictions for every `customer_id` in `sample_submission.csv` by keeping the existing “fill from bin model else fallback to global top” logic intact, so the submission is valid and no longer scores 0.0 due to missing customers. Changes are minimal and do not alter the core scoring logic; they only fix execution and submission completeness. The script write `submission.csv` in the working directory with the required columns and 1,371,980 rows.'
- What this solution (achieved 0.0) has done: 'You’re currently not yielding a score mainly because the script is unlikely to finish within Kaggle’s time/memory limits: it re-reads the 31.5M-row transactions file once per age-bin, which is extremely expensive. To move toward the target score with minimal semantic change, I keep the same per-age-bin scoring logic and ranking, but read `transactions_train.csv` only once, then reuse it by filtering per bin. I also avoid unnecessary cuDF↔pandas roundtrips inside the loop by doing the weekday/shift-day computation in cuDF directly (same computation, just executed on GPU), which preserves the core logic while making it feasible to complete and write a valid `submission.csv`. Finally, I keep your existing “fill all sample_submission customers; bin prediction if available else global fallback” behavior unchanged to ensure a valid submission.'
- What this solution (achieved 0.0) has done: 'I fix the runtime error by replacing the unsupported `cudf.to_timedelta` call with a cuDF-compatible timedelta creation so the per-age-bin training loop runs. I also remove a couple of unnecessary `reset_index()` calls that duplicate work and can blow up memory/time, without changing the underlying logic or ranking. Finally, I keep your “predict per age-bin then fill all sample_submission customers with a global fallback” behavior intact, ensuring the script always writes a valid `submission.csv` with the correct columns and full row count—so the score can move up from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the cuDF `timedelta64[D]` cast that currently crashes the per-age-bin loop by replacing it with cuDF-supported datetime arithmetic (same shift-day logic, just implemented via day offsets and `cudf.to_timedelta`). I also make the submission generation robust by ensuring the pipeline always reaches the final submission creation even if some bins produce no predictions, while keeping your existing “bin prediction + global fallback top-12” semantics unchanged. These changes are execution-focused and should move the score up from 0.0 by producing a valid, fully-populated `submission.csv` with non-empty predictions for every customer. No model/feature logic is changed beyond fixing the incompatible timedelta implementation.'

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

transactions_df = transactions_df.sort_index()

display_df(transactions_df)



## === cell 11
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
if topcustcnt_byage_df.shape[0] > 0 and topcustcnt_byage_df.shape[1] > 0:
    plt.figure(figsize=(10, 6))
    sns.heatmap(topcustcnt_byage_df, cmap="winter", annot=True, cbar=False)
else:
    print("Skipping heatmap: empty matrix")



## === cell 23
bins_unique_list



## === cell 24
bins_unique_list = customers_df["age_bins"].unique().tolist()
bins_unique_list



## === cell 25
sub_master = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
num_customers = sub_master.shape[0]

sub_master_pd = sub_master.to_pandas()
sub_master_pd = sub_master_pd.merge(
    customers_df[["customer_id", "age_bins"]], on="customer_id", how="left"
)

last_date_all = transactions_df.index.max()
start_date_all = last_date_all - np.timedelta64(7, "D")

tx_all = transactions_df.reset_index()[["t_dat", "customer_id", "article_id"]]
tx_all["t_dat"] = cudf.to_datetime(tx_all["t_dat"])

pop7 = (
    tx_all[tx_all["t_dat"] > start_date_all][["article_id"]]
    .value_counts()
    .reset_index()
)
pop7 = pop7.sort_values(by="count", ascending=False)
global_top = pop7.head(N)["article_id"].to_pandas().astype(str).str.zfill(10).tolist()
global_pred_str = " ".join(global_top)
del pop7
gc.collect()

bin_pred_maps = {}

count = COUNT
order = ORDER
for bin_unique in bins_unique_list:
    if str(bin_unique) == "nan":
        temp_customers_pd = customers_df[customers_df["age_bins"].isnull()][
            ["customer_id", "age"]
        ].copy()
    else:
        temp_customers_pd = customers_df[customers_df["age_bins"] == bin_unique][
            ["customer_id", "age"]
        ].copy()

    if temp_customers_pd.shape[0] == 0:
        bin_pred_maps[bin_unique] = {}
        continue

    temp_customers_df = cudf.from_pandas(temp_customers_pd)

    df = tx_all.merge(
        temp_customers_df[["customer_id", "age"]], on="customer_id", how="inner"
    )
    info_df(df, count, order)

    print(f"TRANSACTION SHAPE FOR SCOPE {bin_unique}: {df.shape}\n")

    if df.shape[0] == 0:
        bin_pred_maps[bin_unique] = {}
        del df, temp_customers_df, temp_customers_pd
        gc.collect()
        count = 0
        continue

    df["customer_id"] = df["customer_id"].str[-16:].str.hex_to_int().astype("int64")

    last_date_train = df["t_dat"].max()
    if count:
        print("=" * 30 + f"\n4 LAST_DATE_TRAIN:\n{last_date_train}")

    weekday = df["t_dat"].dt.weekday  # 0=Mon

    delta_days1 = (weekday - 1).astype("int32")
    delta1 = cudf.to_timedelta(delta_days1, unit="D")
    t_dat_shiftday = df["t_dat"] - delta1

    add7 = cudf.to_timedelta(cudf.Series(np.full(len(df), 7, dtype=np.int32)), unit="D")
    t_dat_shiftday = t_dat_shiftday.where(weekday < 2, t_dat_shiftday + add7)

    df["t_dat_shiftday"] = t_dat_shiftday
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
    gc.collect()

    df["quotient"] = df["count_targ"] / df["count"]
    info_df(df, count, order)

    target_sales = (
        df.drop("customer_id", axis=1).groupby("article_id")["quotient"].sum()
    )
    info_df(target_sales, count, order)

    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = [str(article_id).zfill(10) for article_id in general_pred]
    general_pred_str = " ".join(general_pred)

    del target_sales
    gc.collect()

    tmp = df.copy().to_pandas()
    info_df(tmp, count, order)

    tmp["x"] = (
        (last_date_train.to_pandas() - tmp["t_dat"]) / np.timedelta64(1, "D")
    ).astype(int)
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

    tmp = tmp.sort_values(["customer_id", "value"], ascending=False).reset_index(
        drop=True
    )
    tmp["pred_tok"] = tmp["article_id"].astype(np.int64).astype(str).str.zfill(10)

    purchase_pd = (
        tmp.groupby("customer_id")["pred_tok"]
        .apply(lambda s: " ".join(s.tolist()))
        .reset_index()
        .rename(columns={"pred_tok": "prediction"})
    )
    purchase_pd["prediction"] = purchase_pd["prediction"].astype(str).str.strip()
    purchase_pd["prediction"] = purchase_pd["prediction"].str[:131]

    temp_map_pd = temp_customers_pd[["customer_id"]].copy()
    temp_map_pd["customer_id_2"] = (
        temp_map_pd["customer_id"]
        .str[-16:]
        .apply(lambda x: int(x, 16))
        .astype(np.int64)
    )

    purchase_pd = purchase_pd.merge(
        temp_map_pd, left_on="customer_id", right_on="customer_id_2", how="left"
    )
    purchase_pd = (
        purchase_pd[["customer_id_y", "prediction"]]
        .rename(columns={"customer_id_y": "customer_id"})
        .dropna()
    )

    out_path = f"submission_{str(bin_unique)}.csv"
    purchase_pd[["customer_id", "prediction"]].to_csv(out_path, index=False)

    bin_pred_maps[bin_unique] = dict(
        zip(purchase_pd["customer_id"].values, purchase_pd["prediction"].values)
    )

    print(f"SAVED PREDICTION FOR {bin_unique}, SHAPE: {purchase_pd.shape}\n")
    print("-" * 50)

    del df, tmp, purchase_pd, temp_customers_df, temp_customers_pd, temp_map_pd
    gc.collect()

    count = 0

print("FINISHED")
print("=" * 50)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3588018694.py in <cell line: 0>()
     77     # then if weekday>=2 add 7 days
     78     delta_days1 = (weekday - 1).astype("int32")
---> 79     delta1 = cudf.to_timedelta(delta_days1, unit="D")
     80     t_dat_shiftday = df["t_dat"] - delta1
     81 

AttributeError: module 'cudf' has no attribute 'to_timedelta'

## === cell 26
final_pred = []

nan_bin_key = None
for k in bin_pred_maps.keys():
    if str(k) == "nan":
        nan_bin_key = k
        break

for cid, abin in zip(
    sub_master_pd["customer_id"].values, sub_master_pd["age_bins"].values
):
    pred = None

    if pd.isna(abin):
        if nan_bin_key is not None:
            pred = bin_pred_maps[nan_bin_key].get(cid)
    else:
        if abin in bin_pred_maps:
            pred = bin_pred_maps[abin].get(cid)

    if pred is None or not isinstance(pred, str) or pred.strip() == "":
        pred = global_pred_str
    else:
        pred = (pred.strip() + " " + global_pred_str).strip()

    toks = pred.split()
    pred = " ".join(toks[:12])

    final_pred.append(pred)

sub_final = pd.DataFrame(
    {"customer_id": sub_master_pd["customer_id"].values, "prediction": final_pred}
)
assert sub_final.shape[0] == num_customers
sub_final.to_csv("submission.csv", index=False)
display_df(sub_final)



## === cell 27
check = pd.read_csv("submission.csv")
print(check.shape)
print(check.columns.tolist())
print("Unique customers:", check["customer_id"].nunique())
print("Sample customers:", num_customers)
assert check["customer_id"].nunique() == num_customers
assert check.shape[0] == num_customers
check.head()
