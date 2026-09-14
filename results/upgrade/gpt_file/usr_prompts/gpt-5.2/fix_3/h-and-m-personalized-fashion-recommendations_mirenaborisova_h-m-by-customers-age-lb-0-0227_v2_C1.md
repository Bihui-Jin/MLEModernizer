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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime errors by updating the deprecated `json_normalize` import and by making cuDF datetime slicing work (sorting the DatetimeIndex before using `.loc` range slicing). Then I fix the submission-generation logic bug: your per-age-bin submission files currently contain only customers in that age bin (and drop the rest), so concatenating them does not produce a full customer_id superset; instead, we build one final submission by starting from the full sample submission and filling predictions per age bin. Finally, I ensure the written file is exactly `submission.csv` with correct columns and customer_id formatting preserved from the sample.'

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
start = np.datetime64("2020-09-01")
stop = np.datetime64("2020-09-21")
idx = transactions_df.index.values
recent_df = transactions_df[(idx >= start) & (idx <= stop)]
display_df(recent_df)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/310548641.py in <cell line: 0>()
      3 start = np.datetime64("2020-09-01")
      4 stop = np.datetime64("2020-09-21")
----> 5 idx = transactions_df.index.values
      6 recent_df = transactions_df[(idx >= start) & (idx <= stop)]
      7 display_df(recent_df)

/usr/local/lib/python3.11/dist-packages/cudf/core/index.py in values(self)
   1641     @property
   1642     def values(self) -> cupy.ndarray:
-> 1643         return self._column.values
   1644 
   1645     def __contains__(self, item) -> bool:

/usr/local/lib/python3.11/dist-packages/cudf/core/column/datetime.py in values(self)
    399         Return a CuPy representation of the DateTimeColumn.
    400         """
--> 401         raise NotImplementedError(
    402             "DateTime Arrays is not yet implemented in cudf"
    403         )

NotImplementedError: DateTime Arrays is not yet implemented in cudf

## === cell 12
recent_df = recent_df.to_pandas()
recent_df = recent_df.merge(
    customers_df[["customer_id", "age_bins"]], on="customer_id", how="inner"
)
display_df(recent_df)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3557231111.py in <cell line: 0>()
----> 1 recent_df = recent_df.to_pandas()
      2 recent_df = recent_df.merge(
      3     customers_df[["customer_id", "age_bins"]], on="customer_id", how="inner"
      4 )
      5 display_df(recent_df)

NameError: name 'recent_df' is not defined

## === cell 13
recent_df = (
    recent_df.groupby(["age_bins", "article_id"])
    .count()
    .reset_index()
    .rename(columns={"customer_id": "counts"})
)
display_df(recent_df)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/396226873.py in <cell line: 0>()
      1 recent_df = (
----> 2     recent_df.groupby(["age_bins", "article_id"])
      3     .count()
      4     .reset_index()
      5     .rename(columns={"customer_id": "counts"})

NameError: name 'recent_df' is not defined

## === cell 14
bins_unique_list = recent_df["age_bins"].unique().tolist()
bins_unique_list



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3381229415.py in <cell line: 0>()
----> 1 bins_unique_list = recent_df["age_bins"].unique().tolist()
      2 bins_unique_list
      3 

NameError: name 'recent_df' is not defined

## === cell 15
bins_unique_list[0]



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1698074537.py in <cell line: 0>()
----> 1 bins_unique_list[0]
      2 

NameError: name 'bins_unique_list' is not defined

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



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2870722163.py in <cell line: 0>()
      2 count = COUNT
      3 order = ORDER
----> 4 for bins_unique in bins_unique_list:
      5     temp_df = recent_df[recent_df["age_bins"] == bins_unique]
      6     info_df(temp_df, count, order)

NameError: name 'bins_unique_list' is not defined

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
    plt.show()



## === cell 23
bins_unique_list



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3595600388.py in <cell line: 0>()
----> 1 bins_unique_list
      2 

NameError: name 'bins_unique_list' is not defined

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

cust_bins_cudf = cudf.from_pandas(customers_df[["customer_id", "age_bins"]])
sub_all = sub_all.merge(cust_bins_cudf, on="customer_id", how="left")

sub_all["customer_id_2"] = (
    sub_all["customer_id"].str[-16:].str.hex_to_int().astype("int64")
)

sub_all["age_bins_str"] = sub_all["age_bins"].fillna("").astype("str")

sub_all["prediction"] = cudf.Series([None] * num_customers)

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
        sub_mask = sub_all["age_bins"].isnull()
    else:
        temp_customers_df = customers_df[customers_df["age_bins"] == bin_unique]
        sub_mask = sub_all["age_bins_str"] == str(bin_unique)

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

    purchase_df["prediction"] = "0" + purchase_df["article_id"].astype("str") + " "
    info_df(purchase_df, count, order)

    purchase_df = (
        purchase_df.groupby("customer_id").agg({"prediction": sum}).reset_index()
    )
    info_df(purchase_df, count, order)

    purchase_df = cudf.DataFrame(purchase_df)

    sub_bin = sub_all.loc[sub_mask, ["customer_id_2"]].merge(
        purchase_df, left_on="customer_id_2", right_on="customer_id", how="left"
    )
    sub_bin = sub_bin.to_pandas()
    sub_bin["prediction"] = sub_bin["prediction"].fillna(general_pred_str)
    sub_bin["prediction"] = sub_bin["prediction"] + " " + general_pred_str
    sub_bin["prediction"] = sub_bin["prediction"].str.strip()
    sub_bin["prediction"] = sub_bin["prediction"].str[:131]

    sub_all.loc[sub_mask, "prediction"] = cudf.from_pandas(sub_bin["prediction"])

    print(f"FILLED PREDICTION FOR {bin_unique}, CUSTOMERS: {int(sub_mask.sum())}\n")
    print("-" * 50)

    count = 0

if sub_all["prediction"].isnull().any():
    df_full = cudf.read_csv(
        PATH_INPUT + "transactions_train.csv",
        usecols=["article_id"],
        dtype={"article_id": "int32"},
    )
    top_global = df_full["article_id"].value_counts().head(N).to_pandas().index.tolist()
    top_global = ["0" + str(a) for a in top_global]
    top_global_str = " ".join(top_global)
    sub_all["prediction"] = sub_all["prediction"].fillna(top_global_str)

print("FINISHED")
print("=" * 50)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/cudf/core/scalar.py in __call__(self, value, dtype)
    255             # try retrieving an instance from the cache:
--> 256             self.__instances.move_to_end(cache_key)
    257             return self.__instances[cache_key]

KeyError: ('', <class 'str'>, interval[int64, right], <class 'cudf.core.dtypes.IntervalDtype'>)

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/cudf/core/column/categorical.py in _validate_fillna_value(self, fill_value)
   1054                 try:
-> 1055                     fill_value = self._encode(fill_value)
   1056                 except ValueError as err:

/usr/local/lib/python3.11/dist-packages/cudf/core/column/categorical.py in _encode(self, value)
    867     def _encode(self, value) -> ScalarLike:
--> 868         return self.categories.find_first_value(value)
    869 

/usr/local/lib/python3.11/dist-packages/cudf/core/column/column.py in find_first_value(self, value)
    940         """
--> 941         first, _ = self._find_first_and_last(value)
    942         return first

/usr/local/lib/python3.11/dist-packages/cudf/core/column/column.py in _find_first_and_last(self, value)
    914     def _find_first_and_last(self, value: ScalarLike) -> tuple[int, int]:
--> 915         indices = self.indices_of(value)
    916         if n := len(indices):

/usr/local/lib/python3.11/dist-packages/cudf/core/column/column.py in indices_of(self, value)
    907         else:
--> 908             value = as_column(value, dtype=self.dtype, length=1)
    909         mask = value.contains(self)

/usr/local/lib/python3.11/dist-packages/cudf/core/column/column.py in as_column(arbitrary, nan_as_null, dtype, length)
   2267             arbitrary = None
-> 2268         arbitrary = cudf.Scalar(arbitrary, dtype=dtype)
   2269         if length == 0:

/usr/local/lib/python3.11/dist-packages/cudf/core/scalar.py in __call__(self, value, dtype)
    260             # construct it and add to cache:
--> 261             obj = super().__call__(value, dtype=dtype)
    262             try:

/usr/local/lib/python3.11/dist-packages/cudf/core/scalar.py in __init__(self, value, dtype)
    331         else:
--> 332             self._host_value, self._host_dtype = _preprocess_host_value(
    333                 value, dtype

/usr/local/lib/python3.11/dist-packages/cudf/core/scalar.py in _preprocess_host_value(value, dtype)
     74         if value not in {None, NA}:
---> 75             raise ValueError(f"Can not coerce {value} to StructDType")
     76         else:

ValueError: Can not coerce  to StructDType

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1047484023.py in <cell line: 0>()
     15 # BUGFIX: cuDF may keep age_bins as categorical; .astype("str") can raise NotImplementedError.
     16 # Convert once to a plain string column for safe comparisons in the per-bin loop.
---> 17 sub_all["age_bins_str"] = sub_all["age_bins"].fillna("").astype("str")
     18 
     19 sub_all["prediction"] = cudf.Series([None] * num_customers)

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/series.py in fillna(self, value, method, axis, inplace, limit)
   1888                 value = value.reindex(self.index)
   1889             value = {self.name: value._column}
-> 1890         return super().fillna(
   1891             value=value, method=method, axis=axis, inplace=inplace, limit=limit
   1892         )

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/frame.py in fillna(self, value, method, axis, inplace, limit)
    806             )
    807 
--> 808         filled_columns = [
    809             col.fillna(value[name], method) if name in value else col.copy()
    810             for name, col in self._column_labels_and_values

/usr/local/lib/python3.11/dist-packages/cudf/core/frame.py in <listcomp>(.0)
    807 
    808         filled_columns = [
--> 809             col.fillna(value[name], method) if name in value else col.copy()
    810             for name, col in self._column_labels_and_values
    811         ]

/usr/local/lib/python3.11/dist-packages/cudf/core/column/column.py in fillna(self, fill_value, method)
    797                 return self.copy()
    798             else:
--> 799                 fill_value = self._validate_fillna_value(fill_value)
    800 
    801         if fill_value is None and method is None:

/usr/local/lib/python3.11/dist-packages/cudf/core/column/categorical.py in _validate_fillna_value(self, fill_value)
   1055                     fill_value = self._encode(fill_value)
   1056                 except ValueError as err:
-> 1057                     raise ValueError(
   1058                         f"{fill_value=} must be in categories"
   1059                     ) from err

ValueError: fill_value='' must be in categories

## === cell 26
submission = sub_all[["customer_id", "prediction"]].to_pandas()
submission.to_csv("submission.csv", index=False)

display_df(submission)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1792207413.py in <cell line: 0>()
----> 1 submission = sub_all[["customer_id", "prediction"]].to_pandas()
      2 submission.to_csv("submission.csv", index=False)
      3 
      4 display_df(submission)
      5 

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
    557             )
    558         else:
--> 559             data = {k: self._grouped_data[k] for k in key}
    560             if len(data) != len(key):
    561                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/cudf/core/column_accessor.py in <dictcomp>(.0)
    557             )
    558         else:
--> 559             data = {k: self._grouped_data[k] for k in key}
    560             if len(data) != len(key):
    561                 raise ValueError(

KeyError: 'prediction'

## === cell 27
check_df = cudf.read_csv("./submission.csv")
display_df(check_df)
assert (
    check_df.shape[0] == num_customers
), f"Row count mismatch: {check_df.shape[0]} vs {num_customers}"
assert set(check_df.columns) == {"customer_id", "prediction"}

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2018018541.py in <cell line: 0>()
----> 1 check_df = cudf.read_csv("./submission.csv")
      2 display_df(check_df)
      3 assert (
      4     check_df.shape[0] == num_customers
      5 ), f"Row count mismatch: {check_df.shape[0]} vs {num_customers}"

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/io/csv.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, prefix, mangle_dupe_cols, dtype, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, skip_blank_lines, parse_dates, dayfirst, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, comment, delim_whitespace, byte_range, storage_options, bytes_per_thread)
     84         bytes_per_thread = ioutils._BYTES_PER_THREAD_DEFAULT
     85 
---> 86     filepath_or_buffer = ioutils.get_reader_filepath_or_buffer(
     87         path_or_data=filepath_or_buffer,
     88         iotypes=(BytesIO, StringIO),

/usr/local/lib/python3.11/dist-packages/cudf/utils/ioutils.py in get_reader_filepath_or_buffer(path_or_data, mode, fs, iotypes, allow_raw_text_input, storage_options, bytes_per_thread, warn_on_raw_text_input, warn_meta, expand_dir_pattern, prefetch_options)
   1882                     tuple(f".json{c}" for c in compression_extensions)
   1883                 ):
-> 1884                     raise FileNotFoundError(
   1885                         f"{input_sources} could not be resolved to any files"
   1886                     )

FileNotFoundError: ['./submission.csv'] could not be resolved to any files
