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

N/A

# 9. Code solution

## === cell 0
DEBUG = False
PATH_INPUT = r"/kaggle/input/h-and-m-personalized-fashion-recommendations/"



## === cell 1
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
transactions_df = transactions_df.sort_index()  # ensure monotonic index for slicing
display_df(transactions_df)



## === cell 11
mask = (transactions_df.index >= "2020-09-01") & (transactions_df.index <= "2020-09-21")
recent_df = transactions_df[mask]
display_df(recent_df)



## === cell 12
recent_df = recent_df.to_pandas()
recent_df = recent_df.merge(
    customers_df[["customer_id", "age_bins"]], on="customer_id", how="inner"
)
display_df(recent_df)



## === cell 13
recent_df = (
    recent_df.groupby(["age_bins", "article_id"]).size().reset_index(name="counts")
)
display_df(recent_df)



## === cell 14
bins_unique_list = recent_df["age_bins"].unique().tolist()
bins_unique_list



## === cell 15
ages_article_id = {}
count = COUNT
order = ORDER
for bins_unique in bins_unique_list:
    temp_df = recent_df[recent_df["age_bins"] == bins_unique]
    info_df(temp_df, count, order)

    temp_df = temp_df.sort_values(by="counts", ascending=False)
    info_df(temp_df, count, order)

    ages_article_id[bins_unique] = temp_df.head(100)["article_id"].tolist()
    count = 0



## === cell 16
from itertools import islice

for key, value in islice(ages_article_id.items(), 3):
    print(f"{key} {len(value)}")



## === cell 17
topcustcnt_byage_df = pd.DataFrame([ages_article_id])
display_df(topcustcnt_byage_df)



## === cell 18
topcustcnt_byage_df = pd.DataFrame([ages_article_id]).T.rename(columns={0: "top_100"})
display_df(topcustcnt_byage_df)



## === cell 19
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



## === cell 20
topcustcnt_byage_df = topcustcnt_byage_df.drop(columns="top_100")
display_df(topcustcnt_byage_df, head=10)



## === cell 21
if not topcustcnt_byage_df.empty:
    plt.figure(figsize=(10, 6))
    sns.heatmap(topcustcnt_byage_df, cmap="winter", annot=True, cbar=False)



## === cell 22
bins_unique_list = customers_df["age_bins"].unique().tolist()
bins_unique_list



## === cell 23
count = COUNT
order = ORDER
for bin_unique in bins_unique_list:
    df = cudf.read_csv(
        PATH_INPUT + "transactions_train.csv",
        usecols=["t_dat", "customer_id", "article_id"],
        dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
    )
    info_df(df, count, order)

    if pd.isna(bin_unique):
        temp_customers_df = customers_df[customers_df["age_bins"].isnull()]
    else:
        temp_customers_df = customers_df[customers_df["age_bins"] == bin_unique]

    temp_customers_df = temp_customers_df.drop(columns=["age_bins"])
    temp_customers_df = cudf.from_pandas(temp_customers_df)
    info_df(temp_customers_df, count, order)

    df = df.merge(
        temp_customers_df[["customer_id", "age"]], on="customer_id", how="inner"
    )
    info_df(df, count, order)

    print(f"TRANSACTION SHAPE FOR SCOPE {bin_unique}: {df.shape}\n")

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

    mask = tmp["weekday"] >= 2
    tmp.loc[mask, "t_dat_shiftday"] = tmp.loc[
        mask, "t_dat_shiftday"
    ] + pd.TimedeltaIndex(np.ones(mask.sum()) * 7, unit="D")
    info_df(tmp, count, order)

    df["t_dat_shiftday"] = tmp["t_dat_shiftday"].values
    info_df(df, count, order)

    weekly_sales = (
        df.drop("customer_id", axis=1)
        .groupby(["t_dat_shiftday", "article_id"])
        .size()
        .reset_index(name="count")
    )
    info_df(weekly_sales, count, order)

    df = df.merge(weekly_sales, on=["t_dat_shiftday", "article_id"], how="left")
    info_df(df, count, order)

    weekly_sales = weekly_sales.set_index("article_id")
    df = df.merge(
        weekly_sales.loc[weekly_sales["t_dat_shiftday"] == last_date_train, ["count"]],
        on="article_id",
        how="left",
        suffixes=("", "_targ"),
    )
    info_df(df, count, order)

    df["count_targ"].fillna(0, inplace=True)
    df["quotient"] = df["count_targ"] / df["count"]
    info_df(df, count, order)

    target_sales = df.groupby("article_id")["quotient"].sum()
    info_df(target_sales, count, order)

    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = ["0" + str(aid) for aid in general_pred]
    general_pred_str = " ".join(general_pred)

    tmp = df.to_pandas()
    info_df(tmp, count, order)

    tmp["x"] = ((last_date_train - tmp["t_dat"]) / np.timedelta64(1, "D")).astype(int)
    info_df(tmp, count, order)

    tmp["x"] = tmp["x"].clip(lower=0)
    info_df(tmp, count, order)

    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    tmp["y"] = a / np.sqrt(tmp["x"]) + b * np.exp(-c * tmp["x"]) - d
    tmp["y"] = tmp["y"].clip(lower=0)
    tmp["value"] = tmp["quotient"] * tmp["y"]
    info_df(tmp, count, order)

    tmp = tmp.groupby(["customer_id", "article_id"]).agg({"value": "sum"}).reset_index()
    info_df(tmp, count, order)

    tmp = tmp[tmp["value"] > 0]
    info_df(tmp, count, order)

    tmp["rank"] = tmp.groupby("customer_id")["value"].rank("dense", ascending=False)
    info_df(tmp, count, order)

    tmp = tmp[tmp["rank"] <= 12]
    info_df(tmp, count, order)

    purchase_df = tmp.sort_values(
        ["customer_id", "value"], ascending=False
    ).reset_index(drop=True)
    info_df(purchase_df, count, order)

    purchase_df["prediction"] = "0" + purchase_df["article_id"].astype(str) + " "
    purchase_df = (
        purchase_df.groupby("customer_id").agg({"prediction": "sum"}).reset_index()
    )
    info_df(purchase_df, count, order)

    purchase_df = cudf.DataFrame(purchase_df)

    sub = cudf.read_csv(
        PATH_INPUT + "sample_submission.csv",
        usecols=["customer_id"],
        dtype={"customer_id": "string"},
    )
    info_df(sub, count, order)

    sub = sub.merge(
        temp_customers_df[["customer_id", "age"]], on="customer_id", how="inner"
    )
    info_df(sub, count, order)

    sub = sub.merge(
        purchase_df, left_on="customer_id", right_on="customer_id", how="left"
    )
    info_df(sub, count, order)

    sub = sub.to_pandas()

    def fill_predictions(row):
        preds = row["prediction"].split() if isinstance(row["prediction"], str) else []
        needed = 12 - len(preds)
        if needed > 0:
            add = [aid for aid in general_pred if aid not in preds][:needed]
            preds.extend(add)
        return " ".join(preds)

    sub["prediction"] = sub.apply(fill_predictions, axis=1)

    sub["prediction"] = sub["prediction"].apply(lambda x: " ".join(x.split()[:12]))

    sub = sub[["customer_id", "prediction"]]
    sub.to_csv(f"submission_{str(bin_unique)}.csv", index=False)
    info_df(sub, count, order)

    print(f"SAVED PREDICTION FOR {bin_unique}, SHAPE: {sub.shape}\n")
    print("-" * 50)

    count = 0

print("FINISHED")
print("=" * 50)



## === cell 24
bins_unique_list = customers_df["age_bins"].unique().tolist()
sub_dfs = []
for bin_unique in bins_unique_list:
    temp_path = f"submission_{str(bin_unique)}.csv"
    if os.path.exists(temp_path):
        sub_dfs.append(cudf.read_csv(temp_path))

sub_df = cudf.concat(sub_dfs, axis=0).drop_duplicates("customer_id")
full_sample = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv", dtype={"customer_id": "string"}
)
final_sub = full_sample.merge(sub_df, on="customer_id", how="left")

if "prediction" not in final_sub.columns:
    final_sub["prediction"] = ""

final_sub["prediction"] = final_sub["prediction"].fillna("")
final_sub.to_csv("by_cust_age__.csv", index=False)

display_df(final_sub)



## === cell 25
article_counts = transactions_df.groupby("article_id").size().reset_index(name="cnt")
top12_series = article_counts.nlargest(12, "cnt")["article_id"]
top12 = top12_series.to_arrow().to_pylist()
top12 = [str(aid) for aid in top12]  # ensure string type
top12 = ["0" + aid for aid in top12]  # pad with leading zero as in original logic
fallback_pred = " ".join(top12)


def ensure_12(pred):
    """Guarantee exactly 12 article IDs per row."""
    if not isinstance(pred, str) or pred.strip() == "":
        return fallback_pred
    items = pred.split()
    if len(items) >= 12:
        return " ".join(items[:12])
    add = [aid for aid in top12 if aid not in items][: 12 - len(items)]
    return " ".join(items + add)


final_sub["prediction"] = final_sub["prediction"].apply(ensure_12)

final_sub.to_csv("submission.csv", index=False)

check_df = cudf.read_csv("submission.csv")
display_df(check_df)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
TypingError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/cudf/core/indexed_frame.py in _apply(self, func, kernel_getter, *args, **kwargs)
   3534         try:
-> 3535             kernel, retty = _compile_or_get(
   3536                 self, func, args, kernel_getter=kernel_getter

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     50                 )
---> 51             return func(*args, **kwargs)
     52 

/usr/local/lib/python3.11/dist-packages/cudf/core/udf/utils.py in _compile_or_get(frame, func, args, kernel_getter, suffix)
    281 
--> 282     kernel, scalar_return_type = kernel_getter(frame, func, args)
    283     np_return_type = (

/usr/local/lib/python3.11/dist-packages/cudf/core/udf/scalar_function.py in _get_scalar_kernel(sr, func, args)
     54     )
---> 55     scalar_return_type = _get_udf_return_type(sr_type, func, args)
     56 

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     50                 )
---> 51             return func(*args, **kwargs)
     52 

/usr/local/lib/python3.11/dist-packages/cudf/core/udf/utils.py in _get_udf_return_type(argty, func, args)
    103     with _CUDFNumbaConfig():
--> 104         ptx, output_type = cudautils.compile_udf(func, compile_sig)
    105 

/usr/local/lib/python3.11/dist-packages/cudf/utils/cudautils.py in compile_udf(udf, type_signature)
    125     # compilation with Numba
--> 126     ptx_code, return_type = cuda.compile_ptx_for_current_device(
    127         udf, type_signature, device=True

/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/compiler.py in compile_ptx_for_current_device(pyfunc, sig, debug, lineinfo, device, fastmath, opt, abi, abi_info)
    567     cc = get_current_device().compute_capability
--> 568     return compile_ptx(pyfunc, sig, debug=debug, lineinfo=lineinfo,
    569                        device=device, fastmath=fastmath, cc=cc, opt=opt,

/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/compiler.py in compile_ptx(pyfunc, sig, debug, lineinfo, device, fastmath, cc, opt, abi, abi_info)
    556     device function with the C ABI."""
--> 557     return compile(pyfunc, sig, debug=debug, lineinfo=lineinfo, device=device,
    558                    fastmath=fastmath, cc=cc, opt=opt, abi=abi,

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_lock.py in _acquire_compile_lock(*args, **kwargs)
     34             with self:
---> 35                 return func(*args, **kwargs)
     36         return _acquire_compile_lock

/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/compiler.py in compile(pyfunc, sig, debug, lineinfo, device, fastmath, cc, opt, abi, abi_info, output)
    509     cc = cc or config.CUDA_DEFAULT_PTX_CC
--> 510     cres = compile_cuda(pyfunc, return_type, args, debug=debug,
    511                         lineinfo=lineinfo, fastmath=fastmath,

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_lock.py in _acquire_compile_lock(*args, **kwargs)
     34             with self:
---> 35                 return func(*args, **kwargs)
     36         return _acquire_compile_lock

/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/compiler.py in compile_cuda(pyfunc, return_type, args, debug, lineinfo, inline, fastmath, nvvm_options, cc, max_registers, lto)
    225     with target_override('cuda'):
--> 226         cres = compiler.compile_extra(typingctx=typingctx,
    227                                       targetctx=targetctx,

/usr/local/lib/python3.11/dist-packages/numba/core/compiler.py in compile_extra(typingctx, targetctx, func, args, return_type, flags, locals, library, pipeline_class)
    743                               args, return_type, flags, locals)
--> 744     return pipeline.compile_extra(func)
    745 

/usr/local/lib/python3.11/dist-packages/numba/core/compiler.py in compile_extra(self, func)
    437         self.state.lifted_from = None
--> 438         return self._compile_bytecode()
    439 

/usr/local/lib/python3.11/dist-packages/numba/core/compiler.py in _compile_bytecode(self)
    505         assert self.state.func_ir is None
--> 506         return self._compile_core()
    507 

/usr/local/lib/python3.11/dist-packages/numba/core/compiler.py in _compile_core(self)
    484                     if is_final_pipeline:
--> 485                         raise e
    486             else:

/usr/local/lib/python3.11/dist-packages/numba/core/compiler.py in _compile_core(self)
    471                 try:
--> 472                     pm.run(self.state)
    473                     if self.state.cr is not None:

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_machinery.py in run(self, state)
    367                 patched_exception = self._patch_error(msg, e)
--> 368                 raise patched_exception
    369 

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_machinery.py in run(self, state)
    355                 if isinstance(pass_inst, CompilerPass):
--> 356                     self._runPass(idx, pass_inst, state)
    357                 else:

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_lock.py in _acquire_compile_lock(*args, **kwargs)
     34             with self:
---> 35                 return func(*args, **kwargs)
     36         return _acquire_compile_lock

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_machinery.py in _runPass(self, index, pss, internal_state)
    310             with SimpleTimer() as pass_time:
--> 311                 mutated |= check(pss.run_pass, internal_state)
    312             with SimpleTimer() as finalize_time:

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_machinery.py in check(func, compiler_state)
    272         def check(func, compiler_state):
--> 273             mangled = func(compiler_state)
    274             if mangled not in (True, False):

/usr/local/lib/python3.11/dist-packages/numba/core/typed_passes.py in run_pass(self, state)
    111             # Type inference
--> 112             typemap, return_type, calltypes, errs = type_inference_stage(
    113                 state.typingctx,

/usr/local/lib/python3.11/dist-packages/numba/core/typed_passes.py in type_inference_stage(typingctx, targetctx, interp, args, return_type, locals, raise_errors)
     92         # return errors in case of partial typing
---> 93         errs = infer.propagate(raise_errors=raise_errors)
     94         typemap, restype, calltypes = infer.unify(raise_errors=raise_errors)

/usr/local/lib/python3.11/dist-packages/numba/core/typeinfer.py in propagate(self, raise_errors)
   1090                 if not force_lit_args:
-> 1091                     raise errors[0]
   1092                 else:

TypingError: Failed in cuda mode pipeline (step: nopython frontend)
No implementation of function Function(<built-in function isinstance>) found for signature:
 
 >>> isinstance(Masked(string_view), Function(<class 'str'>))
 
There are 2 candidate implementations:
    - Of which 2 did not match due to:
    Overload in function 'ol_isinstance': File: numba/cpython/builtins.py: Line 755.
      With argument(s): '(Masked(string_view), Function(<class 'str'>))':
     Rejected as the implementation raised a specific error:
       NumbaTypeError: isinstance() does not support variables of type "Masked(string_view)".
  raised from /usr/local/lib/python3.11/dist-packages/numba/cpython/builtins.py:785

During: resolving callee type: Function(<built-in function isinstance>)
During: typing of call at /tmp/ipykernel_55/632724091.py (13)


File "../../tmp/ipykernel_55/632724091.py", line 13:
<source missing, REPL/exec in use?>


The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/632724091.py in <cell line: 0>()
     21 
     22 # Apply the fallback / trimming to the final submission
---> 23 final_sub["prediction"] = final_sub["prediction"].apply(ensure_12)
     24 
     25 # Save the final Kaggle submission file

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   2681             raise NotImplementedError("by_row is currently not supported.")
   2682 
-> 2683         result = self._apply(func, _get_scalar_kernel, *args, **kwargs)
   2684         result.name = self.name
   2685         return result

/usr/lib/python3.11/contextlib.py in inner(*args, **kwds)
     79         def inner(*args, **kwds):
     80             with self._recreate_cm():
---> 81                 return func(*args, **kwds)
     82         return inner
     83 

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/indexed_frame.py in _apply(self, func, kernel_getter, *args, **kwargs)
   3537             )
   3538         except Exception as e:
-> 3539             raise ValueError(
   3540                 "user defined function compilation failed."
   3541             ) from e

ValueError: user defined function compilation failed.

## --- ERROR in outputing the csv:
Invalid submission: Submission customer_id must be a superset of answers customer_id
