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

- What this solution (achieved 0.0) has done: 'Your code likely didn’t yield a Kaggle score because it produces multiple per-bin submission files and then concatenates them, creating duplicate customers and/or missing customers (inner-join on age) rather than exactly the sample_submission customer list once. I keep your per-age-bin logic intact but change the workflow to build one final submission by starting from `sample_submission.csv` and then filling predictions per customer from each bin, ensuring each customer appears exactly once. I also make the join keys consistent (hex-int customer ids on both sides) and ensure predictions are exactly 12 unique article_ids (space-separated) to match MAP@12 expectations. Finally, the script always write a single valid `submission.csv` at the end.'
- What this solution (achieved 0.0) has done: 'The crash happens because `purchase_df` is no longer a cuDF DataFrame at that point: after `purchase_df.groupby(...).agg(...).reset_index()` it becomes a pandas DataFrame (due to the use of Python’s built-in `sum` in the aggregation), so calling `.to_pandas()` fails. The minimal fix is to safely convert `purchase_df` to pandas only if it’s a cuDF object; otherwise keep it as-is. This preserves all downstream logic and variables used in cell 27 (`final_sub`, `prediction`), with no changes to model/feature semantics. No other cells need modification.'
- What this solution (achieved 0.0) has done: 'You’re getting 0.0 most likely because the submission’s `prediction` strings contain invalid article_ids due to the `"0"+str(article_id)` prefix (H&M `article_id` are already 10-digit; adding a leading “0” makes them wrong), and also because the per-customer concatenation leaves a trailing space and can produce malformed tokenization. I keep your exact scoring logic (per-age-bin, quotient/recency weighting, top-N fallback) but change only the formatting pipeline: build predictions as *exactly* 10-digit strings with `zfill(10)`, join with `' '.join(...)` (no trailing spaces), and normalize to 12 unique items. This should move the score upward toward your target without changing the underlying recommendation logic. I also ensure the final submission uses the sample submission customer list 1-to-1 (already mostly correct in your code) and keep runtime within limits.'
- What this solution (achieved 0.0) has done: 'The 0.0 score is most consistent with “invalid/empty predictions for many customers” due to bin coverage issues: customers with missing age (NaN) in `customers.csv` won’t match any `age_bins` and therefore never get filled inside the per-bin loop, leaving `prediction` as null/empty until the very end. I keep your exact per-age-bin recommendation logic, but make a minimal change so the bin list explicitly includes a NaN bin (age missing), ensuring those customers get per-bin predictions rather than relying solely on the global fallback. I also make the `purchase_df` string aggregation deterministic and safe (avoid potential cudf/pandas surprises) while keeping the same semantics, so each customer gets a well-formed 12-token string. This should move the score up toward your target without changing the underlying model logic.'
- What this solution (achieved 0.0) has done: 'The crash happens because `purchase_df` is already a pandas `DataFrame` at that point (it was created from `tmp.sort_values(...).reset_index(...)` where `tmp` is pandas), so calling `.to_pandas()` raises `AttributeError`. The minimal fix is to only call `.to_pandas()` when the object is a cuDF DataFrame, and otherwise keep it as-is. This preserves the exact downstream logic (groupby/join) and the variables expected by cell 27. No other computations, rankings, or model semantics are changed.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with the model producing valid-looking strings but *mismatching customers* due to inconsistent `customer_id_2` computation (sample uses last 16 hex chars, while merged transactions use the full `customer_id` string after earlier merges) and with many customers being dropped early by an `inner` merge on age bins. I keep your per-age-bin logic, quotient/recency scoring, and ranking intact, but make customer ID handling deterministic by computing `customer_id_2` the same way everywhere and by merging predictions directly onto `final_sub` for the customers in the current bin (instead of only filling NaNs). I also avoid dropping customers in the “recent_df” age-bin stats by switching that merge to `left` and explicitly keeping a NaN bin, so your fallback/top logic remains valid and more customers get filled. These are minimal plumbing fixes that should move the Kaggle score upward toward your target without changing the underlying recommendation formula.'
- What this solution (achieved 0.0) has done: 'Diagnosis: Cell 26 crashes at `tmp.groupby(["customer_id_2","article_id"])` because `tmp` is created from `df.copy().to_pandas()` and, depending on cudf/pandas conversion behavior, the derived column `customer_id_2` is not reliably present in the pandas copy (it can be dropped/converted unexpectedly). The code then attempts to group on a missing column, raising `KeyError: 'customer_id_2'`. The minimal fix is to defensively (re)create `customer_id_2` inside the pandas `tmp` right after conversion if it is absent, using the existing `customer_id` string column. This keeps all downstream semantics identical and preserves the join key used later with `final_sub`.

Patch summary: In cell 26, immediately after `tmp = df.copy().to_pandas()`, add a small guard that reconstructs `tmp["customer_id_2"]` from `tmp["customer_id"]` when missing, matching the exact hex-suffix logic already used earlier in the cell.

Updated cells: (cell 26 only)

Compatibility notes for cell k+1: No interface/variable changes; `final_sub` remains populated the same way, and `final_sub["prediction"]` is still an object series consumed by cell 27.

Assumptions: `customer_id` is always present in `df` and is a 16+ char hex string as in the H&M dataset; pandas can convert hex via `int(x, 16)` deterministically.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model producing *valid-looking* predictions but Kaggle treating many (or all) of them as non-matching because `article_id` tokens are not in the exact expected 10-digit format for every path (some paths can still yield non-zfilled strings), and because the per-customer aggregation can include duplicates and/or fewer than 12 unique items, reducing MAP@12 sharply. I keep your per-age-bin logic, recency/quotient weighting, and ranking intact, but make the tokenization and final per-customer assembly strictly enforce “12 unique, 10-digit article_id tokens” everywhere (bin-level and global fallback). I also make the weekly target-week merge stable by matching on the last week-start date (`t_dat_shiftday`) rather than a raw `last_date_train` timestamp, which otherwise can silently select zero rows and degrade all bin general predictions. These are plumbing/format fixes only and should move the score upward toward your target without changing the underlying scoring formula.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with many customers getting empty/invalid predictions because the per-bin loop computes `bins_unique_list` incorrectly (it contains `NaN` twice and may include unused categories) and because the “general_pred” for a bin can become empty/NaN when the target-week merge returns no rows, leading to blank fallbacks. I keep your exact per-age-bin recommendation logic, but make two minimal plumbing fixes: (1) build the bin iteration list deterministically from the cut categories + one explicit NaN bin, and (2) guarantee a non-empty per-bin fallback by falling back to the global top articles whenever a bin’s `general_pred` comes out empty. These changes should move the score up toward your target while preserving the model semantics and ensuring every customer gets 12 valid 10-digit article_id tokens. The script still write a single valid `submission.csv` aligned 1-to-1 with `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with most customers receiving empty/invalid predictions because the per-bin transaction merge is `inner` on customers-with-age, which drops all transactions for customers with missing age and can make some bins’ computations degenerate; also the bin loop repeatedly reloads the full 31M-row transactions file, increasing the chance of timeouts/partial runs. I keep your exact scoring/ranking logic intact, but make two minimal plumbing changes: (1) load `transactions_train.csv` once and reuse it per bin (same filtering semantics), and (2) change the per-bin customer merge to `left` and then filter to the bin’s customers by `customer_id_2`, ensuring bin customers aren’t accidentally dropped. Finally, I make the “target week” join robust by falling back to `global_fallback` if the selected `last_shiftday` slice is empty (so `general_pred_str` never ends up blank). This should move your score upward toward the target while preserving the core recommendation logic and producing a valid single `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an invalid/near-empty effective submission, not with “weak recommendations”: the bin loop likely fills many customers with blank/short strings (or non-12 tokens) when a bin has no scoped transactions or the weekly target slice is empty. I keep your exact per-age-bin ranking formula and recency weighting, but add minimal guards so each bin always has a non-empty `general_pred_str` (fallback to global top) and each customer always gets exactly 12 valid 10-digit article_id tokens. I also avoid a silent “empty-bin” path by short-circuiting bins with zero transactions/customers and directly filling those customers from the global fallback. These are plumbing/format fixes only and should move the score upward toward your target.'

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
sub_base["customer_id_2"] = (
    sub_base["customer_id"].str[-16:].str.hex_to_int().astype("int64")
)
num_customers = sub_base.shape[0]
info_df(sub_base, 1, 0)

final_sub = sub_base[["customer_id", "customer_id_2"]].to_pandas()
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
df_all["customer_id_2"] = (
    df_all["customer_id"].str[-16:].str.hex_to_int().astype("int64")
)
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
        bin_cust["customer_id"].str[-16:].str.hex_to_int().astype("int64")
    )
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
            tmp["customer_id"]
            .astype(str)
            .str[-16:]
            .apply(lambda x: int(x, 16))
            .astype("int64")
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



## === cell 28
check_df = cudf.read_csv("./submission.csv")
display_df(check_df)
