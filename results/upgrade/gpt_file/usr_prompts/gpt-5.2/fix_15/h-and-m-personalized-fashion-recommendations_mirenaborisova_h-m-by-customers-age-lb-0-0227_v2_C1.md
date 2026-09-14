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

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime errors by updating the deprecated `json_normalize` import and by making cuDF datetime slicing work (sorting the DatetimeIndex before using `.loc` range slicing). Then I fix the submission-generation logic bug: your per-age-bin submission files currently contain only customers in that age bin (and drop the rest), so concatenating them does not produce a full customer_id superset; instead, we build one final submission by starting from the full sample submission and filling predictions per age bin. Finally, I ensure the written file is exactly `submission.csv` with correct columns and customer_id formatting preserved from the sample.'
- What this solution (achieved 0.0) has done: 'I fix the cuDF datetime filtering bug by avoiding `.index.values` on a DateTimeIndex and instead filtering via a regular datetime column (this unblocks the earlier “recent_df” analysis cells). Then I fix the main submission loop crash caused by filling/astype on a cuDF categorical `age_bins` by precomputing an `age_bins_str` column in pandas and merging it into `sub_all` as a plain string. Finally, I ensure `prediction` is always created and filled for every customer in the sample submission, and that `submission.csv` is written with exactly the required columns and row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with invalid/empty per-customer predictions (formatting/ID alignment), not with “weak model quality”. I keep your exact scoring logic but fix the two biggest correctness issues: (1) your `customer_id_2` conversion uses the *last 16 characters* (not the last 16 hex digits after the `0x` prefix), which breaks joins and leaves most customers with only fallback strings; and (2) you never guarantee exactly 12 unique article_ids per customer (MAP@12 expects 12 ranked items; truncating the string to 131 chars can cut IDs mid-token). I minimally adjust ID conversion to use the correct hex slice and rebuild predictions as a 12-item list (customer-specific first, then age-bin fallback, then global fallback) and then join into a space-separated string, ensuring submission validity and a non-zero MAP@12. Everything else (age bins, recency weighting, quotient logic, weekly shiftday logic, per-bin loop) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a submission-format/ID issue rather than “model quality”, so I keep your exact ranking/scoring logic but fix the two places that can silently break MAP@12: (1) your article_id formatting currently produces 11-digit IDs like `"01234567890"` instead of the required 10-digit `"0123456789"`, making almost all predictions non-matching; and (2) your age-bin loop rebuilds the full transactions table every time, which risks timeouts and incomplete writes—so we read it once and reuse it per bin without changing computations. I also ensure all predicted article_ids are exactly 10 digits and joined with single spaces (no trailing spaces), while keeping your per-bin personalization + fallbacks exactly the same. These minimal fixes should move the score up toward the target band without changing the core approach.'
- What this solution (achieved 0.0) has done: 'I fix the crash in the age-bin loop caused by calling `.to_pandas()` on a pandas Series (the object type flips because `tmp`/`purchase_df` are pandas at that point). Then I ensure `sub_all["prediction"]` is filled robustly by (a) initializing it as an empty string column (not `None`, which can keep nulls in cuDF) and (b) filling any remaining null/empty predictions with the global fallback list. Finally, I keep your exact ranking/quotient/recency logic intact while ensuring every row in `submission.csv` has exactly 12 space-separated 10-digit `article_id` tokens, which should move the score up from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the cuDF groupby aggregation crash by replacing the unsupported Python-lambda aggregation with a deterministic pandas-side groupby join that preserves your exact ranking/value logic. Then I fix the remaining null predictions by ensuring `sub_all["prediction"]` is always a proper string column and by assigning fallback predictions using scalar string assignment (which avoids nulls lingering after partial updates). Finally, I keep your per-age-bin personalization + fallback construction unchanged, but I ensure every row ends with exactly 12 space-separated 10-digit `article_id` tokens so the submission is valid and should score non-zero (moving toward the 0.02244 target).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly caused by customer-id misalignment during the merge: you convert `transactions_train.customer_id` to int64 but keep `purchase_pd["customer_id"]` as the original hex string, so the join produces almost all missing personal predictions (and your fallbacks then rarely match). I keep your exact per-age-bin ranking/quotient/recency logic, but make the `purchase_pd["customer_id"]` key use the same int64 encoding (`customer_id_2`) so the merge actually hits. I also fix the age-bin mask for missing ages: comparing `str(bin_unique)` to `"nan"` is unreliable because `bin_unique` is a pandas `Interval` or `NaN`; we use `pd.isna(bin_unique)` to ensure the right customers get filled. These minimal correctness fixes should move the score up from 0.0 toward your 0.02244 target without changing the model logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with predictions not matching any true `article_id` due to subtle formatting/key issues rather than weak ranking logic. I keep your exact age-bin + quotient/recency ranking approach, but fix two correctness points that can silently destroy MAP@12: (1) ensure article IDs remain *exactly the same dtype/format everywhere* (10-digit strings only at the very end) so merges/rank lists aren’t corrupted by int32→string conversions, and (2) ensure each customer’s prediction list is built from a de-duplicated, ordered list of up to 12 items without accidental truncation or malformed tokens. I also make the customer_id join key consistent by keeping `customer_id_2` in both `sub_all` and the per-bin purchase table as `int64` and sorting before assignment to avoid misaligned writes. These changes are minimal, do not alter the model logic, and should move the score up from 0.0 toward your 0.02244 target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is valid CSV but the predicted `article_id` tokens almost never match the ground truth due to an ID formatting mismatch. I make the smallest change that preserves your exact ranking/recency/age-bin logic: keep all `article_id` values as integers throughout the pipeline, only convert to **exact 10-digit strings at the very end**, and ensure per-customer predictions are built from the *true top-12* integer-ranked items (not from possibly truncated/concatenated strings). This eliminates silent corruption from string joins and guarantees every prediction token is a valid 10-digit article id. Everything else (age bins, weekly shiftday logic, quotient/value scoring, per-bin loop) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with “valid CSV but almost zero label matches”, which typically comes from a subtle ID mismatch rather than weak ranking. I keep your exact age-bin/recency/quotient ranking logic, but fix one critical alignment bug: you merge `weekly_sales` using `last_date_train` (a real transaction max date) against `t_dat_shiftday` (a shifted-to-week-boundary date), which almost never matches and collapses `count_targ` toward 0, destroying the signal that drives both age fallback and personal ranking. I compute `last_shiftday` consistently from `last_date_train` using the same shiftday rule before merging, leaving everything else unchanged. This should move the MAP@12 up from 0.0 toward your 0.02244 target without changing the core method.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is much more consistent with a “silent non-matching prediction” issue than with weak ranking logic, so I keep your exact per-age-bin quotient/recency approach but fix the last remaining high-impact correctness risk: building `purchase_pd` from a cuDF object without explicitly converting to pandas can leave you with a cuDF Series/Index that doesn’t behave like a Python list per customer, causing almost all `personal_ids` to be empty and predictions to collapse to generic fallbacks. I also ensure the merge key types are consistent (`int64`) on both sides for every bin before merging, and I add a final safety pass that re-validates/repairs any row that doesn’t contain exactly 12 valid 10-digit tokens (without changing your ranking). These are minimal changes intended to move the score up from 0.0 toward your 0.02244 target by making sure the personalized part of your pipeline actually reaches the submission.'
- What this solution (achieved 0.0) has done: 'I fix the runtime error by handling the fact that `purchase_df` is already a pandas DataFrame at that point (so `.to_pandas()` is invalid) and ensure the subsequent merge still uses consistent `int64` keys. Then I address the null/empty `prediction` problem by making the fallback fill deterministic and type-safe (assigning a proper string scalar and re-checking empties after all per-bin writes). Finally, I keep your existing ranking/age-bin logic unchanged, but ensure every row ends with a valid 12-token, 10-digit `article_id` string so the submission scores non-zero and moves toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with a “valid CSV but near-zero matches” issue, and the highest-risk culprit here is that you’re only building `top_global` from `transactions_train.article_id` without formatting them to *exact 10-digit strings* consistently everywhere—any accidental int→string conversion drift (or missing zero-padding) kill matches. I keep your exact age-bin + recency/quotient ranking logic, but make one minimal correctness-hardening change: enforce `article_id` as `int64` end-to-end and only format to 10-digit strings at the final join, and also force `sub_all.customer_id` to stay exactly the sample submission string (no accidental normalization). Finally, I add a tiny post-check/repair that re-generates any row whose tokens are not exactly 12 valid 10-digit ids using your same prioritized sources (personal → age fallback → global), which should move the score up from 0.0 toward the target without changing the modeling approach.'

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
customers_df["age_bins_str"] = customers_df["age_bins"].astype("string").fillna("")
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

transactions_tmp = transactions_df.reset_index()[["t_dat", "customer_id", "article_id"]]
recent_df = transactions_tmp[
    (transactions_tmp["t_dat"] >= start) & (transactions_tmp["t_dat"] <= stop)
].copy()
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
    plt.show()



## === cell 23
bins_unique_list



## === cell 24
bins_unique_list = customers_df["age_bins"].unique().tolist()
bins_unique_list




## === cell 25
def article_int_to_str10(x) -> str:
    return f"{int(x):010d}"


def customer_hex_to_int64(s: cudf.Series) -> cudf.Series:
    s = s.astype("string")
    s = s.str.lower()
    s = s.str.replace("0x", "", regex=False).str[-16:]
    return s.str.hex_to_int().astype("int64")


def make_pred12_from_ints(personal_ids, fallback_ids, global_ids, n=12):
    out = []
    seen = set()

    if personal_ids is not None:
        for x in personal_ids:
            try:
                xi = int(x)
            except Exception:
                continue
            if xi not in seen:
                out.append(xi)
                seen.add(xi)
            if len(out) >= n:
                return out[:n]

    for x in fallback_ids:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi not in seen:
            out.append(xi)
            seen.add(xi)
        if len(out) >= n:
            return out[:n]

    for x in global_ids:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi not in seen:
            out.append(xi)
            seen.add(xi)
        if len(out) >= n:
            return out[:n]

    if len(out) == 0:
        out = [100000001] * n
    elif len(out) < n:
        out = out + [out[-1]] * (n - len(out))
    return out[:n]


def compute_shiftday_for_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    wd = ts.dayofweek
    shift = ts - pd.Timedelta(days=(wd - 1))
    if wd >= 2:
        shift = shift + pd.Timedelta(days=7)
    return shift


def sanitize_pred_string(pred: str, global_ids_int, n=12) -> str:
    if pred is None:
        pred = ""
    toks = str(pred).split()
    good = []
    seen = set()
    for t in toks:
        if len(t) == 10 and t.isdigit():
            if t not in seen:
                good.append(t)
                seen.add(t)
        if len(good) >= n:
            break
    if len(good) < n:
        for a in global_ids_int:
            s = article_int_to_str10(a)
            if s not in seen:
                good.append(s)
                seen.add(s)
            if len(good) >= n:
                break
    if len(good) < n:
        good = (
            (good + [good[-1]] * (n - len(good)))
            if good
            else [article_int_to_str10(global_ids_int[0])] * n
        )
    return " ".join(good[:n])


sub_all = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
num_customers = sub_all.shape[0]

cust_bins_cudf = cudf.from_pandas(
    customers_df[["customer_id", "age_bins", "age_bins_str", "age"]]
)
sub_all = sub_all.merge(
    cust_bins_cudf[["customer_id", "age_bins", "age_bins_str"]],
    on="customer_id",
    how="left",
)

sub_all["customer_id_2"] = customer_hex_to_int64(sub_all["customer_id"])

sub_all["prediction"] = cudf.Series([""] * num_customers, dtype="string")

df_all = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
)

df_all["article_id"] = df_all["article_id"].astype("int64")

top_global = df_all["article_id"].value_counts().head(N).to_pandas().index.tolist()
top_global_ids_int = [int(a) for a in top_global]
top_global_str = " ".join(article_int_to_str10(a) for a in top_global_ids_int)

count = COUNT
order = ORDER

for bin_unique in bins_unique_list:
    df = df_all.copy(deep=False)
    info_df(df, count, order)

    if pd.isna(bin_unique):
        temp_customers_df = customers_df[customers_df["age_bins"].isnull()]
        sub_mask = sub_all["age_bins"].isnull()
    else:
        temp_customers_df = customers_df[customers_df["age_bins"] == bin_unique]
        sub_mask = sub_all["age_bins_str"] == str(bin_unique)

    temp_customers_df = temp_customers_df.drop(["age_bins", "age_bins_str"], axis=1)
    temp_customers_df = cudf.from_pandas(temp_customers_df)
    info_df(temp_customers_df, count, order)

    df = df.merge(
        temp_customers_df[["customer_id", "age"]], on="customer_id", how="inner"
    )
    info_df(df, count, order)

    print(f"TRANSACTION SHAPE FOR SCOPE {bin_unique}: {df.shape}\n")

    df["customer_id"] = customer_hex_to_int64(df["customer_id"])
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

    last_shiftday = compute_shiftday_for_timestamp(pd.Timestamp(last_date_train))
    df = df.merge(
        weekly_sales.loc[weekly_sales["t_dat_shiftday"] == last_shiftday, ["count"]],
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

    age_fallback = target_sales.nlargest(N).index.to_pandas().tolist()
    age_fallback_ids_int = [int(article_id) for article_id in age_fallback]
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

    purchase_pd = purchase_df[["customer_id", "article_id"]].copy()
    purchase_pd = purchase_pd.rename(columns={"customer_id": "customer_id_2"})
    purchase_pd["customer_id_2"] = purchase_pd["customer_id_2"].astype("int64")
    purchase_pd["article_id"] = purchase_pd["article_id"].astype("int64")

    purchase_pd = (
        purchase_pd.groupby("customer_id_2")["article_id"]
        .apply(lambda s: s.tolist())
        .reset_index()
        .rename(columns={"article_id": "personal_ids"})
    )
    purchase_pd = purchase_pd.sort_values("customer_id_2").reset_index(drop=True)

    purchase_df_cu = cudf.from_pandas(purchase_pd)
    info_df(purchase_df_cu, count, order)

    sub_ids_cu = sub_all.loc[sub_mask, ["customer_id_2"]].copy()
    sub_ids_cu["customer_id_2"] = sub_ids_cu["customer_id_2"].astype("int64")

    sub_bin = sub_ids_cu.merge(purchase_df_cu, on="customer_id_2", how="left")
    sub_bin = sub_bin.to_pandas()

    sub_bin["personal_ids"] = sub_bin["personal_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    sub_bin["pred_ints"] = sub_bin["personal_ids"].apply(
        lambda ids: make_pred12_from_ints(
            ids, age_fallback_ids_int, top_global_ids_int, n=N
        )
    )

    sub_bin["prediction"] = sub_bin["pred_ints"].apply(
        lambda xs: " ".join(article_int_to_str10(x) for x in xs)
    )
    sub_bin = sub_bin.drop(columns=["personal_ids", "pred_ints"])

    sub_bin = sub_bin.sort_values("customer_id_2").reset_index(drop=True)
    masked_ids = (
        sub_all.loc[sub_mask, ["customer_id_2"]]
        .to_pandas()
        .sort_values("customer_id_2")
        .reset_index(drop=True)
    )
    assert masked_ids.shape[0] == sub_bin.shape[0]
    assert (masked_ids["customer_id_2"].values == sub_bin["customer_id_2"].values).all()

    sub_all.loc[sub_mask, "prediction"] = cudf.Series(
        sub_bin["prediction"].values, dtype="string"
    )

    print(f"FILLED PREDICTION FOR {bin_unique}, CUSTOMERS: {int(sub_mask.sum())}\n")
    print("-" * 50)

    count = 0

sub_all["prediction"] = sub_all["prediction"].astype("string")
sub_all["prediction"] = sub_all["prediction"].fillna("")
empty_mask = (sub_all["prediction"].str.len() == 0) | (sub_all["prediction"].isnull())
if bool(empty_mask.any()):
    sub_all.loc[empty_mask, "prediction"] = cudf.Scalar(top_global_str, dtype="string")

sub_pd_final = sub_all[["customer_id", "prediction"]].to_pandas()
sub_pd_final["prediction"] = sub_pd_final["prediction"].apply(
    lambda p: sanitize_pred_string(p, top_global_ids_int, n=N)
)
sub_all = cudf.from_pandas(sub_pd_final)

print("FINISHED")
print("=" * 50)



## === cell 26
submission = sub_all[["customer_id", "prediction"]].to_pandas()
submission.to_csv("submission.csv", index=False)

display_df(submission)



## === cell 27
check_df = cudf.read_csv("./submission.csv")
display_df(check_df)
assert (
    check_df.shape[0] == num_customers
), f"Row count mismatch: {check_df.shape[0]} vs {num_customers}"
assert set(check_df.columns) == {"customer_id", "prediction"}

assert int(check_df["prediction"].isnull().sum()) == 0

sample_preds = submission["prediction"].head(100).tolist()
for p in sample_preds:
    toks = p.split()
    assert len(toks) == 12
    assert all(len(t) == 10 and t.isdigit() for t in toks)
