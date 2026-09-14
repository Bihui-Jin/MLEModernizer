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

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime errors (the deprecated `json_normalize` import and the cuDF datetime slicing on a non-monotonic index) so the notebook runs end-to-end. Then I fix the core submission logic bug: your per-age-bin loop currently *drops* customers not in that bin (inner merge), so the final concatenated submission is missing many `customer_id`s, causing the “superset” error. The minimal correction is to build one final submission initialized from `sample_submission.csv` and fill predictions for customers that have age info and appear in each bin (left-join), while also handling missing-age customers with a fallback. This keeps your existing scoring logic (weekly quotient + recency weighting + general fallback) but makes the output valid and complete.'
- What this solution (achieved 0.0) has done: 'I fix the cuDF datetime slice error by avoiding value-based `.loc` slicing on a non-monotonic DatetimeIndex and instead filtering with boolean masks, which preserves the intent of the “recent” window without changing the modeling logic. Then I fix the age-bin masking bug that crashes in the per-bin loop: `age_bins` is an interval/categorical type and cannot be compared to `str(bin_unique)` inside cuDF, so I create a stable string key for bins (both in pandas and cuDF) and compare strings consistently. Finally, I keep the existing “fill per bin then fallback to global popular last-7-days” logic, ensuring every customer gets a non-empty prediction so the score moves up from 0.0 and the submission passes the format/superset checks.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a “valid CSV but effectively empty/garbled predictions” failure mode for MAP@12, and in this code the main culprit is prediction string construction and truncation: `str[:131]` can cut article_id tokens mid-way, producing invalid IDs that won’t match any ground-truth article_id, and concatenation can yield duplicate/partial tokens. I keep your per-age-bin modeling exactly the same, but change post-processing to always output exactly up to 12 *valid* 10-digit `article_id` tokens per customer (deduped, preserved order), without any character-based truncation. I also ensure that article_id formatting is robust (`zfill(10)` rather than `"0"+str(...)`) so IDs are always correct length, and apply the same token-safe logic to both bin-level and global fallback predictions. These are minimal changes that should move the score up toward your target without changing the underlying ranking logic.'
- What this solution (achieved 0.0) has done: 'I fix the runtime error in the per-bin loop by ensuring `purchase_df` stays a cuDF DataFrame until we explicitly convert, so `purchase_df["article_id"].to_pandas()` is called on a cuDF Series (not a pandas Series). I also make the customer-id join consistent by converting `tmp["customer_id"]` to the same int64 encoding used elsewhere before grouping/ranking, which prevents silent mismatches that can lead to mostly-empty per-customer predictions (and thus a 0.0 score). Finally, I keep your existing bin-wise model + global-pop fallback logic unchanged, only making the minimal type/merge corrections needed so it runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the immediate cuDF runtime error by removing the duplicate-column selection that triggers `ValueError: Selecting duplicate column labels is not supported`, while preserving your existing bin-wise quotient/recency ranking logic. Then I ensure the per-bin merge uses a consistent `int64` customer key and that the per-customer prediction strings are built only from full 10-digit article_id tokens (no character-based truncation). Finally, I keep your global last-7-days popularity fallback and write a complete `submission.csv` with exactly the sample’s customers and non-empty predictions so the score moves up from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the `KeyError: 'quotient'` by ensuring the pandas `tmp` DataFrame contains the `quotient` values computed in cuDF, using an index-aligned merge on `(t_dat, customer_id, article_id)` to preserve the exact existing logic. I also fix one cuDF join issue where `weekly_sales.loc[...]` is invalid (cuDF `.loc` can’t take a boolean mask like pandas) by replacing it with an equivalent boolean-filter + merge, keeping the same semantics. These changes are execution-blocking bugs; they don’t change the model’s ranking logic, but they allow the full pipeline to run and produce meaningful per-customer predictions, moving the score up from 0.0 toward your target. The submission writing/format and the global popularity fallback logic be kept intact.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 likely because the “age-bin model” is accidentally producing near-random rankings: it compares `weekly_sales["t_dat_shiftday"]` (a “shifted week boundary” date) to `last_date_train` (a raw transaction date), so `weekly_sales_last` becomes empty and `count_targ` is 0 for almost all rows, collapsing `quotient/value` to 0 and forcing almost everyone to fall back to generic popularity. I minimally fix this by computing `last_shiftday_train` as the max of `t_dat_shiftday` and using that for the “target week” slice, preserving your exact logic but making it actually non-zero. I also make one small alignment fix: ensure `df["t_dat_shiftday"]` is datetime in cuDF (not object from pandas values) so the groupby/merge and equality filter behave consistently. These changes should move you upward toward the target (higher-is-better) without changing the overall approach (bin-wise quotient + recency weighting + fallback).'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with “predictions never match ground truth” rather than a format error (your checks pass). The most likely cause in this code is that `customer_id` is encoded to `int64` for ranking, but the mapping back to `sub_all["customer_id_2"]` uses `str.hex_to_int()` which can be inconsistent across cuDF versions/inputs; that silently mismatches customers, yielding almost all customers getting only the global-pop fallback (and for many, effectively irrelevant). I make a minimal, semantics-preserving fix by computing `customer_id_2` via a stable CPU-side hex conversion (pandas) and using the exact same conversion for transaction customer ids, so joins align. This keeps your bin-wise quotient/recency model and fallback logic intact, but should move the score up toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests “predictions never match ground truth,” and in this pipeline the most likely cause is the customer key mismatch: you rank using an int64-encoded customer id (`customer_id_2`) but only convert transaction customer_ids within each bin, risking inconsistent conversions across loops and leaving most customers effectively filled by generic/popular-only predictions. I make a minimal, semantics-preserving change: compute a single stable `customer_id_2` mapping once from the full transactions file (same conversion function as sample) and reuse it for every bin via a merge, keeping your per-bin ranking logic unchanged. I also stop re-reading the 31M-row CSV inside each bin (same data, just filtered), which avoids timeouts and reduces the chance of partial/failed bin fills that can also yield near-zero MAP. Finally, I keep your existing post-processing (token-safe formatting + global 7-day popularity fallback) and still write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the execution-blocking cuDF merge that overflows device memory by avoiding the huge `tx_base.merge(...)` on 31M rows and instead precomputing a stable `customer_id -> customer_id_2` mapping once from the unique customer ids, then mapping `customer_id_2` onto transactions via a vectorized pandas join before converting back to cuDF. This keeps your core bin-wise ranking logic intact (same quotient/recency/value computation), but makes the pipeline run end-to-end within resource limits. I also ensure `tx_base` always contains `customer_id_2` (fixing the downstream `KeyError`) and keep the existing token-safe prediction formatting + global-pop fallback so every customer gets up to 12 valid 10-digit article ids. These changes are aimed at moving your score up from 0.0 toward the target by producing non-garbled, customer-aligned predictions without changing the ranking semantics.'
- What this solution (achieved 0.0) has done: 'I fix the runtime error in the bin-wise loop by ensuring `last_date_train` is converted to a pandas `Timestamp` before doing pandas timedelta arithmetic (it’s currently a `numpy.datetime64`, so `.to_pandas()` fails). I also make the `t_dat` column in `tx_base` explicitly datetime after the pandas round-trip so date math and sorting behave consistently across cuDF/pandas boundaries. These are execution-blocking/correctness fixes that preserve your exact ranking logic (quotient + recency weighting + per-age-bin fill + global popularity fallback). The output still be a full `submission.csv` with all sample customers and up to 12 valid 10-digit `article_id` tokens per row, which should move the score up from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates predictions are not matching ground truth even though the CSV is valid; the most likely cause here is a silent customer-key mismatch introduced by converting `customer_id` to an `int64` via hex parsing (this is not the same encoding as the competition’s `customer_id` strings), which breaks the per-customer ranking/aggregation alignment. I keep your bin-wise quotient/recency ranking logic identical, but replace the hex-based `customer_id_2` with a stable categorical integer code derived from the actual `customer_id` strings, built consistently for both sample customers and transactions. This is a minimal change that directly increases the chance predictions are assigned to the correct customers (moving score up toward 0.02244). I also avoid re-reading transactions for global popularity by reusing the already-loaded `transactions_df`, preserving the same “last-7-days popularity” semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 strongly suggests that predictions are being assigned to the wrong customers (or mostly collapsing to generic fallback) even though the CSV passes basic format checks. The smallest, core-logic-preserving fix is to stop using an arbitrary “customer_id_2” coding that depends on concatenation order and instead use a deterministic hash-derived int64 key computed the same way for both transactions and the sample submission, so all joins/groupbys align reliably. I keep your exact per-age-bin quotient/recency/value ranking and your fallback logic, only replacing the customer key mapping + the few merges that depend on it. This should move the score upward toward your target by ensuring bin-wise personalized predictions actually land on the intended `customer_id`s.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with a customer-key mismatch: you generate predictions keyed by `customer_id_2` (a hash), but in the bin loop you merge on `sub_all.customer_id_2` (hash of sample customers) vs `purchase_df.customer_id` (hash of bin customers), and those hashes are not guaranteed to match across independently computed subsets in a stable way. The minimal, core-logic-preserving fix is to compute a single deterministic `customer_id_2` mapping once from the full set of customer_ids (customers table) and reuse it everywhere (sub_all + tx_base + per-bin customer lists), so all joins/groupbys align. I also keep your bin-wise quotient/recency/value ranking unchanged and only adjust the key construction + merges to ensure personalized predictions are assigned to the correct `customer_id`. This should move the score upward from 0.0 toward your target without changing the model logic.'

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
def format_article_id(a) -> str:
    s = str(a)
    s = re.sub(r"\D", "", s)  # keep digits only, defensive
    return s.zfill(10)


def finalize_pred_tokens(primary_tokens, fallback_tokens, n=12):
    out = []
    seen = set()
    for tok in list(primary_tokens) + list(fallback_tokens):
        if tok is None:
            continue
        t = str(tok).strip()
        if not t:
            continue
        t = format_article_id(t)
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= n:
            break
    return " ".join(out)


def prediction_str_to_tokens(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [t for t in s.split() if t.strip()]




## === cell 7
def customer_id_to_int64(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return pd.util.hash_pandas_object(s, index=False).astype("uint64").astype("int64")




## === cell 8
articles_df = cudf.read_csv(
    PATH_INPUT + "articles.csv",
    usecols=["article_id", "product_group_name", "perceived_colour_master_name"],
)
display_df(articles_df)



## === cell 9
customers_df = cudf.read_csv(
    PATH_INPUT + "customers.csv", usecols=["customer_id", "age"]
)
display_df(customers_df)



## === cell 10
customers_df = customers_df.to_pandas()
bin_list = [-1, 19, 29, 39, 49, 59, 69, 119]
customers_df["age_bins"] = pd.cut(customers_df["age"], bin_list)
customers_df["age_bin_str"] = customers_df["age_bins"].astype(str)  # includes "nan"

cust_key_pd = customers_df[["customer_id"]].drop_duplicates().copy()
cust_key_pd["customer_id_2"] = customer_id_to_int64(cust_key_pd["customer_id"])
cust_key_cu = cudf.from_pandas(cust_key_pd)

display_df(customers_df)
display_df(cust_key_pd)



## === cell 11
age_missing = customers_df[customers_df["age_bins"].isnull()].shape[0]
age_missing



## === cell 12
transactions_df = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
)
transactions_df["t_dat"] = cudf.to_datetime(transactions_df["t_dat"])
transactions_df = transactions_df.sort_values("t_dat")
transactions_df = transactions_df.set_index("t_dat")
display_df(transactions_df)



## === cell 13
start = np.datetime64("2020-09-01")
end = np.datetime64("2020-09-21")
idx = transactions_df.index
recent_df = transactions_df[(idx >= start) & (idx <= end)]
display_df(recent_df)



## === cell 14
recent_df = recent_df.to_pandas()
recent_df = recent_df.merge(
    customers_df[["customer_id", "age_bins"]], on="customer_id", how="inner"
)
display_df(recent_df)



## === cell 15
recent_df = (
    recent_df.groupby(["age_bins", "article_id"])
    .count()
    .reset_index()
    .rename(columns={"customer_id": "counts"})
)
display_df(recent_df)



## === cell 16
bins_unique_list = recent_df["age_bins"].unique().tolist()
bins_unique_list



## === cell 17
if len(bins_unique_list) > 0:
    bins_unique_list[0]



## === cell 18
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



## === cell 19
from itertools import islice

for key, value in islice(ages_article_id.items(), 3):
    print(f"{key} {len(value)}")



## === cell 20
topcustcnt_byage_df = pd.DataFrame([ages_article_id])
display_df(topcustcnt_byage_df)



## === cell 21
topcustcnt_byage_df = pd.DataFrame([ages_article_id]).T.rename(columns={0: "top_100"})
display_df(topcustcnt_byage_df)



## === cell 22
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



## === cell 23
topcustcnt_byage_df = topcustcnt_byage_df.drop(columns="top_100")
display_df(topcustcnt_byage_df, head=10)



## === cell 24
if topcustcnt_byage_df.shape[0] > 0 and topcustcnt_byage_df.shape[1] > 0:
    plt.figure(figsize=(10, 6))
    sns.heatmap(topcustcnt_byage_df, cmap="winter", annot=True, cbar=False)



## === cell 25
bins_unique_list = customers_df["age_bin_str"].unique().tolist()
bins_unique_list



## === cell 26
sub_all = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
num_customers = sub_all.shape[0]
sub_all["prediction"] = cudf.Series([""] * num_customers, dtype="str")

sub_all = sub_all.merge(cust_key_cu, on="customer_id", how="left")

cust_age_cudf = cudf.from_pandas(customers_df[["customer_id", "age", "age_bin_str"]])
sub_all = sub_all.merge(
    cust_age_cudf[["customer_id", "age", "age_bin_str"]], on="customer_id", how="left"
)
info_df(sub_all, COUNT, ORDER)



## === cell 27
tx_base = transactions_df.reset_index()[["t_dat", "customer_id", "article_id"]].copy()
tx_base = tx_base.merge(cust_key_cu, on="customer_id", how="left")
tx_base = tx_base[["t_dat", "customer_id_2", "article_id"]]
tx_base["t_dat"] = cudf.to_datetime(tx_base["t_dat"])
info_df(tx_base, COUNT, ORDER)



## === cell 28
count = COUNT
order = ORDER

for bin_unique in bins_unique_list:
    if str(bin_unique) == "nan":
        temp_customers_pd = customers_df[customers_df["age_bin_str"] == "nan"][
            ["customer_id", "age"]
        ]
        bin_mask = sub_all["age_bin_str"].isnull() | (sub_all["age_bin_str"] == "nan")
    else:
        temp_customers_pd = customers_df[
            customers_df["age_bin_str"] == str(bin_unique)
        ][["customer_id", "age"]]
        bin_mask = sub_all["age_bin_str"] == str(bin_unique)

    temp_customers_df = cudf.from_pandas(temp_customers_pd)
    info_df(temp_customers_df, count, order)

    if temp_customers_df.shape[0] == 0:
        print(f"NO CUSTOMERS FOR {bin_unique} -> SKIP BIN MODEL")
        print("-" * 50)
        count = 0
        continue

    temp_customers_key = temp_customers_df.merge(
        cust_key_cu, on="customer_id", how="left"
    )

    df = tx_base.merge(
        temp_customers_key[["customer_id_2"]], on="customer_id_2", how="inner"
    )
    info_df(df, count, order)

    print(f"TRANSACTION SHAPE FOR SCOPE {bin_unique}: {df.shape}\n")

    if df.shape[0] == 0:
        print(f"NO TRANSACTIONS FOR {bin_unique} -> SKIP BIN MODEL")
        print("-" * 50)
        count = 0
        continue

    last_date_train = df["t_dat"].max()
    last_date_train_pd = pd.to_datetime(last_date_train)

    if count:
        print("=" * 30 + f"\n4 LAST_DATE_TRAIN:\n{last_date_train_pd}")

    tmp = df[["t_dat", "customer_id_2", "article_id"]].copy().to_pandas()
    tmp = tmp.rename(columns={"customer_id_2": "customer_id"})
    tmp["customer_id"] = tmp["customer_id"].astype("int64")
    tmp["t_dat"] = pd.to_datetime(tmp["t_dat"])

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

    df_cu = cudf.from_pandas(
        tmp[["t_dat", "customer_id", "article_id", "t_dat_shiftday"]]
    )
    df_cu["t_dat"] = cudf.to_datetime(df_cu["t_dat"])
    df_cu["t_dat_shiftday"] = cudf.to_datetime(df_cu["t_dat_shiftday"])
    info_df(df_cu, count, order)

    weekly_sales = (
        df_cu.drop("customer_id", axis=1)
        .groupby(["t_dat_shiftday", "article_id"])
        .count()
        .reset_index()
    )
    info_df(weekly_sales, count, order)

    weekly_sales = weekly_sales.rename(columns={"t_dat": "count"})
    info_df(weekly_sales, count, order)

    df_cu = df_cu.merge(weekly_sales, on=["t_dat_shiftday", "article_id"], how="left")
    info_df(df_cu, count, order)

    last_shiftday_train = df_cu["t_dat_shiftday"].max()
    weekly_sales_last = weekly_sales[
        weekly_sales["t_dat_shiftday"] == last_shiftday_train
    ][["article_id", "count"]].rename(columns={"count": "count_targ"})
    info_df(weekly_sales_last, count, order)

    df_cu = df_cu.merge(weekly_sales_last, on="article_id", how="left")
    info_df(df_cu, count, order)

    df_cu["count_targ"].fillna(0, inplace=True)
    df_cu["quotient"] = df_cu["count_targ"] / df_cu["count"]
    info_df(df_cu, count, order)

    target_sales = (
        df_cu.drop("customer_id", axis=1).groupby("article_id")["quotient"].sum()
    )
    info_df(target_sales, count, order)

    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = [format_article_id(article_id) for article_id in general_pred]
    general_pred_str = " ".join(general_pred)
    del target_sales

    df_q = df_cu[["t_dat", "customer_id", "article_id", "quotient"]].to_pandas()
    df_q["customer_id"] = df_q["customer_id"].astype("int64")
    df_q["t_dat"] = pd.to_datetime(df_q["t_dat"])

    tmp = tmp.merge(df_q, on=["t_dat", "customer_id", "article_id"], how="left")
    tmp["quotient"] = tmp["quotient"].fillna(0.0)
    info_df(tmp, count, order)

    tmp["x"] = ((last_date_train_pd - tmp["t_dat"]) / np.timedelta64(1, "D")).astype(
        int
    )
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

    purchase_gdf = cudf.from_pandas(purchase_df)

    purchase_gdf["pred_tok"] = (
        purchase_gdf["article_id"].to_pandas().map(format_article_id).values
    )
    info_df(purchase_gdf, count, order)

    purchase_df = (
        purchase_gdf[["customer_id", "pred_tok"]]
        .groupby("customer_id")
        .agg({"pred_tok": list})
        .reset_index()
    )
    info_df(purchase_df, count, order)

    sub_bin = sub_all.loc[bin_mask, ["customer_id_2"]].merge(
        purchase_df, left_on="customer_id_2", right_on="customer_id", how="left"
    )
    sub_bin = sub_bin.to_pandas()

    fallback_tokens = prediction_str_to_tokens(general_pred_str)
    sub_bin["prediction"] = sub_bin["pred_tok"].apply(
        lambda lst: finalize_pred_tokens(
            lst if isinstance(lst, list) else [], fallback_tokens, n=N
        )
    )
    sub_bin = sub_bin[["prediction"]]

    sub_all_pd = sub_all.to_pandas()
    sub_all_pd.loc[np.array(bin_mask.to_pandas()), "prediction"] = sub_bin[
        "prediction"
    ].values
    sub_all = cudf.from_pandas(sub_all_pd)

    print(f"FILLED PREDICTION FOR BIN {bin_unique}, ROWS: {int(bin_mask.sum())}")
    print("-" * 50)

    count = 0

print("FINISHED BIN MODELS")
print("=" * 50)



## === cell 29
tx = transactions_df.reset_index()[["t_dat", "article_id"]].copy()
tx["t_dat"] = cudf.to_datetime(tx["t_dat"])
max_date = tx["t_dat"].max()
tx = tx[tx["t_dat"] >= (max_date - np.timedelta64(7, "D"))]
pop = (
    tx.groupby("article_id")
    .size()
    .sort_values(ascending=False)
    .head(N)
    .index.to_pandas()
    .tolist()
)

global_pop_tokens = [format_article_id(a) for a in pop]
global_pop_str = " ".join(global_pop_tokens)

sub_all_pd = sub_all.to_pandas()

sub_all_pd["prediction"] = sub_all_pd["prediction"].replace("", np.nan).fillna("")
sub_all_pd["prediction"] = sub_all_pd["prediction"].apply(
    lambda s: finalize_pred_tokens(prediction_str_to_tokens(s), global_pop_tokens, n=N)
)

sub_all_pd = sub_all_pd[["customer_id", "prediction"]]
sub_all_pd.to_csv("submission.csv", index=False)

print(sub_all_pd.shape)
print(sub_all_pd.head())



## === cell 30
check = pd.read_csv(PATH_INPUT + "sample_submission.csv", usecols=["customer_id"])
sub_check = pd.read_csv("submission.csv")

assert (
    sub_check.shape[0] == check.shape[0]
), f"Row count mismatch: {sub_check.shape[0]} vs {check.shape[0]}"
assert set(check["customer_id"]).issubset(
    set(sub_check["customer_id"])
), "Submission customer_id must be a superset of sample_submission customer_id"
assert sub_check["prediction"].isna().sum() == 0, "prediction contains NaN"
assert (
    sub_check["prediction"].str.len() > 0
).all(), "Some prediction strings are empty"
print("submission.csv looks valid:", sub_check.shape)
