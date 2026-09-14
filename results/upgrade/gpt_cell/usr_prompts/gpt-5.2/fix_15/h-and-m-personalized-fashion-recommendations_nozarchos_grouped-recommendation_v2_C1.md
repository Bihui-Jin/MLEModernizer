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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.00544

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Diagnosis: The pivot table in cell 21 creates MultiIndex columns for the aggregated `article_id` outputs, but the renaming logic doesn’t reliably match the second-level label for the custom `aggregate_to_list` function in pandas 2.2 (it may appear as a string like `'<lambda>'`/function name rather than the function object). As a result, `_pvt.rename(columns=col_map)` leaves the original MultiIndex columns in place, so selecting `["article_purchased","article_id_count"]` fails with a KeyError. The minimal deterministic fix is to flatten the pivoted columns into simple string names (as done later in cell 22) and then rename those flattened names to the expected columns.

Patch summary: In cell 21 only, replace the fragile MultiIndex tuple matching with a robust MultiIndex-flattening step and a direct rename from the flattened column names to `article_purchased` and `article_id_count`. This preserves the same pivot logic and downstream semantics while ensuring the expected column names exist for the merge and the final `sample_submission.columns = [...]` assignment.

Updated cells: Only cell 21 is changed.

Compatibility notes for cell k+1: No variables used by cell 22 are modified (`transactions_train` and `maximum_history_date` stay unchanged). The updated `sample_submission` DataFrame still ends with the exact same columns as intended: `["customer_id","age_group","index_name","article_purchased","article_id_count"]`.

Assumptions: Assumes pandas’ pivot_table produces MultiIndex columns that can be safely flattened into strings using the standard `col0_col1` pattern, and that the flattened names for the two aggregations become `article_id_count` and `article_id_aggregate_to_list` in this environment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the submission not matching the required schema: you overwrite `sample_submission` early with `customers[...]`, which can exclude customers that appear in `sample_submission.csv` but not in `customers.csv`, and you also save as `predict.csv` instead of the expected `submission.csv`. I make two minimal fixes: (1) start from the real `sample_submission.csv` customer list and merge `age_group` onto it (so every required customer is predicted), and (2) ensure the final file is written as `submission.csv` with exactly `customer_id,prediction` and aligned to the original sample submission order. These changes preserve your recommendation logic and only fix coverage/alignment so the score moves up toward the target.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with a submission-alignment/coverage issue (missing customers or NaN/invalid prediction strings) rather than the recommendation logic itself. I keep your exact recommendation approach, but make the pipeline always start from the official `sample_submission.csv` customer list and preserve its order, while also guaranteeing every row gets a valid 12-article prediction string. I also make the “top-12 popular fallback” deterministic and precomputed once (instead of recomputed inside a string join), and ensure article_id formatting stays consistent (10-digit zero-padded) across all generated predictions. These minimal fixes should move the score upward toward your target without changing the core modeling logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with invalid/empty predictions for a large fraction of rows after the `query("group_possibility != 0")` and `qty_to_recomend != 0` filters, plus a subtle bug in `articles_remove` where `recomendation.append(row["top_article_id"])` appends the entire list (creating nested lists) when purchase history is missing. I keep your exact recommendation logic, but (1) stop dropping customers during intermediate steps (so we always produce predictions for everyone), and (2) make `articles_remove` robust to missing `article_purchased`/`top_article_id` and avoid nested lists, while still removing already-bought items when possible. Finally, I enforce that every prediction is exactly up to 12 space-separated 10-digit `article_id`s (fallback-filled) and preserve the official sample submission order to ensure Kaggle scores it correctly.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is almost certainly coming from invalid/empty predictions caused by a pandas bug in cell 26: `group_recomendation.rename({"customer_id": "article_raiting"}, inplace=True)` doesn’t rename the column (it renames index labels by default), so later steps don’t behave as intended and many customers end up with poor/empty recommendation lists. I make the minimal fix to rename that column correctly and then keep everything else the same. I also add a tiny safety fallback so `top_article_id` is always a list (not NaN) before `articles_remove`, preventing silent empty outputs for whole groups. These changes preserve your exact approach (recent-popular by age_group/index_name with proportional allocation) but should move the score up from 0.0 toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from invalid predictions for a large fraction of customers (empty/NaN strings or not exactly space-separated article_ids), plus some customers potentially ending up with no group rows after the `query("customer_id > 2")` filter. I keep your exact recommendation logic, but add a deterministic “final guarantee” step that enforces every `prediction` is a non-empty string of up to 12 unique 10-digit `article_id`s, and that the submission is aligned 1:1 (and in-order) with the official `sample_submission.csv`. I also make the “popular fallback” computed from the pre-filtered recent transactions (as you already intend) and ensure all article_ids are consistently zero-padded before joining. These minimal changes should move the score up from 0.0 toward your target without changing your core approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is already far below the target (0.00544), so the smallest safe move is to fix issues that can silently destroy MAP@12 without changing your recommender logic. I keep your exact “age_group × index_name recent popularity + proportional allocation + remove already bought + global top-12 fallback” approach, but correct two places that can yield near-empty/invalid personalized lists: (1) ensure `article_id_count` is filled before computing `group_possibility` so allocation ratios aren’t NaN/zeroed, and (2) keep customers with no history from getting all-zero `qty_to_recomend` by applying a minimal fallback allocation rule. Finally, I make the “recent popularity” computed on the same filtered window you already use, but with a slightly less aggressive threshold (customer_id > 1 instead of > 2) to reduce missing `top_article_id` lists while staying within the same logic.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by reading/processing the full 31.5M-row transactions file and then using multiple expensive `pivot_table` operations plus a row-wise `apply` over the huge expanded submission table. I keep the exact same recommendation logic, but (1) read only the needed columns with fixed dtypes, (2) filter to the last 21 days *immediately* during/after load (so all later groupbys run on a much smaller frame), (3) replace `pivot_table(..., aggregate_to_list)` with equivalent `groupby().agg(list)` and `size()` (much faster), and (4) avoid per-row Python loops by doing the “remove already bought items + take qty” step per-customer using set membership against their purchased list. These changes are provably equivalent to your current semantics and should bring runtime under 600s by shrinking data early and removing the worst pandas bottlenecks.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by reading and processing the full 31.5M-row transactions file and by heavy merges/explodes on large intermediate DataFrames. I keep the exact same recommendation logic, but speed it up by (1) reading only the last 21 days directly via chunked CSV filtering (instead of loading everything then filtering), (2) switching joins to faster `map`-based lookups where semantically equivalent, (3) precomputing zero-filled article_id strings once and reusing them, and (4) eliminating the expensive explode+merge anti-join by using a per-customer `set` for “bought” and doing the same filtering/selection in one linear pass (equivalent to the prior logic). These changes preserve the algorithm and outputs (up to negligible ordering ties), but cut memory traffic and avoid the largest quadratic-style intermediate expansions.'

# 9. Code solution

## === cell 0
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import numpy as np
import os

warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None
pd.options.mode.copy_on_write = True

pd.set_option("mode.string_storage", "pyarrow")




## === cell 1
general_path = "../input/h-and-m-personalized-fashion-recommendations/"




## === cell 2
articles = pd.read_csv(
    general_path + "articles.csv",
    usecols=["article_id", "index_name"],
    dtype={"article_id": "int64", "index_name": "category"},
)
customers = pd.read_csv(
    general_path + "customers.csv",
    usecols=["customer_id", "age", "postal_code"],
    dtype={"customer_id": "string", "age": "float32", "postal_code": "string"},
)
sample_submission = pd.read_csv(
    general_path + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)

tx_path = general_path + "transactions_train.csv"

max_date = None
for chunk in pd.read_csv(
    tx_path,
    usecols=["t_dat"],
    parse_dates=["t_dat"],
    chunksize=2_000_000,
):
    cmax = chunk["t_dat"].max()
    if max_date is None or cmax > max_date:
        max_date = cmax

maximum_history_date = max_date
cutoff_21 = maximum_history_date - pd.Timedelta("21 days")

tx_chunks = []
for chunk in pd.read_csv(
    tx_path,
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"customer_id": "string", "article_id": "int64"},
    parse_dates=["t_dat"],
    chunksize=2_000_000,
):
    chunk = chunk[chunk["t_dat"] > cutoff_21]
    if not chunk.empty:
        tx_chunks.append(chunk)

transactions_train = (
    pd.concat(tx_chunks, ignore_index=True)
    if tx_chunks
    else pd.DataFrame(
        {
            "t_dat": pd.Series([], dtype="datetime64[ns]"),
            "customer_id": pd.Series([], dtype="string"),
            "article_id": pd.Series([], dtype="int64"),
        }
    )
)




## === cell 3
print(transactions_train.head(1))




## === cell 4
print(transactions_train.sample(5, random_state=0))




## === cell 5
articles["article_id"] = (
    articles["article_id"].astype("int64").astype("string").str.zfill(10)
)




## === cell 6
print(f"Maximum history date: {maximum_history_date} | cutoff_21: {cutoff_21}")




## === cell 7
print(f"We have {len(customers)} unique customers")
print(f'And {len(customers["postal_code"].unique())} unique locations')




## === cell 8
customer_per_location = (
    customers.groupby("postal_code", sort=False)["customer_id"]
    .size()
    .to_frame("customer_id")
)




## === cell 9
print(customer_per_location.sort_values(by=["customer_id"]).head())




## === cell 10
customers["location"] = "other"
customers.loc[
    customers["postal_code"]
    == "2c29ae653a9282cce4151bd87643c907644e09541abc28ae87dea0d1f6603b1c",
    "location",
] = "big_city"




## === cell 11
pass




## === cell 12
print(articles.nunique())




## === cell 13
print(articles.sample(1, random_state=0))




## === cell 14
print(articles["index_name"].unique()[:10])




## === cell 15
customers["age_group"] = "<20"
customers.loc[customers["age"] > 20, "age_group"] = "20-45"
customers.loc[customers["age"] > 45, "age_group"] = ">45"




## === cell 16
def aggregate_to_list(row):
    return [i for i in row]




## === cell 17
transactions_train["article_id"] = (
    transactions_train["article_id"].astype("int64").astype("string").str.zfill(10)
)

article_to_index = articles.set_index("article_id")["index_name"]
transactions_train["index_name"] = transactions_train["article_id"].map(
    article_to_index
)

cust_to_age = customers.set_index("customer_id")["age_group"]
transactions_train["age_group"] = transactions_train["customer_id"].map(cust_to_age)




## === cell 18
sub_customers = sample_submission[["customer_id"]].copy()
sub_customers["age_group"] = sub_customers["customer_id"].map(cust_to_age)

grp = transactions_train.groupby(["customer_id", "index_name"], sort=False)[
    "article_id"
]
_pvt = grp.agg(article_purchased=list, article_id_count="size").reset_index()

sample_submission = pd.merge(
    sub_customers,
    _pvt[["customer_id", "index_name", "article_purchased", "article_id_count"]],
    on="customer_id",
    how="left",
)

sample_submission.columns = [
    "customer_id",
    "age_group",
    "index_name",
    "article_purchased",
    "article_id_count",
]




## === cell 19
article_last_sale = (
    transactions_train.groupby("article_id", sort=False)["t_dat"]
    .max()
    .rename("t_dat_max")
)
long_time_have_not_sold = article_last_sale[
    article_last_sale < (maximum_history_date - pd.Timedelta("30 days"))
].index




## === cell 20
len_before = len(transactions_train)
transactions_train = transactions_train[
    ~transactions_train["article_id"].isin(long_time_have_not_sold)
].copy()
print("Removed {:.2%} of articles".format(1 - (len(transactions_train) / len_before)))




## === cell 21
pass




## === cell 22
group_recomendation = (
    transactions_train.groupby(["age_group", "index_name", "article_id"], sort=False)[
        "customer_id"
    ]
    .size()
    .reset_index(name="article_raiting")
)




## === cell 23
group_recomendation.sort_values(
    by=["age_group", "index_name", "article_raiting"], ascending=False, inplace=True
)

group_recomendation = group_recomendation.query("article_raiting > 1").copy()




## === cell 24
top_by_group = (
    group_recomendation.groupby(["age_group", "index_name"], sort=False)["article_id"]
    .agg(list)
    .reset_index(name="top_article_id")
)

sample_submission = pd.merge(
    sample_submission,
    top_by_group,
    on=["age_group", "index_name"],
    how="left",
)

sample_submission["top_article_id"] = sample_submission["top_article_id"].apply(
    lambda x: [str(a).zfill(10) for a in x] if isinstance(x, list) else []
)




## === cell 25
sample_submission["article_id_count"] = sample_submission["article_id_count"].fillna(0)

sample_submission["group_possibility"] = sample_submission.groupby("customer_id")[
    "article_id_count"
].transform("sum")

den = sample_submission["group_possibility"].fillna(0)
sample_submission["article_id_count"] = sample_submission[
    "article_id_count"
] / den.replace(0, pd.NA)

sample_submission["article_id_count"] = (
    sample_submission["article_id_count"].fillna(0) * 12
)
sample_submission["article_id_count"] = (
    sample_submission["article_id_count"].astype(float).round().astype("Int64")
)
sample_submission.rename(columns={"article_id_count": "qty_to_recomend"}, inplace=True)

sample_submission["qty_to_recomend"] = (
    sample_submission["qty_to_recomend"].fillna(0).astype("Int64")
)

mask_nohist = sample_submission["group_possibility"].fillna(0) == 0
if mask_nohist.any():
    first_idx = (
        sample_submission[mask_nohist].groupby("customer_id", sort=False).head(1).index
    )
    sample_submission.loc[first_idx, "qty_to_recomend"] = 12




## === cell 26
print(sample_submission.info())




## === cell 27
def articles_remove(row):
    bought = row.get("article_purchased", None)
    if not isinstance(bought, list):
        bought = []
    else:
        bought = [str(a).zfill(10) for a in bought]

    top_list = row.get("top_article_id", None)
    if not isinstance(top_list, list):
        top_list = []
    else:
        top_list = [str(a).zfill(10) for a in top_list]

    qty = row.get("qty_to_recomend", 0)
    if pd.isna(qty):
        qty = 0
    qty = int(qty)

    recomendation = []
    i = 0
    while len(recomendation) < qty and i < len(top_list):
        current_acticle = top_list[i]
        i += 1
        if current_acticle in bought:
            continue
        recomendation.append(current_acticle)
    return recomendation




## === cell 28
print(
    sample_submission[
        sample_submission["customer_id"]
        == "0118ed570ff6ff085cde55a6e801c6861a4e9ff9a8d9e82ee36b33c8b3af8f59"
    ].head()
)




## === cell 29
cust_bought = sample_submission.groupby("customer_id", sort=False)[
    "article_purchased"
].agg(lambda s: next((x for x in s if isinstance(x, list)), []))

cust_bought_set = cust_bought.apply(
    lambda lst: set(str(a).zfill(10) for a in lst) if isinstance(lst, list) else set()
)

work = sample_submission[["customer_id", "top_article_id", "qty_to_recomend"]].copy()
work["bought_set"] = work["customer_id"].map(cust_bought_set)
work["row_order"] = np.arange(len(work), dtype=np.int32)

cust_to_recs = {}
for cust, top_list, qty, bought_set in zip(
    work["customer_id"].to_numpy(),
    work["top_article_id"].to_numpy(),
    work["qty_to_recomend"].to_numpy(),
    work["bought_set"].to_numpy(),
):
    if cust not in cust_to_recs:
        cust_to_recs[cust] = []
    if not isinstance(top_list, list):
        continue
    q = int(qty) if qty is not pd.NA else 0
    if q <= 0:
        continue
    added = 0
    recs_list = cust_to_recs[cust]
    for a in top_list:
        a = str(a).zfill(10)
        if a in bought_set:
            continue
        recs_list.append(a)
        added += 1
        if added >= q:
            break

all_cust = work[["customer_id"]].drop_duplicates(keep="first")
sample_submission = all_cust.copy()
sample_submission["recomendation"] = sample_submission["customer_id"].map(cust_to_recs)
sample_submission["recomendation"] = sample_submission["recomendation"].apply(
    lambda x: x if isinstance(x, list) else []
)




## === cell 30
def lists_aggregate_to_list(row):
    output = []
    for i in row:
        if isinstance(i, list):
            output += i
    return output




## === cell 31
pass




## === cell 32
top12_popular = (
    transactions_train.groupby("article_id")["customer_id"]
    .count()
    .sort_values(ascending=False)
    .head(12)
    .index.astype(str)
    .str.zfill(10)
    .tolist()
)
fallback_pred = " ".join(top12_popular)

required_customers = (
    sample_submission[["customer_id"]].copy()
    if "prediction" in sample_submission.columns
    else None
)
required_customers = pd.read_csv(
    general_path + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)

unknown_customer_id = required_customers[
    ~required_customers["customer_id"].isin(sample_submission["customer_id"])
].copy()
unknown_customer_id["prediction"] = fallback_pred




## === cell 33
def _format_pred_list(x, fallback_list):
    if not isinstance(x, list):
        return fallback_list
    out = []
    seen = set()
    for a in x:
        a = str(a).zfill(10)
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
        if len(out) >= 12:
            break
    if len(out) == 0:
        return fallback_list
    if len(out) < 12:
        for a in fallback_list:
            if a not in seen:
                out.append(a)
                if len(out) >= 12:
                    break
    return out[:12]


fallback_list = top12_popular

sample_submission["prediction"] = sample_submission["recomendation"].apply(
    lambda x: " ".join(_format_pred_list(x, fallback_list))
)

sample_submission = pd.concat(
    [
        sample_submission[["customer_id", "prediction"]],
        unknown_customer_id[["customer_id", "prediction"]],
    ],
    ignore_index=True,
)

sub_template = required_customers.copy()
sample_submission = pd.merge(
    sub_template, sample_submission, on="customer_id", how="left"
)


def _clean_pred_str(s, fallback_pred_str, fallback_list_):
    if not isinstance(s, str):
        s = ""
    s = s.strip()
    if s == "":
        return fallback_pred_str
    parts = [p for p in s.split(" ") if p != ""]
    cleaned = []
    seen = set()
    for p in parts:
        p = str(p).zfill(10)
        if p in seen:
            continue
        seen.add(p)
        cleaned.append(p)
        if len(cleaned) >= 12:
            break
    if len(cleaned) == 0:
        return fallback_pred_str
    if len(cleaned) < 12:
        for a in fallback_list_:
            if a not in seen:
                cleaned.append(a)
                if len(cleaned) >= 12:
                    break
    return " ".join(cleaned[:12])


sample_submission["prediction"] = sample_submission["prediction"].apply(
    lambda s: _clean_pred_str(s, fallback_pred, fallback_list)
)

sample_submission = sample_submission[["customer_id", "prediction"]]
sample_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())
