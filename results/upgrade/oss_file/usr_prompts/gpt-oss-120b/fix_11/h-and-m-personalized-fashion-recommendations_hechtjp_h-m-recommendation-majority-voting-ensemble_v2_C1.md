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

0.0227384268563794

# 6. Current score

0.02024

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The original script crashes because it tries to import and use **cudf**, which requires a GPU driver that isn’t available in the environment. I replaced all cudf usage with standard **pandas**, added a safe fallback to locate the `sample_submission.csv`, and kept the simple workflow that reads the sample submission and writes it back as `submission.csv`. This restores execution, produces a correctly‑formatted CSV, and provides a valid baseline submission for the competition.'
- What this solution (achieved 0.00296) has done: 'I add a lightweight popularity baseline: after loading the sample submission I locate the training transactions file, count article frequencies in a memory‑efficient chunked loop, pick the 12 most frequent article_ids, and fill every customer's prediction with that space‑separated list. This keeps the original workflow intact while providing non‑trivial predictions, moving the MAP@12 score from 0.0 toward the target 0.0227.'
- What this solution (achieved 0.0) has done: 'Implemented a single‑pass read of the massive `transactions_train.csv` to:
- Update the global article frequency Counter and the per‑customer article Counters together, halving disk I/O.
- Use `dtype=str` to avoid costly `astype` calls.
- Iterate over the grouped `(customer_id, article_id)` counts directly, keeping the exact same counting logic.
- Pre‑compute each customer’s top‑12 list once after the file is processed, then build predictions without extra loops.

These changes preserve the original counting and fallback‑to‑global‑top‑articles logic while substantially reducing runtime, keeping the core algorithm identical.'
- What this solution (achieved 0.00768) has done: 'I fixed the unpacking error in the transaction‑processing loop: the `itertuples` call yields three fields (`customer_id`, `article_id`, `cnt`), so we now iterate directly over those three values and update the per‑customer counters. This resolves the runtime crash, lets the script finish, and produces a valid `submission.csv` with reasonable popularity‑based predictions, moving the MAP@12 score toward the target.'
- What this solution (achieved 0.0) has done: 'I add a lightweight two‑pass read of the transaction data to capture recent purchase behaviour (last 30 days) and use those recent counts before the overall popularity counts when building each customer’s top‑12 list. This only extends the existing counting logic, keeps the same model structure, and is expected to increase MAP@12 toward the target without altering the core workflow.'
- What this solution (achieved 0.0202) has done: 'The fix ensures the transaction dates are read as actual timestamps rather than strings, allowing proper date comparison for the recent‑cutoff logic. This resolves the TypeError and lets the popularity‑based recommendation pipeline run end‑to‑end, producing a valid `submission.csv` that should achieve a higher MAP@12 score.'
- What this solution (achieved 0.01966) has done: 'We give recent purchases a higher weight when ranking each customer’s recommendations. By multiplying recent counts (e.g., × 2) and adding overall counts, the top‑12 list better reflects short‑term interest while keeping the original popularity baseline, which should raise MAP@12 toward the target without altering the core workflow.'
- What this solution (achieved 0.02013) has done: 'I increase the influence of recent purchases by raising the `RECENT_WEIGHT` from 2 to 3. This modest adjustment gives more importance to items bought in the last 30 days, which should improve the MAP@12 score and move it closer to the target without changing the overall algorithm.'
- What this solution (achieved 0.02024) has done: 'I increased the recent‑purchase influence from 3 to 4 and recomputed the global popularity list using a weighted combination of overall and recent counts, so recent trends affect both per‑customer rankings and the fallback list. This modest tweak should raise the MAP@12 a bit, moving the score closer to the target while keeping the original algorithm intact.'

# 9. Code solution

## === cell 0
import warnings, os, pathlib, random, pickle, gc, copy, re, sys, time

warnings.filterwarnings("ignore")
from tqdm import tqdm

tqdm.pandas()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from collections import Counter, defaultdict




## === cell 1
DEBUG = False




## === cell 2
def display_df(df, head=3):
    """Print shape and show first rows of a DataFrame."""
    print(f"The shape of df is {df.shape}.\n")
    display(df.head(head))




## === cell 3
possible_paths = [
    "./sample_submission.csv",
    "./kaggle/data/sample_submission.csv",
    "../input/h-m-personalized-fashion-recommendations/sample_submission.csv",
    "../input/sample_submission.csv",
]

sample_path = None
for p in possible_paths:
    if pathlib.Path(p).exists():
        sample_path = p
        break

if sample_path is None:
    raise FileNotFoundError("sample_submission.csv not found in any expected location.")

df_sample = pd.read_csv(sample_path)
display_df(df_sample, head=3)




## === cell 4
expected_cols = {"customer_id", "prediction"}
if not expected_cols.issubset(set(df_sample.columns)):
    rename_map = {}
    for col in df_sample.columns:
        if col.lower() == "customer_id":
            rename_map[col] = "customer_id"
        elif col.lower() == "prediction":
            rename_map[col] = "prediction"
    df_sample = df_sample.rename(columns=rename_map)

assert expected_cols.issubset(
    set(df_sample.columns)
), "Missing required columns in sample submission."




## === cell 5
train_paths = [
    "./transactions_train.csv",
    "./kaggle/data/transactions_train.csv",
    "../input/h-m-personalized-fashion-recommendations/transactions_train.csv",
    "../input/transactions_train.csv",
]

train_path = None
for p in train_paths:
    if pathlib.Path(p).exists():
        train_path = p
        break

if train_path is None:
    raise FileNotFoundError(
        "transactions_train.csv not found in any expected location."
    )

chunksize = 1_000_000

max_date = pd.Timestamp.min
for chunk in tqdm(
    pd.read_csv(
        train_path,
        usecols=["t_dat"],
        parse_dates=["t_dat"],
        infer_datetime_format=True,
        chunksize=chunksize,
    ),
    desc="Finding max date",
):
    chunk_max = chunk["t_dat"].max()
    if pd.notnull(chunk_max) and chunk_max > max_date:
        max_date = chunk_max

recent_cutoff = max_date - pd.Timedelta(days=30)

article_counter = Counter()  # overall popularity
article_recent_counter = Counter()  # recent popularity
cust_counter = defaultdict(Counter)  # overall per‑customer
cust_recent_counter = defaultdict(Counter)  # recent per‑customer

RECENT_WEIGHT = 4

for chunk in tqdm(
    pd.read_csv(
        train_path,
        usecols=["customer_id", "article_id", "t_dat"],
        dtype={"customer_id": str, "article_id": str},
        parse_dates=["t_dat"],
        infer_datetime_format=True,
        chunksize=chunksize,
    ),
    desc="Processing transactions with recency",
):
    article_counter.update(chunk["article_id"])

    grp = chunk.groupby(["customer_id", "article_id"]).size().reset_index(name="cnt")
    for cid, aid, cnt in grp.itertuples(index=False):
        cust_counter[cid][aid] += cnt

    recent_mask = chunk["t_dat"] >= recent_cutoff
    if recent_mask.any():
        recent_chunk = chunk[recent_mask]

        article_recent_counter.update(recent_chunk["article_id"])

        grp_rec = (
            recent_chunk.groupby(["customer_id", "article_id"])
            .size()
            .reset_index(name="cnt")
        )
        for cid, aid, cnt in grp_rec.itertuples(index=False):
            cust_recent_counter[cid][aid] += cnt

top_recent_articles = [aid for aid, _ in article_recent_counter.most_common(12)]

global_combined = Counter()
for aid, cnt in article_counter.items():
    global_combined[aid] += cnt
for aid, cnt in article_recent_counter.items():
    global_combined[aid] += cnt * RECENT_WEIGHT
top_articles = [aid for aid, _ in global_combined.most_common(12)]

customer_top12 = {}
for cid, overall_ctr in cust_counter.items():
    recent_ctr = cust_recent_counter.get(cid, Counter())

    combined_ctr = Counter()
    for aid, cnt in recent_ctr.items():
        combined_ctr[aid] += cnt * RECENT_WEIGHT
    for aid, cnt in overall_ctr.items():
        combined_ctr[aid] += cnt

    top_user = [aid for aid, _ in combined_ctr.most_common(12)]

    if len(top_user) < 12:
        for aid, _ in recent_ctr.most_common():
            if aid not in top_user:
                top_user.append(aid)
            if len(top_user) == 12:
                break
    if len(top_user) < 12:
        for aid, _ in overall_ctr.most_common():
            if aid not in top_user:
                top_user.append(aid)
            if len(top_user) == 12:
                break
    if len(top_user) < 12:
        for aid in top_recent_articles:
            if aid not in top_user:
                top_user.append(aid)
            if len(top_user) == 12:
                break
    if len(top_user) < 12:
        for aid in top_articles:
            if aid not in top_user:
                top_user.append(aid)
            if len(top_user) == 12:
                break

    customer_top12[cid] = top_user

fallback_top12 = (
    top_recent_articles[:12] if len(top_recent_articles) >= 12 else top_articles[:12]
)

predictions = []
for cid in df_sample["customer_id"]:
    top_user = customer_top12.get(cid, fallback_top12)
    predictions.append(" ".join(top_user))

df_sample["prediction"] = predictions




## === cell 6
if DEBUG:
    df_sample = df_sample.sample(frac=0.001, random_state=7)
    display_df(df_sample, head=3)




## === cell 7
output_path = "submission.csv"
df_sample.to_csv(output_path, index=False)
print(f"Saved {output_path} with {len(df_sample)} rows.")




## === cell 8
df_check = pd.read_csv(output_path)
display_df(df_check, head=3)
