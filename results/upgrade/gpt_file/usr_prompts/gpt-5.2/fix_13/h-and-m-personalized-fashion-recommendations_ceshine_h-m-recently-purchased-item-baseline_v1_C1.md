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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.02048

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your code didn’t yield a Kaggle score because it writes `.csv.gz` files; many Kaggle submission upload flows expect a plain `.csv` (and “Not yielded” often happens when no valid `.csv` is produced or uploaded). I keep the exact same “most common items” and “latest purchased with fallback” logic, but additionally write uncompressed `.csv` files with the same content so you can submit successfully. I also make one minimal robustness fix in `keep_latest_k` by converting `most_frequent_items` to Python ints to avoid any potential numpy-scalar quirks during membership checks, without changing the intended predictions. The rest of the pipeline, filtering, and prediction formatting remain identical.'
- What this solution (achieved 0.0) has done: 'Your current logic should score above 0 if Kaggle accepts the file, so the most likely reason for the 0.0 is a submission formatting issue (most commonly: `article_id` got converted to integers and then zero-padded differently than the original 10-digit article IDs, or the file you uploaded wasn’t the plain `.csv`). I keep the exact same “top-12 popular items” and “last purchases within 7 weeks with fallback to popular” approach, but preserve `article_id` as 10-character strings end-to-end so predictions match the expected ID format. I also ensure we only write plain `.csv` outputs and that every `prediction` has exactly 12 space-separated items. These are minimal changes that directly address MAP@12 validity and should move your score up toward the target.'
- What this solution (achieved 0.0) has done: 'Your current pipeline should score above 0, so the 0.0 strongly suggests a submission validity/format issue rather than a modeling issue. I keep the exact same “top-12 popular in last 2 weeks” and “latest purchases in last 7 weeks with popular fallback” logic, but ensure `article_id` stays as 10-character strings at every step (including inside `keep_latest_k`) and that every row outputs exactly 12 space-separated 10-digit IDs. I also write plain `.csv` submissions (not compressed) and add lightweight sanity checks (row count and nulls) to prevent accidentally uploading an invalid file. These are minimal changes directly aimed at making Kaggle properly parse your submission and thus increase the score toward your target.'
- What this solution (achieved 0.0) has done: 'Your current logic should score above 0, so the 0.0 strongly suggests the uploaded file didn’t match Kaggle’s expected schema/content (most commonly: wrong columns because `customer_id` was dropped earlier, or `article_id` formatting not preserved as 10-digit strings in every path). I keep the exact same “top-12 popular items in last 2 weeks” plus “customer latest items within 7 weeks with popular fallback” logic, but fix the `customer_id` handling so it’s always available when writing submissions. I also make `keep_latest_k` explicitly accept the fallback list to avoid any hidden state issues, and add a final hard validation that the written CSV has the correct columns, row count, and exactly 12 space-separated 10-char article IDs per row. These are minimal, execution- and format-critical fixes aimed at getting a non-zero MAP@12 and moving toward your target score.'
- What this solution (achieved 0.0) has done: 'Your current logic should score above 0, so the 0.0 strongly indicates Kaggle isn’t matching your predictions to the true `article_id`s due to formatting. The smallest fix is to ensure every predicted `article_id` is a 10-digit string everywhere (including the “latest purchased” path) and to prevent any accidental conversion to scientific notation or truncated integers when reading/writing. I also make the “popular fallback” list explicitly derived from already-zero-padded strings, and add one final submission re-load sanity check to confirm the CSV roundtrips with the correct schema. Core logic (popular-in-last-2-weeks baseline + last-7-weeks per customer with fallback) is unchanged.'
- What this solution (achieved 0.0) has done: 'Your current logic should score above 0, so the 0.0 strongly suggests a prediction/content mismatch rather than modeling weakness. The smallest likely issue is customer key mismatching: converting `customer_id` hex to `int64` can overflow and collide (many 16-hex-digit values exceed signed int64), causing the merge to assign wrong predictions or mostly fall back to global popular items. I keep the exact same “popular last 2 weeks” + “latest purchases last 7 weeks with popular fallback” logic, but change the join key to the original `customer_id` string (no overflow risk) and keep the rest identical. I also add a quick collision check on the old int key (informational) and keep the same submission-writing and validation to ensure Kaggle accepts it.'
- What this solution (achieved 0.0) has done: 'Your current approach should score above 0, so the most likely reason for 0.0 is corrupted customer matching caused by converting `customer_id` hex strings into `int` (many exceed int64; collisions/mismatches can make predictions effectively random). To move the score up toward your target with minimal logic changes, I keep your exact “popular last 2 weeks” + “latest purchases last 7 weeks with popular fallback” method, but remove the `customer_id_int` conversion from the merge path and merge strictly on the original `customer_id` string. I also keep `article_id` as zero-padded 10-char strings end-to-end and retain your final submission sanity checks, so Kaggle parses IDs correctly. The output remains a plain `.csv` with the required columns and exactly 12 predictions per customer.'
- What this solution (achieved 0.0) has done: 'Your core logic is already a known baseline that should score non-zero, so the most probable cause of 0.0 is that many (or all) `customer_id`s in the written submission do not exactly match Kaggle’s expected IDs due to hidden whitespace/encoding artifacts. I make one minimal, score-relevant robustness change: normalize `customer_id` (strip whitespace + enforce lowercase) consistently in both `transactions_train` and `sample_submission` before the groupby/merge, keeping all recommendation logic identical. I also add a strict post-write check that the saved CSV’s `customer_id` set exactly matches the sample submission’s `customer_id` set (this catches the “valid file but wrong keys” failure mode that yields ~0). No model/feature/training logic changes are introduced; only key normalization and stronger validation.'
- What this solution (achieved 0.0) has done: 'Your code already implements a reasonable baseline, so a 0.0 score strongly suggests a submission parsing/matching issue rather than weak recommendations. The smallest high-impact fix is to ensure `customer_id` is *not* lowercased: H&M customer IDs are hex-like strings and Kaggle expects exact matches, so changing case can break the join to the ground truth and yield ~0. I keep your exact popular-items + latest-7-weeks-with-fallback logic, but remove the `.str.lower()` normalization everywhere and tighten the final integrity check to ensure the saved CSV customer_ids match the sample submission exactly. This should move your score up toward (and likely past) your target band without changing the model logic.'
- What this solution (achieved 0.0) has done: 'Your current recommendation logic should yield a non-zero MAP@12, so the 0.0 strongly suggests Kaggle couldn’t correctly match your submitted `customer_id`s or `article_id`s due to subtle formatting changes. I keep your exact “top-12 popular from last 2 weeks + latest items from last 7 weeks with popular fallback” logic, but tighten ID handling to avoid any hidden dtype/whitespace issues: enforce `customer_id` as stripped *string* without any case changes, and enforce `article_id` as 10-char zero-padded strings at the point of grouping and at final write. I also ensure the output CSV is written with the exact column names and no index, and add a deterministic final integrity check that `customer_id` order and set match the sample submission (order matters less for Kaggle, but it helps catch accidental misalignment before upload). These are minimal, score-relevant changes aimed at turning the “format-mismatch 0.0” into the expected baseline score closer to your target.'
- What this solution (achieved 0.0) has done: 'Your pipeline logic is already a known non-zero baseline for H&M, so the 0.0 is almost certainly coming from submission-file integrity issues rather than recommendation quality. The main bug is in the final CSV reload check: it references `_chk2` but creates `_chk2 = pd.read_csv(...)` into `_chk2` while assigning to `_chk2["customer_id"] = _chk2[...]` using an undefined variable name in between, which can break execution before you upload the intended file. I fix that variable-name typo and add one minimal, score-relevant safety: ensure `prediction` is written/read as string and that `article_id` is always kept as 10-char strings (already mostly done) so the uploaded CSV matches Kaggle’s expected IDs exactly. No changes are made to the recommendation logic (popular-in-last-2-weeks + last-7-weeks with popular fallback), only to ensure a valid, correctly formatted `.csv` is reliably produced.'
- What this solution (achieved 0.0) has done: 'Your current logic should score non-zero, so the 0.0 is overwhelmingly likely due to the script failing at the very end and/or you uploading an output that wasn’t actually the intended CSV. The concrete bug is in the final reload/sanity-check block: it creates `_chk2` but then uses `_chk2` (undefined), which stops execution and can leave you with an old/invalid file. I fix that typo so the pipeline reliably finishes and writes `submission_recently_purchased.csv`. I also add one minimal, score-relevant safety: write `article_id` as zero-padded 10-char strings at read time (not just after), to prevent any silent dtype/format drift.'

# 9. Code solution

## === cell 0
import gc
import sys
from itertools import chain

import numpy as np
import pandas as pd



## === cell 1
df = pd.read_csv(
    "../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv",
    dtype={"article_id": "string", "customer_id": "string"},
    parse_dates=["t_dat"],
)

df["article_id"] = df["article_id"].astype("string").str.zfill(10)
df["customer_id"] = df["customer_id"].astype("string").str.strip()

print(df.shape)
df.head()



## === cell 2
test_df = pd.read_csv(
    "../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv",
    dtype={"customer_id": "string"},
).drop("prediction", axis=1)

test_df["customer_id"] = test_df["customer_id"].astype("string").str.strip()
test_df.head()



## === cell 3
train_unique = df["customer_id"].nunique()
test_unique = test_df["customer_id"].nunique()
print("Unique customer_id train/test:", train_unique, test_unique)

try:
    _train_int = df["customer_id"].head(200000).apply(lambda x: int(x[-16:], 16))
    _test_int = test_df["customer_id"].head(200000).apply(lambda x: int(x[-16:], 16))
    if (
        _train_int.nunique() < _train_int.shape[0]
        or _test_int.nunique() < _test_int.shape[0]
    ):
        print(
            "WARNING: collisions detected in truncated/int customer_id conversion on a sample."
        )
except Exception as e:
    print("Info: could not compute int conversion sample check:", repr(e))
gc.collect()



## === cell 4
print("Max `t_dat`:", df["t_dat"].max())
active_articles = df.groupby("article_id")["t_dat"].max().reset_index()
active_articles = active_articles[active_articles["t_dat"] >= "2019-09-01"].reset_index(
    drop=True
)
n_classes = active_articles.shape[0] + 1
active_articles.shape, n_classes



## === cell 5
df = df[df["article_id"].isin(active_articles["article_id"])].reset_index(drop=True)
df.shape



## === cell 6
df["week"] = (df["t_dat"].max() - df["t_dat"]).dt.days // 7
print(df["week"].nunique())



## === cell 7
item_counts = df[df.week < 2].article_id.value_counts()
item_counts[:12]



## === cell 8
most_frequent_items = item_counts[:12].index.astype("string").str.zfill(10).tolist()
prediction_str = " ".join(most_frequent_items)
prediction_str



## === cell 9
test_df_out = test_df.copy()
test_df_out["prediction"] = prediction_str
test_df_out[["customer_id", "prediction"]].to_csv(
    "submission_most_common_items.csv", index=False
)

_chk = pd.read_csv(
    "submission_most_common_items.csv",
    dtype={"customer_id": "string", "prediction": "string"},
)
assert list(_chk.columns) == ["customer_id", "prediction"]
assert len(_chk) == len(test_df)
del _chk
gc.collect()



## === cell 10
df_tmp = df[df["week"] <= 7].sort_values(["customer_id", "t_dat"], ascending=False)




## === cell 11
def keep_latest_k(articles, k=12, fallback=None):
    if fallback is None:
        fallback = []
    fallback = [str(x).zfill(10) for x in fallback]

    result = []
    for item in chain((str(a).zfill(10) for a in articles), fallback):
        if item in result:
            continue
        result.append(item)
        if len(result) == k:
            break
    if len(result) < k:
        for item in fallback:
            if item in result:
                continue
            result.append(item)
            if len(result) == k:
                break
    return result




## === cell 12
df_tmp["article_id"] = df_tmp["article_id"].astype("string").str.zfill(10)

df_latest_items = (
    df_tmp.groupby("customer_id", sort=False)
    .agg({"article_id": lambda s: keep_latest_k(s, k=12, fallback=most_frequent_items)})
    .reset_index()
)
df_latest_items["prediction"] = df_latest_items["article_id"].apply(
    lambda x: " ".join(x)
)
df_latest_items.head()



## === cell 13
test_df2 = test_df.merge(
    df_latest_items[["customer_id", "prediction"]],
    how="left",
    on="customer_id",
)
test_df2.head()



## === cell 14
test_df2["prediction"] = test_df2["prediction"].fillna(prediction_str)


def enforce_12(pred):
    parts = [p.zfill(10) for p in str(pred).split() if p]
    seen = set()
    dedup = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)
        if len(dedup) == 12:
            break
    if len(dedup) < 12:
        for p in most_frequent_items:
            p = str(p).zfill(10)
            if p in seen:
                continue
            seen.add(p)
            dedup.append(p)
            if len(dedup) == 12:
                break
    return " ".join(dedup[:12])


test_df2["prediction"] = test_df2["prediction"].apply(enforce_12)

assert (
    "customer_id" in test_df2.columns and "prediction" in test_df2.columns
), "Missing required columns."
assert len(test_df2) == len(test_df), "Row count mismatch vs sample_submission."
assert test_df2["customer_id"].isna().sum() == 0, "Found NaN customer_id."
assert test_df2["prediction"].isna().sum() == 0, "Found NaN predictions."
token_counts = test_df2["prediction"].str.split().str.len()
assert (
    token_counts == 12
).all(), f"Not all rows have 12 predictions. Bad rows: {(token_counts != 12).sum()}"
bad_tokens = test_df2["prediction"].str.split().explode().str.len().ne(10).sum()
assert bad_tokens == 0, f"Found tokens not length-10: {bad_tokens}"

assert (
    test_df2["customer_id"]
    .astype("string")
    .str.strip()
    .equals(test_df["customer_id"].astype("string").str.strip())
), "customer_id order/content mismatch vs sample_submission (may cause 0.0 if uploaded wrong file)."

test_df2.head()



## === cell 15
test_df2[["customer_id", "prediction"]].to_csv(
    "submission_recently_purchased.csv", index=False
)
print("Wrote:", "submission_recently_purchased.csv", "rows:", len(test_df2))
print(test_df2.head(3))

_chk2 = pd.read_csv(
    "submission_recently_purchased.csv",
    dtype={"customer_id": "string", "prediction": "string"},
)

_chk2["customer_id"] = _chk2["customer_id"].astype("string").str.strip()

assert list(_chk2.columns) == ["customer_id", "prediction"]
assert len(_chk2) == len(test_df)
assert _chk2["prediction"].str.split().str.len().eq(12).all()
assert _chk2["prediction"].str.split().explode().str.len().eq(10).all()

assert _chk2["customer_id"].equals(
    test_df["customer_id"]
), "Saved submission customer_id order does not match sample_submission."

del _chk2
gc.collect()
