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

df["customer_id"] = df["customer_id"].astype("string")
df["customer_id_int"] = (
    df["customer_id"].apply(lambda x: int(x[-16:], 16)).astype("int64")
)

print(df.shape)
df.head()



## === cell 2
test_df = pd.read_csv(
    "../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv",
    dtype={"customer_id": "string"},
).drop("prediction", axis=1)

test_df["customer_id"] = test_df["customer_id"].astype("string")
test_df["customer_id_int"] = (
    test_df["customer_id"].apply(lambda x: int(x[-16:], 16)).astype("int64")
)



## === cell 3
print("Max `t_dat`:", df["t_dat"].max())
active_articles = df.groupby("article_id")["t_dat"].max().reset_index()
active_articles = active_articles[active_articles["t_dat"] >= "2019-09-01"].reset_index(
    drop=True
)
n_classes = active_articles.shape[0] + 1
active_articles.shape, n_classes



## === cell 4
df = df[df["article_id"].isin(active_articles["article_id"])].reset_index(drop=True)
df.shape



## === cell 5
df["week"] = (df["t_dat"].max() - df["t_dat"]).dt.days // 7
print(df["week"].nunique())



## === cell 6
item_counts = df[df.week < 2].article_id.value_counts()
item_counts[:12]



## === cell 7
most_frequent_items = [str(x).zfill(10) for x in item_counts[:12].index.tolist()]
prediction_str = " ".join(most_frequent_items)
prediction_str



## === cell 8
test_df_out = test_df.copy()
test_df_out["prediction"] = prediction_str
test_df_out[["customer_id", "prediction"]].to_csv(
    "submission_most_common_items.csv", index=False
)



## === cell 9
df_tmp = df[df["week"] <= 7].sort_values(["customer_id_int", "t_dat"], ascending=False)




## === cell 10
def keep_latest_k(articles, k=12, fallback=None):
    if fallback is None:
        fallback = []
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




## === cell 11
df_latest_items = (
    df_tmp.groupby("customer_id_int")
    .agg({"article_id": lambda s: keep_latest_k(s, k=12, fallback=most_frequent_items)})
    .reset_index()
)
df_latest_items["prediction"] = df_latest_items["article_id"].apply(
    lambda x: " ".join(x)
)
df_latest_items.head()



## === cell 12
test_df2 = test_df.merge(
    df_latest_items[["customer_id_int", "prediction"]],
    how="left",
    on="customer_id_int",
)
test_df2.head()



## === cell 13
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

test_df2.head()



## === cell 14
test_df2[["customer_id", "prediction"]].to_csv(
    "submission_recently_purchased.csv", index=False
)
print("Wrote:", "submission_recently_purchased.csv", "rows:", len(test_df2))
print(test_df2.head(3))
