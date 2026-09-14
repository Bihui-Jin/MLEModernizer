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

- What this solution (achieved 0.0) has done: 'I adjust the heuristic so it focuses on the most recent purchases, which typically improve MAP@12, and I write the submission as a plain .csv file (Kaggle expects that name). Specifically, I change the week filter from ≤ 4 weeks to ≤ 2 weeks in cell 9 and output “submission.csv” without compression in cell 14. These minimal tweaks keep the overall logic unchanged while steering the score closer to the target.'
- What this solution (achieved 0.0) has done: 'I slightly broaden the recent‑purchase window so that the heuristic uses a few more weeks of data (both for the global most‑frequent items and for each customer’s recent items). This small change keeps the overall logic untouched but generally raises MAP@12 enough to move the score toward the target 0.02048.'
- What this solution (achieved 0.0) has done: 'I broaden the recent‑purchase window (from ≤3 weeks to ≤5 weeks) and recompute the global most‑frequent items on that same window. This keeps the original heuristic logic but gives each customer a richer, more up‑to‑date set of candidate articles, which should raise MAP@12 toward the target score while preserving the overall pipeline.'
- What this solution (achieved 0.0) has done: 'I slightly widen the recent‑purchase window that drives both the global most‑frequent list and the per‑customer recent‑item list (from 5 weeks to 8 weeks). This keeps the original heuristic unchanged but gives each customer more up‑to‑date candidate articles, which should raise the MAP@12 from 0.0 toward the target 0.02048 while preserving the overall pipeline.'
- What this solution (achieved 0.0) has done: 'I widen the recent‑purchase window used for both the global most‑frequent list and the per‑customer recent‑item list (from 8 weeks to 12 weeks). This adds more relevant transactions without altering the overall heuristic, so the model stays the same while its MAP@12 should move closer to the target 0.02048.'
- What this solution (achieved 0.0) has done: 'The script now uses a much tighter recent‑purchase window (the last 2 weeks) when building both the global most‑frequent list and the per‑customer recent‑item list. By focusing on the most recent transactions the predictions become more relevant to the test period, which should raise the MAP@12 from 0 toward the target 0.02048 while keeping the original heuristic unchanged.'
- What this solution (achieved 0.0) has done: 'I introduce a modest “recent weeks” parameter (set to 4) and use it consistently when selecting the most‑frequent items and the per‑customer recent purchases. This widens the heuristic window just enough to capture more relevant transactions, which is expected to raise the MAP@12 from 0 toward the target 0.02048 while preserving the original logic.'
- What this solution (achieved 0.0) has done: 'I increase the recent‑weeks window slightly (to 6 weeks) and use the same window when selecting the global most‑frequent items. This modest change keeps the original heuristic untouched while providing more recent transaction data, which should raise MAP@12 from 0 toward the target 0.02048.'
- What this solution (achieved 0.0) has done: 'I increase the recent‑weeks window slightly (to 8 weeks) and change the fallback “most frequent” list to be computed from the whole training set rather than only the recent window. This keeps the original heuristic (per‑customer recent items) while giving a more robust global list, which should raise MAP@12 toward the target 0.02048. The rest of the pipeline and file outputs remain unchanged.'
- What this solution (achieved 0.0) has done: 'I increase the recent‑weeks window (the period used to collect each customer’s latest purchases) from 8 to 12 weeks. This keeps the original heuristic unchanged while giving the model more recent transaction data, which should raise the MAP@12 toward the target 0.02048. No other logic is altered.'
- What this solution (achieved 0.0) has done: 'I tighten the recent‑purchase window to the last 2 weeks (RECENT_WEEKS = 2) and recompute the fallback most‑frequent items based on this recent window, so the fallback predictions are more relevant to the test period. This small change preserves the overall heuristic while moving the MAP@12 score toward the target.'

# 9. Code solution

## === cell 0
import gc
import sys
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder

RECENT_WEEKS = 2




## === cell 1
df = pd.read_csv(
    "../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv",
    dtype={"article_id": str},
    parse_dates=["t_dat"],
)
df["customer_id_int"] = (
    df["customer_id"].apply(lambda x: int(x[-16:], 16)).astype("int64")
)
del df["customer_id"]
df["article_id"] = df["article_id"].astype("int32")
print(df.shape)
df.head()




## === cell 2
test_df = pd.read_csv(
    "../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv"
).drop("prediction", axis=1)
test_df["customer_id_int"] = (
    test_df["customer_id"].apply(lambda x: int(x[-16:], 16)).astype("int64")
)




## === cell 3
print("Max `t_dat`:", df["t_dat"].max())
active_articles = df.groupby("article_id")["t_dat"].max().reset_index()
active_articles = active_articles[
    active_articles["t_dat"] >= "2019-09-01"
].reset_index()
n_classes = active_articles.shape[0] + 1
active_articles.shape, n_classes




## === cell 4
df = df[df["article_id"].isin(active_articles["article_id"])].reset_index(drop=True)
df.shape




## === cell 5
df["week"] = (df["t_dat"].max() - df["t_dat"]).dt.days // 7
print(df["week"].nunique())




## === cell 6
global_item_counts = df.article_id.value_counts()
global_item_counts.head(12)




## === cell 7
most_frequent_items = global_item_counts.index[:12].to_numpy()
prediction_str = " ".join(map("{:010d}".format, most_frequent_items))
prediction_str




## === cell 8
test_df["prediction"] = prediction_str
test_df.to_csv("submission_most_common_items.csv.gz", compression="gzip", index=False)




## === cell 9
df_tmp = df[df["week"] <= RECENT_WEEKS].sort_values(
    ["customer_id_int", "t_dat"], ascending=False
)

recent_global_counts = df_tmp.article_id.value_counts()
most_frequent_items = recent_global_counts.index[:12].to_numpy()
prediction_str = " ".join(map("{:010d}".format, most_frequent_items))




## === cell 10
def keep_latest_k(articles, k=12):
    result = []
    for item in chain(articles, most_frequent_items):
        if item in result:
            continue
        result.append(item)
        if len(result) == k:
            break
    return result




## === cell 11
df_latest_items = (
    df_tmp.groupby("customer_id_int").agg({"article_id": keep_latest_k}).reset_index()
)
df_latest_items["prediction"] = df_latest_items.article_id.apply(
    lambda x: " ".join(map("{:010d}".format, x))
)
df_latest_items.head()




## === cell 12
test_df = test_df.drop("prediction", axis=1).merge(
    df_latest_items[["customer_id_int", "prediction"]], how="left", on="customer_id_int"
)
test_df.head()




## === cell 13
test_df["prediction"] = test_df["prediction"].fillna(prediction_str)
test_df.head()




## === cell 14
test_df[["customer_id", "prediction"]].to_csv("submission.csv", index=False)
print("Submission written to submission.csv, rows:", test_df.shape[0])
