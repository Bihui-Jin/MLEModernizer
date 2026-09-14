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

- What this solution (achieved 0.0) has done: 'A quick fix defines the correct data directory, computes the overall most‑popular 12 articles, and writes them as the prediction for every customer in the sample submission. This resolves the missing PATH_INPUT error and guarantees a valid submission.csv with exactly twelve article IDs per row, satisfying the competition’s format without altering the original modeling approach.'
- What this solution (achieved 0.0) has done: 'The fix removes the unsupported `Series.apply` on a CUDA‑accelerated dataframe and directly assigns the already‑validated fallback prediction string to the submission column, guaranteeing a correct CSV output. This resolves the Numba typing error and produces a usable submission, which should now achieve a non‑zero MAP@12 score closer to the target.'
- What this solution (achieved 0.0) has done: 'I fix the article‑ID formatting so the predicted IDs exactly match the 10‑digit strings used in the ground‑truth data. The previous code prefixed every ID with an extra “0”, producing 11‑digit identifiers that never matched any true article and gave a MAP@12 of 0. By zero‑padding each ID to length 10 (using `zfill(10)`) the fallback predictions become valid, which should raise the score toward the target.'
- What this solution (achieved 0.0) has done: 'The patch adds a lightweight per‑customer popular‑item boost: for each customer we find their most‑frequent purchased article and place it first in the prediction list, then fill the remaining slots with the globally most‑popular 12 articles (skipping duplicates). Customers with no history keep the original global fallback. This small personalization is expected to raise the MAP@12 toward the target without altering the overall modeling approach.'
- What this solution (achieved 0.0) has done: 'I tighten the popularity cues to the most recent buying behaviour, because the test period follows the end of the training data.  
The global top‑12 list and the per‑customer “personal” article are now computed from transactions in the last 30 days of the training set (using the `t_dat` column). This keeps the original modelling idea intact while giving predictions that are far more likely to appear in the upcoming week, moving the MAP@12 score toward the target 0.02244.'

# 9. Code solution

## === cell 0
import sys
import warnings

warnings.filterwarnings("ignore")
import os
import gc
import numpy as np
import pandas as pd
import cudf

PATH_INPUT = "/kaggle/input/h-and-m-personalized-fashion-recommendations/"



## === cell 1
transactions_df = cudf.read_csv(
    os.path.join(PATH_INPUT, "transactions_train.csv"),
    usecols=["article_id", "t_dat"],
    dtype={"article_id": "int32", "t_dat": "str"},
)

transactions_df["t_dat"] = cudf.to_datetime(transactions_df["t_dat"])

max_date = transactions_df["t_dat"].max()
cutoff = max_date - np.timedelta64(30, "D")
recent_transactions = transactions_df[transactions_df["t_dat"] >= cutoff]

article_counts = (
    recent_transactions.groupby("article_id").size().reset_index(name="cnt")
)
top12_series = article_counts.nlargest(12, "cnt")["article_id"]
top12 = [str(aid).zfill(10) for aid in top12_series.to_arrow().to_pylist()]
fallback_pred = " ".join(top12)

cust_trans = cudf.read_csv(
    os.path.join(PATH_INPUT, "transactions_train.csv"),
    usecols=["customer_id", "article_id", "t_dat"],
    dtype={"customer_id": "string", "article_id": "int32", "t_dat": "str"},
)

cust_trans["t_dat"] = cudf.to_datetime(cust_trans["t_dat"])

cust_trans = cust_trans.sort_values(["customer_id", "t_dat"], ascending=[True, False])
cust_top_recent = cust_trans.drop_duplicates(subset=["customer_id"], keep="first")[
    ["customer_id", "article_id"]
]

cust_top_pd = cust_top_recent.to_pandas()
cust_top_pd["article_str"] = cust_top_pd["article_id"].apply(lambda x: str(x).zfill(10))
customer_top_map = dict(zip(cust_top_pd["customer_id"], cust_top_pd["article_str"]))



## === cell 2
sample_sub = cudf.read_csv(
    os.path.join(PATH_INPUT, "sample_submission.csv"),
    dtype={"customer_id": "string"},
)


def build_prediction(cust_id):
    personal = customer_top_map.get(cust_id)
    if personal is None:
        return fallback_pred
    preds = [personal] + [aid for aid in top12 if aid != personal]
    return " ".join(preds[:12])


sample_sub_pd = sample_sub.to_pandas()
sample_sub_pd["prediction"] = sample_sub_pd["customer_id"].apply(build_prediction)

sample_sub = cudf.from_pandas(sample_sub_pd)



## === cell 3
output_path = "submission.csv"
sample_sub.to_csv(output_path, index=False)

display(sample_sub.head())
