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

0.00919

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I simplify the pipeline to guarantee that a valid submission CSV is written. The new version keeps the original imports and basic settings, but replaces the overly‑complex per‑age‑bin processing with a straightforward baseline: predict the globally most frequent 12 articles for every customer. This produces a correctly‑formatted file and should bring the MAP@12 score close to the target (≈0.02244) without altering the core modeling idea.'
- What this solution (achieved 0.00296) has done: 'I fix the ID handling so that the predicted article IDs exactly match the true IDs. The original code read `article_id` as an integer, which strips leading zeros, and then blindly prepended a “0”, corrupting the IDs and yielding a MAP@12 of 0. By reading `article_id` as a string and using the values directly (no artificial prefix), the baseline global‑top‑12 predictions correctly align with the ground‑truth IDs and should produce a score close to the target. No other logic is changed.'
- What this solution (achieved 0.00812) has done: 'I compute each customer’s most‑frequent 12 purchased articles from the training transactions and use that personalized list (filled with the global top items when needed) as the prediction. This keeps the original simple baseline while adding per‑customer relevance, which should raise the MAP@12 score toward the target.'
- What this solution (achieved 0.00919) has done: 'The changes replace the slow per‑customer `groupby.apply` that sorted each group individually with a fully vectorized pipeline: sort once, rank rows, keep the most‑recent 100 purchases per customer, count article frequencies, and extract the top 12 articles per customer via grouped aggregations. This eliminates millions of Python‑level calls, drastically reducing runtime while preserving the exact ranking logic (frequency‑ordered, ties broken by article id). No core logic, model, or output format is altered.'

# 9. Code solution

## === cell 0
import sys
import warnings

warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

PATH_INPUT = Path("../input/h-and-m-personalized-fashion-recommendations")
if not PATH_INPUT.is_dir():
    PATH_INPUT = Path("/kaggle/input/h-and-m-personalized-fashion-recommendations")

N_PRED = 12




## === cell 1
def display_df(df, head=3):
    print(f"SHAPE: {df.shape}\n")
    display(df.head(head))




## === cell 2
transactions_path = PATH_INPUT / "transactions_train.csv"
print(f"Reading transactions from {transactions_path}")

trx = pd.read_csv(
    transactions_path,
    usecols=["customer_id", "article_id", "t_dat"],
    dtype={"customer_id": "string", "article_id": "string"},
    parse_dates=["t_dat"],  # convert once to datetime
)

global_top_articles = trx["article_id"].value_counts().head(N_PRED).index.tolist()
print(f"Global top {N_PRED} articles: {' '.join(global_top_articles)}")

print("Computing per‑customer recent top articles (vectorized)...")

trx_sorted = trx.sort_values(["customer_id", "t_dat"], ascending=[True, False])

trx_sorted["rank"] = trx_sorted.groupby("customer_id", sort=False).cumcount() + 1

recent = trx_sorted[trx_sorted["rank"] <= 100]

counts = (
    recent.groupby(["customer_id", "article_id"], sort=False)
    .size()
    .reset_index(name="cnt")
)

counts = counts.sort_values(
    ["customer_id", "cnt", "article_id"],
    ascending=[True, False, True],
    ignore_index=True,
)

cust_top_series = counts.groupby("customer_id", sort=False)["article_id"].apply(
    lambda x: x.tolist()[:N_PRED]
)

cust_top_dict = cust_top_series.to_dict()
print(f"Computed recent top lists for {len(cust_top_dict)} customers.")




## === cell 3
sample_sub_path = PATH_INPUT / "sample_submission.csv"
print(f"Reading sample submission from {sample_sub_path}")
sub = pd.read_csv(sample_sub_path, dtype={"customer_id": "string"})
if "prediction" not in sub.columns:
    sub["prediction"] = ""


def build_prediction(cust_id):
    personal = cust_top_dict.get(cust_id, [])
    uniq = list(dict.fromkeys(personal))  # preserve order, remove dupes
    for art in global_top_articles:
        if len(uniq) >= N_PRED:
            break
        if art not in uniq:
            uniq.append(art)
    uniq = uniq[:N_PRED]
    return " ".join(uniq)


print("Generating predictions for each customer...")
sub["prediction"] = sub["customer_id"].apply(build_prediction)

sub["prediction"] = sub["prediction"].str[:131].str.strip()

output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path} ({sub.shape[0]} rows)")




## === cell 4
check = pd.read_csv(output_path, dtype={"customer_id": "string"})
display_df(check)
