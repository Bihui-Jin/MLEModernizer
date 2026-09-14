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

3.14

# 3. Installed packages

cuml-cu12==25.2.1
geopandas==0.14.4
libcuml-cu12==25.2.1
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.028964833733073

# 6. Current score

0.02515

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01873) has done: 'I fixed the file‑path handling so the datasets are correctly located, added a fallback list of common Kaggle input directories, and replaced the single global “top‑12” list with a lightweight per‑customer recommendation: each customer gets their own most‑weighted recent articles (up to 12) and, if they have fewer than 12, the list is padded with the overall top‑12 items. This keeps the core logic intact while improving relevance and guaranteeing a valid *.csv* submission. The script now runs end‑to‑end and writes `dev_submission.csv`.'
- What this solution (achieved 0.01519) has done: 'I slightly adjust the recency weighting and broaden the recent‑week window.  
* The recent window is expanded from 5 to 7 weeks so more relevant purchases are considered.  
* We replace the linear weight (`max_week‑week+1`) with an exponential decay (`2**(max_week‑week)`) to give much higher importance to the most recent weeks.  
These minimal changes keep the overall per‑customer top‑12 logic intact while expectedly raising MAP@12 toward the target score.'
- What this solution (achieved 0.02189) has done: 'I replace the limited‑week weighting with a smooth exponential decay that is applied to **all** transaction weeks. This keeps the per‑customer top‑12 logic unchanged but gives each customer a richer, time‑aware score (more recent purchases dominate while older ones still contribute). The global top‑12 list is also recomputed from the full decay‑weighted data, so padding remains sensible. This small change is expected to raise MAP@12 toward the target without altering the core recommendation pipeline.'
- What this solution (achieved 0.02311) has done: 'I strengthen the recency bias by decreasing the decay factor from 0.9 to 0.8, giving more weight to the most recent weeks while keeping all other logic unchanged. This small tweak is expected to raise the MAP@12 score toward the target without altering the core recommendation pipeline or the submission format.'
- What this solution (achieved 0.02362) has done: 'I lower the decay factor from 0.8 to 0.7 so that recent purchases receive even higher importance in the weighted scores. This tiny tweak keeps the overall recommendation pipeline unchanged while strengthening the recency bias, which should lift the MAP@12 toward the target without affecting the submission format.'
- What this solution (achieved 0.02396) has done: 'I lower the decay factor from 0.7 to 0.5 so that recent purchases receive even higher weight, which should modestly raise the MAP@12 score toward the target while keeping the overall recommendation logic unchanged.'
- What this solution (achieved 0.02408) has done: 'I slightly increase the recency bias by changing the decay factor from 0.5 to 0.4. This keeps the original recommendation pipeline untouched while giving more weight to recent weeks, which should raise the MAP@12 score toward the target without altering any other logic.'
- What this solution (achieved 0.02515) has done: 'I tighten the recency focus by keeping only the most recent 12 weeks of transactions before applying the decay weighting. This discards older noise while preserving the existing decay‑bias logic, which should modestly raise MAP@12 toward the target value. The change is limited to the preprocessing cell and does not alter the core recommendation pipeline.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

possible_paths = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data/input",
    "./kaggle/data/input",
    "./input",
    "/kaggle/input",
]

BASE_PATH = None
for p in possible_paths:
    if os.path.isdir(p):
        BASE_PATH = p
        break

if BASE_PATH is None:
    raise FileNotFoundError("Could not locate the competition input directory.")

customers_path = os.path.join(BASE_PATH, "customers.csv")
transactions_path = os.path.join(BASE_PATH, "transactions_train.csv")
articles_path = os.path.join(BASE_PATH, "articles.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

c = pd.read_csv(customers_path)
t = pd.read_csv(transactions_path, parse_dates=["t_dat"])
a = pd.read_csv(articles_path)
sample_sub = pd.read_csv(sample_sub_path)




## === cell 1
t["week_number"] = ((t["t_dat"] - t["t_dat"].min()) / np.timedelta64(1, "W")).astype(
    int
) + 1
max_week = t["week_number"].max()

RECENT_WEEKS = 12  # focus on the last 12 weeks of data
min_recent_week = max_week - RECENT_WEEKS + 1
t = t[t["week_number"] >= min_recent_week].copy()

DECAY_FACTOR = 0.4  # unchanged decay factor
t["weight"] = DECAY_FACTOR ** (max_week - t["week_number"])




## === cell 2
TOP_N = 12

global_article_weights = (
    t.groupby("article_id")["weight"].sum().sort_values(ascending=False)
)
global_top_articles = global_article_weights.head(TOP_N).index.astype(str).tolist()

cust_art = t.groupby(["customer_id", "article_id"])["weight"].sum().reset_index()

cust_art = cust_art.sort_values(["customer_id", "weight"], ascending=[True, False])

per_cust_top = (
    cust_art.groupby("customer_id")["article_id"]
    .apply(lambda x: x.head(TOP_N).astype(str).tolist())
    .to_dict()
)




## === cell 3
prediction_list = []
for cust_id in sample_sub["customer_id"]:
    cust_articles = per_cust_top.get(cust_id, [])
    if len(cust_articles) < TOP_N:
        needed = TOP_N - len(cust_articles)
        padding = [art for art in global_top_articles if art not in cust_articles][
            :needed
        ]
        cust_articles.extend(padding)
    cust_articles = cust_articles[:TOP_N]
    prediction_list.append(" ".join(cust_articles))

submission = pd.DataFrame(
    {"customer_id": sample_sub["customer_id"], "prediction": prediction_list}
)




## === cell 4
output_path = "dev_submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print(submission.head())
