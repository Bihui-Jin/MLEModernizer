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

0.0238574840167817

# 6. Current score

0.01954

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00296) has done: 'I replace the missing‑file blending logic with a simple but valid baseline: read the official sample submission, compute the 12 most frequent articles from the training transactions, assign that same list to every customer, and write the result to `submission.csv`. This fixes the FileNotFound errors, ensures a correctly‑named CSV is produced, and gives a non‑zero MAP@12 score that should be close to the target.'
- What this solution (achieved 0.00749) has done: 'I replace the single‑string baseline with a per‑customer top‑12 list: while reading the training file in chunks I count each article per customer, then for every customer I keep their most frequent 12 articles (padding with the global top‑12 if needed). This keeps the original workflow but gives each customer a personalized prediction, which should raise MAP@12 toward the target score.'
- What this solution (achieved 0.01867) has done: 'I enhance the baseline by giving more weight to a customer’s recent purchases, which usually predict next‑week buying behaviour better than older history. First I compute the global most‑popular 12 articles and also record the latest transaction date. Then I build each customer’s count dictionary using only transactions from the last 30 days (based on the max date found). The prediction routine keeps the same logic –‑ it returns the customer’s most frequent recent items and pads with the global top‑12 when needed –‑ but this recency focus should raise MAP@12 toward the target without altering the overall architecture.'
- What this solution (achieved 0.01954) has done: 'The changes add an overall per‑customer count alongside the recent‑30‑day count, then fill each prediction first with recent items, then with the customer's overall most‑frequent items before finally padding with the global top‑12. This modest enrichment keeps the original workflow but should raise MAP@12 toward the target without altering the core model logic.'
- What this solution (achieved 0.01812) has done: 'I increase the recency window from 30 to 45 days and use the most‑popular articles **within that recent period** for padding predictions — this keeps the original per‑customer recent + overall logic but aligns the fallback list with recent trends, which should raise MAP@12 toward the target while leaving the core workflow unchanged.'
- What this solution (achieved 0.01954) has done: 'I shorten the recency window back to 30 days (which gave a higher MAP in earlier runs) and use the overall‑global top‑12 articles as the final fallback list instead of the recent‑global list. This keeps the same per‑customer recent → overall → global logic while providing a more stable padding, which should raise the MAP score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from collections import defaultdict, Counter




## === cell 1
BASE_DIR = "/kaggle/input/h-and-m-personalized-fashion-recommendations"
sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
submission = pd.read_csv(sample_path).sort_values("customer_id").reset_index(drop=True)




## === cell 2
train_path = os.path.join(BASE_DIR, "transactions_train.csv")
article_counts = {}
max_date = pd.Timestamp.min  # will hold the latest transaction date

chunksize = 10_000_000
for chunk in pd.read_csv(
    train_path, usecols=["article_id", "t_dat"], chunksize=chunksize
):
    for article, cnt in chunk["article_id"].value_counts().items():
        article_counts[article] = article_counts.get(article, 0) + cnt
    chunk_dates = pd.to_datetime(chunk["t_dat"], format="%Y-%m-%d", errors="coerce")
    chunk_max = chunk_dates.max()
    if pd.notnull(chunk_max) and chunk_max > max_date:
        max_date = chunk_max

top12_global = [
    str(a) for a, _ in sorted(article_counts.items(), key=lambda x: -x[1])[:12]
]
top12_global_str = " ".join(top12_global)




## === cell 3
recent_cutoff = max_date - pd.Timedelta(days=30)

cust_counts_recent = defaultdict(Counter)
cust_counts_all = defaultdict(Counter)

chunksize = 10_000_000
for chunk in pd.read_csv(
    train_path,
    usecols=["customer_id", "article_id", "t_dat"],
    dtype={"customer_id": str, "article_id": str, "t_dat": str},
    chunksize=chunksize,
):
    chunk["t_dat"] = pd.to_datetime(chunk["t_dat"], format="%Y-%m-%d", errors="coerce")
    recent_mask = chunk["t_dat"] >= recent_cutoff
    recent_chunk = chunk[recent_mask]

    for cust_id, article_id in zip(
        recent_chunk["customer_id"], recent_chunk["article_id"]
    ):
        cust_counts_recent[cust_id][article_id] += 1

    for cust_id, article_id in zip(chunk["customer_id"], chunk["article_id"]):
        cust_counts_all[cust_id][article_id] += 1


def make_prediction(cust_id):
    recent_common = [
        a for a, _ in cust_counts_recent.get(cust_id, Counter()).most_common(12)
    ]
    if len(recent_common) >= 12:
        return " ".join(recent_common[:12])

    needed = 12 - len(recent_common)
    overall_common = [
        a
        for a, _ in cust_counts_all.get(cust_id, Counter()).most_common()
        if a not in recent_common
    ][:needed]
    recent_common.extend(overall_common)
    needed = 12 - len(recent_common)

    if needed > 0:
        pad = [a for a in top12_global if a not in recent_common][:needed]
        recent_common.extend(pad)

    return " ".join(recent_common) if recent_common else top12_global_str


submission["prediction"] = submission["customer_id"].apply(make_prediction)




## === cell 4
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
