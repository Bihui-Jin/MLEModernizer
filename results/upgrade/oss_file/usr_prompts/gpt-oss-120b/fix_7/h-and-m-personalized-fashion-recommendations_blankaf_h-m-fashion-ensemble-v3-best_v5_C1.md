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

0.0238254097118143

# 6. Current score

0.01167

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00296) has done: 'I replace the failing code that tried to read many non‑existent submission files with a simple, self‑contained pipeline: load the training transactions, compute the 12 most frequent article IDs, assign this same list to every customer in the provided sample submission, and write a valid `submission.csv`. This fixes the runtime errors and guarantees a correctly formatted output while giving a reasonable baseline score that moves toward the target.'
- What this solution (achieved 0.00734) has done: 'The update adds a lightweight per‑customer frequency calculation: while counting the globally most common articles, we also tally each customer’s article purchases in the same chunked loop. After processing, we build a “top‑12 per customer” dictionary and use it to fill the submission, falling back to the global top‑12 list for unseen customers. This personalization should raise MAP@12 toward the target without changing the overall pipeline or model logic.'
- What this solution (achieved 0.01963) has done: 'The fix adds a lightweight recency filter: first we scan the transaction file to find the most recent date, then we only count purchases from the last 30 days when building per‑customer and global top‑12 article lists. This keeps the original per‑customer “most frequent 12” logic but focuses on recent behaviour, which should raise MAP@12 toward the target while preserving the overall pipeline. The script now runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.01698) has done: 'I extend the recent window from 30 to 60 days to capture more relevant purchases and ensure every customer receives exactly 12 article predictions by filling any short personal lists with the global top‑12 items (skipping duplicates). The changes are limited to the date‑threshold calculation, storing per‑customer top‑12 as lists, and a small helper that builds a full 12‑item prediction string before writing the submission file.'
- What this solution (achieved 0.01385) has done: 'I increase the recency window from 60 to 90 days (so more relevant purchases are considered) and replace the per‑customer “most‑frequent” logic with a most‑recent distinct list of up to 12 articles gathered from that window. The fallback to the globally most‑common 12 articles remains unchanged, ensuring every customer receives exactly 12 predictions. These minimal tweaks keep the original pipeline while making the recommendations more timely, which should raise MAP@12 toward the target score.'
- What this solution (achieved 0.01167) has done: 'I increase the recency window from 90 to 120 days, so the per‑customer recent‑article lists are built from a slightly larger, more relevant period. This small change keeps the overall pipeline unchanged while giving the model a bit more recent purchase information, which should raise MAP@12 toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from collections import Counter, defaultdict



## === cell 1
DATA_ROOT = "/kaggle/input/h-and-m-personalized-fashion-recommendations"
transaction_path = os.path.join(DATA_ROOT, "transactions_train.csv")
chunksize = 10_000_000

max_date = None
for chunk in pd.read_csv(
    transaction_path,
    usecols=["t_dat"],
    dtype=str,
    chunksize=chunksize,
):
    dates = pd.to_datetime(chunk["t_dat"], format="%Y-%m-%d", errors="coerce")
    chunk_max = dates.max()
    if pd.isna(chunk_max):
        continue
    if max_date is None or chunk_max > max_date:
        max_date = chunk_max

if max_date is None:
    raise RuntimeError("Unable to determine max transaction date.")

recent_threshold = max_date - pd.Timedelta(days=120)



## === cell 2
article_counter = Counter()
cust_recent_list = defaultdict(list)  # store most‑recent distinct articles per customer

for chunk in pd.read_csv(
    transaction_path,
    usecols=["t_dat", "customer_id", "article_id"],
    dtype=str,
    chunksize=chunksize,
):
    chunk["t_dat"] = pd.to_datetime(chunk["t_dat"], format="%Y-%m-%d", errors="coerce")
    recent_mask = chunk["t_dat"] >= recent_threshold
    recent_chunk = chunk[recent_mask]

    article_counter.update(recent_chunk["article_id"])

    recent_chunk = recent_chunk.sort_values("t_dat", ascending=False)

    for cust_id, art_id in zip(recent_chunk["customer_id"], recent_chunk["article_id"]):
        lst = cust_recent_list[cust_id]
        if art_id not in lst and len(lst) < 12:
            lst.append(art_id)

top_12_articles = [article for article, _ in article_counter.most_common(12)]
cust_top12_list = {cust_id: lst for cust_id, lst in cust_recent_list.items()}



## === cell 3
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
submission_df = pd.read_csv(sample_sub_path, dtype=str)




## === cell 4
def build_prediction(cid: str) -> str:
    """Return a space‑separated string of up to 12 article IDs for a customer."""
    personal = cust_top12_list.get(cid, []).copy()
    if len(personal) < 12:
        for art in top_12_articles:
            if art not in personal:
                personal.append(art)
            if len(personal) == 12:
                break
    return " ".join(personal)


submission_df["prediction"] = submission_df["customer_id"].apply(build_prediction)



## === cell 5
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}. Shape: {submission_df.shape}")
