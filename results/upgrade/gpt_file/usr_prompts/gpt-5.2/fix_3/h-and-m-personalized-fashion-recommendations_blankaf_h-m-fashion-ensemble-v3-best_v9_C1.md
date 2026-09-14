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

0.0239069984319075

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The current notebook is an ensemble/blending script that depends on several pre-made submission files that don’t exist in this environment, which is why it fails immediately. To keep core intent (produce a submission) with minimal changes, I’m replacing those missing-input reads with a lightweight, fully in-notebook baseline that generates valid predictions from the provided `transactions_train.csv`: a “last-week popular items” list and per-customer recent-history fallback. This runs within the Kaggle time limit by reading only needed columns and only the last ~30 days of transactions. The output is a correctly formatted `submission.csv` with `customer_id,prediction` for all customers in `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is almost certainly from an invalid submission format: H&M requires `customer_id` to be the 64-bit hex string from `sample_submission.csv`, but your code force-casts it to `str`, which turns those hex IDs into the literal text `"0x...."` and no longer matches the expected IDs, yielding effectively zero credit. I keep your core recommender logic intact and make the smallest fix: read and preserve `customer_id` exactly as provided (as a string column), and avoid any transformation that could alter it. I also make the date scan and filtering use consistent string dates to avoid subtle comparison issues while keeping runtime similar. The output remain `submission.csv` with the correct columns and 12-or-fewer article IDs per row.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import gc



## === cell 1
DATA_DIRS = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data/h-and-m-personalized-fashion-recommendations",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for d in DATA_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    for d in DATA_DIRS:
        p = os.path.join(d, "h-and-m-personalized-fashion-recommendations", filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in known DATA_DIRS: {DATA_DIRS}"
    )


transactions_path = find_file("transactions_train.csv")
sample_path = find_file("sample_submission.csv")

transactions_path, sample_path



## === cell 2
sub = pd.read_csv(sample_path, usecols=["customer_id"], dtype={"customer_id": "string"})
sub = sub.sort_values("customer_id").reset_index(drop=True)
sub.head()



## === cell 3
usecols = ["t_dat", "customer_id", "article_id"]
dtypes = {"t_dat": "string", "customer_id": "string", "article_id": "int32"}

max_date = None
for chunk in pd.read_csv(
    transactions_path, usecols=["t_dat"], dtype={"t_dat": "string"}, chunksize=2_000_000
):
    cmax = chunk["t_dat"].max()
    if max_date is None or cmax > max_date:
        max_date = cmax

max_date



## === cell 4
DAYS = 30
min_date = (pd.to_datetime(max_date) - pd.Timedelta(days=DAYS)).strftime("%Y-%m-%d")

cust_recent = {}  # customer_id -> list of article_ids (most recent first, unique)
pop_counts = {}  # article_id -> count

for chunk in pd.read_csv(
    transactions_path, usecols=usecols, dtype=dtypes, chunksize=2_000_000
):
    chunk = chunk[chunk["t_dat"] >= min_date]
    if chunk.empty:
        continue

    vc = chunk["article_id"].value_counts()
    for aid, cnt in vc.items():
        pop_counts[aid] = pop_counts.get(aid, 0) + int(cnt)

    chunk = chunk.sort_values(["customer_id", "t_dat"])
    for cid, grp in chunk.groupby("customer_id", sort=False):
        rec = cust_recent.get(cid)
        if rec is None:
            rec = []
        for aid in grp["article_id"].iloc[::-1].to_numpy():
            if aid in rec:
                continue
            rec.append(int(aid))
            if len(rec) >= 12:
                break
        cust_recent[cid] = rec

del chunk, vc
gc.collect()

len(pop_counts), len(cust_recent)



## === cell 5
top_articles = [
    aid for aid, _ in sorted(pop_counts.items(), key=lambda x: (-x[1], x[0]))[:12]
]
top_articles_str = [str(a).zfill(10) for a in top_articles]
top_pred = " ".join(top_articles_str)
top_pred




## === cell 6
def make_pred(cid: str) -> str:
    rec = cust_recent.get(cid)
    if not rec:
        return top_pred
    rec_str = [str(a).zfill(10) for a in rec]
    s = set(rec_str)
    filled = rec_str + [a for a in top_articles_str if a not in s]
    return " ".join(filled[:12])


sub["prediction"] = sub["customer_id"].map(make_pred)
sub.head()



## === cell 7
assert sub.shape[0] > 0
assert list(sub.columns) == ["customer_id", "prediction"]
assert sub["prediction"].isna().sum() == 0
assert (sub["prediction"].str.split().map(len) <= 12).all()

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
out_path
