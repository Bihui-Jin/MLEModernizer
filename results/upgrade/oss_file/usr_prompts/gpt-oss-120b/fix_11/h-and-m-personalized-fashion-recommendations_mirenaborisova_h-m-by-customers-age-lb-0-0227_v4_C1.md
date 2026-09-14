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

- What this solution (achieved 0.0) has done: 'I fixed the import error, ensured the datetime index is sorted before slicing, and changed the merge that creates the submission to a left‑join so every `customer_id` from the sample submission is kept. This restores the pipeline, generates a complete CSV, and guarantees the submission’s `customer_id` set matches the required superset.'
- What this solution (achieved 0.0) has done: 'The script now imports all required libraries, reads the transaction data, determines the most frequent 12 articles overall, and creates a valid submission.csv where every customer receives this popular‑article list. This fixes the previous import and undefined‑variable errors and guarantees a correctly‑formatted submission that achieves a baseline score close to the target.'
- What this solution (achieved 0.0) has done: 'I fix the article‑id formatting (zero‑pad to 10 digits) and replace the naïve “global‑top‑12 for everyone” baseline with a simple per‑customer frequency list that falls back to the global top articles when a customer has fewer than 12 distinct purchases. This small personalization usually lifts MAP@12 from zero toward the target while keeping the original pipeline structure untouched.'
- What this solution (achieved 0.0) has done: 'We switch the per‑customer list from “most frequent articles overall” to the most recent distinct purchases (using the transaction date). This keeps the overall pipeline and fallback‑to‑global‑top logic unchanged while giving predictions that better match the test period, which should raise the MAP@12 from 0 toward the target 0.02244.'
- What this solution (achieved 0.0) has done: 'I keep the original pipeline but add a lightweight per‑customer frequency list and combine it with the recent‑purchase list before falling back to the global top‑12. This modest personalization usually raises MAP@12 toward the target without altering the core logic or introducing new modelling steps.'
- What this solution (achieved 0.0) has done: 'I limit the “recent” recommendation list to the last 30 days of the training period, which better matches the test window while keeping the original per‑customer, frequency, and global fallback logic unchanged. This small change should add a few more relevant items to each customer’s prediction, moving the MAP@12 from 0.0 towards the target 0.02244 without altering the overall pipeline. The rest of the script (imports, data loading, prediction building, and CSV output) remains the same, just with the added recent‑window filter.'
- What this solution (achieved 0.0) has done: 'I tighten the recent‑window to the last 7 days (instead of 30) because purchases made just before the test period are far more likely to reappear, giving a modest boost in MAP@12 while keeping the original pipeline untouched. The only modification is the `recent_window_start` calculation; all other logic, formatting, and fallback behavior remain identical.'
- What this solution (achieved 0.0) has done: 'I keep the overall pipeline unchanged but widen the recent‑purchase window from 7 to 30 days. A larger window captures more items that are likely to appear again in the test period, giving a modest boost to MAP@12 while preserving the existing logic and fallback order.'
- What this solution (achieved 0.0) has done: 'The recent‑purchase window is narrowed from 30 days to the last 7 days so the model prioritises items that are most likely to reappear in the test period, while keeping the existing frequency and global fallback logic unchanged. This small tweak should raise MAP@12 toward the target without altering the core pipeline.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import cudf  # retained for compatibility if needed elsewhere
import matplotlib.pyplot as plt
import seaborn as sns

DEBUG = False
PATH_INPUT = r"../input/h-and-m-personalized-fashion-recommendations/"
N_TOP = 12  # number of articles to recommend per customer



## === cell 1
transactions_path = PATH_INPUT + "transactions_train.csv"
transactions_df = pd.read_csv(
    transactions_path,
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"customer_id": "string", "article_id": "int64"},
    parse_dates=["t_dat"],
)

max_date = transactions_df["t_dat"].max()
recent_window_start = max_date - pd.Timedelta(days=7)
recent_df = transactions_df[transactions_df["t_dat"] >= recent_window_start]

global_top_series = transactions_df["article_id"].value_counts().nlargest(N_TOP).index
global_top_str = [str(a).zfill(10) for a in global_top_series]


def recent_distinct_articles(series):
    rev = series[::-1]  # newest first (original sort is ascending)
    uniq = rev.drop_duplicates()  # keep first occurrence per article
    recent = uniq.iloc[:N_TOP]  # up to N_TOP recent distinct items
    return recent.tolist()


cust_recent_series = (
    recent_df.sort_values("t_dat")
    .groupby("customer_id")["article_id"]
    .apply(recent_distinct_articles)
)

cust_freq_series = transactions_df.groupby("customer_id")["article_id"].apply(
    lambda x: x.value_counts().index.tolist()
)

sample_sub_path = PATH_INPUT + "sample_submission.csv"
sample_sub_df = pd.read_csv(sample_sub_path, dtype={"customer_id": "string"})


def build_prediction(cid):
    recent = cust_recent_series.get(cid, [])
    freq = cust_freq_series.get(cid, [])
    combined = []

    def extend_from(seq):
        for a in seq:
            a_str = str(a).zfill(10) if not isinstance(a, str) else a
            if a_str not in combined:
                combined.append(a_str)
                if len(combined) >= N_TOP:
                    break

    extend_from(recent)  # prioritize recent purchases (last 7 days)
    if len(combined) < N_TOP:
        extend_from(freq)  # then overall frequency
    if len(combined) < N_TOP:
        extend_from(global_top_str)  # finally global most popular

    return " ".join(combined[:N_TOP])


sample_sub_df["prediction"] = sample_sub_df["customer_id"].apply(build_prediction)

submission_file = "submission.csv"
sample_sub_df.to_csv(submission_file, index=False)

print(f"Submission saved to {submission_file}. Preview:")
print(sample_sub_df.head())
