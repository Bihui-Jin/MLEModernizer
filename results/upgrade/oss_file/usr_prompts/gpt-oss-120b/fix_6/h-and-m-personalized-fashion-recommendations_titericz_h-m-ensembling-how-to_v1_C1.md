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

0.023095584863488

# 6. Current score

0.02047

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00296) has done: 'I replace the failing code that tried to read non‑existent public submissions with a minimal, self‑contained pipeline: load the training transactions, compute the 12 most frequent article_id (as a simple popularity baseline), assign this same list to every customer_id in the sample submission, and write a valid `submission.csv`. This fixes the file‑not‑found errors, guarantees a CSV output, and provides a reasonable baseline that should achieve a MAP@12 score close to the target without altering any core modeling logic.'
- What this solution (achieved 0.00794) has done: 'I keep the overall baseline approach but personalize the 12‑item recommendation for each customer. First I compute the global 12 most frequent articles (used as a fallback). Then I group the transaction data by customer_id and store each customer’s own top‑12 articles. When building the submission, each customer receives their personal list padded with the global popular items until 12 articles are present. This small personalization is expected to raise the MAP@12 from 0.00296 closer to the target 0.02309 without altering the core pipeline.'
- What this solution (achieved 0.01995) has done: 'I keep the same overall pipeline but add a simple recency‑based weighting: use the most recent 30 days of transactions to build a “recent” top‑12 list per customer and globally, then fall back to the overall historic lists if needed. This modest change should raise MAP@12 toward the target without altering the core modeling logic.'
- What this solution (achieved 0.01825) has done: 'I slightly broaden the recency window (from 30 to 45 days) when building the recent‑transactions based recommendation lists. This keeps the overall pipeline identical while giving the model a bit more recent data, which should boost MAP@12 toward the target without over‑changing the core logic.'
- What this solution (achieved 0.02047) has done: 'The plan is to move the recency window back to 30 days (the previous 30‑day version achieved a higher MAP) and to give the recent‑global popular items higher priority than the older personal list. This small tweak keeps the overall pipeline intact while nudging the MAP@12 upward toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd




## === cell 1
RECENCY_DAYS = 30

transactions_path = (
    "../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"
)
if not os.path.exists(transactions_path):
    transactions_path = "../input/transactions_train.csv"

transactions = pd.read_csv(
    transactions_path,
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"customer_id": str, "article_id": str},
    parse_dates=["t_dat"],
)

global_top12 = transactions["article_id"].value_counts().head(12).index.tolist()

customer_top_series = transactions.groupby("customer_id")["article_id"].apply(
    lambda x: x.value_counts().head(12).index.tolist()
)
customer_top_dict = customer_top_series.to_dict()

max_date = transactions["t_dat"].max()
cutoff_date = max_date - pd.Timedelta(days=RECENCY_DAYS)
recent_tx = transactions[transactions["t_dat"] >= cutoff_date]

global_top12_recent = recent_tx["article_id"].value_counts().head(12).index.tolist()

customer_recent_series = recent_tx.groupby("customer_id")["article_id"].apply(
    lambda x: x.value_counts().head(12).index.tolist()
)
customer_top_recent_dict = customer_recent_series.to_dict()




## === cell 2
sample_sub_path = (
    "../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv"
)
if not os.path.exists(sample_sub_path):
    sample_sub_path = "../input/sample_submission.csv"

submission = pd.read_csv(sample_sub_path, dtype={"customer_id": str})


def build_prediction(cust_id):
    pred = []

    recent_personal = customer_top_recent_dict.get(cust_id, [])
    for art in recent_personal:
        if len(pred) >= 12:
            break
        pred.append(art)

    if len(pred) < 12:
        for art in global_top12_recent:
            if len(pred) >= 12:
                break
            if art not in pred:
                pred.append(art)

    if len(pred) < 12:
        overall_personal = customer_top_dict.get(cust_id, [])
        for art in overall_personal:
            if len(pred) >= 12:
                break
            if art not in pred:
                pred.append(art)

    if len(pred) < 12:
        for art in global_top12:
            if len(pred) >= 12:
                break
            if art not in pred:
                pred.append(art)

    return " ".join(pred[:12])


submission["prediction"] = submission["customer_id"].apply(build_prediction)




## === cell 3
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} with {len(submission)} rows.")
