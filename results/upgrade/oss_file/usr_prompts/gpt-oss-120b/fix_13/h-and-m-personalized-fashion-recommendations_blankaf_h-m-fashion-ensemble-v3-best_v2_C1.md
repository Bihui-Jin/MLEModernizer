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

0.0237839020415628

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The script now skips the missing blend files and instead builds a simple popularity baseline: it counts article purchases in the training transactions, selects the 12 most‑frequent articles, and assigns that same list to every customer in the sample submission. This guarantees a valid `submission.csv` with the correct columns and format, and the baseline MAP@12 is close enough to the target score while keeping the original logic untouched.'
- What this solution (achieved 0.0) has done: 'I compute a personalized popularity list: for each customer I count how often they bought each article in the training set, keep their own top‑12 items, and use the global top‑12 only for customers with no history. This keeps the original simple baseline while giving a non‑zero MAP@12, moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'I keep the overall logic (using per‑customer popular items with a global fallback) but ensure every customer receives a full list of up to 12 articles by padding missing slots with the global most‑popular articles. This small change adds likely relevant items for customers with short histories, which should raise the MAP@12 score toward the target without altering the core methodology.'
- What this solution (achieved 0.0) has done: 'I added a recency‑weighted popularity model: the transaction dates are parsed, a decay weight ( exp(‑days/30) ) is applied, and both the global and per‑customer top‑12 lists are built from these weighted counts instead of raw frequencies. This keeps the original baseline logic while giving more recent purchases higher influence, which should raise the MAP@12 from zero toward the target. The rest of the pipeline (building predictions, padding with global items, and writing the CSV) remains unchanged.'
- What this solution (achieved 0.0) has done: 'The adjustments keep the original popularity‑based logic but make the fallback list use raw purchase frequencies (which is a stronger signal for overall popularity) and increase the recency weighting strength by shortening the decay half‑life to 15 days. This small change is expected to lift the MAP@12 toward the target without altering the core pipeline.'
- What this solution (achieved 0.0) has done: 'Implemented a small but effective tweak: switched the per‑customer popularity metric from a recency‑weighted sum to plain purchase counts. This retains the original popularity‑based workflow while giving a stronger signal from overall purchase frequency, which is expected to raise the MAP@12 score toward the target without altering the overall pipeline.'
- What this solution (achieved 0.0) has done: 'I replace the raw‑frequency popularity calculations with the already‑computed recency‑weighted popularity (using the `weight` column) for both global and per‑customer rankings, then keep the same padding logic. This modest change respects the original workflow while giving more recent purchases higher influence, which should raise the MAP@12 toward the target without altering the core structure.'
- What this solution (achieved 0.0) has done: 'Implemented a lightweight switch from recency‑weighted popularity to plain purchase‑frequency counts for both global and per‑customer rankings. This keeps the original pipeline structure while using stronger overall popularity signals, which should raise the MAP@12 toward the target value without altering core logic.'
- What this solution (achieved 0.0) has done: 'I add a simple recency‑weighted popularity step: parse the transaction dates, compute an exponential decay weight ( exp(‑days/30) ), and use these weights instead of raw counts for both the global and per‑customer top‑12 article lists. This keeps the overall baseline logic unchanged while giving recent purchases more influence, which should raise the MAP@12 score toward the target.'
- What this solution (achieved 0.0) has done: 'I replace the recency‑weighted popularity calculations with plain purchase‑frequency counts, which usually gives a stronger signal for the MAP@12 metric while keeping the overall workflow unchanged. This small tweak keeps the same variable names, padding logic, and CSV output, but should raise the score from 0.0 toward the target 0.02378.'
- What this solution (achieved 0.0) has done: 'I replace the simple purchase‑frequency counts with a recency‑weighted popularity model: each transaction gets an exponential decay weight based on how many days before the latest date it occurred ( weight = exp(‑days/30) ). The weighted sums are used to compute both the global top‑12 articles and each customer’s personal top‑12 list, then the same padding logic is applied. This keeps the overall baseline structure while giving recent purchases more influence, which should raise the MAP@12 toward the target.'
- What this solution (achieved 0.0) has done: 'I replace the recency‑weighted popularity with a simple purchase‑count metric. By setting every transaction’s weight to 1, both the global top‑12 list and each customer’s personal top‑12 list are built from raw frequencies, which generally gives a stronger signal for MAP@12 while keeping the overall workflow unchanged. This small tweak is expected to raise the score toward the target without altering any other logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import gc

DATA_PATH = "/kaggle/input/h-and-m-personalized-fashion-recommendations"
transactions_path = os.path.join(DATA_PATH, "transactions_train.csv")
sample_sub_path = os.path.join(DATA_PATH, "sample_submission.csv")

transactions = pd.read_csv(
    transactions_path,
    dtype={"article_id": str, "customer_id": str, "t_dat": str},
)

transactions["article_id"] = transactions["article_id"].str.zfill(10)

transactions["weight"] = np.float32(1.0)

global_top_articles = (
    transactions.groupby("article_id")["weight"].sum().nlargest(12).index.tolist()
)

cust_article_weights = (
    transactions.groupby(["customer_id", "article_id"])["weight"]
    .sum()
    .reset_index(name="wgt")
)

cust_article_weights = cust_article_weights.sort_values(
    ["customer_id", "wgt"], ascending=[True, False]
)

cust_top12 = cust_article_weights.groupby("customer_id").head(12)

cust_pred_series = cust_top12.groupby("customer_id")["article_id"].apply(
    lambda ids: " ".join(ids)
)
cust_pred_dict = cust_pred_series.to_dict()

del transactions, cust_article_weights, cust_top12
gc.collect()




## === cell 1
submission = pd.read_csv(sample_sub_path, dtype={"customer_id": str})


def build_prediction(customer_id):
    """
    Return up to 12 article IDs for a given customer:
    start with the customer's own top items (plain counts),
    then pad with globally popular items.
    """
    personal = cust_pred_dict.get(customer_id, "")
    personal_ids = personal.split() if personal else []
    for art in global_top_articles:
        if len(personal_ids) >= 12:
            break
        if art not in personal_ids:
            personal_ids.append(art)
    return " ".join(personal_ids[:12])


submission["prediction"] = submission["customer_id"].apply(build_prediction)




## === cell 2
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
