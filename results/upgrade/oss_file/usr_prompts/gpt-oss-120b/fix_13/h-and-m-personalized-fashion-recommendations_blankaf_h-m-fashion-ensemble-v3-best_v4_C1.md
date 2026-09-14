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

0.0237733507340115

# 6. Current score

0.02043

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00296) has done: 'The script failed because it referenced non‑existent relative paths ( `../input/fashions/...` ). I added robust path handling that points to the actual Kaggle input directory (`/kaggle/input/h-and-m-personalized-fashion-recommendations`). With the correct locations the sample submission loads, the most‑frequent 12 articles are computed, and a valid `submission.csv` is written.'
- What this solution (achieved 0.00734) has done: 'I replace the single global‑most‑popular list with per‑customer most‑frequent articles (using the training data) and keep the global list only as a fallback for customers with no history. This simple personalization should raise MAP@12 considerably toward the target while preserving the overall workflow and output format.'
- What this solution (achieved 0.01963) has done: 'I add a short recency filter: first determine the most recent transaction date, then only count purchases from the last 30 days to build both the global and per‑customer top‑12 lists. This keeps the overall logic unchanged (still using most‑frequent recent items) but focuses predictions on items that actually appeared shortly before the test period, which should raise MAP@12 toward the target. The rest of the pipeline (loading sample submission, applying predictions, writing `submission.csv`) remains identical.'
- What this solution (achieved 0.02043) has done: 'I keep the overall workflow unchanged but improve the per‑customer predictions by ensuring each customer receives exactly 12 article IDs. After gathering the recent 30‑day purchase frequencies, I fill any shortfall with the global most‑popular recent items (avoiding duplicates). This simple padding provides more predictions per customer, which typically raises MAP@12 and moves the score closer to the target.'
- What this solution (achieved 0.01988) has done: 'I add overall purchase counters (not just recent) for each customer and use them to fill any remaining slots after the recent‑based top‑12 list, before falling back to the global recent popular items. This keeps the original workflow but gives each customer a richer personalized list, which should raise MAP@12 enough to reach the target range while preserving the core logic.'
- What this solution (achieved 0.0178) has done: 'I extend the recent‑window from 30 to 45 days to give each customer more recent signal, compute an overall‑global top‑12 list (using the whole training period) and use it as the final fallback instead of the recent‑only list. These small tweaks keep the original workflow intact while adding a bit more coverage, which should raise MAP@12 toward the target without altering the core modeling logic.'
- What this solution (achieved 0.02043) has done: 'I tighten the recency window from 45 to 30 days (so the recent signal is fresher) and reorder the fallback logic: after a customer’s own recent items we first fill with the globally‑most‑recent popular items, then with that customer’s overall historic items, and finally with the overall‑global popular items. I also change the default prediction for completely unseen customers to use the global recent top‑12 list instead of the overall list. These small adjustments keep the original workflow intact while providing fresher and better‑aligned recommendations, which should raise MAP@12 toward the target.'
- What this solution (achieved 0.0167) has done: 'I increase the recency window from 30 to 60 days so that recent counters capture more relevant purchases, and I reorder the fallback logic to prioritize a customer’s overall historic items before using the globally‑recent popular items. These small, low‑risk tweaks keep the original workflow intact while giving the model a better chance to raise MAP@12 into the target range.'
- What this solution (achieved 0.01988) has done: 'I restore the recency window from 60 days back to 30 days, which the earlier experiments showed improves MAP@12 while keeping the overall logic unchanged. This small change should raise the score closer to the target without altering any core modeling steps or the submission format.'
- What this solution (achieved 0.01812) has done: 'I increase the recency window from 30 days to 45 days, giving each customer more recent purchase information while preserving the overall prediction flow and fallback logic. This small change is expected to raise MAP@12 toward the target without altering the core algorithm.'
- What this solution (achieved 0.02043) has done: 'I shorten the recency window to the last 30 days (instead of 45) to focus on fresher purchases and reorder the fallback logic so that after a customer’s own recent items we first add globally‑recent popular articles, then the customer’s overall historic items, and finally the global overall top items. These minimal adjustments keep the original counting‑based approach while providing more relevant recommendations, which should raise MAP@12 toward the target.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict

possible_bases = [
    Path("/kaggle/input/h-and-m-personalized-fashion-recommendations"),
    Path("./kaggle/input/h-and-m-personalized-fashion-recommendations"),
    Path("../input/h-and-m-personalized-fashion-recommendations"),
]
base_path = None
for p in possible_bases:
    if p.is_dir():
        base_path = p
        break
if base_path is None:
    raise FileNotFoundError(
        "Competition data directory not found in expected locations."
    )

sample_path = base_path / "sample_submission.csv"
train_path = base_path / "transactions_train.csv"

sample_sub = pd.read_csv(sample_path, dtype={"customer_id": str})
sample_sub = sample_sub.sort_values("customer_id").reset_index(drop=True)




## === cell 1
chunksize = 1_000_000
max_date = None

for chunk in pd.read_csv(
    train_path,
    usecols=["t_dat"],
    dtype={"t_dat": str},
    chunksize=chunksize,
):
    chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
    cur_max = chunk["t_dat"].max()
    if max_date is None or cur_max > max_date:
        max_date = cur_max

cutoff_date = max_date - pd.Timedelta(days=30)

global_recent_counter = Counter()  # recent global popularity
overall_counter = Counter()  # overall global popularity
cust_recent_counters = defaultdict(Counter)  # recent per‑customer
cust_overall_counters = defaultdict(Counter)  # overall per‑customer

for chunk in pd.read_csv(
    train_path,
    usecols=["customer_id", "article_id", "t_dat"],
    dtype={"customer_id": str, "article_id": str, "t_dat": str},
    chunksize=chunksize,
):
    chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])

    overall_counter.update(chunk["article_id"])
    for cust_id, art_id in zip(chunk["customer_id"], chunk["article_id"]):
        cust_overall_counters[cust_id][art_id] += 1

    recent = chunk[chunk["t_dat"] >= cutoff_date]
    global_recent_counter.update(recent["article_id"])
    for cust_id, art_id in zip(recent["customer_id"], recent["article_id"]):
        cust_recent_counters[cust_id][art_id] += 1

global_recent_top12 = [art for art, _ in global_recent_counter.most_common(12)]
global_overall_top12 = [art for art, _ in overall_counter.most_common(12)]

cust_pred_str = {}
all_cust_ids = set(cust_recent_counters.keys()) | set(cust_overall_counters.keys())
for cust_id in all_cust_ids:
    recent_cnt = cust_recent_counters.get(cust_id, Counter())
    overall_cnt = cust_overall_counters.get(cust_id, Counter())

    top12 = [art for art, _ in recent_cnt.most_common()]

    if len(top12) < 12:
        for art in global_recent_top12:
            if art not in top12:
                top12.append(art)
                if len(top12) == 12:
                    break

    if len(top12) < 12:
        for art, _ in overall_cnt.most_common():
            if art not in top12:
                top12.append(art)
                if len(top12) == 12:
                    break

    if len(top12) < 12:
        for art in global_overall_top12:
            if art not in top12:
                top12.append(art)
                if len(top12) == 12:
                    break

    cust_pred_str[cust_id] = " ".join(top12[:12])




## === cell 2
def get_prediction(row):
    return cust_pred_str.get(row["customer_id"], " ".join(global_recent_top12))


sample_sub["prediction"] = sample_sub.apply(get_prediction, axis=1)




## === cell 3
output_path = "submission.csv"
sample_sub.to_csv(output_path, index=False)
