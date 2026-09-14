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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'Your notebook fails because it tries to read public-submission CSVs from `../input/hm-public-submissions`, which isn’t available in this environment; that prevents `sub0/sub1/sub2` from being created and cascades into NameErrors. I replace those missing inputs with three lightweight, legitimate baseline recommenders derived from `transactions_train.csv` (global top-12, last-week top-12, and exponential-decay top-12), then keep your exact blending logic (same `cust_blend` and weights) to produce the final predictions. I also make the script robust to the two possible dataset root paths you listed and ensure `article_id` is formatted as 10-digit strings in the `prediction` field. Finally, it always write a valid `submission-blend-1.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data/h-and-m-personalized-fashion-recommendations",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for b in BASE_CANDIDATES:
    if os.path.exists(os.path.join(b, "transactions_train.csv")) and os.path.exists(
        os.path.join(b, "sample_submission.csv")
    ):
        DATA_ROOT = b
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find transactions_train.csv and sample_submission.csv under expected Kaggle paths."
    )

TRANS_PATH = os.path.join(DATA_ROOT, "transactions_train.csv")
SAMPLE_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print("TRANS_PATH:", TRANS_PATH)
print("SAMPLE_PATH:", SAMPLE_PATH)



## === cell 2
transactions = pd.read_csv(
    TRANS_PATH,
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"t_dat": "string", "customer_id": "string", "article_id": "int64"},
)
sample = pd.read_csv(SAMPLE_PATH, dtype={"customer_id": "string"})
sample = sample.sort_values("customer_id").reset_index(drop=True)

transactions["article_id_str"] = transactions["article_id"].astype(str).str.zfill(10)

transactions.shape, sample.shape



## === cell 3

transactions["t_dat"] = pd.to_datetime(transactions["t_dat"], errors="coerce")
max_date = transactions["t_dat"].max()

top_global = transactions["article_id_str"].value_counts().head(12).index.tolist()
pred_global = " ".join(top_global)

week_start = max_date - pd.Timedelta(days=6)
week_tx = transactions.loc[transactions["t_dat"] >= week_start, "article_id_str"]
top_week = week_tx.value_counts().head(12).index.tolist()
if len(top_week) == 0:
    top_week = top_global
pred_week = " ".join(top_week)

halflife_days = 7.0
age_days = (max_date - transactions["t_dat"]).dt.days.astype("float32").clip(lower=0)
weights = np.exp(-np.log(2.0) * (age_days / halflife_days)).astype("float32")

decay_scores = (
    pd.DataFrame(
        {"article_id_str": transactions["article_id_str"].values, "w": weights}
    )
    .groupby("article_id_str", sort=False)["w"]
    .sum()
    .sort_values(ascending=False)
)
top_decay = decay_scores.head(12).index.tolist()
pred_decay = " ".join(top_decay)

sub0 = sample[["customer_id"]].copy()
sub0["prediction"] = pred_global

sub1 = sample[["customer_id"]].copy()
sub1["prediction"] = pred_week

sub2 = sample[["customer_id"]].copy()
sub2["prediction"] = pred_decay

sub0.shape, sub1.shape, sub2.shape



## === cell 4
print((sub0["prediction"] == sub1["prediction"]).mean())
print((sub0["prediction"] == sub2["prediction"]).mean())
print((sub1["prediction"] == sub2["prediction"]).mean())



## === cell 5
print(sub0.head())
print()
print(sub1.head())
print()
print(sub2.head())



## === cell 6
sub0 = sub0.sort_values("customer_id").reset_index(drop=True)
sub1 = sub1.sort_values("customer_id").reset_index(drop=True)
sub2 = sub2.sort_values("customer_id").reset_index(drop=True)

sub0.columns = ["customer_id", "prediction0"]
sub0["prediction1"] = sub1["prediction"].values
sub0["prediction2"] = sub2["prediction"].values
del sub1, sub2
sub0.head()




## === cell 7
def cust_blend(dt, W=[1, 1, 1]):

    REC = []
    REC.append(dt["prediction0"].split())
    REC.append(dt["prediction1"].split())
    REC.append(dt["prediction2"].split())

    res = {}
    for M in range(len(REC)):
        for n, v in enumerate(REC[M]):
            if v in res:
                res[v] += W[M] / (n + 1)
            else:
                res[v] = W[M] / (n + 1)

    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())

    return " ".join(res[:12])


sub0["prediction"] = sub0.apply(cust_blend, W=[1.05, 1.00, 0.95], axis=1)
sub0.head()



## === cell 8
print((sub0["prediction"] == sub0["prediction0"]).mean())
print((sub0["prediction"] == sub0["prediction1"]).mean())
print((sub0["prediction"] == sub0["prediction2"]).mean())



## === cell 9
sub0 = sub0[["customer_id", "prediction"]].copy()
sub0 = sample[["customer_id"]].merge(sub0, on="customer_id", how="left")

if sub0["prediction"].isna().any():
    sub0["prediction"] = sub0["prediction"].fillna(pred_global)

sub0.to_csv("submission-blend-1.csv", index=False)
print("Wrote submission-blend-1.csv with shape:", sub0.shape)
print(sub0.head())
