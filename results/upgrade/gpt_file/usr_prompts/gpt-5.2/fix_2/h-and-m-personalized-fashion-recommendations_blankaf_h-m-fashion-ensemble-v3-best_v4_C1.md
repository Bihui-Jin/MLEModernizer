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

0.00568

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.00568) has done: 'Your notebook fails because it tries to read several pre-made submission files from `../input/fashions/`, which do not exist in this Kaggle environment. To keep the blending core logic intact while making it runnable end-to-end, I add a small fallback that generates reasonable base submissions directly from `transactions_train.csv` (global top-12 and last-week top-12) whenever those external files are missing. Then the existing `cust_blend` rank-weighted ensembling run as written, and we always output a valid `submission.csv` with the required columns and all customers from `sample_submission.csv`. This should yield a non-trivial MAP@12 (typically around the target band for simple popularity-based baselines) instead of “no submission produced”.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import gc

DATA_DIR_CANDIDATES = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data",
    "/kaggle/input",
]


def _find_file(filename: str) -> str:
    for base in DATA_DIR_CANDIDATES:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    for base in DATA_DIR_CANDIDATES:
        p = os.path.join(base, "h-and-m-personalized-fashion-recommendations", filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in known Kaggle input locations."
    )


SAMPLE_PATH = _find_file("sample_submission.csv")
TRANS_PATH = _find_file("transactions_train.csv")

sample = pd.read_csv(SAMPLE_PATH)
sample = sample.sort_values("customer_id").reset_index(drop=True)

trans = pd.read_csv(
    TRANS_PATH,
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"t_dat": "string", "customer_id": "string", "article_id": "int32"},
)



## === cell 1


def build_global_top12(
    transactions: pd.DataFrame, sample_df: pd.DataFrame
) -> pd.DataFrame:
    top = transactions["article_id"].value_counts().head(12).index.astype(str).tolist()
    pred = " ".join(top)
    out = sample_df[["customer_id"]].copy()
    out["prediction"] = pred
    return out


def build_last_week_top12(
    transactions: pd.DataFrame, sample_df: pd.DataFrame
) -> pd.DataFrame:
    last_date = transactions["t_dat"].max()
    last_dt = pd.to_datetime(last_date)
    start_dt = (last_dt - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    last_week = transactions[transactions["t_dat"] >= start_dt]
    top = last_week["article_id"].value_counts().head(12).index.astype(str).tolist()
    if len(top) == 0:
        return build_global_top12(transactions, sample_df)
    pred = " ".join(top)
    out = sample_df[["customer_id"]].copy()
    out["prediction"] = pred
    return out


def safe_read_submission(
    path: str, sample_df: pd.DataFrame, fallback_kind: str
) -> pd.DataFrame:
    """
    Try reading an existing submission file; if missing, generate a fallback.
    Ensures:
      - all customers from sample_submission
      - sorted by customer_id
      - columns: customer_id, prediction (string)
    """
    if path is not None and os.path.exists(path):
        df = pd.read_csv(path)
        if "customer_id" not in df.columns or "prediction" not in df.columns:
            raise ValueError(
                f"{path} must contain columns ['customer_id','prediction']"
            )
        df["prediction"] = df["prediction"].astype(str)
        df = df.merge(sample_df[["customer_id"]], on="customer_id", how="right")
        df["prediction"] = df["prediction"].fillna("")
        df = df.sort_values("customer_id").reset_index(drop=True)
        return df[["customer_id", "prediction"]]

    if fallback_kind == "global":
        df = build_global_top12(trans, sample_df)
    elif fallback_kind == "last_week":
        df = build_last_week_top12(trans, sample_df)
    else:
        df = build_global_top12(trans, sample_df)

    df["prediction"] = df["prediction"].astype(str)
    df = df.sort_values("customer_id").reset_index(drop=True)
    return df




## === cell 2

sub0 = (
    safe_read_submission(path=None, sample_df=sample, fallback_kind="global")
    .sort_values("customer_id")
    .reset_index(drop=True)
)
sub1 = (
    safe_read_submission(path=None, sample_df=sample, fallback_kind="last_week")
    .sort_values("customer_id")
    .reset_index(drop=True)
)
sub2 = (
    safe_read_submission(path=None, sample_df=sample, fallback_kind="global")
    .sort_values("customer_id")
    .reset_index(drop=True)
)
sub3 = (
    safe_read_submission(path=None, sample_df=sample, fallback_kind="last_week")
    .sort_values("customer_id")
    .reset_index(drop=True)
)



## === cell 3
sub0.columns = ["customer_id", "prediction0"]
sub0["prediction1"] = sub1["prediction"]
sub0["prediction2"] = sub2["prediction"]
sub0["prediction3"] = sub3["prediction"].astype(str)



## === cell 4
del sub1, sub2, sub3
gc.collect()
sub0.head()




## === cell 5
def cust_blend(dt, W=[1, 1, 1, 1]):
    REC = []
    REC.append(str(dt["prediction0"]).split())
    REC.append(str(dt["prediction1"]).split())
    REC.append(str(dt["prediction2"]).split())
    REC.append(str(dt["prediction3"]).split())
    res = {}
    for M in range(len(REC)):
        for n, v in enumerate(REC[M]):
            if v in res:
                res[v] += W[M] / (n + 1)
            else:
                res[v] = W[M] / (n + 1)
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return " ".join(res[:12])


sub0["prediction"] = sub0.apply(cust_blend, W=[0.25, 0.25, 0.25, 0.25], axis=1)
sub0.head()



## === cell 6
del sub0["prediction0"]
del sub0["prediction1"]
del sub0["prediction2"]
del sub0["prediction3"]
gc.collect()



## === cell 7
sub00 = (
    safe_read_submission(path=None, sample_df=sample, fallback_kind="last_week")
    .sort_values("customer_id")
    .reset_index(drop=True)
)
sub1 = (
    safe_read_submission(path=None, sample_df=sample, fallback_kind="global")
    .sort_values("customer_id")
    .reset_index(drop=True)
)
sub2 = (
    safe_read_submission(path=None, sample_df=sample, fallback_kind="last_week")
    .sort_values("customer_id")
    .reset_index(drop=True)
)
sub3 = (
    safe_read_submission(path=None, sample_df=sample, fallback_kind="global")
    .sort_values("customer_id")
    .reset_index(drop=True)
)



## === cell 8
sub00.columns = ["customer_id", "prediction0"]
sub00["prediction1"] = sub1["prediction"]
sub00["prediction2"] = sub2["prediction"]
sub00["prediction3"] = sub3["prediction"].astype(str)

del sub1, sub2, sub3
gc.collect()
sub00.head()



## === cell 9
sub00["prediction"] = sub00.apply(cust_blend, W=[0.25, 0.25, 0.25, 0.25], axis=1)
sub00.head()



## === cell 10
del sub00["prediction0"]
del sub00["prediction1"]
del sub00["prediction2"]
del sub00["prediction3"]
gc.collect()



## === cell 11
sub1 = (
    safe_read_submission(path=None, sample_df=sample, fallback_kind="global")
    .sort_values("customer_id")
    .reset_index(drop=True)
)
sub1["prediction"] = sub1["prediction"].astype(str)
sub0["prediction"] = sub0["prediction"].astype(str)
sub00["prediction"] = sub00["prediction"].astype(str)

sub1.columns = ["customer_id", "prediction0"]
sub1["prediction1"] = sub0["prediction"]
sub1["prediction2"] = sub00["prediction"]
del sub0, sub00
gc.collect()




## === cell 12
def cust_blend(dt, W=[1, 1, 1]):
    REC = []
    REC.append(str(dt["prediction0"]).split())
    REC.append(str(dt["prediction1"]).split())
    REC.append(str(dt["prediction2"]).split())
    res = {}
    for M in range(len(REC)):
        for n, v in enumerate(REC[M]):
            if v in res:
                res[v] += W[M] / (n + 1)
            else:
                res[v] = W[M] / (n + 1)
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return " ".join(res[:12])


sub1["prediction"] = sub1.apply(cust_blend, W=[0.415, 0.565, 0.22], axis=1)

del sub1["prediction0"]
del sub1["prediction1"]
del sub1["prediction2"]



## === cell 13
sub1 = sub1.merge(sample[["customer_id"]], on="customer_id", how="right")
sub1["prediction"] = sub1["prediction"].fillna("").astype(str)
sub1 = sub1.sort_values("customer_id").reset_index(drop=True)
sub1 = sub1[["customer_id", "prediction"]]

sub1.to_csv("submission.csv", index=False)
print(sub1.head())
print("Wrote submission.csv with shape:", sub1.shape)
