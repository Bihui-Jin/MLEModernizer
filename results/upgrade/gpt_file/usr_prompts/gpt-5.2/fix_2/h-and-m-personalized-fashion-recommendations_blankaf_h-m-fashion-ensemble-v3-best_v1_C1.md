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

0.0158

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0158) has done: 'Your notebook fails because it tries to read several pre-made “../input/fashions/*.csv” blend submissions that do not exist in this Kaggle environment, so nothing downstream is defined and no `submission.csv` is ever written. I replace those missing external inputs with locally-computed baseline submissions derived from `transactions_train.csv`, keeping the same blending core logic (weighted reciprocal-rank vote + top-12). To keep runtime under control, I generate candidates from the last ~8 weeks of transactions and compute a few simple-but-strong baselines (global top, recent global top, customer recent top, age-segment top) and then blend them using your existing `cust_blend` pattern. Finally, I align exactly to `sample_submission.csv` customers and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import gc

DATA_DIR_CANDIDATES = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data/h-and-m-personalized-fashion-recommendations",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(fname: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    for d in DATA_DIR_CANDIDATES:
        for root, _, files in os.walk(d):
            if fname in files:
                return os.path.join(root, fname)
    raise FileNotFoundError(f"Could not find {fname} under {DATA_DIR_CANDIDATES}")


transactions_path = _find_file("transactions_train.csv")
customers_path = _find_file("customers.csv")
sample_sub_path = _find_file("sample_submission.csv")

transactions_path, customers_path, sample_sub_path



## === cell 1

sample = pd.read_csv(sample_sub_path, usecols=["customer_id"])
sample["customer_id"] = sample["customer_id"].astype(str)

customers = pd.read_csv(customers_path, usecols=["customer_id", "age"])
customers["customer_id"] = customers["customer_id"].astype(str)

usecols = ["t_dat", "customer_id", "article_id"]
trans = pd.read_csv(
    transactions_path,
    usecols=usecols,
    dtype={"customer_id": "string", "article_id": "int32"},
)
trans["t_dat"] = pd.to_datetime(trans["t_dat"], errors="coerce")
trans["customer_id"] = trans["customer_id"].astype(str)

max_date = trans["t_dat"].max()
cutoff_recent = max_date - pd.Timedelta(days=56)
trans_recent = trans.loc[trans["t_dat"] >= cutoff_recent].copy()

top_global = trans["article_id"].value_counts().head(12).index.astype(str).tolist()
top_global_str = " ".join(top_global)

top_recent = (
    trans_recent["article_id"].value_counts().head(12).index.astype(str).tolist()
)
top_recent_str = " ".join(top_recent)

cust_recent = (
    trans_recent.groupby(["customer_id", "article_id"]).size().reset_index(name="cnt")
)
cust_recent = cust_recent.sort_values(["customer_id", "cnt"], ascending=[True, False])
cust_recent_top = (
    cust_recent.groupby("customer_id")["article_id"]
    .apply(lambda s: " ".join(s.astype(str).head(12)))
    .to_dict()
)

cust_age = customers.copy()
cust_age["age"] = pd.to_numeric(cust_age["age"], errors="coerce")
bins = [-np.inf, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, np.inf]
labels = list(range(len(bins) - 1))
cust_age["age_bin"] = pd.cut(
    cust_age["age"].fillna(-1), bins=bins, labels=labels
).astype("Int64")

trans_age = trans.merge(
    cust_age[["customer_id", "age_bin"]], on="customer_id", how="left"
)
age_top = (
    trans_age.groupby(["age_bin", "article_id"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["age_bin", "cnt"], ascending=[True, False])
)
age_top_map = (
    age_top.groupby("age_bin")["article_id"]
    .apply(lambda s: " ".join(s.astype(str).head(12)))
    .to_dict()
)

base = sample.merge(customers[["customer_id", "age"]], on="customer_id", how="left")
base["age"] = pd.to_numeric(base["age"], errors="coerce")
base["age_bin"] = pd.cut(base["age"].fillna(-1), bins=bins, labels=labels).astype(
    "Int64"
)

sub0 = sample.copy()
sub0["prediction"] = top_recent_str  # acts like "submission231" (recent popularity)

sub1 = sample.copy()
sub1["prediction"] = (
    sub0["customer_id"].map(cust_recent_top).fillna(top_recent_str)
)  # acts like "sub-age" but personalized

sub2 = sample.copy()
sub2["prediction"] = top_global_str  # acts like "sub-weekly" (global popularity)

sub3 = sample.copy()
sub3 = sub3.merge(base[["customer_id", "age_bin"]], on="customer_id", how="left")
sub3["prediction"] = (
    sub3["age_bin"].map(age_top_map).fillna(top_global_str)
)  # segment popularity
sub3 = sub3[["customer_id", "prediction"]].copy()

sub0 = sub0.sort_values("customer_id").reset_index(drop=True)
sub1 = sub1.sort_values("customer_id").reset_index(drop=True)
sub2 = sub2.sort_values("customer_id").reset_index(drop=True)
sub3 = sub3.sort_values("customer_id").reset_index(drop=True)

del trans_age, age_top
gc.collect()

sub0.head()



## === cell 2
sub0.columns = ["customer_id", "prediction0"]
sub0["prediction1"] = sub1["prediction"].astype(str)
sub0["prediction2"] = sub2["prediction"].astype(str)
sub0["prediction3"] = sub3["prediction"].astype(str)

del sub1, sub2, sub3
gc.collect()
sub0.head()




## === cell 3
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


sub0["prediction"] = sub0.apply(cust_blend, W=[0.62, 0.54, 0.50, 0.50], axis=1)
sub0[["customer_id", "prediction"]].head()



## === cell 4
del sub0["prediction0"]
del sub0["prediction1"]
del sub0["prediction2"]
del sub0["prediction3"]
gc.collect()

sub0.head()



## === cell 5

cutoff_7d = max_date - pd.Timedelta(days=7)
trans_7d = trans.loc[trans["t_dat"] >= cutoff_7d]
top_7d = trans_7d["article_id"].value_counts().head(12).index.astype(str).tolist()
top_7d_str = " ".join(top_7d)

cutoff_14d = max_date - pd.Timedelta(days=14)
trans_14d = trans.loc[trans["t_dat"] >= cutoff_14d]
top_14d = trans_14d["article_id"].value_counts().head(12).index.astype(str).tolist()
top_14d_str = " ".join(top_14d)

cust_all = (
    trans.groupby(["customer_id", "article_id"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["customer_id", "cnt"], ascending=[True, False])
)
cust_all_top = (
    cust_all.groupby("customer_id")["article_id"]
    .apply(lambda s: " ".join(s.astype(str).head(12)))
    .to_dict()
)

sub00 = sample.copy()
sub00["prediction"] = top_14d_str

sub1 = sample.copy()
sub1["prediction"] = top_7d_str

sub2 = sample.copy()
sub2["prediction"] = sub2["customer_id"].map(cust_all_top).fillna(top_recent_str)

sub3 = sample.copy()
sub3["prediction"] = sub3["customer_id"].map(cust_recent_top).fillna(top_recent_str)

sub00 = sub00.sort_values("customer_id").reset_index(drop=True)
sub1 = sub1.sort_values("customer_id").reset_index(drop=True)
sub2 = sub2.sort_values("customer_id").reset_index(drop=True)
sub3 = sub3.sort_values("customer_id").reset_index(drop=True)

del trans_7d, trans_14d, cust_all
gc.collect()

sub00.head()



## === cell 6
sub00.columns = ["customer_id", "prediction0"]
sub00["prediction1"] = sub1["prediction"].astype(str)
sub00["prediction2"] = sub2["prediction"].astype(str)
sub00["prediction3"] = sub3["prediction"].astype(str)

del sub1, sub2, sub3
gc.collect()
sub00.head()



## === cell 7
sub00["prediction"] = sub00.apply(cust_blend, W=[0.48, 0.42, 0.40, 0.34], axis=1)
sub00[["customer_id", "prediction"]].head()



## === cell 8
del sub00["prediction0"]
del sub00["prediction1"]
del sub00["prediction2"]
del sub00["prediction3"]
gc.collect()

sub00.head()



## === cell 9
sub1 = sample.copy()
sub1["prediction"] = sub1["customer_id"].map(cust_recent_top).fillna(top_recent_str)
sub1 = sub1.sort_values("customer_id").reset_index(drop=True)

sub1["prediction"] = sub1["prediction"].astype(str)
sub0["prediction"] = sub0["prediction"].astype(str)
sub00["prediction"] = sub00["prediction"].astype(str)

sub1.columns = ["customer_id", "prediction0"]
sub1["prediction1"] = sub0["prediction"]
sub1["prediction2"] = sub00["prediction"]

del sub0, sub00
gc.collect()

sub1.head()




## === cell 10
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
gc.collect()

sub1.head()



## === cell 11
out = sample.merge(sub1[["customer_id", "prediction"]], on="customer_id", how="left")
out["prediction"] = out["prediction"].fillna(top_recent_str).astype(str)

out.to_csv("submission.csv", index=False)
print(out.head())
print("Wrote submission.csv with shape:", out.shape)
