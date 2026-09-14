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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your current notebook is an ensembling script that expects several external submission CSVs in `../input/fashions/`, but those files don’t exist in this Kaggle environment, so it fails before producing any `submission.csv`. To keep the core intent (produce predictions) while making minimal, runnable changes, I replace the missing-input ensemble with a standard, fast H&M baseline that generates per-customer recommendations from the training transactions (recent popularity + customer’s recent purchases fallback). This fixes the FileNotFound errors, ensures correct submission formatting and alignment with `sample_submission.csv`, and should yield a reasonable MAP@12 baseline (typically in the right ballpark of your target) without adding heavy models or long training.'
- What this solution (achieved 0.0) has done: 'Your current script is already a valid “recent purchases + global popularity” baseline, so a 0.0 score strongly suggests a submission-format mismatch rather than model quality. I make two minimal, score-relevant fixes: (1) ensure `customer_id` is written as a plain string column (not pandas `string` dtype that can sometimes serialize unexpectedly), and (2) guarantee every row has exactly 12 space-separated `article_id`s by padding with global popular items (some customers can end up with fewer than 12 if `top_global` is too short or overlaps heavily). I also add a couple of lightweight sanity checks on the serialized CSV to catch the kind of issue that can lead to 0.0 even when predictions look fine. Core logic (recent deduped customer history + recent global popularity fallback) remains the same.'
- What this solution (achieved 0.0) has done: 'Your current logic should not score 0.0, so the most likely cause is an ID-format mismatch: H&M `customer_id` must be a 64-hex string, and reading/writing it as pandas `string` can preserve hidden whitespace or NA-like values; if Kaggle can’t match customer_ids, you effectively get 0.0. I make minimal, score-relevant changes to (1) force `customer_id` to a clean, stripped Python `str` consistently in both sample and transactions, (2) build `cust_hist` with those same cleaned IDs so lookups always hit, and (3) add a strict sanity check that all customer_ids are 64-hex before writing. Core recommendation logic (recent unique purchases then global popularity fallback to 12 items) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import gc



## === cell 1
BASE_DIRS = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/input",
    "/kaggle/data/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for b in BASE_DIRS:
        p = os.path.join(b, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {filename} in any of: {BASE_DIRS}")


transactions_path = find_file("transactions_train.csv")
sample_path = find_file("sample_submission.csv")

transactions_path, sample_path



## === cell 2
dtypes = {
    "customer_id": "object",  # read as object to avoid pandas string/NA semantics
    "article_id": "int64",
    "sales_channel_id": "int8",
    "price": "float32",
}
usecols = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]

trans = pd.read_csv(transactions_path, usecols=usecols, dtype=dtypes)
trans["t_dat"] = pd.to_datetime(trans["t_dat"], errors="coerce")

sample = pd.read_csv(
    sample_path, usecols=["customer_id"], dtype={"customer_id": "object"}
)

trans["customer_id"] = trans["customer_id"].astype(str).str.strip()
sample["customer_id"] = sample["customer_id"].astype(str).str.strip()

sample.shape, trans.shape



## === cell 3
max_date = trans["t_dat"].max()
recent_days = 14
recent_start = max_date - pd.Timedelta(days=recent_days)

recent = trans.loc[trans["t_dat"] >= recent_start, ["article_id"]]

top_global = (
    recent["article_id"].value_counts().head(500).index.astype("int64").tolist()
)

len(top_global), top_global[:5], max_date, recent_start



## === cell 4
trans_sorted = trans.sort_values(["customer_id", "t_dat"], ascending=[True, False])

cust_hist = (
    trans_sorted.drop_duplicates(["customer_id", "article_id"], keep="first")
    .groupby("customer_id")["article_id"]
    .apply(lambda s: s.head(50).astype("int64").tolist())
)

del trans_sorted
gc.collect()

cust_hist.head()



## === cell 5
top_global_str = [f"{a:010d}" for a in top_global]


def make_pred(cid: str) -> str:
    items = cust_hist.get(cid, None)
    rec = []
    seen = set()

    if items is not None:
        for a in items:
            s = f"{int(a):010d}"
            if s not in seen:
                seen.add(s)
                rec.append(s)
            if len(rec) >= 12:
                return " ".join(rec[:12])

    for s in top_global_str:
        if s not in seen:
            seen.add(s)
            rec.append(s)
        if len(rec) >= 12:
            break

    if len(rec) < 12:
        for s in top_global_str:
            if len(rec) >= 12:
                break
            if s not in seen:
                seen.add(s)
                rec.append(s)

    return " ".join(rec[:12])


sample_ids = sample["customer_id"].tolist()
preds = [make_pred(cid) for cid in sample_ids]

submission = pd.DataFrame(
    {
        "customer_id": np.array(sample_ids, dtype=object),
        "prediction": np.array(preds, dtype=object),
    }
)
submission.head(), submission.shape



## === cell 6
assert submission["customer_id"].isna().sum() == 0
assert submission["prediction"].isna().sum() == 0
assert submission.shape[0] == sample.shape[0]
assert submission["customer_id"].nunique() == sample["customer_id"].nunique()
assert (submission["prediction"].str.split().str.len() == 12).all()
assert (submission["prediction"].str.contains(r"^\d{10}( \d{10}){11}$")).all()

assert (
    submission["customer_id"].str.match(r"^[0-9a-f]{64}$").all()
), "customer_id format is not 64-hex"

submission.to_csv("submission.csv", index=False)

chk = pd.read_csv(
    "submission.csv", dtype={"customer_id": "object", "prediction": "object"}
)
assert list(chk.columns) == ["customer_id", "prediction"]
assert chk.shape[0] == sample.shape[0]
assert chk["customer_id"].astype(str).str.strip().str.match(r"^[0-9a-f]{64}$").all()

print("Wrote submission.csv with shape:", chk.shape)
print(chk.head())



## === cell 7
del (
    trans,
    recent,
    cust_hist,
    preds,
    submission,
    sample,
    top_global,
    top_global_str,
    chk,
    sample_ids,
)
gc.collect()
