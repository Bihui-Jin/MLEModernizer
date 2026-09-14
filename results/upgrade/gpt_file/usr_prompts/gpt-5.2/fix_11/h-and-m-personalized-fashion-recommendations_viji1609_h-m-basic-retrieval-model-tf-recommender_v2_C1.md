# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.01387

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The initial crash is caused by an incompatibility between `tensorflow_recommenders` and the installed `protobuf==6.x` (it triggers the `MessageFactory.GetPrototype` AttributeError before `data_dir` is defined, cascading into all later NameErrors). The minimal robust fix is to avoid importing TFRS entirely and instead implement the same core two-tower retrieval training objective directly in Keras (customer/article embeddings with sampled-softmax over article IDs), which preserves the model architecture and training semantics closely while running in this environment. We then build a brute-force top‑K scorer over all article embeddings to generate 12 recommendations per customer, and write a valid `submission.csv` with the required columns and formatting. This produces an end-to-end runnable pipeline and should yield a non-trivial MAP@12 score (better than “no submission”), while keeping the overall approach (two-tower retrieval with embeddings trained from transaction pairs) intact.'
- What this solution (achieved 0.0) has done: 'I remove the protobuf-triggering import that crashes before `data_dir` is set, since this solution already uses pure Keras and doesn’t need `tensorflow_recommenders`. Then I fix the embedding-weight access bug by explicitly building the two towers (so the Embedding layer has variables) and by referencing the embedding matrix via `.embeddings` after build, avoiding the empty `.weights` list. Finally, I fix the inference shape issue by ensuring customer/article towers always output 2D `[B, D]` tensors (including for single examples) and keep the submission formatting exactly as required (`customer_id`, `prediction`) written to `submission.csv`.'
- What this solution (achieved 0.0) has done: 'The crash comes from a `protobuf`/TensorFlow internal incompatibility triggered at import-time; the safest minimal fix is to force the pure‑Python protobuf implementation *before* importing TensorFlow so the notebook can run. Then, to avoid producing an all‑zero score due to recommending arbitrary items, we keep your two‑tower sampled-softmax training but add a minimal, standard post-processing blend: fill each customer’s 12 predictions with their most recent purchases (from the last weeks of training) and backfill with global popular items, which is score-positive for MAP@12 while preserving the core model. Finally, we ensure article IDs are always 10‑digit strings and that the submission has the exact required columns and `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'I fix the import-time crash by forcing TensorFlow to use the pure‑Python protobuf implementation *before* importing TensorFlow and by restarting the protobuf runtime if it was already loaded, which resolves the `MessageFactory.GetPrototype` error that prevents any training/inference from running. Then I keep your two-tower sampled-softmax training exactly as-is, but fix a scoring bug: `cand_ids_np[topk_idx]` fails because `cand_ids_np` is a Python list and `topk_idx` is a 2D numpy array; converting `cand_ids_np` to a 1D numpy array makes indexing correct and prevents inference from silently failing or producing invalid predictions. Finally, I ensure the submission always contains 10-digit `article_id` strings and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'The crash happens immediately on importing TensorFlow because the forced pure‑python protobuf runtime still leaves an incompatible protobuf 6.x installed, which triggers `MessageFactory.GetPrototype` during TF init. The smallest robust fix is to avoid TensorFlow entirely and keep the same two‑tower retrieval core logic using a lightweight count-based association model: build customer→article interaction strengths with time-decay and then rank articles per customer (plus the existing recent-history and global-popular backfills). This run end-to-end in the provided environment, create a valid `submission.csv`, and should move the score up from 0.0 toward the target since it’s a standard strong baseline for this competition. Paths and submission formatting remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is almost certainly coming from a submission/customer_id mismatch (H&M requires the *hex* `customer_id` strings from `sample_submission.csv`, but your training-based dictionaries use the raw 64-bit IDs from `transactions_train.csv`, so nearly everyone falls back to generic popular items). I keep your exact core logic (recent-history + time-decayed customer×article weights + popular backfill), but add the minimal, standard fix: map transaction `customer_id` to the submission’s hex IDs using `customers.csv` (its index is the hex IDs and `customer_id` column is the raw numeric). This makes the personalization actually apply to the scored customers and should lift the score from 0.0 toward your target without changing the model/approach. I also ensure types are consistent and that we still write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

np.random.seed(42)

DATA_DIR_CANDIDATES = [
    Path("../input/h-and-m-personalized-fashion-recommendations"),
    Path("/kaggle/input/h-and-m-personalized-fashion-recommendations"),
    Path("/kaggle/data/h-and-m-personalized-fashion-recommendations"),
    Path("/kaggle/data"),
    Path("/kaggle/input"),
]
data_dir = next(
    (p for p in DATA_DIR_CANDIDATES if (p / "transactions_train.csv").exists()), None
)
if data_dir is None:
    raise FileNotFoundError(
        "Could not find transactions_train.csv in expected Kaggle locations. "
        f"Tried: {DATA_DIR_CANDIDATES}"
    )

print("Using data_dir:", data_dir)




## === cell 1
cust_map_df = pd.read_csv(data_dir / "customers.csv", usecols=["customer_id"])
cust_map_df = cust_map_df.reset_index().rename(
    columns={"index": "customer_hex", "customer_id": "customer_raw"}
)
cust_map_df["customer_hex"] = cust_map_df["customer_hex"].astype(str)
cust_map_df["customer_raw"] = pd.to_numeric(
    cust_map_df["customer_raw"], errors="coerce"
).astype("Int64")

raw_to_hex = (
    cust_map_df.dropna(subset=["customer_raw"])
    .set_index("customer_raw")["customer_hex"]
    .to_dict()
)

print("Customer mapping size:", len(raw_to_hex))
print("Example mapping:", next(iter(raw_to_hex.items())))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
/tmp/ipykernel_11/4177954893.py in <cell line: 0>()
     18 
     19 print("Customer mapping size:", len(raw_to_hex))
---> 20 print("Example mapping:", next(iter(raw_to_hex.items())))
     21 
     22 

StopIteration: 

## === cell 2
usecols = ["t_dat", "customer_id", "article_id"]
train0 = pd.read_csv(data_dir / "transactions_train.csv", usecols=usecols)

train0 = train0[train0["t_dat"] >= "2020-09-01"].copy()

train0["article_id"] = train0["article_id"].astype(str).str.zfill(10)
train0["t_dat"] = pd.to_datetime(train0["t_dat"], errors="coerce")

train0["customer_id"] = pd.to_numeric(train0["customer_id"], errors="coerce").astype(
    "Int64"
)
train0["customer_id"] = train0["customer_id"].map(raw_to_hex).astype("string")
train0 = train0.dropna(subset=["customer_id"]).copy()
train0["customer_id"] = train0["customer_id"].astype(str)

print(train0.head())
print("Train rows (after customer_id mapping):", len(train0))




## === cell 3
article_df = pd.read_csv(data_dir / "articles.csv", usecols=["article_id"])
article_df["article_id"] = article_df["article_id"].astype(str).str.zfill(10)
print(article_df.head(), "articles:", len(article_df))




## === cell 4
pop_window_start = train0["t_dat"].max() - pd.Timedelta(days=30)
popular_articles = (
    train0.loc[train0["t_dat"] >= pop_window_start, "article_id"]
    .value_counts()
    .head(200)
    .index.astype(str)
    .str.zfill(10)
    .tolist()
)

cust_window_start = train0["t_dat"].max() - pd.Timedelta(days=60)
tmp = train0.loc[
    train0["t_dat"] >= cust_window_start, ["customer_id", "t_dat", "article_id"]
].copy()
tmp = tmp.sort_values(["customer_id", "t_dat"], ascending=[True, False])

cust_recent = {}
for cid, g in tmp.groupby("customer_id", sort=False):
    seen = set()
    recs = []
    for a in g["article_id"].astype(str).str.zfill(10).tolist():
        if a not in seen:
            seen.add(a)
            recs.append(a)
        if len(recs) >= 12:
            break
    cust_recent[cid] = recs

print("Popular list size:", len(popular_articles), "Example:", popular_articles[:5])
print("Customers with recent history (60d):", len(cust_recent))




## === cell 5
max_date = train0["t_dat"].max()
age_days = (max_date - train0["t_dat"]).dt.days.clip(lower=0).astype(np.int16)

decay = np.exp(-age_days / 14.0).astype(np.float32)

agg = train0[["customer_id", "article_id"]].copy()
agg["w"] = decay
cust_art = (
    agg.groupby(["customer_id", "article_id"], sort=False)["w"].sum().reset_index()
)

TOPN_PER_CUST = 100
cust_art = cust_art.sort_values(["customer_id", "w"], ascending=[True, False])
cust_top = cust_art.groupby("customer_id", sort=False).head(TOPN_PER_CUST)

cust_model_recs = {}
for cid, g in cust_top.groupby("customer_id", sort=False):
    cust_model_recs[cid] = g["article_id"].astype(str).str.zfill(10).tolist()

print("Built model recs for customers:", len(cust_model_recs))
some_cid = next(iter(cust_model_recs.keys()))
print("Example customer:", some_cid, "model recs:", cust_model_recs[some_cid][:12])




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
/tmp/ipykernel_11/3555394583.py in <cell line: 0>()
     19 
     20 print("Built model recs for customers:", len(cust_model_recs))
---> 21 some_cid = next(iter(cust_model_recs.keys()))
     22 print("Example customer:", some_cid, "model recs:", cust_model_recs[some_cid][:12])
     23 

StopIteration: 

## === cell 6
sub = pd.read_csv(data_dir / "sample_submission.csv")
sub["customer_id"] = sub["customer_id"].astype(str)

cust_values = sub["customer_id"].values.astype(str)
pred_strings = np.empty(len(cust_values), dtype=object)

for i, cid in enumerate(cust_values):
    recs = []
    seen = set()

    for a in cust_recent.get(cid, []):
        a = str(a).zfill(10)
        if a not in seen:
            seen.add(a)
            recs.append(a)
        if len(recs) >= 12:
            break

    if len(recs) < 12:
        for a in cust_model_recs.get(cid, []):
            a = str(a).zfill(10)
            if a not in seen:
                seen.add(a)
                recs.append(a)
            if len(recs) >= 12:
                break

    if len(recs) < 12:
        for a in popular_articles:
            a = str(a).zfill(10)
            if a not in seen:
                seen.add(a)
                recs.append(a)
            if len(recs) >= 12:
                break

    pred_strings[i] = " ".join(recs[:12])

sub["prediction"] = pred_strings
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
