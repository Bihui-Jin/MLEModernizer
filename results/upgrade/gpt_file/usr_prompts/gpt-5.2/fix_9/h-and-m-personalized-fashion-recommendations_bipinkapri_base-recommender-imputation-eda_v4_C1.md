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

0.00368

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime error caused by `DataFrame.append` being removed in pandas 2.x by replacing it with `pd.concat`, which allow the pipeline to complete and actually write a submission file. I also ensure both parts being concatenated have exactly the required columns (`customer_id`, `prediction`) and that `prediction` strings contain at most 12 space-separated `article_id`s (to match MAP@12 semantics and avoid invalid-format issues). Finally, I build predictions for *all* customers in `sample_submission.csv` (not just customers.csv), because Kaggle expects every `customer_id` from the sample submission to be present; customers missing from your customer table get a safe global-popularity fallback.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an ID-format mismatch: the submission `customer_id` values are the 64-bit hex strings from `sample_submission.csv`, but your merges use `customers.csv` IDs as-is (often read differently if not forced to string), causing almost everyone to fall back to the same global list and yielding near-zero MAP@12. I make minimal, metric-relevant fixes: enforce `customer_id` as `str` consistently across all loaded tables, normalize transactions `t_dat` to datetime (no logic change) and then compute “global popular” from the last 7 days (a very small change that usually improves MAP@12 without altering the recommender structure). I also ensure `age_bucket` never becomes NaN by assigning a safe bucket for any missing ages so age-based fallback doesn’t collapse to NaN. The rest of your pipeline (section-based recommendations + age-bucket fallback + global fallback, max 12 items) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with `customer_id` being formatted differently between `sample_submission.csv` and the tables used to build recommendations, causing almost everyone to get an effectively “wrong/empty” prediction. I make a minimal, metric-relevant fix by forcing `customer_id` to be a plain Python `str` everywhere and stripping any accidental whitespace, while keeping your exact recommendation logic unchanged. I also ensure the final submission has exactly one row per `customer_id` from the sample, in the same order, and that every prediction is non-empty and capped to 12 items (as MAP@12 expects). These changes should lift the score from 0.0 toward your target without changing the core recommender approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with a submission alignment/ID mismatch: you build recommendations for some customers, but you never guarantee exactly one final row per `customer_id` in the sample with clean, space-delimited 10-digit `article_id`s (duplicates/empties can silently tank MAP@12). I make minimal, metric-relevant fixes: (1) enforce one-row-per-customer by aggregating/deduplicating predictions after concatenation, (2) ensure every predicted token is a 10-digit `article_id` and remove duplicates while keeping order (better MAP@12 than repeated items), and (3) build the global fallback from the most recent 7 days and use it whenever a customer’s prediction becomes empty after cleaning. Core logic (section-based for “active” customers, age-bucket fallback otherwise) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with your predictions being largely irrelevant to the *last-week* target window: you compute “popular articles per section” over the full history, which tends to underperform badly on this competition and can collapse close to 0. I keep your exact recommender structure (section-popularity → customer top sections; else age-bucket → popular sections; else global fallback), but compute `pop_art` and `Age_pref_sec` from the most recent 7 days only (the same 7-day window you already use for `global_fallback`). I also ensure customers missing from `customers.csv` get a safe `age_bucket` before merging to avoid NaN-based empty predictions, and keep the same submission formatting/cleaning so the file stays valid. These are minimal, metric-aligned changes that should increase the score toward your target without changing the overall logic.'
- What this solution (achieved 0.0) has done: 'I fix the runtime error in the groupby-transform step by ensuring the `recommend` column contains only strings (filling missing merges with empty strings) before joining, which prevents floats/NaNs from breaking `" ".join(...)`. I also keep your recommendation logic identical but make the aggregation robust by using a deterministic groupby-agg join instead of transform (same semantics: concatenate the three section-level recommendation strings per customer). Finally, I keep the strict submission requirements: one row per `customer_id` from `sample_submission.csv`, `prediction` always non-empty, deduplicated, and capped to 12 article_ids, then write `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with the predictions being computed on the *wrong time window*: you build customer preference sections from all history (`MD2`), while the rest of your logic (popularity and global fallback) uses the last 7 days, so the section choices are stale and can collapse MAP@12. I keep your exact recommender structure (customer top sections → section-popular articles; else age-bucket sections → section-popular articles; else global fallback) and only make the minimal metric-aligned change to compute `cust_pref_sec` from `MD2_recent` (same 7-day window already used elsewhere). I also enforce the same recent-window consistency for the “number of sections per customer” split (list1 vs list2) so customers are routed correctly. Submission formatting, deduplication, and the 12-item cap stay unchanged and it still write `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission formatting mismatch rather than a modeling issue: H&M expects the column to be named exactly `prediction`, and any extra/renamed column (e.g., `predictions`) can yield a scored-but-empty/invalid submission that returns 0. I make a minimal fix to guarantee the output schema is exactly `customer_id,prediction`, with one row per `customer_id` in `sample_submission.csv` and in the same order. I also harden `clean_prediction` so it always outputs exactly up to 12 valid 10-digit `article_id`s and never returns an empty string (which can silently tank MAP@12). Core recommendation logic (recent-window section popularity + age-bucket fallback + global fallback) stays unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm



## === cell 1
articles = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv",
    dtype={"article_id": "int64"},
)
transactions = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv",
    dtype={"article_id": "int64", "customer_id": "string"},
)
customer = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv",
    dtype={"customer_id": "string"},
)



## === cell 2
articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)

transactions["article_id"] = transactions["article_id"].astype(str).str.zfill(10)
transactions["customer_id"] = transactions["customer_id"].astype(str).str.strip()

customer["customer_id"] = customer["customer_id"].astype(str).str.strip()

transactions["t_dat"] = pd.to_datetime(transactions["t_dat"], errors="coerce")



## === cell 3
customer.nunique()



## === cell 4
100 * customer.isnull().sum() / customer.shape[0]



## === cell 5
100 * customer["fashion_news_frequency"].value_counts() / customer.shape[0]



## === cell 6
100 * customer["club_member_status"].value_counts() / customer.shape[0]



## === cell 7
customer.drop(labels=["FN", "Active", "fashion_news_frequency"], inplace=True, axis=1)



## === cell 8
customer["age"].hist()



## === cell 9
customer["age"] = customer["age"].fillna(customer["age"].median())



## === cell 10
customer["club_member_status"] = customer["club_member_status"].fillna(
    customer["club_member_status"].mode().values[0]
)




## === cell 11
def make_buckets(x):
    if x >= 16 and x <= 24:
        return "Youth"
    elif x > 24 and x <= 40:
        return "Young Adults"
    elif x > 40 and x <= 64:
        return "Middle Age Adults"
    elif x > 64:
        return "Seniors"




## === cell 12
customer["age_bucket"] = customer["age"].apply(lambda x: make_buckets(x))
customer["age_bucket"] = customer["age_bucket"].fillna("Young Adults")



## === cell 13
transactions.drop(labels=["sales_channel_id"], inplace=True, axis=1)



## === cell 14
merged_data1 = pd.merge(left=transactions, right=customer, how="left", on="customer_id")



## === cell 15
del transactions



## === cell 16
MD2 = pd.merge(
    left=merged_data1,
    right=articles[
        [
            "article_id",
            "prod_name",
            "index_group_name",
            "graphical_appearance_name",
            "index_name",
            "section_name",
            "garment_group_name",
        ]
    ],
    on="article_id",
    how="left",
)



## === cell 17
MD2.shape



## === cell 18
del merged_data1



## === cell 19
Sales_age_group = MD2.groupby(["age_bucket"])["price"].sum().reset_index()



## === cell 20
sns.barplot(
    x="age_bucket",
    y="price",
    data=Sales_age_group.sort_values(by=["price"], ascending=False),
)
plt.xticks(rotation=90)



## === cell 21
transations_per_group = MD2["age_bucket"].value_counts().reset_index()
transations_per_group.columns = ["age_bucket", "orders"]



## === cell 22
sns.barplot(x="age_bucket", y="orders", data=transations_per_group)
plt.xticks(rotation=90)



## === cell 23
prod_name = MD2["prod_name"].value_counts().reset_index()
prod_name.columns = ["prod_name", "orders"]



## === cell 24
sns.barplot(x="prod_name", y="orders", data=prod_name.head(10))
plt.xticks(rotation=90)



## === cell 25
index_group_name = MD2["index_group_name"].value_counts().reset_index()
index_group_name.columns = ["index_group_name", "orders"]

sns.barplot(x="index_group_name", y="orders", data=index_group_name.head(10))
plt.xticks(rotation=90)



## === cell 26
graphical_appearance_name = (
    MD2["graphical_appearance_name"].value_counts().reset_index()
)
graphical_appearance_name.columns = ["graphical_appearance_name", "orders"]

sns.barplot(
    x="graphical_appearance_name", y="orders", data=graphical_appearance_name.head(10)
)
plt.xticks(rotation=90)



## === cell 27
index_name = MD2["index_name"].value_counts().reset_index()
index_name.columns = ["index_name", "orders"]

sns.barplot(x="index_name", y="orders", data=index_name)
plt.xticks(rotation=90)



## === cell 28
MD2["section_name"].value_counts()

section_name = MD2["section_name"].value_counts().reset_index()
section_name.columns = ["section_name", "orders"]

sns.barplot(x="section_name", y="orders", data=section_name.head(10))
plt.xticks(rotation=90)



## === cell 29
garment_group_name = MD2["garment_group_name"].value_counts().reset_index()
garment_group_name.columns = ["garment_group_name", "orders"]

sns.barplot(x="garment_group_name", y="orders", data=garment_group_name.head(10))
plt.xticks(rotation=90)



## === cell 30
max_date = MD2["t_dat"].max()
if pd.notna(max_date):
    MD2_recent = MD2[MD2["t_dat"] >= (max_date - pd.Timedelta(days=7))].copy()
    if MD2_recent.shape[0] == 0:
        MD2_recent = MD2
else:
    MD2_recent = MD2



## === cell 31
Age_pref_sec = (
    MD2_recent.groupby(["age_bucket", "section_name"])["article_id"]
    .count()
    .reset_index()
)
Age_pref_sec.sort_values(by=["age_bucket", "article_id"], ascending=False, inplace=True)



## === cell 32
Age_pref_sec.groupby(["age_bucket"]).head(4).reset_index(drop=True)



## === cell 33
cust_pref_sec = (
    MD2_recent.groupby(["customer_id", "section_name"])["article_id"]
    .count()
    .reset_index()
)



## === cell 34
cust_pref_sec.sort_values(
    by=["customer_id", "article_id"], ascending=False, inplace=True
)



## === cell 35
cust_pref_sec.reset_index(drop=True, inplace=True)



## === cell 36
cust_no_sec = (
    cust_pref_sec.groupby(["customer_id"])["section_name"].count().reset_index()
)



## === cell 37
cust_no_sec.sort_values(by="section_name", inplace=True)



## === cell 38
Customer_list_1 = (
    cust_no_sec[cust_no_sec.section_name > 2]["customer_id"].unique().tolist()
)



## === cell 39
Customer_list_2 = (
    cust_no_sec[cust_no_sec.section_name <= 2]["customer_id"].unique().tolist()
)



## === cell 40
cust_pref_sec_list1 = cust_pref_sec[
    cust_pref_sec["customer_id"].isin(Customer_list_1)
].reset_index(drop=True)



## === cell 41
cust_pref_sec_list1 = cust_pref_sec_list1.groupby("customer_id").head(3)



## === cell 42
pop_art = (
    MD2_recent.groupby(["section_name", "article_id"])["customer_id"]
    .count()
    .reset_index()
)



## === cell 43
pop_art.sort_values(by=["section_name", "customer_id"], ascending=False, inplace=True)



## === cell 44
pop_art = pop_art.groupby("section_name").head(4).reset_index(drop=True)



## === cell 45
cust_pref_sec_list1



## === cell 46
pop_art.drop(labels="customer_id", axis=1, inplace=True)



## === cell 47
Age_pref_sec = Age_pref_sec.groupby(["age_bucket"]).head(3).reset_index(drop=True)
Age_pref_sec.reset_index(drop=True, inplace=True)



## === cell 48
Age_pref_sec.drop(labels="article_id", axis=1, inplace=True)



## === cell 49
Age_pref_sec



## === cell 50
pop_art



## === cell 51
pop_art_recom = []
for i in tqdm(pop_art.section_name.unique().tolist()):
    artcle = pop_art[pop_art.section_name == i]["article_id"].unique().tolist()
    artcle = [str(i) for i in artcle]
    recommend = " ".join(artcle)
    pop_art_recom.append({"section_name": i, "recommend": recommend})



## === cell 52
recom_sec = pd.DataFrame(pop_art_recom)



## === cell 53
recom_sec.head()



## === cell 54
age_recom = []
for i in Age_pref_sec.age_bucket.unique().tolist():
    section = (
        Age_pref_sec[Age_pref_sec.age_bucket == i]["section_name"].unique().tolist()
    )
    rec2 = (
        recom_sec[recom_sec.section_name.isin(section)]["recommend"].unique().tolist()
    )
    rec3 = " ".join(rec2)
    age_recom.append({"age_bucket": i, "recommend": rec3})



## === cell 55
age_recoom = pd.DataFrame(age_recom)



## === cell 56
sample = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv",
    dtype={"customer_id": "string"},
)
sample["customer_id"] = sample["customer_id"].astype(str).str.strip()

Submission = sample[["customer_id"]].merge(
    customer[["customer_id", "age_bucket"]], on="customer_id", how="left"
)

Submission["age_bucket"] = Submission["age_bucket"].fillna("Young Adults")



## === cell 57
Submission.nunique()



## === cell 58
cust_pref_sec_list1.drop(labels=["article_id"], axis=1, inplace=True)



## === cell 59
recom_type1_merge = pd.merge(
    left=cust_pref_sec_list1, right=recom_sec, on="section_name", how="left"
)



## === cell 60
recom_type1_merge["recommend"] = recom_type1_merge["recommend"].fillna("").astype(str)

recom_type1_merge = (
    recom_type1_merge.groupby("customer_id", as_index=False)["recommend"]
    .agg(lambda x: " ".join([s for s in x.tolist() if s]))
    .rename(columns={"recommend": "prediction"})
)



## === cell 61
recom_type1_merge.drop_duplicates(subset=["customer_id"], keep="first", inplace=True)



## === cell 62
recom_type1_merge = recom_type1_merge[["customer_id", "prediction"]]



## === cell 63
rec_merge2 = Submission[
    ~Submission.customer_id.isin(recom_type1_merge.customer_id.unique().tolist())
]



## === cell 64
rec_merge2.reset_index(drop=True, inplace=True)



## === cell 65
rec_merge2.head()



## === cell 66
age_recoom.head()



## === cell 67
rec_merge2 = pd.merge(left=rec_merge2, right=age_recoom, on="age_bucket", how="left")



## === cell 68
rec_merge2.drop(labels="age_bucket", axis=1, inplace=True)



## === cell 69
rec_merge2.rename(columns={"recommend": "prediction"}, inplace=True)



## === cell 70
rec_merge2.shape[0] + recom_type1_merge.shape[0]



## === cell 71
max_date = MD2["t_dat"].max()
if pd.notna(max_date):
    recent = MD2[MD2["t_dat"] >= (max_date - pd.Timedelta(days=7))]
    if recent.shape[0] > 0:
        global_top12 = recent["article_id"].value_counts().head(12).index.tolist()
    else:
        global_top12 = MD2["article_id"].value_counts().head(12).index.tolist()
else:
    global_top12 = MD2["article_id"].value_counts().head(12).index.tolist()

global_top12 = [str(x).zfill(10) for x in global_top12]
global_fallback = " ".join(global_top12)


def clean_prediction(pred: str, fallback: str) -> str:
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        pred = ""
    tokens = str(pred).split()
    cleaned = []
    seen = set()
    for t in tokens:
        t = "".join(ch for ch in str(t) if ch.isdigit())
        if not t:
            continue
        t = t.zfill(10)
        if t not in seen:
            seen.add(t)
            cleaned.append(t)
        if len(cleaned) >= 12:
            break
    if len(cleaned) == 0:
        tokens = str(fallback).split()
        cleaned = []
        seen = set()
        for t in tokens:
            t = "".join(ch for ch in str(t) if ch.isdigit()).zfill(10)
            if t and (t not in seen):
                seen.add(t)
                cleaned.append(t)
            if len(cleaned) >= 12:
                break
    return " ".join(cleaned[:12])




## === cell 72
final_submission = pd.concat([recom_type1_merge, rec_merge2], ignore_index=True)
final_submission = final_submission[["customer_id", "prediction"]]

final_submission["customer_id"] = (
    final_submission["customer_id"].astype(str).str.strip()
)
final_submission["prediction"] = final_submission["prediction"].fillna("").astype(str)

final_submission = final_submission.groupby("customer_id", as_index=False)[
    "prediction"
].agg(lambda x: " ".join([s for s in x.tolist() if s]))

final_submission["prediction"] = final_submission["prediction"].apply(
    lambda s: clean_prediction(s, global_fallback)
)



## === cell 73
final_submission.reset_index(drop=True, inplace=True)



## === cell 74
final_submission.head(2)["prediction"].iloc[0]



## === cell 75
final_submission.head()



## === cell 76
final_submission = sample[["customer_id"]].merge(
    final_submission, on="customer_id", how="left"
)
final_submission["prediction"] = final_submission["prediction"].apply(
    lambda s: clean_prediction(s, global_fallback)
)

final_submission = final_submission[["customer_id", "prediction"]]
final_submission.to_csv("submission.csv", index=False)

print(final_submission.head())
print("Wrote submission.csv with shape:", final_submission.shape)
print("Columns:", final_submission.columns.tolist())
print("Any null predictions:", final_submission["prediction"].isna().any())
print(
    "Example prediction token count:",
    len(final_submission["prediction"].iloc[0].split()),
)
