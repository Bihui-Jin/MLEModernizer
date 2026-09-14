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
pillow==11.3.0
seaborn==0.12.2
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

0.00775

# 6. Current score

0.00909

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00909) has done: 'I fix the runtime errors caused by deprecated `DataFrame.append`, multi-index column handling after `unstack()`, and incorrect assumptions about the available `sex_attribute` categories. I make the customer “attribute” derivation robust by selecting the most-purchased `sex_attribute` per customer (with a safe default), ensuring the merge back into `customers_df` works cleanly. Then I ensure the recommendation table is built consistently (top items per `age_id` + `attribute`) and that every row in `sample_submission.csv` gets a valid 12-item prediction string (falling back to global top-12 if needed). Finally, the script write a valid `submission.csv` with the required columns and formatting.'
- What this solution (achieved 0.00654) has done: 'Your current score (0.00909) is higher than the target (0.00775), so we should *slightly reduce* performance in a controlled way to move closer to the target band without changing the core approach. The smallest safe lever here is the recency window: using a longer window (earlier start date) typically makes recommendations more popularity-biased and less aligned to the immediate next-week purchases, which should lower MAP@12 a bit. I only change the cutoff date used to filter `transactions_df` (keeping the same aggregation, ranking, and submission construction), and keep everything else identical to preserve semantics and stability. The submission format/writing stays the same.'
- What this solution (achieved 0.00909) has done: 'To move your current MAP@12 (0.00654) upward toward the target (0.00775), I make the smallest change that usually improves this baseline without altering the core recommendation logic: tighten the recency window so the “top-12 per (age_id, attribute)” reflects purchases closer to the evaluation week. This keeps the same aggregation/ranking approach, just on a more relevant slice of transactions. I also compute the global fallback top-12 from the same filtered window for consistency, and keep submission formatting identical.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from datetime import datetime, date, timedelta

from collections import Counter, defaultdict
from PIL import Image
from pathlib import Path

path = Path("/kaggle/input/h-and-m-personalized-fashion-recommendations/")



## === cell 1
transactions_df = pd.read_csv(
    path / "transactions_train.csv", dtype={"article_id": str}
)
articles_df = pd.read_csv(path / "articles.csv", dtype={"article_id": str})
customers_df = pd.read_csv(path / "customers.csv")
submission = pd.read_csv(path / "sample_submission.csv")



## === cell 2
transactions_df["t_dat"] = pd.to_datetime(transactions_df["t_dat"])



## === cell 3
_ = customers_df.head(3)



## === cell 4
_ = customers_df.isnull().sum()



## === cell 5
age_id = 0
rows = []
age = 16

for _ in range(53):
    if age < 30:
        for a in [age, age + 1]:
            rows.append({"age": a, "age_id": age_id})
        age += 2
        age_id += 1
    elif age < 60:
        for a in [age, age + 1, age + 2, age + 3, age + 4]:
            rows.append({"age": a, "age_id": age_id})
        age += 5
        age_id += 1
    else:
        rows.append({"age": age, "age_id": age_id})
        age += 1

age_group = pd.DataFrame(rows).drop_duplicates(subset=["age"]).reset_index(drop=True)



## === cell 6
_ = age_group.head(10)



## === cell 7
customers_df = pd.merge(customers_df, age_group, on="age", how="left")
customers_df = customers_df.drop(
    ["FN", "Active", "club_member_status", "fashion_news_frequency", "postal_code"],
    axis=1,
)



## === cell 8
_ = articles_df.head(3)



## === cell 9
_ = (articles_df["index_group_name"].unique(), articles_df["index_group_no"].unique())



## === cell 10
sex_category = articles_df[["index_group_no", "index_group_name"]].copy()



## === cell 11
plt.figure(figsize=(5, 5))
plt.rcParams["font.size"] = 12
vc = sex_category["index_group_name"].value_counts().sort_values(ascending=False)
plt.pie(
    vc.values,
    labels=vc.index.tolist(),
    startangle=90,
    counterclock=False,
    autopct="%1.1f%%",
)
plt.show()



## === cell 12
del sex_category



## === cell 13
articles_category_df = articles_df[["article_id", "index_group_no"]].copy()
articles_category_df.columns = ["article_id", "sex_attribute"]



## === cell 14
transactions_df = pd.merge(
    transactions_df, articles_category_df, on="article_id", how="left"
)



## === cell 15
cust_attr = (
    transactions_df.dropna(subset=["sex_attribute"])
    .groupby(["customer_id", "sex_attribute"])["article_id"]
    .size()
    .reset_index(name="cnt")
)
idx = cust_attr.groupby("customer_id")["cnt"].idxmax()
cust_attr = cust_attr.loc[idx, ["customer_id", "sex_attribute"]].reset_index(drop=True)

sex_map = {
    1: "Ladieswear",
    2: "Ladieswear",
    3: "Menswear",
    4: "Kids & Baby",
    5: "Sport",
    6: "Divided",
    7: "Divided",
    8: "Ladieswear",
    9: "Menswear",
    10: "Kids & Baby",
}
cust_attr["attribute"] = cust_attr["sex_attribute"].map(sex_map).fillna("Ladieswear")

normalize_map = {
    "Ladieswear": "Woman",
    "Divided": "Young",
    "Menswear": "Man",
    "Kids & Baby": "Have-kids",
    "Sport": "Sports-person",
}
cust_attr["attribute"] = cust_attr["attribute"].map(normalize_map).fillna("Woman")
cust_sex1 = cust_attr[["customer_id", "attribute"]].copy()



## === cell 16
_ = cust_sex1["attribute"].value_counts(dropna=False)



## === cell 17
cust_sex1["attribute"] = cust_sex1["attribute"].astype(str)



## === cell 18
_ = cust_sex1.attribute.unique()



## === cell 19
allowed = {"Woman", "Young", "Man", "Have-kids", "Sports-person"}
cust_sex1.loc[~cust_sex1["attribute"].isin(list(allowed)), "attribute"] = "Woman"



## === cell 20
_ = cust_sex1.attribute.unique()



## === cell 21
plt.figure(figsize=(5, 5))
plt.rcParams["font.size"] = 12
vc2 = cust_sex1["attribute"].value_counts().sort_values(ascending=False)
plt.pie(
    vc2.values,
    labels=vc2.index.tolist(),
    startangle=90,
    counterclock=False,
    autopct="%1.1f%%",
)
plt.show()



## === cell 22
_ = cust_sex1["attribute"].value_counts().sort_values(ascending=False)



## === cell 23
customers_df = pd.merge(customers_df, cust_sex1, on="customer_id", how="left")



## === cell 24
_ = customers_df.isnull().sum()



## === cell 25
customers_df["attribute"] = customers_df["attribute"].fillna("Woman")



## === cell 26
age_mean = (
    customers_df[["age", "attribute"]]
    .groupby("attribute")["age"]
    .mean()
    .round()
    .reset_index()
)
age_mean.columns = ["attribute", "age_mean"]



## === cell 27
customers_df = pd.merge(customers_df, age_mean, on="attribute", how="left")
customers_df.loc[customers_df["age"].isnull(), "age"] = customers_df.loc[
    customers_df["age"].isnull(), "age_mean"
]
customers_df = customers_df.drop(["age_mean"], axis=1)
customers_df = customers_df.drop(["age_id"], axis=1, errors="ignore")
customers_df["age"] = customers_df["age"].round().astype("Int64")
customers_df = pd.merge(customers_df, age_group, on="age", how="left")



## === cell 28
transactions_df = pd.merge(transactions_df, customers_df, on="customer_id", how="left")



## === cell 29
del cust_sex1



## === cell 30
transactions_df = transactions_df.loc[
    transactions_df["t_dat"] >= pd.to_datetime("2020-09-01")
].copy()



## === cell 31
transactions_df["article_id"] = transactions_df["article_id"].astype(str)
grp = (
    transactions_df.dropna(subset=["age_id", "attribute", "article_id"])
    .groupby(["age_id", "attribute", "article_id"])["customer_id"]
    .size()
    .reset_index(name="count")
)
last_dat = (
    transactions_df.dropna(subset=["age_id", "attribute", "article_id"])
    .groupby(["age_id", "attribute", "article_id"])["t_dat"]
    .max()
    .reset_index(name="last_t_dat")
)
grp = grp.merge(last_dat, on=["age_id", "attribute", "article_id"], how="left")
grp = grp.sort_values(
    ["age_id", "attribute", "count", "last_t_dat"], ascending=[True, True, False, False]
)

topk = grp.groupby(["age_id", "attribute"]).head(12).copy()
recommend_sex_age = (
    topk.groupby(["age_id", "attribute"])["article_id"]
    .apply(lambda s: " ".join(s.tolist()))
    .reset_index()
)
recommend_sex_age.columns = ["age_id", "attribute", "prediction"]



## === cell 32
_ = recommend_sex_age.head(5)




## === cell 33
def show_images(article_ids, cols=1, rows=-1):
    if isinstance(article_ids, int) or isinstance(article_ids, str):
        article_ids = [article_ids]
    article_count = len(article_ids)
    if rows < 0:
        rows = (article_count // cols) + 1
    plt.figure(figsize=(3 + 3.5 * cols, 3 + 5 * rows))
    for i in range(article_count):
        article_id = ("0" + str(article_ids[i]))[-10:]
        plt.subplot(rows, cols, i + 1)
        plt.axis("off")
        plt.title(article_id)
        try:
            image = Image.open(path / f"images/{article_id[:3]}/{article_id}.jpg")
            plt.imshow(image)
        except Exception:
            pass
    plt.show()




## === cell 34
show_images("0915526001")



## === cell 35
global_top12 = transactions_df["article_id"].value_counts().head(12).index.tolist()
global_fallback_pred = " ".join(global_top12)




## === cell 36
def clean_pred(x: str) -> str:
    if not isinstance(x, str) or not x.strip():
        return global_fallback_pred
    items = x.strip().split()
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            out.append(it)
            seen.add(it)
        if len(out) == 12:
            break
    if len(out) < 12:
        for it in global_top12:
            if it not in seen:
                out.append(it)
                seen.add(it)
            if len(out) == 12:
                break
    return " ".join(out)




## === cell 37
submission_out = submission[["customer_id"]].copy()
submission_out = pd.merge(
    submission_out,
    customers_df[["customer_id", "age_id", "attribute"]],
    on="customer_id",
    how="left",
)
submission_out = pd.merge(
    submission_out, recommend_sex_age, on=["age_id", "attribute"], how="left"
)

submission_out["prediction"] = submission_out["prediction"].apply(clean_pred)

submission_out = submission_out[["customer_id", "prediction"]]
submission_out.to_csv("submission.csv", index=False)
submission_out.head()
