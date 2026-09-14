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

0.00823

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the pandas 2.x incompatibility by replacing deprecated `DataFrame.append` usage with a list-accumulate + `pd.concat`, which unblocks creation of `age_group`. Then I fix the customer “attribute” (sex group) derivation by correctly flattening the `unstack()` MultiIndex columns and renaming them safely based on the actual `index_group_no` values present, which resolves the downstream merge errors. Finally, I make the recommendation string construction deterministic and ensure the submission has the exact required columns and `.csv` suffix, with a safe fallback to global top-12 articles for customers missing age/attribute matches so a valid file is always produced.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash caused by trying to index into an empty `recommend_sex_age` dataframe in the image-plotting cells, so the notebook runs end-to-end and always writes `submission.csv`. To move the score up from 0.0 toward the target, I correct the recommendation construction so it produces **up to 12 space-separated article_ids** per customer (the current code concatenates arbitrarily many IDs and then truncates by characters, which harms MAP@12). I keep the same core logic (age_id + attribute segmentation with recent transactions) but change the aggregation to a deterministic “top-12 by count (and recency tie-break)” list. I also ensure `prediction` is never empty by falling back to a global top-12 list.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission formatting/type issue rather than model quality, so the smallest score-improving change is to guarantee that every predicted `article_id` is **exactly 10 digits** (zero-padded) and that each row contains **at most 12 space-separated IDs**. I keep your exact segmentation logic (age_id + attribute + recent transactions) and only adjust the string construction and final submission sanitization to match Kaggle’s expected `article_id` format. I also enforce uniqueness while preserving order so repeated IDs don’t waste MAP@12 slots. These changes should move the score up from 0.0 toward (and likely above) your target band without altering the core recommender logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with a submission validity issue (wrong `article_id` token format), not the segmentation logic itself. I keep your exact recommender logic (age_id + attribute segments, top-12 by count with recency tie-break) and only make minimal fixes to ensure every predicted token is a real H&M `article_id` formatted as **exactly 10 digits**, and that no invalid IDs (e.g., “0000000000”) slip in. Concretely: I enforce `article_id` as a 10-char string everywhere, filter out null/invalid IDs before ranking, and sanitize the final `prediction` so it always contains up to 12 valid 10-digit IDs (fallback to global top-12 that is also sanitized). This should move the score up from 0.0 toward your target without changing the core approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is almost certainly coming from an ID-format mismatch: your code *validates* IDs against `articles.csv` in `ALL_ARTICLES_10`, but `articles.csv` is read as `str` without zero-padding, so nearly all real 10-digit predictions get filtered out and you fall back to arbitrary `ALL_ARTICLES_10` order (hurting score and can even produce mostly invalid tokens). I make a minimal, core-logic-preserving fix by zero-padding `articles_df["article_id"]` once up-front (and in `articles_category_df`) so the validity filter aligns with Kaggle’s expected 10-digit `article_id`. I also make the global fallback deterministic and strong by using only the (now-correctly-padded) global top-12, avoiding the unordered `set` spillover. These changes keep your segmentation/recent-popularity logic identical, but should move the score up from 0.0 toward (and likely beyond) your 0.00823 target band.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is still most consistent with a *silent submission validity issue*: the current pipeline creates `recommend_sex_age` with a column named `article_id` that contains a **space-separated string**, and later merges it into `submission` under the same name as the true `article_id` field; this makes it easy to accidentally propagate malformed/non-10-digit tokens (or empty strings) without noticing. I keep your exact segmentation and “top-12 by count with recency tie-break” logic, but make one minimal structural fix: store the recommendation string under a dedicated column (e.g., `rec`) so it cannot collide with real `article_id` columns. I also make the global fallback list guaranteed-valid by building it directly from the already-sanitized `transactions_df["article_id"]` (which is in `ALL_ARTICLES_10`) and remove the redundant `if a in ALL_ARTICLES_10` filter that can empty it unexpectedly. These changes should move your score up from 0.0 toward (and likely into/above) your 0.00823 target band without changing the recommender’s core behavior.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is still most consistent with a submission token validity mismatch: `transactions_df["article_id"]` is read as `str` but never zero-padded to 10 digits before being checked against `ALL_ARTICLES_10`, so many IDs can be turned into NaN and you end up with mostly fallback predictions that don’t match labels well. I make the smallest core-logic-preserving fix by zfilling `transactions_df["article_id"]` to 10 digits immediately after reading, so segment counts and global top-12 are built from correctly formatted IDs. I also build `ALL_ARTICLES_10` from `articles_df` after zfill (already done) and keep the rest of your segmentation + top-12-by-count/recency tie-break logic identical. This should move the score up from 0.0 toward your 0.00823 target without changing the recommender approach.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with a “valid file but wrong key/type alignment” issue: `sample_submission.customer_id` is a zero-padded hex-like string, while your `customers_df.customer_id` coming from `customers.csv` is typically read as an integer-like value unless forced, so the merge produces all-NaN `age_id/attribute` and you fall back to the same global list for everyone (often scoring ~0). I make the minimal fix by reading `customer_id` as `str` everywhere and normalizing it (`strip().lower()`) before any merges, without changing your segmentation logic or ranking. I also add a tiny safety normalization for `transactions_df.customer_id` to match the same format, so the customer attribute/age joins actually work. This should move the score upward toward your 0.00823 target while preserving your existing recommender approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is still most consistent with a key-format mismatch: Kaggle’s `customer_id` is **case-sensitive** (it’s a hex-like string), and lowercasing it breaks the join between your predictions and the evaluation keys. I make the smallest fix by **removing `.str.lower()`** everywhere for `customer_id` so the submission keys exactly match `sample_submission.csv`. I also keep your existing `article_id` zero-padding/sanitization logic unchanged, and ensure the output CSV is still correctly formatted with `customer_id,prediction`. This should move the score upward toward your 0.00823 target without changing the recommender’s core segmentation/popularity logic.'

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
    path / "transactions_train.csv",
    dtype={"article_id": str, "customer_id": str},
)
articles_df = pd.read_csv(path / "articles.csv", dtype={"article_id": str})
customers_df = pd.read_csv(path / "customers.csv", dtype={"customer_id": str})
submission = pd.read_csv(path / "sample_submission.csv", dtype={"customer_id": str})

for _df, _col in [
    (transactions_df, "customer_id"),
    (customers_df, "customer_id"),
    (submission, "customer_id"),
]:
    _df[_col] = _df[_col].astype(str).str.strip()

transactions_df["article_id"] = transactions_df["article_id"].astype(str).str.zfill(10)



## === cell 2
transactions_df["t_dat"] = pd.to_datetime(transactions_df["t_dat"])



## === cell 3
customers_df



## === cell 4
customers_df.isnull().sum()



## === cell 5
customers_df["age"].plot.hist(bins=50)



## === cell 6
age_id = 0
age = 16
parts = []

for _ in range(53):
    if age < 30:
        parts.append(pd.DataFrame({"age": [age, age + 1], "age_id": [age_id, age_id]}))
        age += 2
        age_id += 1
    elif age < 60:
        parts.append(
            pd.DataFrame(
                {
                    "age": [age, age + 1, age + 2, age + 3, age + 4],
                    "age_id": [age_id, age_id, age_id, age_id, age_id],
                }
            )
        )
        age += 5
        age_id += 1
    else:
        parts.append(pd.DataFrame({"age": [age], "age_id": [age_id]}))
        age += 1

age_group = pd.concat(parts, ignore_index=True)



## === cell 7
age_group



## === cell 8
customers_df = pd.merge(customers_df, age_group, on="age", how="left")
customers_df = customers_df.drop(
    ["FN", "Active", "club_member_status", "fashion_news_frequency", "postal_code"],
    axis=1,
)
customers_df



## === cell 9
articles_df["article_id"] = articles_df["article_id"].astype(str).str.zfill(10)



## === cell 10
articles_df.head(5)



## === cell 11
print(articles_df["index_group_name"].unique())
print(articles_df["index_group_no"].unique())



## === cell 12
sex_category = articles_df[["index_group_no", "index_group_name"]].reset_index(
    drop=True
)
display(sex_category["index_group_name"].value_counts())



## === cell 13
sex_category_list = sex_category["index_group_name"].value_counts().index.to_list()
plt.figure(figsize=(5, 5))
plt.rcParams["font.size"] = 12
plt.pie(
    sex_category["index_group_name"].value_counts().sort_values(ascending=False),
    labels=sex_category_list,
    startangle=90,
    counterclock=False,
    autopct="%1.1f%%",
)
plt.show()



## === cell 14
del sex_category_list



## === cell 15
articles_category_df = pd.DataFrame(articles_df[["article_id", "index_group_no"]])
articles_category_df.columns = ["article_id", "sex_attribute"]
articles_category_df



## === cell 16
transactions_df = pd.merge(
    transactions_df, articles_category_df, on="article_id", how="left"
)
transactions_df



## === cell 17
cust_sex = (
    transactions_df[["customer_id", "sex_attribute", "article_id"]]
    .groupby(["customer_id", "sex_attribute"])["article_id"]
    .count()
    .unstack(fill_value=0)
)

default_name_map = {
    1: "Woman",
    2: "Young",
    3: "Man",
    4: "Have-kids",
    5: "Sports-person",
}
cust_sex = cust_sex.rename(
    columns=lambda c: default_name_map.get(int(c), str(c)) if pd.notna(c) else "Unknown"
)

cust_sex



## === cell 18
cust_sex["attribute"] = cust_sex.apply(lambda x: list(x[x == x.max()].index), axis=1)
cust_sex



## === cell 19
cust_sex1 = pd.DataFrame(cust_sex[["attribute"]]).reset_index()
cust_sex1["attribute"] = cust_sex1["attribute"].apply(",".join).astype(str)
del cust_sex
cust_sex1



## === cell 20
print(cust_sex1.attribute.unique())



## === cell 21
valid_attrs = {"Woman", "Young", "Man", "Have-kids", "Sports-person"}
cust_sex1.loc[~cust_sex1["attribute"].isin(valid_attrs), "attribute"] = "Woman"
cust_sex1



## === cell 22
print(cust_sex1.attribute.unique())



## === cell 23
temp = cust_sex1["attribute"].value_counts().index.to_list()
plt.figure(figsize=(5, 5))
plt.rcParams["font.size"] = 12
plt.pie(
    cust_sex1["attribute"].value_counts().sort_values(ascending=False),
    labels=temp,
    startangle=90,
    counterclock=False,
    autopct="%1.1f%%",
)
plt.show()



## === cell 24
print(cust_sex1["attribute"].value_counts().sort_values(ascending=False))



## === cell 25
customers_df = pd.merge(customers_df, cust_sex1, on="customer_id", how="left")
customers_df



## === cell 26
customers_df.isnull().sum()



## === cell 27
customers_df["attribute"] = customers_df["attribute"].fillna("Woman")



## === cell 28
age_mean = (
    customers_df[["age", "attribute"]]
    .groupby("attribute")["age"]
    .mean()
    .round()
    .reset_index()
)
age_mean.columns = ["attribute", "age_mean"]
age_mean



## === cell 29
customers_df = pd.merge(customers_df, age_mean, on="attribute", how="left")
customers_df.loc[customers_df["age"].isnull(), "age"] = customers_df.loc[
    customers_df["age"].isnull(), "age_mean"
]
customers_df = customers_df.drop(["age_mean", "age_id"], axis=1)
customers_df = pd.merge(customers_df, age_group, on="age", how="left")
customers_df.isnull().sum()



## === cell 30
transactions_df = pd.merge(transactions_df, customers_df, on="customer_id", how="left")
transactions_df.isnull().sum()



## === cell 31
del cust_sex1



## === cell 32
transactions_df = transactions_df.loc[
    transactions_df.t_dat >= pd.to_datetime("2020-09-15")
]
transactions_df



## === cell 33
ALL_ARTICLES_10 = set(articles_df["article_id"].astype(str).str.zfill(10))


def _aid10_or_nan(x) -> str:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    s10 = s.zfill(10)[-10:]
    return s10 if s10 in ALL_ARTICLES_10 else np.nan


def _top12_unique_valid(series: pd.Series) -> str:
    seen = set()
    out = []
    for v in series:
        a = _aid10_or_nan(v)
        if a is None or (isinstance(a, float) and np.isnan(a)) or pd.isna(a):
            continue
        if a not in seen:
            seen.add(a)
            out.append(a)
        if len(out) == 12:
            break
    return " ".join(out)


transactions_df["article_id"] = transactions_df["article_id"].map(_aid10_or_nan)
transactions_df = transactions_df.loc[transactions_df["article_id"].notna()].copy()

transactions_df["attribute"] = transactions_df["attribute"].fillna("Woman")
transactions_df["age_id"] = (
    transactions_df["age_id"].fillna(age_group["age_id"].max()).astype(int)
)

seg_counts = (
    transactions_df.groupby(["age_id", "attribute", "article_id"])
    .agg(count=("customer_id", "size"), last_date=("t_dat", "max"))
    .reset_index()
)

seg_counts = seg_counts.sort_values(
    ["age_id", "attribute", "count", "last_date", "article_id"],
    ascending=[True, True, False, False, True],
)

recommend_sex_age = (
    seg_counts.groupby(["age_id", "attribute"])["article_id"]
    .apply(_top12_unique_valid)
    .reset_index(name="rec")
)

recommend_sex_age




## === cell 34
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
            image = Image.open(f"{path}/images/{article_id[:3]}/{article_id}.jpg")
            plt.imshow(image)
        except Exception:
            pass




## === cell 35
show_images("0915526001")



## === cell 36
top12_global = transactions_df["article_id"].value_counts().head(12).index.tolist()
top12_global = [str(a).zfill(10) for a in top12_global]  # already valid, keep format
if len(top12_global) == 0:
    top12_global = list(sorted(ALL_ARTICLES_10))[:12]
top12_global = (top12_global + top12_global)[:12]  # deterministic length-12 safeguard
top12_global_str = " ".join(top12_global)



## === cell 37
submission = pd.read_csv(path / "sample_submission.csv", dtype={"customer_id": str})
submission["customer_id"] = submission["customer_id"].astype(str).str.strip()

submission = submission[["customer_id"]]
submission = pd.merge(
    submission,
    customers_df[["customer_id", "age_id", "attribute"]],
    on="customer_id",
    how="left",
)
submission["attribute"] = submission["attribute"].fillna("Woman")
submission["age_id"] = (
    submission["age_id"].fillna(age_group["age_id"].max()).astype(int)
)

submission = pd.merge(
    submission, recommend_sex_age, on=["age_id", "attribute"], how="left"
)

submission["prediction"] = submission["rec"].fillna(top12_global_str).astype(str)


def _sanitize_pred(s: str) -> str:
    toks = [] if pd.isna(s) else str(s).split()
    out = []
    seen = set()
    for t in toks:
        a = _aid10_or_nan(t)
        if a is None or (isinstance(a, float) and np.isnan(a)) or pd.isna(a):
            continue
        if a not in seen:
            seen.add(a)
            out.append(a)
        if len(out) == 12:
            break
    if len(out) == 0:
        return top12_global_str
    if len(out) < 12:
        for a in top12_global:
            if a not in seen:
                out.append(a)
                seen.add(a)
            if len(out) == 12:
                break
    return " ".join(out[:12])


submission["prediction"] = submission["prediction"].map(_sanitize_pred)
submission.loc[submission["prediction"].str.len().eq(0), "prediction"] = (
    top12_global_str
)

submission = submission.drop(["age_id", "attribute", "rec"], axis=1)
submission = submission[["customer_id", "prediction"]]
submission.to_csv("submission.csv", index=False)
submission



## === cell 38
recommend_sex_age.loc[(recommend_sex_age["age_id"] == 7), :]



## === cell 39
if len(recommend_sex_age) > 0:
    example_row = min(38, len(recommend_sex_age) - 1)
    show_images(recommend_sex_age.iloc[example_row, 2].split(), 6)



## === cell 40
if len(recommend_sex_age) > 0:
    example_row = min(36, len(recommend_sex_age) - 1)
    show_images(recommend_sex_age.iloc[example_row, 2].split(), 6)
