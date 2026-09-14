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

- What this solution (achieved 0.0) has done: 'The crash happens because `unstack()` produced a DataFrame with **0 columns** (no non-null `sex_attribute` groups to unstack), so assigning 5 column names triggers a length mismatch. The minimal fix is to make the unstack deterministic by explicitly unstacking the `sex_attribute` level, then **reindexing** to the expected 5 category columns (0..4) and filling missing ones with zeros. Finally, we set the human-readable column names as before, so downstream logic in cell 17 still works unchanged. This preserves the original aggregation semantics (counts per customer per sex_attribute) while avoiding the empty-column edge case.'
- What this solution (achieved 0.0) has done: 'Diagnosis: Cell 40 crashes because it assumes `recommend_sex_age` has at least 39 rows, but after filtering/deduplication earlier it can be empty (or smaller), so `recommend_sex_age.iloc[38, 2]` raises an out-of-bounds `IndexError`. The fix is to guard the `.iloc` access and only call `show_images` when the requested row exists and the `article_id` string is non-null. This keeps the visualization intent but makes the notebook deterministic and non-crashing.

Patch summary: Add a minimal bounds/null check in cell 40 before indexing `recommend_sex_age` and calling `show_images`. No changes to data processing, model logic, or outputs; only prevents an invalid `.iloc` access.

Updated cells: Only cell 40 is modified.

Compatibility notes for cell k+1: Cell 41 remains unchanged; this patch does not alter `recommend_sex_age` or `show_images`, and introduces no new dependencies. If `recommend_sex_age` is empty, cell 40 safely do nothing instead of crashing; cell 41 may still crash if it also indexes out of bounds (unchanged per instructions).

Assumptions: It is acceptable for the visualization cell to no-op when the requested row does not exist, since it is purely exploratory and not part of submission generation.'
- What this solution (achieved 0.0) has done: 'Diagnosis: Cell 41 indexes `recommend_sex_age.iloc[36, 2]`, but `recommend_sex_age` can be empty (or have fewer than 37 rows) after the date filter and deduplication, so positional index 36 is out of bounds and raises `IndexError`.  
Patch summary: Add a simple bounds and NaN check in cell 41 (similar to the existing guard in cell 40) before attempting to split and show images, preventing the crash while preserving the same visualization behavior when data exists.  
Updated cells: Only cell 41 is modified.  
Compatibility notes for cell k+1: No variables or interfaces are changed; this only guards execution, so any later cells that rely on `recommend_sex_age` remain unaffected.  
Assumptions: It is acceptable to skip plotting when the requested row/column is unavailable or missing (consistent with cell 40’s guarded plotting).'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most likely coming from invalid/empty predictions for many customers (NaNs in `prediction` after the merges), which Kaggle effectively scores as no correct items. I keep your exact recommendation logic, but make the submission deterministic and valid by (1) ensuring every customer gets a non-empty 12-item fallback (global top articles from the same recent window) when their segment prediction is missing, and (2) enforcing exactly 12 space-separated article_ids in the required string format. This is a minimal post-processing change that preserves your core aggregation approach while moving MAP@12 upward toward the target. The output path/name stays `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is consistent with the submission having *mostly blank/invalid predictions* after the merge, because `customers_df` is merged into `submission` only for customers that exist in `customers.csv`; any mismatch or missing `age_id/attribute` leaves NaNs and then many rows can end up with empty/space-prefixed ids. I keep your same “sex+age segment popular items” logic, but make the join and post-processing robust by (1) filling missing `age_id/attribute` *before* the `recommend_sex_age` merge so every customer lands in some segment, and (2) building the global fallback from properly stripped `article_id` values (no leading spaces) and enforcing exactly 12 valid 10-digit ids. This should move MAP@12 upward toward your target without changing your core aggregation approach. The output stays `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is consistent with the submission `prediction` strings not matching the competition’s expected `article_id` formatting (10-digit numeric strings) because the training pipeline intentionally prepends a leading space to `article_id`, and then later `("0"+p)[-10:]` truncates the *wrong* last 10 characters (often leaving non-numeric like `" 091552600"`). I keep your exact “sex+age segment popular items” aggregation logic, but make the submission-building step robust by stripping spaces and extracting digits before enforcing 10-digit formatting. I also build the global fallback top-12 from clean 10-digit ids (not the space-prefixed keys) to ensure every customer always has 12 valid items. These minimal post-processing fixes should move the score upward toward the 0.00823 target without changing your core modeling approach.'

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
transactions_df



## === cell 3
customers_df



## === cell 4
customers_df.isnull().sum()



## === cell 5
customers_df["age"].plot.hist(bins=50)



## === cell 6
age_id = 0
age_group = pd.DataFrame(columns=["age", "age_id"])
age = 16

for i in range(53):
    if age < 30:
        temp_group = pd.DataFrame({"age": [age, age + 1], "age_id": [age_id, age_id]})
        age_group = pd.concat([age_group, temp_group], ignore_index=True)
        age += 2
        age_id += 1
    elif age < 60:
        temp_group = pd.DataFrame(
            {
                "age": [age, age + 1, age + 2, age + 3, age + 4],
                "age_id": [age_id, age_id, age_id, age_id, age_id],
            }
        )
        age_group = pd.concat([age_group, temp_group], ignore_index=True)
        age += 5
        age_id += 1
    else:
        temp_group = pd.DataFrame({"age": [age], "age_id": [age_id]})
        age_group = pd.concat([age_group, temp_group], ignore_index=True)
        age += 1



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
articles_df.head(5)



## === cell 10
print(articles_df["index_group_name"].unique())
print(articles_df["index_group_no"].unique())



## === cell 11
sex_category = articles_df[["index_group_no", "index_group_name"]].reset_index()
display(sex_category["index_group_name"].value_counts())



## === cell 12
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
plt.show



## === cell 13
del sex_category_list



## === cell 14
articles_category_df = pd.DataFrame(articles_df[["article_id", "index_group_no"]])
articles_category_df.columns = ["article_id", "sex_attribute"]
articles_category_df



## === cell 15
transactions_df = pd.merge(
    transactions_df, articles_category_df, on="article_id", how="left"
)
transactions_df



## === cell 16
cust_sex = (
    transactions_df[["customer_id", "sex_attribute", "article_id"]]
    .groupby(["customer_id", "sex_attribute"])
    .count()
    .unstack("sex_attribute")
)

cust_sex.columns = cust_sex.columns.get_level_values(-1)

cust_sex = cust_sex.reindex(columns=[0, 1, 2, 3, 4], fill_value=0)

cust_sex.columns = ["Woman", "Young", "Man", "Have-kids", "Sports-person"]
cust_sex



## === cell 17
cust_sex["attribute"] = cust_sex.apply(lambda x: list(x[x == x.max()].index), axis=1)
cust_sex



## === cell 18
cust_sex1 = pd.DataFrame(cust_sex[["attribute"]]).reset_index()
cust_sex1["attribute"] = cust_sex1["attribute"].apply(",".join).astype(str)
del cust_sex
cust_sex1



## === cell 19
print(cust_sex1.attribute.unique())



## === cell 20
cust_sex1.loc[
    ~(
        (cust_sex1["attribute"] == "Woman")
        | (cust_sex1["attribute"] == "Young")
        | (cust_sex1["attribute"] == "Man")
        | (cust_sex1["attribute"] == "Have-kids")
        | (cust_sex1["attribute"] == "Sports-person")
    ),
    "attribute",
] = "Woman"
cust_sex1



## === cell 21
print(cust_sex1.attribute.unique())



## === cell 22
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
plt.show



## === cell 23
print(cust_sex1["attribute"].value_counts().sort_values(ascending=False))



## === cell 24
customers_df = pd.merge(customers_df, cust_sex1, on="customer_id", how="left")
customers_df



## === cell 25
customers_df.isnull().sum()



## === cell 26
customers_df["attribute"].fillna("Woman", inplace=True)



## === cell 27
age_mean = (
    customers_df[["age", "attribute"]].groupby("attribute").mean().round().reset_index()
)
age_mean.columns = ["attribute", "age_mean"]
age_mean



## === cell 28
customers_df = pd.merge(customers_df, age_mean, on="attribute", how="left")
customers_df.loc[(customers_df["age"].isnull()), "age"] = customers_df["age_mean"]
customers_df = customers_df.drop(["age_mean", "age_id"], axis=1)
customers_df = pd.merge(customers_df, age_group, on="age", how="left")
customers_df.isnull().sum()



## === cell 29
transactions_df = pd.merge(transactions_df, customers_df, on="customer_id", how="left")
transactions_df.isnull().sum()



## === cell 30
del cust_sex1



## === cell 31
transactions_df = transactions_df.loc[
    transactions_df.t_dat >= pd.to_datetime("2020-09-15")
]  # changed from 2020-09-01
transactions_df



## === cell 32
transactions_df.article_id = " " + transactions_df.article_id.astype("str")
temp = (
    transactions_df.groupby(["age_id", "attribute", "article_id"])["customer_id"]
    .agg("count")
    .reset_index()
)
temp.columns = ["age_id", "attribute", "article_id", "count"]
transactions_df = transactions_df.merge(
    temp, on=["age_id", "attribute", "article_id"], how="left"
)
transactions_df



## === cell 33
transactions_df = transactions_df.sort_values(["count", "t_dat"], ascending=False)
transactions_df = transactions_df.drop_duplicates(["age_id", "attribute", "article_id"])
transactions_df




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
            image = Image.open(
                f"/kaggle/input/h-and-m-personalized-fashion-recommendations/images/{article_id[:3]}/{article_id}.jpg"
            )
            plt.imshow(image)
        except:
            pass




## === cell 35
show_images("0915526001")



## === cell 36
recommend_sex_age = pd.DataFrame(
    transactions_df.groupby(["age_id", "attribute"]).article_id.sum().reset_index()
)
recommend_sex_age["len"] = recommend_sex_age["article_id"].apply(lambda x: len(x))
recommend_sex_age



## === cell 37
recommend_sex_age["article_id"] = recommend_sex_age["article_id"].str.strip()
recommend_sex_age["article_id"] = recommend_sex_age["article_id"].str[:131]
recommend_sex_age = recommend_sex_age.drop(["len"], axis=1)



## === cell 38
import re


def _to_article10(x: str) -> str:
    s = "" if pd.isna(x) else str(x)
    digits = re.sub(r"\D", "", s)
    if digits == "":
        return ""
    return digits[-10:].zfill(10)


def _clean_pred_to_12(pred: str, fallback12: str) -> str:
    if pd.isna(pred) or str(pred).strip() == "":
        pred = fallback12
    parts = [p for p in str(pred).split() if p]
    parts = [_to_article10(p) for p in parts]
    parts = [p for p in parts if p]  # drop empties if any

    if len(parts) < 12:
        fb_parts = [p for p in str(fallback12).split() if p]
        fb_parts = [_to_article10(p) for p in fb_parts]
        fb_parts = [p for p in fb_parts if p]
        parts = (parts + fb_parts)[:12]
    else:
        parts = parts[:12]
    return " ".join(parts)


submission = pd.read_csv(
    "../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv"
)
submission = submission[["customer_id"]]
submission = pd.merge(submission, customers_df, on="customer_id", how="left")

submission["attribute"] = submission["attribute"].fillna("Woman")

_default_age_id = (
    int(customers_df["age_id"].mode().iloc[0])
    if customers_df["age_id"].notna().any()
    else 0
)
submission["age_id"] = submission["age_id"].fillna(_default_age_id).astype(int)

submission = pd.merge(
    submission, recommend_sex_age, on=["age_id", "attribute"], how="left"
)

_global_top_raw = (
    transactions_df.groupby("article_id")["count"]
    .sum()
    .sort_values(ascending=False)
    .index.astype(str)
    .tolist()
)
_global_top_clean = []
_seen = set()
for a in _global_top_raw:
    aa = _to_article10(a)
    if aa and aa not in _seen:
        _global_top_clean.append(aa)
        _seen.add(aa)
_global_top_12 = " ".join(_global_top_clean[:12])

if len(_global_top_clean) < 12:
    _overall_top = (
        pd.read_csv(
            path / "transactions_train.csv",
            usecols=["article_id"],
            dtype={"article_id": str},
        )
        .groupby("article_id")
        .size()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    _overall_top_clean = []
    _seen2 = set()
    for a in _overall_top:
        aa = _to_article10(a)
        if aa and aa not in _seen2:
            _overall_top_clean.append(aa)
            _seen2.add(aa)
    _global_top_12 = " ".join(_overall_top_clean[:12])

submission["prediction"] = submission["article_id"].apply(
    lambda x: _clean_pred_to_12(x, _global_top_12)
)

submission = submission.drop(["age", "age_id", "attribute", "article_id"], axis=1)
submission.columns = ("customer_id", "prediction")
submission.to_csv("submission.csv", index=False)
submission



## === cell 39
recommend_sex_age.loc[(recommend_sex_age["age_id"] == 7),]



## === cell 40
if (
    len(recommend_sex_age) > 38
    and recommend_sex_age.shape[1] > 2
    and pd.notna(recommend_sex_age.iloc[38, 2])
):
    show_images(recommend_sex_age.iloc[38, 2].split(), 6)



## === cell 41
if (
    len(recommend_sex_age) > 36
    and recommend_sex_age.shape[1] > 2
    and pd.notna(recommend_sex_age.iloc[36, 2])
):
    show_images(recommend_sex_age.iloc[36, 2].split(), 6)
