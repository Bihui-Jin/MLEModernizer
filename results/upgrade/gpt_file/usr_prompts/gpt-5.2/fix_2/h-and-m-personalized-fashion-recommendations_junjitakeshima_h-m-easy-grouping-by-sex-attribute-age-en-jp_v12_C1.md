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

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the pandas 2.x incompatibility by replacing deprecated `DataFrame.append` usage with a list-accumulate + `pd.concat`, which unblocks creation of `age_group`. Then I fix the customer “attribute” (sex group) derivation by correctly flattening the `unstack()` MultiIndex columns and renaming them safely based on the actual `index_group_no` values present, which resolves the downstream merge errors. Finally, I make the recommendation string construction deterministic and ensure the submission has the exact required columns and `.csv` suffix, with a safe fallback to global top-12 articles for customers missing age/attribute matches so a valid file is always produced.'

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
articles_df.head(5)



## === cell 10
print(articles_df["index_group_name"].unique())
print(articles_df["index_group_no"].unique())



## === cell 11
sex_category = articles_df[["index_group_no", "index_group_name"]].reset_index(
    drop=True
)
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
plt.show()



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
valid_attrs = {"Woman", "Young", "Man", "Have-kids", "Sports-person"}
cust_sex1.loc[~cust_sex1["attribute"].isin(valid_attrs), "attribute"] = "Woman"
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
plt.show()



## === cell 23
print(cust_sex1["attribute"].value_counts().sort_values(ascending=False))



## === cell 24
customers_df = pd.merge(customers_df, cust_sex1, on="customer_id", how="left")
customers_df



## === cell 25
customers_df.isnull().sum()



## === cell 26
customers_df["attribute"] = customers_df["attribute"].fillna("Woman")



## === cell 27
age_mean = (
    customers_df[["age", "attribute"]]
    .groupby("attribute")["age"]
    .mean()
    .round()
    .reset_index()
)
age_mean.columns = ["attribute", "age_mean"]
age_mean



## === cell 28
customers_df = pd.merge(customers_df, age_mean, on="attribute", how="left")
customers_df.loc[customers_df["age"].isnull(), "age"] = customers_df.loc[
    customers_df["age"].isnull(), "age_mean"
]
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
]
transactions_df



## === cell 32
transactions_df["attribute"] = transactions_df["attribute"].fillna("Woman")
transactions_df["age_id"] = (
    transactions_df["age_id"].fillna(age_group["age_id"].max()).astype(int)
)

transactions_df["article_id"] = " " + transactions_df["article_id"].astype(str)

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
            image = Image.open(f"{path}/images/{article_id[:3]}/{article_id}.jpg")
            plt.imshow(image)
        except Exception:
            pass




## === cell 35
show_images("0915526001")



## === cell 36
recommend_sex_age = pd.DataFrame(
    transactions_df.groupby(["age_id", "attribute"])["article_id"].sum().reset_index()
)
recommend_sex_age["len"] = recommend_sex_age["article_id"].apply(lambda x: len(x))
recommend_sex_age



## === cell 37
recommend_sex_age["article_id"] = recommend_sex_age["article_id"].str.strip()
recommend_sex_age["article_id"] = recommend_sex_age["article_id"].str[:131]
recommend_sex_age = recommend_sex_age.drop(["len"], axis=1)



## === cell 38
top12_global = (
    transactions_df["article_id"]
    .value_counts()
    .index.to_series()
    .str.strip()
    .head(12)
    .tolist()
)
top12_global_str = " ".join(top12_global)



## === cell 39
submission = pd.read_csv(path / "sample_submission.csv")
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

submission["prediction"] = submission["article_id"].fillna(top12_global_str)
submission = submission.drop(["age_id", "attribute", "article_id"], axis=1)

submission = submission[["customer_id", "prediction"]]
submission.to_csv("submission.csv", index=False)
submission



## === cell 40
recommend_sex_age.loc[(recommend_sex_age["age_id"] == 7), :]



## === cell 41
show_images(recommend_sex_age.iloc[38, 2].split(), 6)



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3526519941.py in <cell line: 0>()
----> 1 show_images(recommend_sex_age.iloc[38, 2].split(), 6)
      2 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1181             key = tuple(com.apply_if_callable(x, self.obj) for x in key)
   1182             if self._is_scalar_access(key):
-> 1183                 return self.obj._get_value(*key, takeable=self._takeable)
   1184             return self._getitem_tuple(key)
   1185         else:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _get_value(self, index, col, takeable)
   4210         if takeable:
   4211             series = self._ixs(col, axis=1)
-> 4212             return series._values[index]
   4213 
   4214         series = self._get_item_cache(col)

IndexError: index 38 is out of bounds for axis 0 with size 0

## === cell 42
show_images(recommend_sex_age.iloc[36, 2].split(), 6)

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2149485251.py in <cell line: 0>()
----> 1 show_images(recommend_sex_age.iloc[36, 2].split(), 6)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1181             key = tuple(com.apply_if_callable(x, self.obj) for x in key)
   1182             if self._is_scalar_access(key):
-> 1183                 return self.obj._get_value(*key, takeable=self._takeable)
   1184             return self._getitem_tuple(key)
   1185         else:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _get_value(self, index, col, takeable)
   4210         if takeable:
   4211             series = self._ixs(col, axis=1)
-> 4212             return series._values[index]
   4213 
   4214         series = self._get_item_cache(col)

IndexError: index 36 is out of bounds for axis 0 with size 0
