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

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fixed the pandas `.append` deprecation, guarded sections that previously crashed, and replaced the complex attribute‑based recommendation logic with a simple baseline that predicts the 12 most frequent articles for every customer. This ensures the notebook runs end‑to‑end and creates a valid `submission.csv` file, while still providing a reasonable MAP@12 score that moves toward the target.'

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
age_id = 0
age_group = pd.DataFrame(columns=["age", "age_id"])
age = 16

for _ in range(53):
    if age < 30:
        temp_group = pd.DataFrame({"age": [age, age + 1], "age_id": [age_id, age_id]})
        age_group = pd.concat([age_group, temp_group], ignore_index=True)
        age += 2
        age_id += 1
    elif age < 60:
        temp_group = pd.DataFrame(
            {"age": [age, age + 1, age + 2, age + 3, age + 4], "age_id": [age_id] * 5}
        )
        age_group = pd.concat([age_group, temp_group], ignore_index=True)
        age += 5
        age_id += 1
    else:
        temp_group = pd.DataFrame({"age": [age], "age_id": [age_id]})
        age_group = pd.concat([age_group, temp_group], ignore_index=True)
        age += 1



## === cell 6
age_group



## === cell 7
customers_df = pd.merge(customers_df, age_group, on="age", how="left")
customers_df = customers_df.drop(
    ["FN", "Active", "club_member_status", "fashion_news_frequency", "postal_code"],
    axis=1,
)
customers_df



## === cell 8
articles_df.head(5)



## === cell 9
print(articles_df["index_group_name"].unique())
print(articles_df["index_group_no"].unique())



## === cell 10
sex_category = articles_df[["index_group_no", "index_group_name"]].reset_index()
display(sex_category["index_group_name"].value_counts())



## === cell 11
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



## === cell 12
del sex_category_list



## === cell 13
articles_category_df = pd.DataFrame(articles_df[["article_id", "index_group_no"]])
articles_category_df.columns = ["article_id", "sex_attribute"]
articles_category_df



## === cell 14
transactions_df = pd.merge(
    transactions_df, articles_category_df, on="article_id", how="left"
)
transactions_df



## === cell 15
cust_sex = pd.DataFrame(columns=["customer_id", "attribute"])
cust_sex



## === cell 16
try:
    cust_sex["attribute"] = cust_sex.apply(
        lambda x: list(x[x == x.max()].index), axis=1
    )
except Exception:
    pass
cust_sex



## === cell 17
try:
    cust_sex1 = pd.DataFrame(cust_sex[["attribute"]]).reset_index()
    cust_sex1["attribute"] = cust_sex1["attribute"].apply(",".join).astype(str)
except Exception:
    cust_sex1 = pd.DataFrame()
cust_sex1



## === cell 18
if not cust_sex1.empty:
    print(cust_sex1.attribute.unique())
else:
    print("cust_sex1 empty")



## === cell 19
if not cust_sex1.empty:
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



## === cell 20
if not cust_sex1.empty:
    print(cust_sex1.attribute.unique())
else:
    print("No attribute values")



## === cell 21
if not cust_sex1.empty:
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



## === cell 22
if not cust_sex1.empty:
    print(cust_sex1["attribute"].value_counts().sort_values(ascending=False))
else:
    print("No attribute distribution")



## === cell 23
customers_df = customers_df.copy()
customers_df



## === cell 24
customers_df.isnull().sum()



## === cell 25
pass



## === cell 26
pass



## === cell 27
pass



## === cell 28
transactions_df = pd.merge(transactions_df, customers_df, on="customer_id", how="left")
transactions_df.isnull().sum()



## === cell 29
if "cust_sex1" in globals():
    del cust_sex1



## === cell 30
transactions_df = transactions_df.loc[
    transactions_df.t_dat >= pd.to_datetime("2020-09-15")
]
transactions_df



## === cell 31
pass



## === cell 32
pass




## === cell 33
def show_images(article_ids, cols=1, rows=-1):
    if isinstance(article_ids, (int, str)):
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
        except Exception:
            pass




## === cell 34
show_images("0915526001")



## === cell 35
top_articles = transactions_df["article_id"].value_counts().head(12).index.tolist()
top_pred_str = " ".join(top_articles)



## === cell 36
pass



## === cell 37
submission_df = pd.read_csv(path / "sample_submission.csv")[["customer_id"]].copy()
submission_df["prediction"] = top_pred_str
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
submission_df.head()



## === cell 38
submission_df.head()



## === cell 39
show_images(top_articles, cols=4)



## === cell 40
print(f"Submission file written to {submission_path}")
