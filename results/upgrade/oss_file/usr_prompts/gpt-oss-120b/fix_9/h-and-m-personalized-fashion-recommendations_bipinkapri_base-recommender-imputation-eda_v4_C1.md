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

- What this solution (achieved 0.0) has done: 'Implemented missing imports, added tqdm, seaborn, and matplotlib.pyplot, and renumbered cells to start from 1 as required. This resolves the NameError issues, enables all downstream processing, and ensures the script runs end‑to‑end, producing a valid `submission.csv` with the correct columns for the competition. No core logic was altered, preserving the original recommendation methodology while making the pipeline functional.'
- What this solution (achieved 0.0) has done: 'I add a small post‑processing step that guarantees every row has a non‑empty prediction string and caps the recommendation list at 12 items (the competition limit). This minor change keeps the original recommendation logic intact while preventing empty or overly long predictions that caused a zero MAP@12 score. The new cells trim the `prediction` column after it is built and before saving the CSV.'
- What this solution (achieved 0.0) has done: 'I add a NumPy import and, after building the recommendation list, fill any empty predictions with the globally most‑popular 12 articles. This small fallback keeps the existing logic but ensures every customer has a non‑empty prediction, which should raise the MAP@12 from 0 toward the target 0.00368 while staying within the original framework.'
- What this solution (achieved 0.0) has done: 'I keep the overall recommendation pipeline unchanged but add a tiny fallback that uses each customer’s most‑recently bought 12 articles (from the training transactions) instead of the generic global‑top list when the current prediction is empty or only the global fallback. This small extra step preserves the original logic, adds only a few lines, and is expected to lift the MAP@12 from 0 toward the target 0.00368 without altering model architecture or core processing.'
- What this solution (achieved 0.0) has done: 'We make the prediction‑selection step robust to missing or empty strings by handling NaN values and falling back to recent purchases or the global top‑12 list, then ensure every row has a non‑empty, 12‑item prediction string. This small change keeps the original recommendation logic untouched while adding the necessary fallback to raise the MAP@12 score toward the target.'
- What this solution (achieved 0.0) has done: 'I tighten the recent‑purchase fallback so it returns up to 12 **unique** article IDs in recency order, which should give a few more correct hits and move the MAP@12 score toward the target. This only changes the recent‑prediction construction in the final cell and leaves the rest of the pipeline untouched.'
- What this solution (achieved 0.0) has done: 'I add a tiny post‑processing step that pads each prediction list up to 12 items using the global most‑popular articles (skipping those already present). This keeps the original recommendation logic untouched, ensures every submission line contains a full 12‑item list, and should raise the MAP@12 from 0 toward the target 0.00368 without altering the model or major pipeline steps.'

# 9. Code solution

## === cell 0
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

articles = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv"
)
transactions = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"
)
customer = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv"
)



## === cell 1
articles["article_id"] = articles["article_id"].astype(str)
transactions["article_id"] = transactions["article_id"].astype(str)
articles["article_id"] = articles["article_id"].apply(lambda x: x.zfill(10))
transactions["article_id"] = transactions["article_id"].apply(lambda x: x.zfill(10))



## === cell 2
customer.nunique()



## === cell 3
100 * customer.isnull().sum() / customer.shape[0]



## === cell 4
100 * customer["fashion_news_frequency"].value_counts() / customer.shape[0]



## === cell 5
100 * customer["club_member_status"].value_counts() / customer.shape[0]



## === cell 6
customer.drop(labels=["FN", "Active", "fashion_news_frequency"], inplace=True, axis=1)



## === cell 7
customer["age"].hist()



## === cell 8
customer["age"] = customer["age"].fillna(customer["age"].median())



## === cell 9
customer["club_member_status"] = customer["club_member_status"].fillna(
    customer["club_member_status"].mode().values[0]
)




## === cell 10
def make_buckets(x):
    if 16 <= x <= 24:
        return "Youth"
    elif 24 < x <= 40:
        return "Young Adults"
    elif 40 < x <= 64:
        return "Middle Age Adults"
    elif x > 64:
        return "Seniors"




## === cell 11
customer["age_bucket"] = customer["age"].apply(lambda x: make_buckets(x))



## === cell 12
transactions.drop(labels=["sales_channel_id"], inplace=True, axis=1)



## === cell 13
merged_data1 = pd.merge(left=transactions, right=customer, how="left", on="customer_id")



## === cell 14
del transactions



## === cell 15
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



## === cell 16
MD2.shape



## === cell 17
del merged_data1



## === cell 18
Sales_age_group = MD2.groupby(["age_bucket"])["price"].sum().reset_index()



## === cell 19
sns.barplot(
    x="age_bucket",
    y="price",
    data=Sales_age_group.sort_values(by=["price"], ascending=False),
)
plt.xticks(rotation=90)



## === cell 20
transations_per_group = MD2["age_bucket"].value_counts().reset_index()
transations_per_group.columns = ["age_bucket", "orders"]



## === cell 21
sns.barplot(x="age_bucket", y="orders", data=transations_per_group)
plt.xticks(rotation=90)



## === cell 22
prod_name = MD2["prod_name"].value_counts().reset_index()
prod_name.columns = ["prod_name", "orders"]



## === cell 23
sns.barplot(x="prod_name", y="orders", data=prod_name.head(10))
plt.xticks(rotation=90)



## === cell 24
index_group_name = MD2["index_group_name"].value_counts().reset_index()
index_group_name.columns = ["index_group_name", "orders"]
sns.barplot(x="index_group_name", y="orders", data=index_group_name.head(10))
plt.xticks(rotation=90)



## === cell 25
graphical_appearance_name = (
    MD2["graphical_appearance_name"].value_counts().reset_index()
)
graphical_appearance_name.columns = ["graphical_appearance_name", "orders"]
sns.barplot(
    x="graphical_appearance_name", y="orders", data=graphical_appearance_name.head(10)
)
plt.xticks(rotation=90)



## === cell 26
index_name = MD2["index_name"].value_counts().reset_index()
index_name.columns = ["index_name", "orders"]
sns.barplot(x="index_name", y="orders", data=index_name)
plt.xticks(rotation=90)



## === cell 27
MD2["section_name"].value_counts()
section_name = MD2["section_name"].value_counts().reset_index()
section_name.columns = ["section_name", "orders"]
sns.barplot(x="section_name", y="orders", data=section_name.head(10))
plt.xticks(rotation=90)



## === cell 28
garment_group_name = MD2["garment_group_name"].value_counts().reset_index()
garment_group_name.columns = ["garment_group_name", "orders"]
sns.barplot(x="garment_group_name", y="orders", data=garment_group_name.head(10))
plt.xticks(rotation=90)



## === cell 29
Age_pref_sec = (
    MD2.groupby(["age_bucket", "section_name"])["article_id"].count().reset_index()
)
Age_pref_sec.sort_values(by=["age_bucket", "article_id"], ascending=False, inplace=True)



## === cell 30
Age_pref_sec.groupby(["age_bucket"]).head(4).reset_index(drop=True)



## === cell 31
cust_pref_sec = (
    MD2.groupby(["customer_id", "section_name"])["article_id"].count().reset_index()
)



## === cell 32
cust_pref_sec.sort_values(
    by=["customer_id", "article_id"], ascending=False, inplace=True
)



## === cell 33
cust_pref_sec.reset_index(drop=True, inplace=True)



## === cell 34
cust_no_sec = (
    cust_pref_sec.groupby(["customer_id"])["section_name"].count().reset_index()
)



## === cell 35
cust_no_sec.sort_values(by="section_name", inplace=True)



## === cell 36
Customer_list_1 = (
    cust_no_sec[cust_no_sec.section_name > 2]["customer_id"].unique().tolist()
)



## === cell 37
Customer_list_2 = (
    cust_no_sec[cust_no_sec.section_name <= 2]["customer_id"].unique().tolist()
)



## === cell 38
cust_pref_sec_list1 = cust_pref_sec[
    cust_pref_sec["customer_id"].isin(Customer_list_1)
].reset_index(drop=True)



## === cell 39
cust_pref_sec_list1 = cust_pref_sec_list1.groupby("customer_id").head(3)



## === cell 40
pop_art = (
    MD2.groupby(["section_name", "article_id"])["customer_id"].count().reset_index()
)



## === cell 41
pop_art.sort_values(by=["section_name", "customer_id"], ascending=False, inplace=True)



## === cell 42
pop_art = pop_art.groupby("section_name").head(4).reset_index(drop=True)



## === cell 43
cust_pref_sec_list1



## === cell 44
pop_art.drop(labels="customer_id", axis=1, inplace=True)



## === cell 45
Age_pref_sec = Age_pref_sec.groupby(["age_bucket"]).head(3).reset_index(drop=True)
Age_pref_sec.reset_index(drop=True, inplace=True)



## === cell 46
Age_pref_sec.drop(labels="article_id", axis=1, inplace=True)



## === cell 47
Age_pref_sec



## === cell 48
pop_art



## === cell 49
pop_art_recom = []
for i in tqdm(pop_art.section_name.unique().tolist()):
    artcle = pop_art[pop_art.section_name == i]["article_id"].unique().tolist()
    artcle = [str(i) for i in artcle]
    recommend = " ".join(artcle)
    pop_art_recom.append({"section_name": i, "recommend": recommend})



## === cell 50
recom_sec = pd.DataFrame(pop_art_recom)



## === cell 51
recom_sec.head()



## === cell 52
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



## === cell 53
age_recoom = pd.DataFrame(age_recom)



## === cell 54
Submission = customer[["customer_id", "age_bucket"]]



## === cell 55
Submission.nunique()



## === cell 56
cust_pref_sec_list1.drop(labels=["article_id"], axis=1, inplace=True)



## === cell 57
recom_type1_merge = pd.merge(
    left=cust_pref_sec_list1, right=recom_sec, on="section_name", how="left"
)



## === cell 58
recom_type1_merge["prediction"] = recom_type1_merge.groupby(["customer_id"])[
    "recommend"
].transform(lambda x: " ".join(x))



## === cell 59
recom_type1_merge.drop_duplicates(subset=["customer_id"], keep="first", inplace=True)



## === cell 60
recom_type1_merge.drop(labels=["section_name", "recommend"], axis=1, inplace=True)



## === cell 61
rec_merge2 = Submission[
    ~Submission.customer_id.isin(recom_type1_merge.customer_id.unique().tolist())
]



## === cell 62
rec_merge2.reset_index(drop=True, inplace=True)



## === cell 63
rec_merge2.head()



## === cell 64
age_recoom.head()



## === cell 65
rec_merge2 = pd.merge(left=rec_merge2, right=age_recoom, on="age_bucket", how="left")



## === cell 66
rec_merge2.drop(labels="age_bucket", axis=1, inplace=True)



## === cell 67
rec_merge2.rename(columns={"recommend": "prediction"}, inplace=True)



## === cell 68
rec_merge2.shape[0] + recom_type1_merge.shape[0]



## === cell 69
final_submission = pd.concat([recom_type1_merge, rec_merge2], ignore_index=True)
global_top = MD2["article_id"].value_counts().head(12).index.tolist()
global_top_str = " ".join(global_top)

recent_pred = (
    MD2.sort_values("t_dat")
    .groupby("customer_id")["article_id"]
    .apply(
        lambda x: " ".join(
            x.tail(12)  # take last 12 purchases
            .astype(str)  # ensure string type
            .drop_duplicates(keep="last")  # keep most recent occurrence only
        )
    )
    .reset_index(name="recent")
)

final_submission = final_submission.merge(recent_pred, on="customer_id", how="left")


def choose_pred(row):
    pred = row["prediction"]
    recent = row["recent"]
    if (
        pd.isnull(pred)
        or not isinstance(pred, str)
        or pred.strip() == ""
        or pred.strip() == global_top_str
    ):
        return (
            recent
            if isinstance(recent, str) and recent.strip() != ""
            else global_top_str
        )
    return pred


final_submission["prediction"] = final_submission.apply(choose_pred, axis=1)

final_submission.drop(columns=["recent"], inplace=True)

final_submission["prediction"] = final_submission["prediction"].apply(
    lambda x: " ".join(x.split()[:12])
)




## === cell 70
def pad_to_12(pred):
    items = pred.split()
    if len(items) < 12:
        for aid in global_top:
            if aid not in items:
                items.append(aid)
            if len(items) == 12:
                break
    return " ".join(items[:12])


final_submission["prediction"] = final_submission["prediction"].apply(pad_to_12)



## === cell 71
final_submission.head()



## === cell 72
final_submission.to_csv("submission.csv", index=False)
