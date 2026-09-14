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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.00544

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the pandas merge/pivot issues caused by MultiIndex columns from `pivot_table`, which currently prevents the pipeline from creating the customer×index_name history features. I also correct the column renaming/typos (`article_raiting`, missing `columns=` in `rename`, and swapped column names) so later steps (`qty_to_recomend`, `article_purchased`, `recomendation`) exist and are consistent. Finally, I ensure we start from the real `sample_submission.csv` (so we predict for exactly the required customers) and write a valid `submission.csv` with columns `customer_id` and `prediction` (keeping the same underlying recommendation logic).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from an invalid submission format: `customer_id` must be the *hashed string IDs* from `sample_submission.csv`, but your pipeline merges with `customers.csv` (which uses different IDs) and ends up producing mostly/all default or misaligned predictions. I keep your exact recommendation logic, but switch to building customer features directly from `transactions_train` (derive `age_group` per `customer_id` from transactions), so the IDs always match the submission customers. I also ensure every customer gets exactly up to 12 valid 10-digit `article_id`s, padding with the global top-12 if your per-customer list is shorter. These are minimal, execution-safe changes aimed at moving the score upward toward your target band.'

# 9. Code solution

## === cell 0
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")



## === cell 1
general_path = "../input/h-and-m-personalized-fashion-recommendations/"



## === cell 2
articles = pd.read_csv(general_path + "articles.csv")
customers = pd.read_csv(general_path + "customers.csv")
sample_submission = pd.read_csv(general_path + "sample_submission.csv")
transactions_train = pd.read_csv(general_path + "transactions_train.csv")



## === cell 3
transactions_train.info()



## === cell 4
transactions_train.sample(5)



## === cell 5
transactions_train["article_id"] = (
    transactions_train["article_id"].astype(str).str.zfill(10)
)



## === cell 6
articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)



## === cell 7
transactions_train["t_dat"] = pd.to_datetime(
    transactions_train["t_dat"], format="%Y-%m-%d"
)



## === cell 8
maximum_history_date = transactions_train["t_dat"].max()



## === cell 9
print(f"We have {len(customers)} unique customers")
print(f'And {len(customers["postal_code"].unique())} unique locations')



## === cell 10
customer_per_location = customers.pivot_table(
    index="postal_code", aggfunc={"customer_id": "count"}
)



## === cell 11
customer_per_location.sort_values(by=["customer_id"])



## === cell 12
customers["location"] = "other"
customers.loc[
    customers["postal_code"]
    == "2c29ae653a9282cce4151bd87643c907644e09541abc28ae87dea0d1f6603b1c",
    "location",
] = "big_city"



## === cell 13
transactions_train["price"].hist(bins=30, figsize=(16, 5))
plt.title("Prices")
plt.show()



## === cell 14
articles.nunique()



## === cell 15
articles.sample(1)



## === cell 16
articles["index_name"].unique()



## === cell 17
customers["age_group"] = "<20"
customers.loc[customers["age"] > 20, "age_group"] = "20-45"
customers.loc[customers["age"] > 45, "age_group"] = ">45"




## === cell 18
def aggregate_to_list(x):
    return list(x)




## === cell 19
transactions_train = pd.merge(
    transactions_train,
    articles[["article_id", "index_name"]],
    on="article_id",
    how="left",
)



## === cell 20
cust_age = (
    transactions_train.groupby("customer_id", as_index=False).agg(age=("age", "median"))
    if "age" in transactions_train.columns
    else None
)

cust_age_group = customers[["customer_id", "age"]].copy()
cust_age_group["age_group"] = "<20"
cust_age_group.loc[cust_age_group["age"] > 20, "age_group"] = "20-45"
cust_age_group.loc[cust_age_group["age"] > 45, "age_group"] = ">45"

tx_customers = pd.DataFrame({"customer_id": transactions_train["customer_id"].unique()})
cust_age_group = tx_customers.merge(
    cust_age_group[["customer_id", "age_group"]], on="customer_id", how="left"
)

cust_age_group["age_group"] = cust_age_group["age_group"].fillna("20-45")

transactions_train = transactions_train.merge(
    cust_age_group, on="customer_id", how="left"
)

base_sub = sample_submission[["customer_id"]].merge(
    cust_age_group, on="customer_id", how="left"
)
base_sub["age_group"] = base_sub["age_group"].fillna("20-45")

cust_index_hist = transactions_train.groupby(
    ["customer_id", "index_name"], as_index=False
).agg(
    article_purchased=("article_id", aggregate_to_list),
    article_id_count=("article_id", "count"),
)

sample_submission = base_sub.merge(cust_index_hist, on="customer_id", how="left")



## === cell 21
articles_sale_interval = transactions_train.pivot_table(
    index="article_id", aggfunc={"t_dat": ["min", "max"]}
).reset_index()
articles_sale_interval.columns = [
    i[0] if (pd.isna(i[1]) or i[1] == "") else i[0] + "_" + i[1]
    for i in articles_sale_interval.columns
]
long_time_have_not_sold = articles_sale_interval[
    articles_sale_interval["t_dat_max"] < maximum_history_date - pd.Timedelta("30 days")
]



## === cell 22
len_before = len(transactions_train)
transactions_train = transactions_train[
    ~transactions_train["article_id"].isin(long_time_have_not_sold["article_id"])
]
print("Removed {:.2%} of articles".format(1 - (len(transactions_train) / len_before)))



## === cell 23
transactions_train = transactions_train[
    transactions_train["t_dat"]
    > transactions_train["t_dat"].max() - pd.Timedelta("21 days")
]



## === cell 24
group_recomendation = transactions_train.groupby(
    ["age_group", "index_name", "article_id"], as_index=False
).agg(article_raiting=("customer_id", "count"))



## === cell 25
group_recomendation.sort_values(
    by=["age_group", "index_name", "article_raiting"], ascending=False, inplace=True
)
group_recomendation = group_recomendation[
    group_recomendation["article_raiting"] > 2
].copy()



## === cell 26
top_per_group = group_recomendation.groupby(
    ["age_group", "index_name"], as_index=False
).agg(top_article_id=("article_id", aggregate_to_list))

sample_submission = sample_submission.merge(
    top_per_group, on=["age_group", "index_name"], how="left"
)



## === cell 27
sample_submission["article_id_count"] = sample_submission["article_id_count"].fillna(0)

sample_submission["group_possibility"] = sample_submission.groupby("customer_id")[
    "article_id_count"
].transform("sum")
sample_submission = sample_submission[
    sample_submission["group_possibility"] != 0
].copy()

sample_submission["article_id_count"] = (
    sample_submission["article_id_count"] / sample_submission["group_possibility"]
)
sample_submission["article_id_count"] = sample_submission["article_id_count"] * 12
sample_submission["qty_to_recomend"] = (
    sample_submission["article_id_count"].astype(float).round().astype("Int64")
)



## === cell 28
sample_submission.info()



## === cell 29
sample_submission = sample_submission[sample_submission["qty_to_recomend"] != 0].copy()




## === cell 30
def articles_remove(row):
    bought = row["article_purchased"]
    top = row["top_article_id"]
    qty = int(row["qty_to_recomend"]) if pd.notna(row["qty_to_recomend"]) else 0

    if not isinstance(bought, list):
        bought = []
    if not isinstance(top, list):
        top = []

    recomendation = []
    i = 0
    while len(recomendation) < qty and i < len(top):
        current_article = top[i]
        i += 1
        if current_article in bought:
            continue
        recomendation.append(current_article)
    return recomendation




## === cell 31
sample_submission[
    sample_submission["customer_id"]
    == "0118ed570ff6ff085cde55a6e801c6861a4e9ff9a8d9e82ee36b33c8b3af8f59"
]



## === cell 32
sample_submission["recomendation"] = sample_submission.apply(articles_remove, axis=1)




## === cell 33
def lists_aggregate_to_list(row):
    output = []
    for i in row:
        if isinstance(i, list):
            output += i
    return output




## === cell 34
sample_submission = sample_submission.groupby("customer_id", as_index=False).agg(
    recomendation=("recomendation", lists_aggregate_to_list)
)



## === cell 35
global_top12 = (
    transactions_train.groupby("article_id", as_index=False)
    .agg(cnt=("customer_id", "count"))
    .sort_values("cnt", ascending=False)["article_id"]
    .head(12)
    .tolist()
)
global_top12_str = " ".join(global_top12)

needed_customers = pd.read_csv(
    general_path + "sample_submission.csv", usecols=["customer_id"]
)
sample_submission = needed_customers.merge(
    sample_submission, on="customer_id", how="left"
)




## === cell 36
def to_pred_str(x):
    if isinstance(x, list) and len(x) > 0:
        seen = set()
        out = []
        for a in x:
            if a not in seen:
                out.append(a)
                seen.add(a)
            if len(out) == 12:
                break
        if len(out) < 12:
            for a in global_top12:
                if a not in seen:
                    out.append(a)
                    seen.add(a)
                if len(out) == 12:
                    break
        return " ".join(out)
    return global_top12_str


sample_submission["prediction"] = sample_submission["recomendation"].apply(to_pred_str)



## === cell 37
sample_submission = sample_submission[["customer_id", "prediction"]]



## === cell 38
sample_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())
