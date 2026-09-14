# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

pd.options.mode.copy_on_write = True




## === cell 1
general_path = "../input/h-and-m-personalized-fashion-recommendations/"




## === cell 2
articles = pd.read_csv(
    general_path + "articles.csv",
    usecols=["article_id", "index_name"],
    dtype={"article_id": "int64", "index_name": "category"},
)
customers = pd.read_csv(
    general_path + "customers.csv",
    usecols=["customer_id", "age", "postal_code"],
    dtype={"customer_id": "string", "age": "float32", "postal_code": "string"},
)
sample_submission = pd.read_csv(
    general_path + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)

transactions_train = pd.read_csv(
    general_path + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"customer_id": "string", "article_id": "int64"},
    parse_dates=["t_dat"],
)
maximum_history_date = transactions_train["t_dat"].max()

cutoff_21 = maximum_history_date - pd.Timedelta("21 days")
transactions_train = transactions_train[transactions_train["t_dat"] > cutoff_21].copy()




## === cell 3
print(transactions_train.head(1))




## === cell 4
print(transactions_train.sample(5, random_state=0))




## === cell 5
articles["article_id"] = articles["article_id"].map(lambda x: f"{x:010d}")




## === cell 6
print(f"Maximum history date: {maximum_history_date} | cutoff_21: {cutoff_21}")




## === cell 7
print(f"We have {len(customers)} unique customers")
print(f'And {len(customers["postal_code"].unique())} unique locations')




## === cell 8
customer_per_location = (
    customers.groupby("postal_code", sort=False)["customer_id"]
    .size()
    .to_frame("customer_id")
)




## === cell 9
print(customer_per_location.sort_values(by=["customer_id"]).head())




## === cell 10
customers["location"] = "other"
customers.loc[
    customers["postal_code"]
    == "2c29ae653a9282cce4151bd87643c907644e09541abc28ae87dea0d1f6603b1c",
    "location",
] = "big_city"




## === cell 11
pass




## === cell 12
print(articles.nunique())




## === cell 13
print(articles.sample(1, random_state=0))




## === cell 14
print(articles["index_name"].unique()[:10])




## === cell 15
customers["age_group"] = "<20"
customers.loc[customers["age"] > 20, "age_group"] = "20-45"
customers.loc[customers["age"] > 45, "age_group"] = ">45"




## === cell 16
def aggregate_to_list(row):
    return [i for i in row]




## === cell 17
transactions_train["article_id"] = transactions_train["article_id"].map(
    lambda x: f"{x:010d}"
)

transactions_train = pd.merge(
    transactions_train,
    articles[["article_id", "index_name"]],
    on="article_id",
    how="left",
)




## === cell 18
transactions_train = pd.merge(
    transactions_train,
    customers[["customer_id", "age_group"]],
    on="customer_id",
    how="left",
)




## === cell 19
sub_customers = sample_submission[["customer_id"]].copy()
sub_customers = pd.merge(
    sub_customers, customers[["customer_id", "age_group"]], on="customer_id", how="left"
)

grp = transactions_train.groupby(["customer_id", "index_name"], sort=False)[
    "article_id"
]
_pvt = grp.agg(article_purchased=list, article_id_count="size").reset_index()

sample_submission = pd.merge(
    sub_customers,
    _pvt[["customer_id", "index_name", "article_purchased", "article_id_count"]],
    on="customer_id",
    how="left",
)

sample_submission.columns = [
    "customer_id",
    "age_group",
    "index_name",
    "article_purchased",
    "article_id_count",
]




## === cell 20
articles_sale_interval = (
    transactions_train.groupby("article_id", sort=False)["t_dat"]
    .agg(t_dat_min="min", t_dat_max="max")
    .reset_index()
)
long_time_have_not_sold = articles_sale_interval[
    articles_sale_interval["t_dat_max"] < maximum_history_date - pd.Timedelta("30 days")
]




## === cell 21
len_before = len(transactions_train)
transactions_train = transactions_train[
    ~transactions_train["article_id"].isin(long_time_have_not_sold["article_id"])
].copy()
print("Removed {:.2%} of articles".format(1 - (len(transactions_train) / len_before)))




## === cell 22
pass




## === cell 23
group_recomendation = (
    transactions_train.groupby(["age_group", "index_name", "article_id"], sort=False)[
        "customer_id"
    ]
    .size()
    .reset_index(name="article_raiting")
)




## === cell 24
group_recomendation.sort_values(
    by=["age_group", "index_name", "article_raiting"], ascending=False, inplace=True
)

group_recomendation = group_recomendation.query("article_raiting > 1").copy()




## === cell 25
top_by_group = (
    group_recomendation.groupby(["age_group", "index_name"], sort=False)["article_id"]
    .agg(list)
    .reset_index(name="top_article_id")
)

sample_submission = pd.merge(
    sample_submission,
    top_by_group,
    on=["age_group", "index_name"],
    how="left",
)

sample_submission["top_article_id"] = sample_submission["top_article_id"].apply(
    lambda x: [str(a).zfill(10) for a in x] if isinstance(x, list) else []
)




## === cell 26
sample_submission["article_id_count"] = sample_submission["article_id_count"].fillna(0)

sample_submission["group_possibility"] = sample_submission.groupby("customer_id")[
    "article_id_count"
].transform("sum")

den = sample_submission["group_possibility"].fillna(0)
sample_submission["article_id_count"] = sample_submission[
    "article_id_count"
] / den.replace(0, pd.NA)

sample_submission["article_id_count"] = (
    sample_submission["article_id_count"].fillna(0) * 12
)
sample_submission["article_id_count"] = (
    sample_submission["article_id_count"].astype(float).round().astype("Int64")
)
sample_submission.rename(columns={"article_id_count": "qty_to_recomend"}, inplace=True)

sample_submission["qty_to_recomend"] = (
    sample_submission["qty_to_recomend"].fillna(0).astype("Int64")
)

mask_nohist = sample_submission["group_possibility"].fillna(0) == 0
if mask_nohist.any():
    first_idx = (
        sample_submission[mask_nohist].groupby("customer_id", sort=False).head(1).index
    )
    sample_submission.loc[first_idx, "qty_to_recomend"] = 12




## === cell 27
print(sample_submission.info())




## === cell 28
def articles_remove(row):
    bought = row.get("article_purchased", None)
    if not isinstance(bought, list):
        bought = []
    else:
        bought = [str(a).zfill(10) for a in bought]

    top_list = row.get("top_article_id", None)
    if not isinstance(top_list, list):
        top_list = []
    else:
        top_list = [str(a).zfill(10) for a in top_list]

    qty = row.get("qty_to_recomend", 0)
    if pd.isna(qty):
        qty = 0
    qty = int(qty)

    recomendation = []
    i = 0
    while len(recomendation) < qty and i < len(top_list):
        current_acticle = top_list[i]
        i += 1
        if current_acticle in bought:
            continue
        recomendation.append(current_acticle)
    return recomendation




## === cell 29
print(
    sample_submission[
        sample_submission["customer_id"]
        == "0118ed570ff6ff085cde55a6e801c6861a4e9ff9a8d9e82ee36b33c8b3af8f59"
    ].head()
)




## === cell 30
cust_bought = sample_submission.groupby("customer_id", sort=False)[
    "article_purchased"
].agg(lambda s: next((x for x in s if isinstance(x, list)), []))

work = sample_submission[["customer_id", "top_article_id", "qty_to_recomend"]].copy()
work["bought_list"] = work["customer_id"].map(cust_bought)
work["bought_set"] = work["bought_list"].apply(
    lambda x: set(x) if isinstance(x, list) else set()
)

work["row_order"] = np.arange(len(work), dtype=np.int32)

exp = work.explode("top_article_id", ignore_index=False)
exp = exp[exp["top_article_id"].notna()].copy()

exp = exp[~exp.apply(lambda r: r["top_article_id"] in r["bought_set"], axis=1)]

exp["pos_in_row"] = exp.groupby(["customer_id", "row_order"], sort=False).cumcount()
exp = exp[exp["pos_in_row"] < exp["qty_to_recomend"].astype(int)]

exp.sort_values(["customer_id", "row_order", "pos_in_row"], kind="stable", inplace=True)

recs = (
    exp.groupby("customer_id", sort=False)["top_article_id"]
    .agg(list)
    .reset_index(name="recomendation")
)

all_cust = work[["customer_id"]].drop_duplicates(keep="first")
sample_submission = all_cust.merge(recs, on="customer_id", how="left")
sample_submission["recomendation"] = sample_submission["recomendation"].apply(
    lambda x: x if isinstance(x, list) else []
)




## === cell 31
def lists_aggregate_to_list(row):
    output = []
    for i in row:
        if isinstance(i, list):
            output += i
    return output




## === cell 32
pass




## === cell 33
top12_popular = (
    transactions_train.groupby("article_id")["customer_id"]
    .count()
    .sort_values(ascending=False)
    .head(12)
    .index.astype(str)
    .str.zfill(10)
    .tolist()
)
fallback_pred = " ".join(top12_popular)

required_customers = pd.read_csv(
    general_path + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
unknown_customer_id = required_customers[
    ~required_customers["customer_id"].isin(sample_submission["customer_id"])
].copy()
unknown_customer_id["prediction"] = fallback_pred




## === cell 34
def _format_pred_list(x, fallback_list):
    if not isinstance(x, list):
        return fallback_list
    out = []
    seen = set()
    for a in x:
        a = str(a).zfill(10)
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
        if len(out) >= 12:
            break
    if len(out) == 0:
        return fallback_list
    if len(out) < 12:
        for a in fallback_list:
            if a not in seen:
                out.append(a)
                if len(out) >= 12:
                    break
    return out[:12]


fallback_list = top12_popular

sample_submission["prediction"] = sample_submission["recomendation"].apply(
    lambda x: " ".join(_format_pred_list(x, fallback_list))
)

sample_submission = pd.concat(
    [
        sample_submission[["customer_id", "prediction"]],
        unknown_customer_id[["customer_id", "prediction"]],
    ],
    ignore_index=True,
)

sub_template = required_customers.copy()
sample_submission = pd.merge(
    sub_template, sample_submission, on="customer_id", how="left"
)


def _clean_pred_str(s, fallback_pred_str, fallback_list_):
    if not isinstance(s, str):
        s = ""
    s = s.strip()
    if s == "":
        return fallback_pred_str
    parts = [p for p in s.split(" ") if p != ""]
    cleaned = []
    seen = set()
    for p in parts:
        p = str(p).zfill(10)
        if p in seen:
            continue
        seen.add(p)
        cleaned.append(p)
        if len(cleaned) >= 12:
            break
    if len(cleaned) == 0:
        return fallback_pred_str
    if len(cleaned) < 12:
        for a in fallback_list_:
            if a not in seen:
                cleaned.append(a)
                if len(cleaned) >= 12:
                    break
    return " ".join(cleaned[:12])


sample_submission["prediction"] = sample_submission["prediction"].apply(
    lambda s: _clean_pred_str(s, fallback_pred, fallback_list)
)

sample_submission = sample_submission[["customer_id", "prediction"]]
sample_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())
