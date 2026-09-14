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
lightgbm==4.6.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.0204628930615779

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I remove the internet `wget` dependency and replace it with a small local `apk` implementation so the notebook runs offline on Kaggle. I fix the broken parquet input paths by reading the provided CSVs from `/kaggle/input/h-and-m-personalized-fashion-recommendations/` and then recreate the missing `week` feature from `t_dat` using the standard “days since max date” mapping used for this competition. I keep the same candidate-generation + LightGBM ranker core logic, but fix type mismatches (customer hex string vs int) and ensure predictions are generated for every `customer_id` in `sample_submission.csv`. Finally, I write a valid submission with a `.csv` suffix (not only `.csv.gz`) to guarantee Kaggle accepts it.'

# 9. Code solution

## === cell 0
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


def apk(actual, predicted, k=12):
    """
    actual: list of ground-truth item ids (strings or ints)
    predicted: list of predicted item ids (strings or ints)
    """
    if actual is None:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    actual_set = set(actual)
    if len(actual_set) == 0:
        return 0.0
    for i, p in enumerate(predicted):
        if p in actual_set and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual_set), k)


def customer_hex_id_to_int(series: pd.Series) -> pd.Series:
    return series.str[-16:].apply(hex_id_to_int)


def hex_id_to_int(s: str) -> int:
    return int(s[-16:], 16)


def article_id_str_to_int(series: pd.Series) -> pd.Series:
    return series.astype("int32")


def article_id_int_to_str(series: pd.Series) -> pd.Series:
    return series.astype("int64").astype(str).str.zfill(10)


class Categorize(BaseEstimator, TransformerMixin):
    def __init__(self, min_examples=0):
        self.min_examples = min_examples
        self.categories = []

    def fit(self, X, y=None):
        for i in range(X.shape[1]):
            vc = X.iloc[:, i].value_counts(dropna=False)
            self.categories.append(vc[vc > self.min_examples].index.tolist())
        return self

    def transform(self, X):
        data = {
            X.columns[i]: pd.Categorical(
                X.iloc[:, i], categories=self.categories[i]
            ).codes
            for i in range(X.shape[1])
        }
        return pd.DataFrame(data=data)


def calculate_apk(list_of_preds, list_of_gts):
    apks = []
    for preds, gt in zip(list_of_preds, list_of_gts):
        apks.append(apk(gt, preds, k=12))
    return float(np.mean(apks)) if len(apks) else 0.0


def eval_sub(sub_csv, skip_cust_with_no_purchases=True):
    sub = pd.read_csv(sub_csv)
    validation_set = pd.read_parquet("data/validation_ground_truth.parquet")

    apks = []
    no_purchases_pattern = []
    for pred, gt in zip(
        sub.prediction.str.split(), validation_set.prediction.str.split()
    ):
        if skip_cust_with_no_purchases and (gt == no_purchases_pattern):
            continue
        apks.append(apk(gt, pred, k=12))
    return float(np.mean(apks)) if len(apks) else 0.0




## === cell 1
import os



## === cell 2
transactions_path = (
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"
)
customers_path = (
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv"
)
articles_path = (
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv"
)

transactions = pd.read_csv(
    transactions_path,
    usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
    dtype={
        "t_dat": "string",
        "customer_id": "string",
        "article_id": "int32",
        "price": "float32",
        "sales_channel_id": "int8",
    },
)
customers = pd.read_csv(
    customers_path, dtype={"customer_id": "string", "postal_code": "string"}
)
articles = pd.read_csv(articles_path, dtype={"article_id": "int32"})

transactions["t_dat"] = pd.to_datetime(transactions["t_dat"], errors="coerce")
max_date = transactions["t_dat"].max()
transactions["week"] = ((max_date - transactions["t_dat"]).dt.days // 7).astype("int16")



## === cell 3
test_week = int(transactions.week.max() + 1)
transactions = transactions[transactions.week > transactions.week.max() - 10].copy()



## === cell 4
articles["article_id"] = articles["article_id"].astype("int32")
customers["customer_id"] = customers["customer_id"].astype("string")
transactions["customer_id"] = transactions["customer_id"].astype("string")



## === cell 5
pass



## === cell 6
c2weeks = transactions.groupby("customer_id")["week"].unique()



## === cell 7
c2weeks2shifted_weeks = {}
for c_id, weeks in c2weeks.items():
    c2weeks2shifted_weeks[c_id] = {}
    weeks_sorted = np.sort(weeks)
    for i in range(weeks_sorted.shape[0] - 1):
        c2weeks2shifted_weeks[c_id][int(weeks_sorted[i])] = int(weeks_sorted[i + 1])
    c2weeks2shifted_weeks[c_id][int(weeks_sorted[-1])] = int(test_week)



## === cell 8
candidates_last_purchase = transactions.copy()



## === cell 9
map_rows = []
for c_id, d in c2weeks2shifted_weeks.items():
    for w, sw in d.items():
        map_rows.append((c_id, w, sw))
week_map = pd.DataFrame(
    map_rows, columns=["customer_id", "week", "week_shifted"]
).astype({"customer_id": "string", "week": "int16", "week_shifted": "int16"})

candidates_last_purchase = candidates_last_purchase.merge(
    week_map, on=["customer_id", "week"], how="left"
)
candidates_last_purchase["week"] = candidates_last_purchase["week_shifted"].astype(
    "int16"
)
candidates_last_purchase.drop(columns=["week_shifted"], inplace=True)



## === cell 10
pass



## === cell 11
mean_price = transactions.groupby(["week", "article_id"])["price"].mean()



## === cell 12
sales = (
    transactions.groupby("week")["article_id"]
    .value_counts()
    .groupby("week")
    .rank(method="dense", ascending=False)
    .groupby("week")
    .head(12)
    .rename("bestseller_rank")
    .astype("int16")
)



## === cell 13
bestsellers_previous_week = pd.merge(
    sales, mean_price, on=["week", "article_id"]
).reset_index()
bestsellers_previous_week.week = (bestsellers_previous_week.week + 1).astype("int16")



## === cell 14
unique_transactions = (
    transactions.groupby(["week", "customer_id"])
    .head(1)
    .drop(columns=["article_id", "price"])
    .copy()
)



## === cell 15
candidates_bestsellers = pd.merge(
    unique_transactions,
    bestsellers_previous_week,
    on="week",
)



## === cell 16
test_set_transactions = unique_transactions.drop_duplicates("customer_id").reset_index(
    drop=True
)
test_set_transactions.week = test_week



## === cell 17
candidates_bestsellers_test_week = pd.merge(
    test_set_transactions,
    bestsellers_previous_week,
    on="week",
)



## === cell 18
candidates_bestsellers = pd.concat(
    [candidates_bestsellers, candidates_bestsellers_test_week], ignore_index=True
)
candidates_bestsellers.drop(columns="bestseller_rank", inplace=True)



## === cell 19
pass



## === cell 20
transactions["purchased"] = 1



## === cell 21
data = pd.concat(
    [transactions, candidates_last_purchase, candidates_bestsellers], ignore_index=True
)
data["purchased"] = data["purchased"].fillna(0).astype("int8")



## === cell 22
data.drop_duplicates(["customer_id", "article_id", "week"], inplace=True)



## === cell 23
data.purchased.mean()



## === cell 24
pass



## === cell 25
data = pd.merge(
    data,
    bestsellers_previous_week[["week", "article_id", "bestseller_rank"]],
    on=["week", "article_id"],
    how="left",
)



## === cell 26
data = data[data.week != data.week.min()].copy()
data["bestseller_rank"] = data["bestseller_rank"].fillna(999).astype("int16")



## === cell 27
data = pd.merge(data, articles, on="article_id", how="left")
data = pd.merge(data, customers, on="customer_id", how="left")



## === cell 28
data.sort_values(["week", "customer_id"], inplace=True)
data.reset_index(drop=True, inplace=True)



## === cell 29
train = data[data.week != test_week].copy()
test = (
    data[data.week == test_week]
    .drop_duplicates(["customer_id", "article_id", "sales_channel_id"])
    .copy()
)



## === cell 30
train_baskets = train.groupby(["week", "customer_id"])["article_id"].count().values



## === cell 31
columns_to_use = [
    "article_id",
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_code",
    "index_group_no",
    "section_no",
    "garment_group_no",
    "FN",
    "Active",
    "club_member_status",
    "fashion_news_frequency",
    "age",
    "postal_code",
    "bestseller_rank",
]



## === cell 32
for col in [
    "index_code",
    "club_member_status",
    "fashion_news_frequency",
    "postal_code",
]:
    if col in train.columns:
        train[col] = train[col].astype("category").cat.codes.astype("int32")
        test[col] = test[col].astype("category").cat.codes.astype("int32")

for col in columns_to_use:
    if col not in train.columns:
        train[col] = 0
        test[col] = 0

train_X = train[columns_to_use].copy()
train_y = train["purchased"].astype("int8").copy()
test_X = test[columns_to_use].copy()

train_X = train_X.fillna(0)
test_X = test_X.fillna(0)



## === cell 33
pass



## === cell 34
from lightgbm.sklearn import LGBMRanker



## === cell 35
ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    n_estimators=1,
    importance_type="gain",
    verbose=-1,
)



## === cell 36
ranker = ranker.fit(
    train_X,
    train_y,
    group=train_baskets,
)



## === cell 37
fi_sum = ranker.feature_importances_.sum()
if fi_sum > 0:
    for i in ranker.feature_importances_.argsort()[::-1][:10]:
        print(columns_to_use[i], ranker.feature_importances_[i] / fi_sum)



## === cell 38
pass



## === cell 39
test["preds"] = ranker.predict(test_X)

c_id2predicted_article_ids = (
    test.sort_values(["customer_id", "preds"], ascending=False)
    .groupby("customer_id")["article_id"]
    .apply(list)
    .to_dict()
)

bestsellers_last_week = bestsellers_previous_week[
    bestsellers_previous_week.week == bestsellers_previous_week.week.max()
]["article_id"].tolist()



## === cell 40
pass



## === cell 41
sub = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv",
    dtype={"customer_id": "string"},
)



## === cell 42
preds = []
for c_id in sub["customer_id"].values:
    pred = c_id2predicted_article_ids.get(c_id, [])
    pred = pred + bestsellers_last_week
    preds.append(pred[:12])



## === cell 43
preds = [
    " ".join(pd.Series(ps, dtype="int64").astype(str).str.zfill(10).tolist())
    for ps in preds
]
sub["prediction"] = preds



## === cell 44
sub_name = "basic_model_submission"
sub.to_csv(f"{sub_name}.csv", index=False)
sub.to_csv(f"{sub_name}.csv.gz", index=False)
print("Wrote:", f"{sub_name}.csv")
print(sub.head())
