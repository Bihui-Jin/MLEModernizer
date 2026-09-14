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

# 5. Code solution

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


class CustomerIdEncoder:
    def __init__(self):
        self._map = None

    def fit(self, *series_list: pd.Series):
        all_ids = pd.concat(
            [s.astype("string") for s in series_list], ignore_index=True
        )
        uniques = pd.Index(all_ids.unique())
        codes = np.arange(len(uniques), dtype=np.uint32)
        self._map = pd.Series(codes, index=uniques)
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        if self._map is None:
            raise ValueError("CustomerIdEncoder is not fitted.")
        out = series.astype("string").map(self._map)
        if out.isna().any():
            missing = series.astype("string")[out.isna()].head(5).tolist()
            raise ValueError(
                f"Unknown customer_id(s) encountered (examples): {missing}"
            )
        return out.astype("uint32")


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
sample_sub_path = (
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv"
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
sub = pd.read_csv(sample_sub_path, dtype={"customer_id": "string"})

transactions["t_dat"] = pd.to_datetime(transactions["t_dat"], errors="coerce")
max_date = transactions["t_dat"].max()
transactions["week"] = ((max_date - transactions["t_dat"]).dt.days // 7).astype("int16")




## === cell 3
all_cust_ids = pd.Index(
    pd.concat(
        [
            transactions["customer_id"],
            customers["customer_id"],
            sub["customer_id"],
        ],
        ignore_index=True,
    ).unique()
)
cust_map = pd.Series(
    np.arange(len(all_cust_ids), dtype=np.uint32),
    index=pd.Index(all_cust_ids.astype("string")).sort_values(),
)

transactions["customer_id_int"] = (
    transactions["customer_id"].map(cust_map).astype("uint32")
)
customers["customer_id_int"] = customers["customer_id"].map(cust_map).astype("uint32")
sub["customer_id_int"] = sub["customer_id"].map(cust_map).astype("uint32")

if sub["customer_id_int"].isna().any():
    missing = sub.loc[sub["customer_id_int"].isna(), "customer_id"].head(5).tolist()
    raise ValueError(
        f"Unmapped customer_id(s) in sample submission (examples): {missing}"
    )

assert sub["customer_id_int"].nunique() == len(
    sub
), "customer_id_int collision in sample submission!"




## === cell 4
test_week = int(transactions.week.max() + 1)
transactions = transactions[transactions.week > transactions.week.max() - 10].copy()




## === cell 5
articles["article_id"] = articles["article_id"].astype("int32")
customers["customer_id"] = customers["customer_id"].astype("string")
transactions["customer_id"] = transactions["customer_id"].astype("string")




## === cell 6
pass




## === cell 7
cust_week = transactions[["customer_id_int", "week"]].drop_duplicates()
cust_week = cust_week.sort_values(["customer_id_int", "week"], kind="mergesort")
cust_week["week_shifted"] = cust_week.groupby("customer_id_int", sort=False)[
    "week"
].shift(-1)
cust_week["week_shifted"] = cust_week["week_shifted"].fillna(test_week).astype("int16")
week_map = cust_week.astype({"customer_id_int": "uint32", "week": "int16"})




## === cell 8
candidates_last_purchase = transactions[
    ["customer_id_int", "week", "article_id", "price", "sales_channel_id"]
].copy()




## === cell 9
candidates_last_purchase = candidates_last_purchase.merge(
    week_map, on=["customer_id_int", "week"], how="left"
)

if candidates_last_purchase["week_shifted"].isna().any():
    candidates_last_purchase["week_shifted"] = candidates_last_purchase[
        "week_shifted"
    ].fillna(test_week)

candidates_last_purchase["week"] = candidates_last_purchase["week_shifted"].astype(
    "int16"
)
candidates_last_purchase.drop(columns=["week_shifted"], inplace=True)

if candidates_last_purchase["week"].isna().any():
    raise ValueError(
        "candidates_last_purchase contains NaN week after shifting; mapping failed."
    )




## === cell 10
pass




## === cell 11
mean_price = transactions.groupby(["week", "article_id"])["price"].mean()




## === cell 12
wk_art_cnt = (
    transactions.groupby(["week", "article_id"], sort=False)
    .size()
    .rename("cnt")
    .reset_index()
)
wk_art_cnt.sort_values(
    ["week", "cnt", "article_id"], ascending=[True, False, True], inplace=True
)
wk_art_cnt["bestseller_rank"] = (
    wk_art_cnt.groupby("week", sort=False)["cnt"]
    .rank(method="dense", ascending=False)
    .astype("int16")
)
sales = wk_art_cnt.loc[
    wk_art_cnt["bestseller_rank"] <= 12, ["week", "article_id", "bestseller_rank"]
].set_index(["week", "article_id"])["bestseller_rank"]




## === cell 13
bestsellers_previous_week = pd.merge(
    sales, mean_price, on=["week", "article_id"]
).reset_index()
bestsellers_previous_week.week = (bestsellers_previous_week.week + 1).astype("int16")




## === cell 14
unique_transactions = (
    transactions.groupby(["week", "customer_id_int"])
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
test_set_transactions = unique_transactions.drop_duplicates(
    "customer_id_int"
).reset_index(drop=True)
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
data.drop_duplicates(["customer_id_int", "article_id", "week"], inplace=True)




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
article_cols_needed = [
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
]
data = pd.merge(data, articles[article_cols_needed], on="article_id", how="left")
data = pd.merge(
    data,
    customers[
        [
            "customer_id_int",
            "FN",
            "Active",
            "club_member_status",
            "fashion_news_frequency",
            "age",
            "postal_code",
        ]
    ],
    on="customer_id_int",
    how="left",
)




## === cell 28
data = data.merge(
    mean_price.rename("mean_price").reset_index(), on=["week", "article_id"], how="left"
)
data["mean_price"] = data["mean_price"].fillna(0.0).astype("float32")




## === cell 29
data.sort_values(["week", "customer_id_int", "article_id"], inplace=True)
data.reset_index(drop=True, inplace=True)




## === cell 30
train = data[data.week != test_week].copy()
test = (
    data[data.week == test_week]
    .drop_duplicates(["customer_id_int", "article_id", "sales_channel_id"])
    .copy()
)




## === cell 31
train_baskets = (
    train.groupby(["week", "customer_id_int"], sort=False)
    .size()
    .to_numpy(dtype="int32")
)
if int(train_baskets.sum()) != int(len(train)):
    raise ValueError(
        f"Group sum ({train_baskets.sum()}) != n_train_rows ({len(train)}); group definition mismatch."
    )




## === cell 32
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
    "mean_price",
]




## === cell 33
cat_cols = [
    "article_id",
    "index_code",
    "club_member_status",
    "fashion_news_frequency",
    "postal_code",
]

for col in columns_to_use:
    if col not in train.columns:
        train[col] = 0
        test[col] = 0

train_X = train[columns_to_use].copy()
train_y = train["purchased"].astype("int8").copy()
test_X = test[columns_to_use].copy()

train_X = train_X.fillna(0)
test_X = test_X.fillna(0)

for c in cat_cols:
    if c in train_X.columns:
        train_X[c] = train_X[c].astype("category")
        test_X[c] = test_X[c].astype("category")

categorizer = Categorize(min_examples=0)
combined_cat = pd.concat(
    [train_X[cat_cols], test_X[cat_cols]], axis=0, ignore_index=True
)
categorizer.fit(combined_cat)

combined_cat_codes = categorizer.transform(combined_cat).astype("int32")
train_X.loc[:, cat_cols] = combined_cat_codes.iloc[: len(train_X)].to_numpy()
test_X.loc[:, cat_cols] = combined_cat_codes.iloc[len(train_X) :].to_numpy()

train_X = train_X.astype("float32")
test_X = test_X.astype("float32")




## === cell 34
pass




## === cell 35
from lightgbm.sklearn import LGBMRanker




## === cell 36
ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    importance_type="gain",
    verbose=-1,
    random_state=42,
    n_jobs=-1,
)




## === cell 37
ranker = ranker.fit(
    train_X,
    train_y,
    group=train_baskets,
)




## === cell 38
fi_sum = ranker.feature_importances_.sum()
if fi_sum > 0:
    for i in ranker.feature_importances_.argsort()[::-1][:10]:
        print(columns_to_use[i], ranker.feature_importances_[i] / fi_sum)




## === cell 39
pass




## === cell 40
test["preds"] = ranker.predict(test_X)

if "bestseller_rank" in test.columns:
    test["preds"] = test["preds"] + (
        1e-6 * (1000.0 - test["bestseller_rank"].astype("float32"))
    )

test_ranked = test.sort_values(["customer_id_int", "preds"], ascending=[True, False])
test_ranked["rnk"] = test_ranked.groupby("customer_id_int").cumcount()
test_ranked = test_ranked[test_ranked["rnk"] < 12]

c_id2predicted_article_ids = (
    test_ranked.groupby("customer_id_int")["article_id"].apply(list).to_dict()
)

bestsellers_last_week = bestsellers_previous_week[
    bestsellers_previous_week.week == bestsellers_previous_week.week.max()
]["article_id"].tolist()




## === cell 41
pass




## === cell 42
bestsellers_last_week_str = (
    pd.Series(bestsellers_last_week, dtype="int64").astype(str).str.zfill(10).tolist()
)

preds = []
for c_id_int in sub["customer_id_int"].values:
    pred_ints = c_id2predicted_article_ids.get(int(c_id_int), [])
    if len(pred_ints):
        pred_str = (
            pd.Series(pred_ints, dtype="int64").astype(str).str.zfill(10).tolist()
        )
    else:
        pred_str = []
    pred_str = pred_str + bestsellers_last_week_str
    seen = set()
    dedup = []
    for a in pred_str:
        if a not in seen:
            seen.add(a)
            dedup.append(a)
        if len(dedup) >= 12:
            break
    preds.append(" ".join(dedup))

sub["prediction"] = preds
sub.drop(columns=["customer_id_int"], inplace=True)

non_empty_rate = float((sub["prediction"].str.len() > 0).mean())
if non_empty_rate < 0.999:
    raise ValueError(
        f"Too many empty predictions ({non_empty_rate:.4f}); pipeline issue."
    )




## === cell 43
sub = sub[["customer_id", "prediction"]]
sub_name = "basic_model_submission"
sub.to_csv(f"{sub_name}.csv", index=False)
sub.to_csv(f"{sub_name}.csv.gz", index=False)
print("Wrote:", f"{sub_name}.csv")
print(sub.head())
print("Non-empty prediction rate:", (sub["prediction"].str.len() > 0).mean())
print("Columns:", list(sub.columns))
