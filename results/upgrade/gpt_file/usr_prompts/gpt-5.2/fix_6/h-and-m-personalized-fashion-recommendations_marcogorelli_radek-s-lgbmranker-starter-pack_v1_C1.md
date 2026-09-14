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

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the internet `wget` dependency and replace it with a small local `apk` implementation so the notebook runs offline on Kaggle. I fix the broken parquet input paths by reading the provided CSVs from `/kaggle/input/h-and-m-personalized-fashion-recommendations/` and then recreate the missing `week` feature from `t_dat` using the standard “days since max date” mapping used for this competition. I keep the same candidate-generation + LightGBM ranker core logic, but fix type mismatches (customer hex string vs int) and ensure predictions are generated for every `customer_id` in `sample_submission.csv`. Finally, I write a valid submission with a `.csv` suffix (not only `.csv.gz`) to guarantee Kaggle accepts it.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a submission-format mismatch: `article_id` must be 10-digit zero-padded strings, but your `c_id2predicted_article_ids` stores plain `int32` and then the final formatting can silently fail when lists contain non-ints/NaNs or when category-coding differs between train/test. I keep your exact candidate generation + LightGBM ranker setup, but make two minimal, high-impact fixes: (1) enforce consistent categorical encoding between train and test by fitting category codes on the concatenation (prevents train/test code mismatch that can collapse ranking quality), and (2) enforce `article_id` string formatting earlier and robustly (so every customer gets valid 10-char ids). These changes are small, do not alter the model architecture or training approach, and should move the score up toward your target (0.02046) from 0.0 by ensuring valid predictions and more stable feature semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with a “valid CSV but effectively empty/constant predictions” failure mode coming from the ranker not learning anything (right now it uses `n_estimators=1` with DART, which often yields near-random/flat scores) and from group construction not matching the ranking query definition (groups should be per `(week, customer_id)` row counts, not counts of purchased-only rows). I keep your exact candidate generation + LGBMRanker approach, but (1) compute LightGBM `group` sizes from the same `train` rows you feed into `train_X` (per week-customer query), and (2) increase `n_estimators` modestly (still within the same model family/objective) so the ranker produces meaningful ordering and moves score up toward the 0.02046 target from 0.0. I also add a tiny safety alignment check to ensure the group vector sums exactly to `len(train_X)` (preventing silent mis-training). Submission formatting stays the same (10-digit zero-padded `article_id` strings, 12 max per customer).'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with the LightGBM ranker being trained with *misaligned query groups*: `group` must correspond to the exact order of rows in `train_X`, but you currently compute group sizes on `train.groupby(["week","customer_id"])` while later training rows are only sorted by `["week","customer_id"]` at the `data` stage (and can be perturbed by merges/dedup), which can silently corrupt the ranking objective and yield near-constant predictions. I make the smallest safe change by recomputing `train_baskets` **from the already-sorted `train` used to build `train_X`**, ensuring group sizes match row order exactly (sum check remains). I also ensure the per-customer ranking is correct by sorting predictions with `preds` descending (your current `ascending=False` on `preds` is correct, but it also sets `customer_id` descending; we sort `customer_id` ascending for stable grouping). These changes keep your exact candidate generation + LGBMRanker approach and should move the score up toward the 0.02046 target from 0.0 by fixing the ranking training semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an evaluation key mismatch: the H&M submission `customer_id` values must be the 64-bit hex strings, but your model is trained/predicting on the raw `transactions_train.csv` `customer_id` strings, so the lookup into `c_id2predicted_article_ids` can fail for most/all rows and you effectively submit only bestsellers. I keep the same candidate generation + LGBMRanker setup, but add a minimal, deterministic normalization step that converts both transaction and submission customer_ids to a shared integer key for joins/grouping, then map predictions back to the original submission customer_id order. This preserves your model/feature core logic while making the per-customer predictions actually attach to the right customers, which should move MAP@12 up toward your ~0.020 target. I also ensure the final submission is built from the sample submission and that article_ids are always formatted as 10-digit strings.'

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
transactions["customer_id_int"] = customer_hex_id_to_int(transactions["customer_id"])
customers["customer_id_int"] = customer_hex_id_to_int(customers["customer_id"])



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
c2weeks = transactions.groupby("customer_id_int")["week"].unique()



## === cell 8
c2weeks2shifted_weeks = {}
for c_id, weeks in c2weeks.items():
    c2weeks2shifted_weeks[c_id] = {}
    weeks_sorted = np.sort(weeks)
    for i in range(weeks_sorted.shape[0] - 1):
        c2weeks2shifted_weeks[c_id][int(weeks_sorted[i])] = int(weeks_sorted[i + 1])
    c2weeks2shifted_weeks[c_id][int(weeks_sorted[-1])] = int(test_week)



## === cell 9
candidates_last_purchase = transactions.copy()



## === cell 10
map_rows = []
for c_id, d in c2weeks2shifted_weeks.items():
    for w, sw in d.items():
        map_rows.append((c_id, w, sw))
week_map = pd.DataFrame(
    map_rows, columns=["customer_id_int", "week", "week_shifted"]
).astype({"customer_id_int": "int64", "week": "int16", "week_shifted": "int16"})

candidates_last_purchase = candidates_last_purchase.merge(
    week_map, on=["customer_id_int", "week"], how="left"
)
candidates_last_purchase["week"] = candidates_last_purchase["week_shifted"].astype(
    "int16"
)
candidates_last_purchase.drop(columns=["week_shifted"], inplace=True)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_11/4095617589.py in <cell line: 0>()
     10     week_map, on=["customer_id_int", "week"], how="left"
     11 )
---> 12 candidates_last_purchase["week"] = candidates_last_purchase["week_shifted"].astype(
     13     "int16"
     14 )

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
     99 
    100     elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
--> 101         return _astype_float_to_int_nansafe(arr, dtype, copy)
    102 
    103     elif arr.dtype == object:

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_float_to_int_nansafe(values, dtype, copy)
    143     """
    144     if not np.isfinite(values).all():
--> 145         raise IntCastingNaNError(
    146             "Cannot convert non-finite values (NA or inf) to integer"
    147         )

IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer

## === cell 11
pass



## === cell 12
mean_price = transactions.groupby(["week", "article_id"])["price"].mean()



## === cell 13
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



## === cell 14
bestsellers_previous_week = pd.merge(
    sales, mean_price, on=["week", "article_id"]
).reset_index()
bestsellers_previous_week.week = (bestsellers_previous_week.week + 1).astype("int16")



## === cell 15
unique_transactions = (
    transactions.groupby(["week", "customer_id_int"])
    .head(1)
    .drop(columns=["article_id", "price"])
    .copy()
)



## === cell 16
candidates_bestsellers = pd.merge(
    unique_transactions,
    bestsellers_previous_week,
    on="week",
)



## === cell 17
test_set_transactions = unique_transactions.drop_duplicates(
    "customer_id_int"
).reset_index(drop=True)
test_set_transactions.week = test_week



## === cell 18
candidates_bestsellers_test_week = pd.merge(
    test_set_transactions,
    bestsellers_previous_week,
    on="week",
)



## === cell 19
candidates_bestsellers = pd.concat(
    [candidates_bestsellers, candidates_bestsellers_test_week], ignore_index=True
)
candidates_bestsellers.drop(columns="bestseller_rank", inplace=True)



## === cell 20
pass



## === cell 21
transactions["purchased"] = 1



## === cell 22
data = pd.concat(
    [transactions, candidates_last_purchase, candidates_bestsellers], ignore_index=True
)
data["purchased"] = data["purchased"].fillna(0).astype("int8")



## === cell 23
data.drop_duplicates(["customer_id_int", "article_id", "week"], inplace=True)



## === cell 24
data.purchased.mean()



## === cell 25
pass



## === cell 26
data = pd.merge(
    data,
    bestsellers_previous_week[["week", "article_id", "bestseller_rank"]],
    on=["week", "article_id"],
    how="left",
)



## === cell 27
data = data[data.week != data.week.min()].copy()
data["bestseller_rank"] = data["bestseller_rank"].fillna(999).astype("int16")



## === cell 28
data = pd.merge(data, articles, on="article_id", how="left")
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



## === cell 29
data.sort_values(["week", "customer_id_int"], inplace=True)
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
    train.groupby(["week", "customer_id_int"], sort=False).size().values.astype("int32")
)
if train_baskets.sum() != len(train):
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
]



## === cell 33
cat_cols = ["index_code", "club_member_status", "fashion_news_frequency", "postal_code"]
for col in cat_cols:
    if col in train.columns and col in test.columns:
        combined = pd.concat([train[[col]], test[[col]]], axis=0, ignore_index=True)
        codes = combined[col].astype("category").cat.codes.astype("int32").values
        train[col] = codes[: len(train)]
        test[col] = codes[len(train) :]

for col in columns_to_use:
    if col not in train.columns:
        train[col] = 0
        test[col] = 0

train_X = train[columns_to_use].copy()
train_y = train["purchased"].astype("int8").copy()
test_X = test[columns_to_use].copy()

train_X = train_X.fillna(0)
test_X = test_X.fillna(0)



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

c_id2predicted_article_ids = (
    test.sort_values(["customer_id_int", "preds"], ascending=[True, False])
    .groupby("customer_id_int")["article_id"]
    .apply(list)
    .to_dict()
)

bestsellers_last_week = bestsellers_previous_week[
    bestsellers_previous_week.week == bestsellers_previous_week.week.max()
]["article_id"].tolist()



## === cell 41
pass



## === cell 42
sub = pd.read_csv(
    "/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv",
    dtype={"customer_id": "string"},
)

sub["customer_id_int"] = customer_hex_id_to_int(sub["customer_id"])



## === cell 43
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



## === cell 44
sub_name = "basic_model_submission"
sub.to_csv(f"{sub_name}.csv", index=False)
sub.to_csv(f"{sub_name}.csv.gz", index=False)
print("Wrote:", f"{sub_name}.csv")
print(sub.head())
print("Non-empty prediction rate:", (sub["prediction"].str.len() > 0).mean())
