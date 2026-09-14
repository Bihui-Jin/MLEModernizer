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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the pandas merge/pivot issues caused by MultiIndex columns from `pivot_table`, which currently prevents the pipeline from creating the customer×index_name history features. I also correct the column renaming/typos (`article_raiting`, missing `columns=` in `rename`, and swapped column names) so later steps (`qty_to_recomend`, `article_purchased`, `recomendation`) exist and are consistent. Finally, I ensure we start from the real `sample_submission.csv` (so we predict for exactly the required customers) and write a valid `submission.csv` with columns `customer_id` and `prediction` (keeping the same underlying recommendation logic).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from an invalid submission format: `customer_id` must be the *hashed string IDs* from `sample_submission.csv`, but your pipeline merges with `customers.csv` (which uses different IDs) and ends up producing mostly/all default or misaligned predictions. I keep your exact recommendation logic, but switch to building customer features directly from `transactions_train` (derive `age_group` per `customer_id` from transactions), so the IDs always match the submission customers. I also ensure every customer gets exactly up to 12 valid 10-digit `article_id`s, padding with the global top-12 if your per-customer list is shorter. These are minimal, execution-safe changes aimed at moving the score upward toward your target band.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the submission being almost entirely “global top-12” because your `sample_submission` table gets filtered down to only customers with history (`group_possibility != 0`), and most test customers have no matching history rows after your 21-day cut, so they end up blank and then padded. I keep your exact recommendation logic, but stop dropping customers and instead compute `qty_to_recomend` safely per customer while retaining all customers/rows, so customers with partial/empty group history still get a properly-constructed list. I also ensure `index_name` NAs don’t silently break grouping/merging by filling them with a sentinel before the groupby, which increases the number of usable history rows without changing the approach. Finally, I keep the “pad to 12 with global top-12” behavior, but now it more often be a mix of personalized + global, which should move the score upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission being effectively “non-personalized” because the per-customer history (`cust_index_hist`) is built before your 21-day cutoff, but the group recommendations (`top_per_group`) are built after the cutoff; this mismatch makes many customers’ `index_name` rows fail to join to any `top_article_id`, yielding mostly empty recommendations that get padded to the global top-12. I keep your exact approach (age_group + index_name group top articles, exclude already-bought, allocate up to 12 by per-index share, then pad with global top-12), but rebuild the customer×index_name history *after* the same filtering (removing stale articles and applying the 21-day window) so the joins have consistent support. I also compute `maximum_history_date` after removing stale articles (same semantics, just consistent reference) so the “not sold in 30 days” filter is applied relative to the same dataset. These are minimal, execution-safe changes intended to move the score upward toward your target band without changing the model/logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with a submission that is effectively “all global top-12” for most customers because the join to `top_per_group` is done on `["age_group","index_name"]` while `index_name` is not defined for customers in `base_sub` (so almost all `top_article_id` become NaN and recommendations end up empty/padded). I keep your exact logic (age_group + index_name group top articles, exclude already-bought, allocate up to 12 by per-index share, then pad with global top-12), but I merge `top_per_group` using only `age_group` and *then* filter each customer’s group top list down to the relevant `index_name` rows via a mapping. This preserves the same recommendation approach while making the personalization actually connect, which should move the score up toward your target band. I also ensure the per-customer aggregation always produces a list (even when no history) so no customers accidentally collapse to NaN recommendations.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with your recommendations not matching the evaluation’s `article_id` format (the metric expects 10-digit string IDs), because `top_article_id` is built from `group_recomendation` where `article_id` can silently revert to integer dtype after groupby/agg. I make a minimal, execution-safe fix by forcing `article_id` to remain a zero-padded string in *all* downstream tables used for recommendation lists, so the predicted strings match Kaggle’s expected IDs. I also ensure any list-like columns coming from groupby are consistently Python lists of strings (not numpy arrays/ints), without changing your recommendation logic. This should move the score up toward your target by making the submission valid and actually matchable to ground truth.'

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
transactions_train["index_name"] = transactions_train["index_name"].fillna(
    "UNKNOWN_INDEX"
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



## === cell 20
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



## === cell 21
len_before = len(transactions_train)
transactions_train = transactions_train[
    ~transactions_train["article_id"].isin(long_time_have_not_sold["article_id"])
]
print("Removed {:.2%} of articles".format(1 - (len(transactions_train) / len_before)))



## === cell 22
transactions_train = transactions_train[
    transactions_train["t_dat"]
    > transactions_train["t_dat"].max() - pd.Timedelta("21 days")
]



## === cell 23
group_recomendation = transactions_train.groupby(
    ["age_group", "index_name", "article_id"], as_index=False
).agg(article_raiting=("customer_id", "count"))



## === cell 24
group_recomendation.sort_values(
    by=["age_group", "index_name", "article_raiting"], ascending=False, inplace=True
)
group_recomendation = group_recomendation[
    group_recomendation["article_raiting"] > 2
].copy()



## === cell 25
group_recomendation["article_id"] = (
    group_recomendation["article_id"].astype(str).str.zfill(10)
)

top_per_group = group_recomendation.groupby(
    ["age_group", "index_name"], as_index=False
).agg(top_article_id=("article_id", aggregate_to_list))



## === cell 26
cust_index_hist = transactions_train.groupby(
    ["customer_id", "index_name"], as_index=False
).agg(
    article_purchased=("article_id", aggregate_to_list),
    article_id_count=("article_id", "count"),
)

sample_submission = base_sub.merge(cust_index_hist, on="customer_id", how="left")



## === cell 27
sample_submission["article_id_count"] = sample_submission["article_id_count"].fillna(0)

has_hist_row = sample_submission["index_name"].notna()
sample_submission["group_possibility"] = (
    sample_submission.loc[has_hist_row]
    .groupby("customer_id")["article_id_count"]
    .transform("sum")
)
sample_submission["group_possibility"] = sample_submission["group_possibility"].fillna(
    0
)

den = sample_submission["group_possibility"].replace(0, pd.NA)
share = (sample_submission["article_id_count"] / den).fillna(0)

sample_submission["qty_float"] = share * 12.0
sample_submission["qty_floor"] = sample_submission["qty_float"].apply(
    lambda v: int(v) if pd.notna(v) else 0
)

sample_submission["frac"] = (
    sample_submission["qty_float"] - sample_submission["qty_floor"]
).fillna(0.0)

floor_sum = (
    sample_submission.loc[has_hist_row]
    .groupby("customer_id")["qty_floor"]
    .transform("sum")
    .fillna(0)
)
sample_submission["floor_sum"] = floor_sum
sample_submission["slots_left"] = (
    (12 - sample_submission["floor_sum"]).clip(lower=0).astype(int)
)

sample_submission["rank_frac"] = (
    sample_submission.loc[has_hist_row]
    .groupby("customer_id")["frac"]
    .rank(method="first", ascending=False)
)
sample_submission["rank_frac"] = sample_submission["rank_frac"].fillna(10**9)

sample_submission["qty_to_recomend"] = sample_submission["qty_floor"]
add_one = has_hist_row & (
    sample_submission["rank_frac"] <= sample_submission["slots_left"]
)
sample_submission.loc[add_one, "qty_to_recomend"] = (
    sample_submission.loc[add_one, "qty_to_recomend"] + 1
)
sample_submission["qty_to_recomend"] = sample_submission["qty_to_recomend"].astype(
    "Int64"
)

sample_submission.drop(
    columns=["qty_float", "qty_floor", "frac", "floor_sum", "slots_left", "rank_frac"],
    inplace=True,
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_11/795733636.py in <cell line: 0>()
     35 sample_submission["floor_sum"] = floor_sum
     36 sample_submission["slots_left"] = (
---> 37     (12 - sample_submission["floor_sum"]).clip(lower=0).astype(int)
     38 )
     39 

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

## === cell 28
sample_submission.info()



## === cell 29
top_dict = {}
for (ag, idx), grp in top_per_group.groupby(["age_group", "index_name"]):
    lst = grp["top_article_id"].iloc[0]
    if not isinstance(lst, list):
        lst = list(lst)
    lst = [str(a).zfill(10) for a in lst]
    top_dict.setdefault(ag, {})[idx] = lst


def assign_top_article_id(row):
    ag = row["age_group"]
    idx = row["index_name"]
    if pd.isna(idx):
        return []
    return top_dict.get(ag, {}).get(idx, [])


sample_submission["top_article_id"] = sample_submission.apply(
    assign_top_article_id, axis=1
)




## === cell 30
def articles_remove(row):
    bought = row["article_purchased"]
    top = row["top_article_id"]
    qty = int(row["qty_to_recomend"]) if pd.notna(row["qty_to_recomend"]) else 0

    if not isinstance(bought, list):
        bought = []
    if not isinstance(top, list):
        top = []

    bought = [str(a).zfill(10) for a in bought]
    top = [str(a).zfill(10) for a in top]

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




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'qty_to_recomend'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1683540338.py in <cell line: 0>()
----> 1 sample_submission["recomendation"] = sample_submission.apply(articles_remove, axis=1)
      2 
      3 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/1826274484.py in articles_remove(row)
      2     bought = row["article_purchased"]
      3     top = row["top_article_id"]
----> 4     qty = int(row["qty_to_recomend"]) if pd.notna(row["qty_to_recomend"]) else 0
      5 
      6     if not isinstance(bought, list):

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'qty_to_recomend'

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



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1794690465.py in <cell line: 0>()
----> 1 sample_submission = sample_submission.groupby("customer_id", as_index=False).agg(
      2     recomendation=("recomendation", lists_aggregate_to_list)
      3 )
      4 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
   1430 
   1431         op = GroupByApply(self, func, args=args, kwargs=kwargs)
-> 1432         result = op.agg()
   1433         if not is_dict_like(func) and result is not None:
   1434             # GH #52849

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg(self)
    188 
    189         if is_dict_like(func):
--> 190             return self.agg_dict_like()
    191         elif is_list_like(func):
    192             # we require a list, but not a 'str'

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg_dict_like(self)
    421         Result of aggregation.
    422         """
--> 423         return self.agg_or_apply_dict_like(op_name="agg")
    424 
    425     def compute_dict_like(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg_or_apply_dict_like(self, op_name)
   1606             obj, "as_index", True, condition=hasattr(obj, "as_index")
   1607         ):
-> 1608             result_index, result_data = self.compute_dict_like(
   1609                 op_name, selected_obj, selection, kwargs
   1610             )

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in compute_dict_like(self, op_name, selected_obj, selection, kwargs)
    460         is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))
    461         func = cast(AggFuncTypeDict, self.func)
--> 462         func = self.normalize_dictlike_arg(op_name, selected_obj, func)
    463 
    464         is_non_unique_col = (

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in normalize_dictlike_arg(self, how, obj, func)
    661             cols = Index(list(func.keys())).difference(obj.columns, sort=True)
    662             if len(cols) > 0:
--> 663                 raise KeyError(f"Column(s) {list(cols)} do not exist")
    664 
    665         aggregator_types = (list, tuple, dict)

KeyError: "Column(s) ['recomendation'] do not exist"

## === cell 35
global_top12 = (
    transactions_train.groupby("article_id", as_index=False)
    .agg(cnt=("customer_id", "count"))
    .sort_values("cnt", ascending=False)["article_id"]
    .head(12)
    .tolist()
)
global_top12 = [str(a).zfill(10) for a in global_top12]
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
        x = [str(a).zfill(10) for a in x]
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



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'recomendation'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1273978881.py in <cell line: 0>()
     21 
     22 
---> 23 sample_submission["prediction"] = sample_submission["recomendation"].apply(to_pred_str)
     24 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'recomendation'

## === cell 37
sample_submission = sample_submission[["customer_id", "prediction"]]



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3095693906.py in <cell line: 0>()
----> 1 sample_submission = sample_submission[["customer_id", "prediction"]]
      2 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['prediction'] not in index"

## === cell 38
sample_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame must have 'customer_id' and 'prediction' columns.
