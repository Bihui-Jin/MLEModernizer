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

0.00482

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Diagnosis: The crash happens because `transactions_train.pivot_table(..., aggfunc={'article_id': ['count', aggregate_to_list]})` produces a DataFrame with MultiIndex column levels (e.g., `('article_id','count')`, `('article_id','aggregate_to_list')`). Pandas 2.x disallows merging a left DataFrame with single-level columns against a right DataFrame with multi-level columns, raising `MergeError: Not allowed to merge between different levels`.  
Patch summary: Flatten the pivot_table result’s MultiIndex columns to single-level names *before* calling `pd.merge`, keeping the same data/aggregation logic and then applying the existing final renaming to preserve downstream expectations. This is a minimal localized fix in cell 21 only.  
Updated cells: Only cell 21 is changed.  
Compatibility notes for cell k+1: The resulting `sample_submission` still ends with exactly the same column names as intended by the original code (`customer_id, age_group, index_name, article_purchased, article_id_count`), so later cells can use it unchanged.  
Assumptions: The original intent was to merge on `customer_id` only (as coded) and then rename the resulting columns; we do not change that behavior, only make the merge legal by flattening the right-hand columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most likely from a submission-format mismatch: the competition expects the file to be named `submission.csv` and (more importantly) to contain exactly the `customer_id` values from `sample_submission.csv` (all 1,371,980 rows) in the same universe, while your code currently builds from `customers.csv` and writes `predict.csv`. I make the smallest changes to (1) build predictions starting from the provided sample submission customer list, (2) ensure every customer in the sample gets a prediction (fall back to global top-12 for missing), and (3) write the required `submission.csv`. This preserves your core logic (same pivots, same recommendation construction), but fixes alignment/coverage so the submission is valid and should move the score up toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with `article_id` formatting being wrong in the final `prediction` strings: the H&M competition expects 10-digit article IDs (string, often with leading zeros), but after filtering/pivots your pipeline can still emit IDs that don’t match the expected format exactly, causing almost no matches at evaluation time. I make the smallest change that preserves your logic: enforce a single canonical `article_id` representation (10-digit zero-padded string) everywhere, and especially right before writing predictions (both for known and unknown customers). This should move the score upward toward your target without changing your recommendation approach. I also keep your current “start from sample_submission and fill missing with global top-12” behavior to ensure full coverage and a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your pipeline is producing a valid `submission.csv`, so the 0.0 MAP@12 is most likely coming from recommendation strings that don’t match the evaluation IDs (format/coverage) or from your `articles_remove` logic accidentally inserting a nested list (so the final joined strings contain `[`/`]`/quotes instead of pure 10-digit IDs). I make two minimal, score-relevant fixes: (1) ensure `group_recomendation` actually renames the count column (currently `rename(..., inplace=True)` does nothing without `columns=`), and (2) make `articles_remove` always return a *flat list of article_id strings* (never a list-of-lists), while keeping your exact approach (exclude previously bought items, allocate per-group quantities, fill unknowns with global top-12). These changes keep your core logic intact but should turn “non-matching” predictions into valid article IDs, moving the score up toward your target. I also enforce 10-digit zero-padded `article_id` strings at the point they are used for ranking/recommendation to avoid any accidental integer conversion.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with a silent coverage/format issue: some customers end up with empty or too-short prediction lists after the `query("group_possibility != 0")` / `qty_to_recomend != 0` filtering, and the final fill only happens for fully-missing rows (NaN) rather than empty strings. I keep your exact recommendation logic, but add a minimal “guarantee 12 items” post-processing step that (a) pads each customer’s list with the global top-12 (computed once) and (b) removes duplicates while preserving order. This ensures every row has 12 valid, zero-padded 10-digit `article_id`s, which should move the score up toward your target without changing the core model/aggregation approach. I also reuse the same precomputed global top-12 string for unknown customers and for padding to keep behavior consistent and fast.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with making essentially “random”/too-generic predictions for almost all customers; your current pipeline filters to the last 30 days and then also removes long-not-sold items, which typically pushes you toward a near-global-top12 solution and yields a very low score. To move toward the target (0.00482) with minimal logic change, I keep your exact approach but adjust the *recency window* to 7 days (matching the evaluation horizon) and reduce the “not sold for 30 days” removal threshold to 7 days so you don’t discard recently relevant items. This keeps the same pivots, aggregations, and recommendation construction, but makes the “global top-12” and group tops more aligned with the next-week purchases, which should increase MAP@12 from 0.0 toward your target. The submission format/coverage safeguards remain intact and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with an ID-matching issue: `customer_id` in this competition is a 64-hex string, and if it’s read as a number/object inconsistently (or with whitespace), merges can silently fail so most customers fall back to the same global list, yielding near-zero relevance. I make the smallest score-relevant changes to enforce consistent dtypes (`customer_id` as string everywhere; `article_id` as zero-padded 10-char string everywhere) at read time and right before every merge/prediction build, without changing your recommendation logic. I also ensure the final submission is strictly aligned to `sample_submission.csv` order and that every row has exactly 12 unique article IDs (your existing padding already does this; we just make it robust to dtype/missing-list edge cases). This should move the score upward toward your target while keeping the same overall pipeline and semantics.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with `article_purchased` being computed on the full history *before* you apply the final 7-day filter, which causes `articles_remove()` to exclude many of the actually-relevant popular items and can leave lots of customers with near-empty lists that then get padded by a generic global top-12 (very low MAP@12). To move toward the target with minimal logic change, I only change the order of operations: first apply the “remove long-not-sold” and “keep last 7 days” filters, then build the per-customer purchase lists and counts from that same 7-day window. Everything else (same pivots, same group recommendation build, same exclusion-and-pad strategy, same submission building from `sample_submission`) stays intact, but the exclusions now match the evaluation horizon and should lift MAP@12 above 0.0.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with ID mismatch in the final `prediction` strings: the evaluator expects **space-separated 10-digit numeric article IDs**, but your pipeline can emit `'<NA>'`/`nan`/other non-IDs after merges and list aggregation, which yields no matches and thus 0.0. I keep your exact recommendation logic, but add a minimal, centralized sanitation step that (1) filters recommendation lists to only valid 10-digit digit-only IDs, (2) guarantees exactly 12 unique IDs by padding from the global top-12, and (3) ensures the final submission is aligned to `sample_submission.csv` order. These changes are purely post-processing/format-hardening and should move the score upward toward your target while preserving your core approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with the model making predictions from the wrong “reference week”: you filter the last 7 days of transactions, but those last 7 days are actually inside the training period and not aligned to the (hidden) evaluation week right after the training ends. With minimal change (no change to your pivot/merge/recommendation logic), I shift the 7‑day window back by 7 days so you build “next-week style” popularity and customer history from the week immediately preceding the final week. I also make the “long-not-sold” filter consistent with that same reference date so you don’t discard items that were active right before the prediction period. This should move the score up from 0.0 toward your target while preserving the same core approach and still writing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is still most consistent with “almost no customers get personalized predictions” due to the `query("group_possibility != 0")` + `qty_to_recomend != 0` filters dropping most customers, so the submission degenerates to the same global top-12 for nearly everyone. To move the score up toward 0.00482 with minimal semantic change, I keep your exact group-based recommendation construction but stop dropping customers: instead, I keep all rows and set `qty_to_recomend=0` where needed so they be padded later. I also make the quantity allocation stable (force at least 1 for customers with any history, cap at 12) so you actually emit some per-customer items before padding. Finally, I keep your existing “pad to 12 with global top-12” post-processing so submission validity/coverage stays guaranteed.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most plausibly caused by using the wrong reference week: the code currently builds popularity/personalization from the week ending **7 days before** the last training date, which is misaligned with the actual prediction week (the 7 days immediately after training ends). With minimal change and identical core logic, I switch the `reference_date` to the true end-of-training date (`maximum_history_date`) so the “recent 7 days” window matches the evaluation horizon. I also make the “long-not-sold” cutoff anchored to the same end date (still 7 days) to avoid removing items that are active right before the prediction period. Everything else (pivots, group recommendations, removal/padding, submission formatting) stays the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your submission likely scores 0.0 because the “remove long-not-sold” filter is computed on the full history but applied before the final 7‑day window, which can delete many items that are actually active right before the prediction period and leave most customers with generic/global-only predictions. To move the score up toward 0.00482 with minimal change (no architecture/loop changes), I compute the “long-not-sold” set using only history up to the reference end date, then apply both filters consistently (anchored to the same `reference_date`) before building per-customer purchase lists and group recommendations. This keeps your exact pivot/merge/recommendation construction, but makes the candidate item universe and exclusions aligned with the evaluation horizon. Everything else (format hardening, padding to 12, and writing `submission.csv` aligned to `sample_submission.csv`) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 strongly suggests the evaluator is not matching your predictions to ground truth due to an ID representation mismatch, not that the recommender has no signal. The smallest score-relevant change is to enforce the competition’s exact dtype/format for *both* `customer_id` (64-hex string) and `article_id` (10-digit string) at every read/merge boundary, and to ensure every aggregated list contains only canonical 10-digit strings (no ints, no floats, no “.0”). I also compute `global_top12` using a cheaper/value_counts-based path (same semantics: top by count) and reuse it consistently for padding, while keeping your entire pivot/group recommendation logic unchanged. These are minimal format/robustness fixes aimed specifically at moving the score up from 0.0 toward 0.00482 without changing the modeling approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with your recommender producing (almost) no relevant overlap because the per-customer recommendations are built only from the very last 7 days, while the evaluation week is the 7 days *after* the training period ends; with “last-week-only” popularity/group lists you often end up too sparse and too global. To move upward toward your target with minimal semantic change, I keep your exact pipeline but widen the transaction window used to build customer history and group top lists from 7 days to 30 days (while still anchoring to the same end date), which is a common minimal lift for this competition and should increase personalization coverage. I also make one small, score-relevant robustness fix: ensure `group_recomendation` is built from canonical 10-digit article IDs (so list aggregation and later comparisons don’t silently mismatch). Everything else (same pivots, same allocation, same exclusion, same padding-to-12, same submission writing) stays intact and still writes a valid `submission.csv`.'

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
articles = pd.read_csv(general_path + "articles.csv", dtype={"article_id": "string"})
customers = pd.read_csv(
    general_path + "customers.csv",
    dtype={"customer_id": "string", "postal_code": "string"},
)
sample_submission = pd.read_csv(
    general_path + "sample_submission.csv", dtype={"customer_id": "string"}
)
transactions_train = pd.read_csv(
    general_path + "transactions_train.csv",
    dtype={"customer_id": "string", "article_id": "string", "sales_channel_id": "int8"},
)



## === cell 3
transactions_train.info()



## === cell 4
transactions_train.sample(5)



## === cell 5
transactions_train["article_id"] = (
    transactions_train["article_id"].astype("string").str.zfill(10)
)



## === cell 6
articles["article_id"] = articles["article_id"].astype("string").str.zfill(10)



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
def aggregate_to_list(row):
    return [i for i in row]




## === cell 19
transactions_train = pd.merge(
    transactions_train,
    articles[["article_id", "index_name"]],
    on="article_id",
    how="left",
)



## === cell 20
transactions_train = pd.merge(
    transactions_train,
    customers[["customer_id", "age_group"]],
    on="customer_id",
    how="left",
)



## === cell 21
reference_date = maximum_history_date
history_days = (
    30  # was effectively 7; widen to improve overlap without changing approach.
)

tx_hist = transactions_train[transactions_train["t_dat"] <= reference_date].copy()

articles_sale_interval = tx_hist.pivot_table(
    index="article_id", aggfunc={"t_dat": ["min", "max"]}
).reset_index()
articles_sale_interval.columns = [
    i[0] if (pd.isna(i[1]) or i[1] == "") else i[0] + "_" + i[1]
    for i in articles_sale_interval.columns
]

long_time_have_not_sold = articles_sale_interval[
    articles_sale_interval["t_dat_max"] < reference_date - pd.Timedelta("7 days")
]



## === cell 22
len_before = len(transactions_train)
transactions_train = transactions_train[
    ~transactions_train["article_id"].isin(long_time_have_not_sold["article_id"])
]
print("Removed {:.2%} of articles".format(1 - (len(transactions_train) / len_before)))



## === cell 23
transactions_train = transactions_train[
    (
        transactions_train["t_dat"]
        > reference_date - pd.Timedelta(f"{history_days} days")
    )
    & (transactions_train["t_dat"] <= reference_date)
]



## === cell 24
sample_submission = sample_submission[["customer_id"]].copy()
sample_submission["customer_id"] = sample_submission["customer_id"].astype("string")

sample_submission = sample_submission.merge(
    customers[["customer_id", "age_group"]].copy(),
    on="customer_id",
    how="left",
)

pivot_df = transactions_train.pivot_table(
    index=["customer_id", "index_name"],
    aggfunc={"article_id": ["count", aggregate_to_list]},
).reset_index()

pivot_df.columns = [
    (
        (c[0] if (pd.isna(c[1]) or c[1] == "") else f"{c[0]}_{c[1]}")
        if isinstance(c, tuple)
        else c
    )
    for c in pivot_df.columns
]

sample_submission = pd.merge(sample_submission, pivot_df, on="customer_id", how="left")

sample_submission.columns = [
    "customer_id",
    "age_group",
    "index_name",
    "article_purchased",
    "article_id_count",
]



## === cell 25
transactions_train["article_id"] = (
    transactions_train["article_id"].astype("string").str.zfill(10)
)

group_recomendation = transactions_train.pivot_table(
    index=["age_group", "index_name", "article_id"], aggfunc={"customer_id": "count"}
).reset_index()



## === cell 26
group_recomendation.sort_values(
    by=["age_group", "index_name", "customer_id"], ascending=False, inplace=True
)
group_recomendation.query("customer_id > 2", inplace=True)

group_recomendation.rename(columns={"customer_id": "article_raiting"}, inplace=True)



## === cell 27
sample_submission = pd.merge(
    sample_submission,
    group_recomendation.pivot_table(
        index=["age_group", "index_name"], aggfunc={"article_id": aggregate_to_list}
    ).reset_index(),
    on=["age_group", "index_name"],
    how="left",
)
sample_submission.rename(columns={"article_id": "top_article_id"}, inplace=True)



## === cell 28
sample_submission["group_possibility"] = sample_submission.groupby("customer_id")[
    "article_id_count"
].transform("sum")

sample_submission["group_possibility"] = sample_submission["group_possibility"].fillna(
    0
)

sample_submission["qty_to_recomend"] = 0

mask_hist = sample_submission["group_possibility"] > 0
sample_submission.loc[mask_hist, "qty_to_recomend"] = (
    (
        (
            sample_submission.loc[mask_hist, "article_id_count"]
            / sample_submission.loc[mask_hist, "group_possibility"]
        )
        * 12.0
    )
    .round()
    .astype("Int64")
)

sample_submission.loc[mask_hist, "qty_to_recomend"] = (
    sample_submission.loc[mask_hist, "qty_to_recomend"]
    .fillna(0)
    .clip(lower=1, upper=12)
    .astype("Int64")
)



## === cell 29
sample_submission.info()



## === cell 30
sample_submission = sample_submission.copy()




## === cell 31
def _canon_article_id(x) -> str | None:
    if x is None:
        return None
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = s.zfill(10)
    return s




## === cell 32
def articles_remove(row):
    bought = row["article_purchased"]
    if isinstance(bought, list):
        bought = [_canon_article_id(x) for x in bought]
        bought = [x for x in bought if x is not None]
    else:
        bought = []

    top = row["top_article_id"]
    if isinstance(top, list):
        top = [_canon_article_id(x) for x in top]
        top = [x for x in top if x is not None]
    else:
        top = []

    qty = row["qty_to_recomend"]
    if pd.isna(qty):
        qty = 0
    qty = int(qty)

    recomendation = []
    i = 0
    while len(recomendation) < qty and i < len(top):
        current_article = top[i]
        i += 1
        if current_article in bought:
            continue
        recomendation.append(current_article)
    return recomendation




## === cell 33
sample_submission[
    sample_submission["customer_id"]
    == "0118ed570ff6ff085cde55a6e801c6861a4e9ff9a8d9e82ee36b33c8b3af8f59"
]



## === cell 34
sample_submission["recomendation"] = sample_submission.apply(articles_remove, axis=1)




## === cell 35
def lists_aggregate_to_list(row):
    output = []
    for i in row:
        output += i
    return output




## === cell 36
sample_submission = sample_submission.pivot_table(
    index="customer_id", aggfunc={"recomendation": lists_aggregate_to_list}
).reset_index()



## === cell 37
all_required_customers = pd.read_csv(
    general_path + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)

unknown_customer_id = all_required_customers[
    ~all_required_customers["customer_id"].isin(sample_submission["customer_id"])
].copy()



## === cell 38
unknown_customer_id = unknown_customer_id[["customer_id"]]



## === cell 39
global_top12 = (
    transactions_train["article_id"]
    .astype("string")
    .str.zfill(10)
    .value_counts()
    .index[:12]
    .to_list()
)
global_top12 = [x for x in global_top12 if isinstance(x, str)]
global_top12_str = " ".join(global_top12)

unknown_customer_id["prediction"] = global_top12_str



## === cell 40
sample_submission["recomendation"] = sample_submission["recomendation"].apply(
    lambda lst: [_canon_article_id(x) for x in lst] if isinstance(lst, list) else []
)




## === cell 41
def _is_valid_article_id(x: str) -> bool:
    if x is None:
        return False
    x = str(x)
    return len(x) == 10 and x.isdigit()


def pad_to_12_unique(lst, pad):
    if not isinstance(lst, list):
        lst = []
    seen = set()
    out = []
    for x in lst:
        x = _canon_article_id(x)
        if not _is_valid_article_id(x):
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) == 12:
            return out
    for x in pad:
        x = _canon_article_id(x)
        if not _is_valid_article_id(x):
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) == 12:
            break
    return out


sample_submission["recomendation"] = sample_submission["recomendation"].apply(
    lambda lst: pad_to_12_unique(lst, global_top12)
)
sample_submission["prediction"] = sample_submission["recomendation"].str.join(" ")



## === cell 42
sample_submission = pd.concat(
    [sample_submission, unknown_customer_id], ignore_index=True
)



## === cell 43
sample_submission.drop("recomendation", axis=1, inplace=True)

sample_submission = all_required_customers.merge(
    sample_submission, on="customer_id", how="left"
)

sample_submission["prediction"] = sample_submission["prediction"].fillna(
    global_top12_str
)
sample_submission.loc[sample_submission["prediction"].eq(""), "prediction"] = (
    global_top12_str
)


def _fix_prediction_string(s: str) -> str:
    if not isinstance(s, str):
        s = ""
    parts = [p for p in s.split() if _is_valid_article_id(p)]
    parts = pad_to_12_unique(parts, global_top12)
    return " ".join(parts)


sample_submission["prediction"] = sample_submission["prediction"].apply(
    _fix_prediction_string
)

sample_submission = sample_submission[["customer_id", "prediction"]]

sample_submission.to_csv("submission.csv", index=False)
print(sample_submission.head())
print("Wrote submission.csv with rows:", len(sample_submission))
print("Example prediction:", sample_submission["prediction"].iloc[0])
print(
    "Example prediction length:", len(sample_submission["prediction"].iloc[0].split())
)
