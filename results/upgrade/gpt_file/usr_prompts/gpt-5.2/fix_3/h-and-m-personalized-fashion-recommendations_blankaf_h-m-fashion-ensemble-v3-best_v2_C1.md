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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.0237839020415628

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The notebook currently fails because it tries to read several external “../input/fashions/*.csv” files that do not exist in this Kaggle environment, so nothing downstream is defined. To keep the core idea (a blended recommender submission) but make it runnable end-to-end, I replace those missing inputs with locally-generated candidate “submissions” built from the provided `transactions_train.csv` (popular items + simple customer-history variants). I also fix the invalid `del sub1, sub2, sub3,` syntax and make the blend robust to missing/empty predictions while keeping the same blending logic and output format. The result writes a valid `submission.csv` with `customer_id,prediction` for all customers in `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an invalid/empty submission for many rows (e.g., blank predictions or misalignment), even though a CSV is written. I make minimal fixes that keep your blended-popularity core logic intact but (1) ensure every customer gets exactly 12 article_ids, (2) remove slow/fragile per-row `.apply()` usage that can timeout or partially execute in Kaggle, and (3) guarantee the final submission order matches `sample_submission.csv` exactly. These changes should move MAP@12 up toward your target by making the submission valid and complete, without changing the fundamental recommendation approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import gc



## === cell 1

DATA_DIR_CANDIDATES = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data/h-and-m-personalized-fashion-recommendations",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    if os.path.exists(filename):
        return filename
    raise FileNotFoundError(f"Could not find {filename} in known Kaggle input paths.")


transactions_path = _find_file("transactions_train.csv")
sample_sub_path = _find_file("sample_submission.csv")

sample_sub = pd.read_csv(sample_sub_path, dtype={"customer_id": "string"})
sample_sub = sample_sub.reset_index(drop=True)

trans = pd.read_csv(
    transactions_path,
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"customer_id": "string", "article_id": "int64", "t_dat": "string"},
)
trans["t_dat"] = pd.to_datetime(trans["t_dat"], errors="coerce")
trans = trans.dropna(subset=["t_dat"])
trans["article_id_str"] = trans["article_id"].astype(str).str.zfill(10)

last_date = trans["t_dat"].max()
w1_start = last_date - pd.Timedelta(days=7)
w2_start = last_date - pd.Timedelta(days=30)

gc.collect()




## === cell 2
def _top_articles(df: pd.DataFrame, k: int = 12) -> list[str]:
    vc = df["article_id_str"].value_counts()
    return vc.index[:k].tolist()


top_all_12 = _top_articles(trans, 12)
top_7_12 = _top_articles(trans.loc[trans["t_dat"] >= w1_start], 12)
top_30_12 = _top_articles(trans.loc[trans["t_dat"] >= w2_start], 12)

trans_sorted = trans.sort_values(["customer_id", "t_dat"])
last_unique = trans_sorted.drop_duplicates(
    ["customer_id", "article_id_str"], keep="last"
)
last_unique = last_unique.sort_values(["customer_id", "t_dat"])

cust_recent = last_unique.groupby("customer_id")["article_id_str"].apply(
    lambda s: s.tail(12).tolist()
)

trans_30 = trans.loc[trans["t_dat"] >= w2_start]
cust_freq_30 = (
    trans_30.groupby(["customer_id", "article_id_str"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["customer_id", "cnt"], ascending=[True, False])
    .groupby("customer_id")["article_id_str"]
    .apply(lambda s: s.head(12).tolist())
)

del trans_sorted, last_unique, trans_30
gc.collect()




## === cell 3
def _make_pred_string(items: list[str]) -> str:
    if not items:
        return ""
    return " ".join(items[:12])


def _merge_with_fallback(
    primary: list[str], fallback: list[str], k: int = 12
) -> list[str]:
    seen = set()
    out = []
    for x in primary:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
            if len(out) >= k:
                return out
    for x in fallback:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
            if len(out) >= k:
                return out
    return out


def _ensure_k(items: list[str], fallback: list[str], k: int = 12) -> list[str]:
    out = []
    seen = set()
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
            if len(out) >= k:
                return out
    for x in fallback:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
            if len(out) >= k:
                return out
    return out


gc.collect()




## === cell 4
def cust_blend_row(pred_lists, W):
    res = {}
    for M, rec in enumerate(pred_lists):
        for n, v in enumerate(rec):
            if not v:
                continue
            res[v] = res.get(v, 0.0) + W[M] / (n + 1)
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return res[:12]


cust_ids = sample_sub["customer_id"].tolist()

pred0_list = []
pred1_list = []
pred2_list = []
pred3_list = []

for cid in cust_ids:
    r = cust_recent.get(cid, [])
    f = cust_freq_30.get(cid, [])
    pred0_list.append(
        _ensure_k(_merge_with_fallback(r, top_all_12, 12), top_all_12, 12)
    )
    pred1_list.append(_ensure_k(_merge_with_fallback(f, top_30_12, 12), top_30_12, 12))
    pred2_list.append(_ensure_k(top_7_12, top_all_12, 12))
    pred3_list.append(_ensure_k(top_all_12, top_all_12, 12))

W_a = [0.62, 0.54, 0.50, 0.50]
blend_a = []
for i in range(len(cust_ids)):
    blend_a.append(
        cust_blend_row(
            [pred0_list[i], pred1_list[i], pred2_list[i], pred3_list[i]], W_a
        )
    )

sub0 = pd.DataFrame(
    {
        "customer_id": cust_ids,
        "prediction": [" ".join(x) for x in blend_a],
    }
)

gc.collect()



## === cell 5
predA_list = []
predB_list = []
predC_list = []
predD_list = []

for cid in cust_ids:
    r = cust_recent.get(cid, [])
    f = cust_freq_30.get(cid, [])
    predA_list.append(_ensure_k(_merge_with_fallback(r, top_7_12, 12), top_7_12, 12))
    predB_list.append(
        _ensure_k(_merge_with_fallback(f, top_all_12, 12), top_all_12, 12)
    )
    predC_list.append(_ensure_k(top_30_12, top_all_12, 12))
    predD_list.append(_ensure_k(top_all_12, top_all_12, 12))

W_b = [0.48, 0.42, 0.40, 0.34]
blend_b = []
for i in range(len(cust_ids)):
    blend_b.append(
        cust_blend_row(
            [predA_list[i], predB_list[i], predC_list[i], predD_list[i]], W_b
        )
    )

sub00 = pd.DataFrame(
    {
        "customer_id": cust_ids,
        "prediction": [" ".join(x) for x in blend_b],
    }
)

gc.collect()



## === cell 6
tmp = pd.read_csv(
    transactions_path,
    usecols=["customer_id", "article_id"],
    dtype={"customer_id": "string", "article_id": "int64"},
)
tmp["article_id_str"] = tmp["article_id"].astype(str).str.zfill(10)

cust_freq_all = (
    tmp.groupby(["customer_id", "article_id_str"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["customer_id", "cnt"], ascending=[True, False])
    .groupby("customer_id")["article_id_str"]
    .apply(lambda s: s.head(12).tolist())
)
del tmp
gc.collect()

base_pred_list = []
for cid in cust_ids:
    f = cust_freq_all.get(cid, [])
    base_pred_list.append(
        _ensure_k(_merge_with_fallback(f, top_all_12, 12), top_all_12, 12)
    )

base_pred_str = [" ".join(x) for x in base_pred_list]

sub1 = pd.DataFrame(
    {
        "customer_id": cust_ids,
        "prediction0": base_pred_str,
        "prediction1": sub0["prediction"].astype("string").tolist(),
        "prediction2": sub00["prediction"].astype("string").tolist(),
    }
)

del sub0, sub00
gc.collect()




## === cell 7
def _split_pred(s: str) -> list[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    return s.split()


W_final = [0.31, 0.50, 0.19]
final_preds = []
for i in range(len(sub1)):
    p0 = _ensure_k(_split_pred(sub1.at[i, "prediction0"]), top_all_12, 12)
    p1 = _ensure_k(_split_pred(sub1.at[i, "prediction1"]), top_all_12, 12)
    p2 = _ensure_k(_split_pred(sub1.at[i, "prediction2"]), top_all_12, 12)
    final_list = cust_blend_row([p0, p1, p2], W_final)
    final_preds.append(" ".join(final_list))

sub1 = sub1[["customer_id"]].copy()
sub1["prediction"] = final_preds

sub1 = sample_sub[["customer_id"]].merge(sub1, on="customer_id", how="left")
mask_empty = sub1["prediction"].isna() | (
    sub1["prediction"].astype(str).str.strip() == ""
)
if mask_empty.any():
    sub1.loc[mask_empty, "prediction"] = " ".join(_ensure_k(top_all_12, top_all_12, 12))


def _fix_len_12(pred: str) -> str:
    items = _split_pred(pred)
    items = _ensure_k(items, top_all_12, 12)
    return " ".join(items[:12])


sub1["prediction"] = sub1["prediction"].map(_fix_len_12)

sub1.to_csv("submission.csv", index=False)
print(sub1.head())
print("Wrote submission.csv with shape:", sub1.shape)
print("Example prediction length:", len(sub1.loc[0, "prediction"].split()))
