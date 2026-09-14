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
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that Kaggle deems effectively empty/invalid for scoring (often due to wrong types/formatting, non-12-length predictions, or customer_id mismatches). I keep your exact blended-popularity logic, but make two minimal fixes that directly affect MAP@12 validity: (1) ensure `customer_id` is written as the original 64-hex string (no pandas `string` NA coercion issues) and (2) enforce that every prediction is exactly 12 space-separated *10-digit* `article_id`s (zero-padded), with a fast, deterministic fill. I also remove the second full read of `transactions_train.csv` by reusing the already-loaded `trans` to reduce risk of partial execution/timeouts that can lead to truncated outputs. These changes should move you up from 0.0 toward your target without changing the recommender core.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with Kaggle scoring your submission as effectively invalid (most commonly because `article_id` tokens aren’t in the exact expected 10-digit numeric format for many/all rows). I make the smallest changes that preserve your blended-popularity logic but enforce that *every* predicted token is a 10-digit, zero-padded numeric `article_id`, and that each row contains exactly 12 unique IDs in the original sample-submission order. I also avoid any possibility of non-digit tokens leaking into the blend by normalizing lists at the source (before blending), not just at the final string step. These changes should move the score up from 0.0 toward your target without changing the recommender approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a submission that is valid syntactically but effectively unscored due to customer_id/article_id formatting mismatches (common in this comp when `customer_id` loses leading zeros, or `article_id` is not consistently 10-digit). I keep your exact blended-popularity logic, but force (1) `customer_id` to be read/written as a zero-padded 64-hex string everywhere, and (2) all `article_id` tokens to be normalized to 10-digit strings at the source and again just before writing. I also add a final strict reindex to `sample_submission` order without a merge (to avoid duplicate-row surprises) and make the per-customer retrieval deterministic and fast by using precomputed dicts (same semantics, fewer pandas corner cases). These are minimal changes aimed at moving the score up from 0.0 toward your target by ensuring Kaggle can correctly align and score every row.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 strongly suggests Kaggle is effectively scoring you as “no correct predictions”, which is consistent with a common H&M pitfall: `customer_id` must be the original 8-hex ID (as in `sample_submission.csv`), not zero-padded to 64. I minimally change the code to stop `zfill(64)` everywhere and instead preserve the exact `customer_id` strings from the sample submission (and normalize transactions to match), keeping your blended-popularity logic unchanged. I also add a tiny safety step to strip whitespace consistently and ensure the final `customer_id` column exactly matches the sample submission order and values. These changes should move the score up from 0.0 toward your target without altering the recommender core.'
- What this solution (achieved 0.0) has done: 'Your current code likely scores 0.0 because `customer_id` is read as pandas `string` (Arrow-backed) and can introduce subtle mismatches vs the exact sample submission IDs during dict lookups/joins, causing most customers to fall back to generic popularity only (and sometimes malformed alignment in older Kaggle scoring). I make minimal, evaluation-relevant changes to force `customer_id` to be plain Python `str` everywhere (sample, transactions, dict keys, and output) while keeping your exact blended-popularity logic unchanged. I also ensure `article_id_str` is generated from the raw integer `article_id` safely and deterministically, and keep the strict final assertions so the produced `submission.csv` is guaranteed valid. These changes should move the score up from 0.0 toward your target by restoring correct per-customer personalization and exact ID matching.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with Kaggle being unable to score your personalization because `customer_id` keys don’t match between `transactions_train.csv` and `sample_submission.csv` (this competition uses 64-hex customer IDs), causing most customers to fall back to generic popularity and effectively behave like “no relevant hits” for many users. I make the smallest, evaluation-relevant change: normalize `customer_id` to a strict 64-hex lowercase string (strip + lower + zfill(64)) consistently for sample submission, transactions, and all dict keys, while preserving your exact blending/re-ranking logic. I also ensure the final output writes the original sample submission `customer_id` strings (as Kaggle expects), by carrying both “raw id” and “normalized id” and reindexing safely. These changes should move MAP@12 up from 0.0 toward your target without altering the recommender core.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most plausibly coming from a customer_id normalization mismatch: the H&M competition customer_id is already a fixed 64-hex string, so forcing `zfill(64)` can create IDs that don’t exist and break personalization (and in some cases break alignment/scoring expectations). I make the minimal change of normalizing customer_id by strict `strip().lower()` only (no zero-fill), so transaction keys match sample submission keys exactly. To keep your core blended-popularity logic unchanged, everything else stays the same, but I also add a tiny safety to ensure we never end up with an empty global top-12 list (which could otherwise propagate blanks). This should move MAP@12 up from 0.0 toward your target by restoring valid per-customer lookups and correct ID matching.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an ID-mismatch issue causing Kaggle to score essentially no correct hits; in this competition `customer_id` should remain the original 64-hex string exactly as in `sample_submission.csv`, and we should not change case (hex is case-sensitive in string matching contexts). I make a minimal change to stop lowercasing/normalizing customer IDs and instead use the raw `customer_id` strings from both transactions and sample submission for all dict keys/lookups, while keeping your blended-popularity logic and weights identical. I also add a tiny guard to ensure customer IDs are consistently stripped (but not lowercased) to prevent whitespace-only mismatches. Everything else (candidate generation, blending, ensuring 12 predictions, output schema/order) stays the same to preserve core logic while moving the score up toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with a customer_id key mismatch: you build all personalization dicts keyed by `trans["customer_id"]`, but you look them up using `sample_sub["customer_id_norm"]` which currently just strips whitespace from `sample_submission.csv` and does not normalize in the same way as transactions. I make the smallest evaluation-relevant fix by (1) using a single shared `_norm_cust_id()` for both transactions and sample submission, (2) building dicts with the normalized IDs and also normalizing the lookup IDs in the same way, while preserving your blending logic/weights and candidate generation. I also add a strict sanity check that the normalized IDs align 1:1 between sample and output to prevent silent fallback-to-popularity for almost everyone. This should move you up from 0.0 toward your target by restoring per-customer history/frequency recommendations without changing the recommender core.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with Kaggle not matching your `customer_id` keys during scoring (even if the CSV is syntactically valid), so the smallest, most score-relevant change is to normalize `customer_id` identically in both `transactions_train.csv` and `sample_submission.csv` using strict `strip()` only (no case changes, no zero-fill) and to build/look up all personalization dicts using that same normalized key. I also make one minimal MAP@12-relevant adjustment to avoid duplicate-heavy recommendation lists: ensure each per-customer candidate list is unique while preserving order before blending (this keeps the same blended-popularity core logic but improves effective coverage of 12 distinct items). Finally, I keep your final strict reindex to `sample_submission` order and the exact 12×10-digit formatting guarantees so the submission is fully scorable.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with Kaggle not being able to match your `customer_id` values during scoring (even though the CSV is syntactically valid), so the smallest score-relevant change is to normalize `customer_id` identically for transactions and sample submission using a shared `strip().lower()` key for all lookups while still writing the *original* `customer_id` strings to the submission. I also make a minimal, MAP@12-relevant improvement that doesn’t change your approach: include a 7-day frequency-per-customer list as an additional blended signal (same candidate/blending logic), which typically increases correct hits without changing the “blended popularity + history” core. Finally, I keep your strict 12×10-digit formatting and exact sample-submission ordering assertions so the output is guaranteed valid and fully scorable.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an evaluation alignment problem: you normalize `customer_id` to lowercase for transactions and lookups, but the competition’s `customer_id` strings are effectively case-sensitive identifiers, so this can break all personalization and can even make scoring behave like “no correct hits.” I make the minimal change to normalize IDs by `strip()` only (no `.lower()`), and ensure we use the exact same normalized key for both `transactions_train.csv` and `sample_submission.csv`. To reduce duplicate-heavy recommendation lists (which hurts MAP@12), I also enforce uniqueness once inside `cust_blend_row` deterministically (core blending logic unchanged). Everything else (candidate sources, weights, blending approach, output formatting and strict assertions) stays the same so it runs end-to-end and produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a silent customer_id key mismatch: you build `cust_recent/cust_freq_*` dicts keyed by *transaction* customer IDs, but you look them up using `sample_submission` IDs; if either side has whitespace/case inconsistencies, almost everyone falls back to generic popularity and can score effectively 0. I make the smallest evaluation-relevant change by enforcing a single shared normalization function for `customer_id` (strip only, no case change) and applying it identically to both transactions and sample submission *before* grouping/dict creation and before lookup. I also add a lightweight guard to ensure we never generate any all-zero filler article (`0000000000`) in top lists (which can happen if transactions were read incorrectly), because that would produce invalid/irrelevant predictions and hurt MAP@12. Core blending logic, weights, candidate sources, and output formatting remain unchanged.'

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


def _norm_cust_id(x: object) -> str:
    if x is None:
        return ""
    return str(x).strip()


transactions_path = _find_file("transactions_train.csv")
sample_sub_path = _find_file("sample_submission.csv")

sample_sub = pd.read_csv(sample_sub_path, dtype={"customer_id": "object"})
sample_sub["customer_id"] = sample_sub["customer_id"].astype(str).map(_norm_cust_id)
sample_sub["customer_id_norm"] = sample_sub["customer_id"].map(_norm_cust_id)
sample_sub = sample_sub.reset_index(drop=True)

trans = pd.read_csv(
    transactions_path,
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"customer_id": "object", "article_id": "int64", "t_dat": "object"},
)

trans["customer_id"] = trans["customer_id"].astype(str).map(_norm_cust_id)

trans["t_dat"] = pd.to_datetime(trans["t_dat"], errors="coerce")
trans = trans.dropna(subset=["t_dat"])

trans["article_id_str"] = trans["article_id"].astype("int64").astype(str).str.zfill(10)

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


def _drop_zero_item(lst: list[str]) -> list[str]:
    return [x for x in lst if x != "0" * 10]


top_all_12 = _drop_zero_item(top_all_12)
top_7_12 = _drop_zero_item(top_7_12)
top_30_12 = _drop_zero_item(top_30_12)

if len(top_all_12) < 12:
    top_all_12 = _drop_zero_item(_top_articles(trans, 12))
if len(top_7_12) < 12:
    top_7_12 = _drop_zero_item(_top_articles(trans, 12))
    if len(top_7_12) < 12:
        top_7_12 = top_all_12
if len(top_30_12) < 12:
    top_30_12 = _drop_zero_item(_top_articles(trans, 12))
    if len(top_30_12) < 12:
        top_30_12 = top_all_12

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

cust_recent = {str(k): v for k, v in cust_recent.to_dict().items()}
cust_freq_30 = {str(k): v for k, v in cust_freq_30.to_dict().items()}

del trans_sorted, last_unique, trans_30
gc.collect()




## === cell 3
def _split_pred(s: str) -> list[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    return s.split()


def _norm_item(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    if not x:
        return ""
    if x.isdigit():
        return x.zfill(10)
    return ""


def _norm_list(items) -> list[str]:
    out = []
    for x in items:
        y = _norm_item(x)
        if y:
            out.append(y)
    return out


def _unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


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


def _fix_len_12_str(pred: str, fallback12: list[str]) -> str:
    items = _norm_list(_split_pred(pred))
    items = _ensure_k(items, fallback12, 12)
    return " ".join(items[:12])


gc.collect()




## === cell 4
def cust_blend_row(pred_lists, W):
    pred_lists = [_unique_keep_order(_norm_list(pl)) for pl in pred_lists]
    res = {}
    for M, rec in enumerate(pred_lists):
        for n, v in enumerate(rec):
            if not v:
                continue
            res[v] = res.get(v, 0.0) + W[M] / (n + 1)
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return res[:12]


cust_ids_raw = sample_sub["customer_id"].astype(str).tolist()
cust_ids_norm = sample_sub["customer_id_norm"].astype(str).tolist()

pred0_list = []
pred1_list = []
pred2_list = []
pred3_list = []

for cid in cust_ids_norm:
    r = cust_recent.get(cid, [])
    f = cust_freq_30.get(cid, [])
    r = _unique_keep_order(_norm_list(r))
    f = _unique_keep_order(_norm_list(f))
    pred0_list.append(
        _ensure_k(_merge_with_fallback(r, top_all_12, 12), top_all_12, 12)
    )
    pred1_list.append(_ensure_k(_merge_with_fallback(f, top_30_12, 12), top_30_12, 12))
    pred2_list.append(_ensure_k(top_7_12, top_all_12, 12))
    pred3_list.append(_ensure_k(top_all_12, top_all_12, 12))

W_a = [0.62, 0.54, 0.50, 0.50]
blend_a = []
for i in range(len(cust_ids_norm)):
    blend_a.append(
        cust_blend_row(
            [pred0_list[i], pred1_list[i], pred2_list[i], pred3_list[i]], W_a
        )
    )

sub0 = pd.DataFrame(
    {"customer_id_norm": cust_ids_norm, "prediction": [" ".join(x) for x in blend_a]}
)

gc.collect()



## === cell 5
predA_list = []
predB_list = []
predC_list = []
predD_list = []

for cid in cust_ids_norm:
    r = cust_recent.get(cid, [])
    f = cust_freq_30.get(cid, [])
    r = _unique_keep_order(_norm_list(r))
    f = _unique_keep_order(_norm_list(f))
    predA_list.append(_ensure_k(_merge_with_fallback(r, top_7_12, 12), top_7_12, 12))
    predB_list.append(
        _ensure_k(_merge_with_fallback(f, top_all_12, 12), top_all_12, 12)
    )
    predC_list.append(_ensure_k(top_30_12, top_all_12, 12))
    predD_list.append(_ensure_k(top_all_12, top_all_12, 12))

W_b = [0.48, 0.42, 0.40, 0.34]
blend_b = []
for i in range(len(cust_ids_norm)):
    blend_b.append(
        cust_blend_row(
            [predA_list[i], predB_list[i], predC_list[i], predD_list[i]], W_b
        )
    )

sub00 = pd.DataFrame(
    {"customer_id_norm": cust_ids_norm, "prediction": [" ".join(x) for x in blend_b]}
)

gc.collect()



## === cell 6
trans_7 = trans.loc[trans["t_dat"] >= w1_start]
cust_freq_7 = (
    trans_7.groupby(["customer_id", "article_id_str"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["customer_id", "cnt"], ascending=[True, False])
    .groupby("customer_id")["article_id_str"]
    .apply(lambda s: s.head(12).tolist())
)
cust_freq_7 = {str(k): v for k, v in cust_freq_7.to_dict().items()}
del trans_7
gc.collect()

cust_freq_all = (
    trans.groupby(["customer_id", "article_id_str"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["customer_id", "cnt"], ascending=[True, False])
    .groupby("customer_id")["article_id_str"]
    .apply(lambda s: s.head(12).tolist())
)
cust_freq_all = {str(k): v for k, v in cust_freq_all.to_dict().items()}
gc.collect()

base_pred_list = []
freq7_pred_list = []
for cid in cust_ids_norm:
    f_all = _unique_keep_order(_norm_list(cust_freq_all.get(cid, [])))
    f_7 = _unique_keep_order(_norm_list(cust_freq_7.get(cid, [])))
    base_pred_list.append(
        _ensure_k(_merge_with_fallback(f_all, top_all_12, 12), top_all_12, 12)
    )
    freq7_pred_list.append(
        _ensure_k(_merge_with_fallback(f_7, top_7_12, 12), top_7_12, 12)
    )

base_pred_str = [" ".join(x) for x in base_pred_list]
freq7_pred_str = [" ".join(x) for x in freq7_pred_list]

sub1 = pd.DataFrame(
    {
        "customer_id_norm": cust_ids_norm,
        "prediction0": base_pred_str,
        "prediction1": sub0["prediction"].astype(str).tolist(),
        "prediction2": sub00["prediction"].astype(str).tolist(),
        "prediction3": freq7_pred_str,
    }
)

del sub0, sub00
gc.collect()



## === cell 7
W_final = [0.28, 0.45, 0.17, 0.10]

final_preds = []
for i in range(len(sub1)):
    p0 = _ensure_k(
        _unique_keep_order(_norm_list(_split_pred(sub1.at[i, "prediction0"]))),
        top_all_12,
        12,
    )
    p1 = _ensure_k(
        _unique_keep_order(_norm_list(_split_pred(sub1.at[i, "prediction1"]))),
        top_all_12,
        12,
    )
    p2 = _ensure_k(
        _unique_keep_order(_norm_list(_split_pred(sub1.at[i, "prediction2"]))),
        top_all_12,
        12,
    )
    p3 = _ensure_k(
        _unique_keep_order(_norm_list(_split_pred(sub1.at[i, "prediction3"]))),
        top_all_12,
        12,
    )
    final_list = cust_blend_row([p0, p1, p2, p3], W_final)
    final_preds.append(" ".join(final_list))

sub_out = pd.DataFrame(
    {
        "customer_id": cust_ids_raw,  # keep original strings for Kaggle submission
        "customer_id_norm": cust_ids_norm,  # normalized key for alignment
        "prediction": final_preds,
    }
)

sub_out = (
    sub_out.set_index("customer_id_norm")
    .reindex(sample_sub["customer_id_norm"])
    .reset_index(drop=True)
)
sub_out["customer_id"] = sample_sub["customer_id"].values
sub_out = sub_out[["customer_id", "prediction"]]

sub_out["prediction"] = sub_out["prediction"].fillna("")
sub_out["prediction"] = sub_out["prediction"].map(
    lambda s: _fix_len_12_str(s, top_all_12)
)

mask_empty = sub_out["prediction"].astype(str).str.strip() == ""
if mask_empty.any():
    fill = " ".join(_ensure_k(top_all_12, top_all_12, 12))
    sub_out.loc[mask_empty, "prediction"] = fill

sub_out["customer_id"] = sub_out["customer_id"].astype(str).map(_norm_cust_id)
sub_out["prediction"] = (
    sub_out["prediction"]
    .astype(str)
    .map(lambda s: " ".join(_unique_keep_order(_norm_list(_split_pred(s)))[:12]))
)
sub_out["prediction"] = sub_out["prediction"].map(
    lambda s: _fix_len_12_str(s, top_all_12)
)

assert sub_out.shape[0] == sample_sub.shape[0]
assert (
    sub_out["customer_id"].astype(str).values
    == sample_sub["customer_id"].astype(str).values
).all()
assert (sub_out["prediction"].str.split().map(len) == 12).all()
assert sub_out["prediction"].str.split().explode().str.fullmatch(r"\d{10}").all()

sub_out.to_csv("submission.csv", index=False)
print(sub_out.head())
print("Wrote submission.csv with shape:", sub_out.shape)
print("Example prediction length:", len(sub_out.loc[0, "prediction"].split()))
print("Any empty predictions:", (sub_out["prediction"].str.strip() == "").any())
print("All lengths==12:", (sub_out["prediction"].str.split().map(len) == 12).all())
