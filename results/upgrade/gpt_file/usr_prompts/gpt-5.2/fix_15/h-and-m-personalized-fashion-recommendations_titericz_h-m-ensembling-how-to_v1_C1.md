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

0.023095584863488

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your notebook fails because it tries to read public-submission CSVs from `../input/hm-public-submissions`, which isn’t available in this environment; that prevents `sub0/sub1/sub2` from being created and cascades into NameErrors. I replace those missing inputs with three lightweight, legitimate baseline recommenders derived from `transactions_train.csv` (global top-12, last-week top-12, and exponential-decay top-12), then keep your exact blending logic (same `cust_blend` and weights) to produce the final predictions. I also make the script robust to the two possible dataset root paths you listed and ensure `article_id` is formatted as 10-digit strings in the `prediction` field. Finally, it always write a valid `submission-blend-1.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current approach predicts the same 12 items for every customer, which typically lands near MAP@12 ≈ 0.0 because it ignores personalization. To move the score upward toward your target while preserving the same blending core logic, I keep your three recommenders but make the “last-week” and “decay” candidates customer-specific (computed from each customer’s own recent purchase history, with safe fallbacks to global lists). This is a minimal change that aligns better with MAP@12 because it increases the chance relevant items appear in the top ranks for each customer. I also keep sorting/alignment and ensure every `prediction` is exactly up to 12 zero-padded article IDs, writing the same `submission-blend-1.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is likely coming from a submission-format mismatch: the competition expects `customer_id` as the original 16-hex string IDs, but your pipeline reads `customer_id` as plain string, so on Kaggle it can silently mis-align with the sample (often due to dtype/parsing differences) and score ~0. I make the pipeline robust by explicitly enforcing the exact `customer_id`/`article_id` string formats used by the competition (including preserving leading zeros and canonical hex customer IDs), and I also ensure each customer’s personalized lists are de-duplicated while keeping order (a small MAP@12 lift because repeats waste slots). These are minimal changes that keep your same three recommenders and the exact same blending logic/weights, but increase the chance that each row is scored correctly and contains 12 distinct items. The script still run end-to-end within constraints and write `submission-blend-1.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an ID-format mismatch: Kaggle expects the original `customer_id` strings exactly as in `sample_submission.csv`, but your code lowercases them (and the transactions), which can break row alignment and yield effectively unscored predictions. I remove the `.str.lower()` normalization and instead preserve `customer_id` verbatim from both files, while still sorting/merging by the sample to guarantee perfect ordering. I also keep your exact three-candidate generation and the same blending weights, but make the per-customer week/decay lists fall back more safely and ensure each prediction is exactly up to 12 distinct 10-digit `article_id`s. These changes are minimal, keep your core logic intact, and should move the score up toward the target by producing correctly aligned, valid submissions.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission alignment problem rather than model quality: MAP@12 be ~0 if rows don’t match the sample’s `customer_id` order exactly. I make two minimal changes that preserve your recommenders and blending logic: (1) stop sorting `sample` early and instead force the final submission to exactly follow `sample_submission.csv` row order, and (2) ensure `customer_id` is preserved verbatim (no normalization) while keeping `article_id` as 10-digit strings and predictions de-duplicated/padded to 12. This should move the score upward toward your target without changing the core approach or weights. The script still write `submission-blend-1.csv` end-to-end.'
- What this solution (achieved 0.0) has done: 'Your current pipeline likely scores 0.0 because the submission is being generated, but it’s effectively “unpersonalized” for a very large fraction of customers (cold-start), and the few personalized lists you create are based only on items (no “similar-item” generalization), so MAP@12 can collapse close to zero. I keep your exact three-source + `cust_blend` logic and weights, but make the customer-specific lists much denser by adding a minimal, legitimate “co-visit” fallback: for each customer’s recent items, recommend globally most co-purchased items (computed from the last-week transactions only). This preserves the overall approach (still candidate-generation + blending) while increasing the chance that relevant items appear in the top 12 for more customers, moving the score up toward your target. I also remove the unnecessary sorting of `sub0/sub1/sub2` before alignment (it can silently misalign if any duplicates occur) and ensure the final rows always follow `sample_submission.csv` order.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most plausibly caused by an ID dtype/format mismatch: `customer_id` is a 16-character hex string and must be read/written exactly as-is, and `article_id` must be a zero-padded 10-digit string; any accidental numeric parsing or formatting drift can lead to effective misalignment and near-zero MAP@12. I make minimal changes to force canonical dtypes on read (`customer_id` as pandas `string`, `article_id` as `Int64`), avoid any implicit casts, and ensure the final submission strictly follows the `sample_submission.csv` order. I also remove a subtle correctness risk where `.get()` is called on a Series with non-unique index semantics by converting the per-customer week-items mapping to a plain dict, and I keep your blending logic/weights identical while ensuring each prediction is exactly 12 distinct 10-digit article IDs.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission being rejected/ignored by the scorer due to an ID formatting issue rather than the recommender quality. I make the smallest changes that enforce canonical formats end-to-end: `customer_id` preserved exactly as in `sample_submission.csv` (no whitespace/normalization beyond a safe strip) and `article_id` always emitted as a 10-digit zero-padded string, with no `<NA>`/float artifacts. I also ensure that the per-customer maps (`cust_week`, `cust_decay`) are real Python dicts (not Series with potentially surprising index behavior) and that every row produces exactly 12 distinct items in rank order. Core logic (three candidate sources + your `cust_blend` weights and scoring) stays the same.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with the submission not matching Kaggle’s expected ID formats rather than the recommender quality. I make two minimal, score-relevant fixes: (1) enforce `article_id` as a canonical 10-digit string from the moment we load it (so no `Int64 -> string` `<NA>` artifacts), and (2) ensure every intermediate prediction list is de-duplicated and capped to 12 before blending (so the blender doesn’t waste ranks on repeats, which directly hurts MAP@12). Core logic stays the same: same three candidate sources (global / per-customer last-week+covisit fallback / per-customer decay) and the same blending function + weights. The output still be written as `submission-blend-1.csv` in exact `sample_submission.csv` row order.'
- What this solution (achieved 0.0) has done: 'Your current score of 0.0 is most consistent with a submission that Kaggle can’t properly score due to ID formatting/alignment issues rather than purely weak recommendations. I keep your exact three-source candidate generation and the same blending (`cust_blend` with the same weights), but I enforce canonical ID formats more strictly: read/write `customer_id` exactly as in `sample_submission.csv`, ensure `article_id` is always a zero-padded 10-digit string, and prevent any accidental whitespace/case drift. I also make `_fix_pred` robust to any non-10-digit tokens and guarantee exactly 12 distinct items, which directly avoids wasting MAP@12 ranks on duplicates/invalid IDs. These are minimal, score-relevant changes that should move your score upward toward the target while preserving your core logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with Kaggle not being able to match rows (customer_id formatting/alignment) or with invalid/empty predictions slipping through for a large share of customers. I keep your exact three-source candidate generation and the same `cust_blend` logic/weights, but make two minimal score-relevant fixes: (1) enforce canonical `customer_id` formatting (lowercase 16-hex) consistently for both `transactions` and `sample`, while still writing predictions in the sample’s exact row order; and (2) make co-visitation computation deterministic and correct by iterating over `week_items_by_cust_dict.values()` (your current `week_items_by_cust.values` is a Series of lists, and the loop is easy to get wrong/fragile). I also ensure every prediction is exactly 12 distinct 10-digit article_ids with safe fallbacks, which prevents wasting MAP@12 slots on duplicates/invalid tokens. These changes are small, keep your approach intact, and should move the score upward toward your target by producing a properly scored submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a customer_id mismatch against Kaggle’s expected IDs (you’re lowercasing and asserting equality to the lowercased sample, which can produce a file that looks valid locally but doesn’t align with the platform’s original IDs). I keep your exact candidate-generation and blending logic, but preserve `customer_id` verbatim from `sample_submission.csv` for the output, and only use a separate canonicalized key for joins/mappings. I also build all per-customer dicts on the canonical key so lookups work, then map back to the original sample order for submission. These are minimal, score-relevant changes that should move MAP@12 upward toward your target without changing the recommender logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most plausibly coming from a row-ID mismatch on Kaggle caused by lowercasing `customer_id` into `customer_id_key` and then joining on that key; while customer IDs are hex, Kaggle scoring expects the submission to align *exactly* to the sample’s original IDs, and any unexpected normalization/join behavior can effectively make predictions “not for the right customers”. I make the smallest score-relevant change: stop lowercasing entirely and use the original `customer_id` as the join key (still stripped for safety), keeping all your candidate generation + blending logic and weights identical. I also keep your de-duplication and 10-digit `article_id` formatting as-is, since it’s MAP@12-relevant and low-risk. This should move you off 0.0 toward your target by ensuring Kaggle can correctly map each row’s predictions to the intended customer.'
- What this solution (achieved 0.0) has done: 'Your 0.0 MAP@12 is most consistent with Kaggle effectively scoring “no correct customers” due to a `customer_id` mismatch between `sample_submission.csv` and `transactions_train.csv` (the transaction IDs are uppercase hex in this dataset, while the sample is lowercase). I keep your exact three-source candidate generation (global / per-customer week+covisit fallback / per-customer decay) and the same blending logic/weights, but I change only the join key to be a canonical lowercase+stripped `customer_id_key` for both sample and transactions, while still writing the original `customer_id` from the sample to the submission. This is a minimal, score-relevant fix that should move you off 0.0 toward your target by ensuring personalization lookups actually hit. I also keep your de-duplication and 10-digit formatting untouched to preserve evaluation semantics and avoid wasting MAP@12 slots.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data/h-and-m-personalized-fashion-recommendations",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for b in BASE_CANDIDATES:
    if os.path.exists(os.path.join(b, "transactions_train.csv")) and os.path.exists(
        os.path.join(b, "sample_submission.csv")
    ):
        DATA_ROOT = b
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find transactions_train.csv and sample_submission.csv under expected Kaggle paths."
    )

TRANS_PATH = os.path.join(DATA_ROOT, "transactions_train.csv")
SAMPLE_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print("TRANS_PATH:", TRANS_PATH)
print("SAMPLE_PATH:", SAMPLE_PATH)




## === cell 2
def _canon_customer_id(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.lower()
    return s


sample = pd.read_csv(SAMPLE_PATH, dtype={"customer_id": "string"})
sample["customer_id"] = sample["customer_id"].astype("string").str.strip()
sample["customer_id_key"] = _canon_customer_id(sample["customer_id"])

transactions = pd.read_csv(
    TRANS_PATH,
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"t_dat": "string", "customer_id": "string", "article_id": "int64"},
)
transactions["customer_id"] = transactions["customer_id"].astype("string").str.strip()
transactions["customer_id_key"] = _canon_customer_id(transactions["customer_id"])

transactions["article_id_str"] = (
    transactions["article_id"].map(lambda x: f"{int(x):010d}").astype("string")
)

print("transactions:", transactions.shape, "sample:", sample.shape)
print("customer_id example (sample original):", sample["customer_id"].iloc[0])
print("customer_id example (sample key):", sample["customer_id_key"].iloc[0])
print("customer_id example (transactions raw):", transactions["customer_id"].iloc[0])
print(
    "customer_id example (transactions key):", transactions["customer_id_key"].iloc[0]
)
print("article_id_str example:", transactions["article_id_str"].iloc[0])



## === cell 3
transactions = transactions.loc[transactions["article_id_str"].notna()].copy()

transactions["t_dat"] = pd.to_datetime(transactions["t_dat"], errors="coerce")
max_date = transactions["t_dat"].max()

top_global = transactions["article_id_str"].value_counts().head(12).index.tolist()
pred_global = " ".join(top_global)

week_start = max_date - pd.Timedelta(days=6)
week_tx = transactions.loc[
    transactions["t_dat"] >= week_start, ["customer_id_key", "article_id_str"]
].copy()


def _topk_unique_from_counts(s, k=12):
    vc = s.value_counts()
    return vc.head(k).index.tolist()


cust_week = week_tx.groupby("customer_id_key")["article_id_str"].apply(
    lambda s: _topk_unique_from_counts(s, k=12)
)

halflife_days = 7.0
age_days = (max_date - transactions["t_dat"]).dt.days.astype("float32").clip(lower=0)
weights = np.exp(-np.log(2.0) * (age_days / halflife_days)).astype("float32")
transactions["w"] = weights

decay_scores_global = (
    transactions.groupby("article_id_str", sort=False)["w"]
    .sum()
    .sort_values(ascending=False)
)
top_decay_global = decay_scores_global.head(12).index.tolist()
pred_decay_global = " ".join(top_decay_global)

cust_decay = (
    transactions.groupby(["customer_id_key", "article_id_str"], sort=False)["w"]
    .sum()
    .reset_index()
)
cust_decay = cust_decay.sort_values(["customer_id_key", "w"], ascending=[True, False])
cust_decay = cust_decay.groupby("customer_id_key")["article_id_str"].apply(
    lambda s: s.head(12).tolist()
)

cust_week_dict = cust_week.to_dict()
cust_decay_dict = cust_decay.to_dict()

week_baskets = week_tx.drop_duplicates(["customer_id_key", "article_id_str"])
week_items_by_cust = week_baskets.groupby("customer_id_key")["article_id_str"].apply(
    list
)
week_items_by_cust_dict = week_items_by_cust.to_dict()

popular_week = week_tx["article_id_str"].value_counts().index.tolist()

co_counts = {}
for items in week_items_by_cust_dict.values():
    if not isinstance(items, list) or len(items) < 2:
        continue
    uniq = list(dict.fromkeys(items))
    L = len(uniq)
    for i in range(L):
        a = uniq[i]
        da = co_counts.get(a)
        if da is None:
            da = {}
            co_counts[a] = da
        for j in range(L):
            if i == j:
                continue
            b = uniq[j]
            da[b] = da.get(b, 0) + 1

co_top = {}
for a, d in co_counts.items():
    top = sorted(d.items(), key=lambda x: (-x[1], x[0]))[:12]
    co_top[a] = [b for b, _ in top]


def _covisit_fallback_for_customer(cust_key, base_list, k=12):
    if not isinstance(base_list, list) or len(base_list) == 0:
        return None
    out = []
    seen = set()
    for a in base_list:
        if a not in seen:
            out.append(a)
            seen.add(a)
        if len(out) >= k:
            return out[:k]
    for a in base_list:
        for b in co_top.get(a, []):
            if b not in seen:
                out.append(b)
                seen.add(b)
            if len(out) >= k:
                return out[:k]
    for b in popular_week:
        if b not in seen:
            out.append(b)
            seen.add(b)
        if len(out) >= k:
            return out[:k]
    for b in top_global:
        if b not in seen:
            out.append(b)
            seen.add(b)
        if len(out) >= k:
            return out[:k]
    return out[:k]


sub0 = sample[["customer_id", "customer_id_key"]].copy()
sub0["prediction"] = pred_global

sub1 = sample[["customer_id", "customer_id_key"]].copy()
sub1_week = sub1["customer_id_key"].map(cust_week_dict)

sub1["prediction"] = [
    (
        " ".join(x)
        if isinstance(x, list) and len(x) > 0
        else (
            " ".join(
                _covisit_fallback_for_customer(
                    ckey, week_items_by_cust_dict.get(ckey, None), k=12
                )
            )
            if isinstance(week_items_by_cust_dict.get(ckey, None), list)
            and len(week_items_by_cust_dict.get(ckey, None)) > 0
            else pred_global
        )
    )
    for ckey, x in zip(sub1["customer_id_key"].values, sub1_week.values)
]

sub2 = sample[["customer_id", "customer_id_key"]].copy()
sub2["prediction"] = (
    sub2["customer_id_key"]
    .map(cust_decay_dict)
    .apply(
        lambda x: (
            " ".join(x) if isinstance(x, list) and len(x) > 0 else pred_decay_global
        )
    )
)

print("sub0/sub1/sub2:", sub0.shape, sub1.shape, sub2.shape)



## === cell 4
print((sub0["prediction"] == sub1["prediction"]).mean())
print((sub0["prediction"] == sub2["prediction"]).mean())
print((sub1["prediction"] == sub2["prediction"]).mean())



## === cell 5
print(sub0.head())
print()
print(sub1.head())
print()
print(sub2.head())



## === cell 6
sub0 = sub0[["customer_id", "customer_id_key", "prediction"]].copy()
sub0.columns = ["customer_id", "customer_id_key", "prediction0"]
sub0["prediction1"] = sub1["prediction"].values
sub0["prediction2"] = sub2["prediction"].values
del sub1, sub2
sub0.head()




## === cell 7
def cust_blend(dt, W=[1, 1, 1]):
    REC = []
    REC.append(dt["prediction0"].split())
    REC.append(dt["prediction1"].split())
    REC.append(dt["prediction2"].split())

    res = {}
    for M in range(len(REC)):
        for n, v in enumerate(REC[M]):
            if v in res:
                res[v] += W[M] / (n + 1)
            else:
                res[v] = W[M] / (n + 1)

    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return " ".join(res[:12])


def _fix_pred(s):
    toks = [t for t in str(s).split() if t and t not in {"<NA>", "nan", "None"}]

    out = []
    seen = set()
    for t in toks:
        td = "".join(ch for ch in t if ch.isdigit())
        if not td:
            continue
        if len(td) > 10:
            td = td[-10:]
        td = td.zfill(10)
        if td not in seen:
            seen.add(td)
            out.append(td)
        if len(out) == 12:
            break

    if len(out) < 12:
        for t in top_global:
            if t not in seen:
                out.append(t)
                seen.add(t)
            if len(out) == 12:
                break
    return " ".join(out[:12])


sub0["prediction0"] = sub0["prediction0"].map(_fix_pred)
sub0["prediction1"] = sub0["prediction1"].map(_fix_pred)
sub0["prediction2"] = sub0["prediction2"].map(_fix_pred)

sub0["prediction"] = sub0.apply(cust_blend, W=[1.05, 1.00, 0.95], axis=1)
sub0.head()



## === cell 8
print((sub0["prediction"] == sub0["prediction0"]).mean())
print((sub0["prediction"] == sub0["prediction1"]).mean())
print((sub0["prediction"] == sub0["prediction2"]).mean())



## === cell 9
sub0 = sub0[["customer_id", "customer_id_key", "prediction"]].copy()

final = sample[["customer_id", "customer_id_key"]].merge(
    sub0[["customer_id_key", "prediction"]],
    on="customer_id_key",
    how="left",
    sort=False,
)

if final["prediction"].isna().any():
    final["prediction"] = final["prediction"].fillna(pred_global)

final["prediction"] = final["prediction"].map(_fix_pred)

final = final[["customer_id", "prediction"]].copy()

assert final.shape[0] == sample.shape[0]
assert (
    final["customer_id"].astype("string").values
    == sample["customer_id"].astype("string").values
).all()

final.to_csv("submission-blend-1.csv", index=False)
print("Wrote submission-blend-1.csv with shape:", final.shape)
print(final.head())
print("Null predictions:", final["prediction"].isna().sum())
print(
    "Min/Max tokens:",
    final["prediction"].map(lambda x: len(str(x).split())).min(),
    final["prediction"].map(lambda x: len(str(x).split())).max(),
)
