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

0.0239973603881963

# 6. Current score

0.01485

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your notebook fails immediately because it tries to load multiple pre-made submission files from `../input/fashions/`, which do not exist in this Kaggle environment, so none of the later blending code can run. I replace those missing inputs with an equivalent, self-contained pipeline that builds predictions directly from the provided `transactions_train.csv` and writes a valid `submission.csv` matching `sample_submission.csv` customer ordering. To keep the core intent (a popularity/recency-style recommender and simple ensembling), I generate 3 candidate lists (overall top, last-7-days top, last-28-days top) and blend them with your same rank-based weighted `cust_blend`. This run end-to-end within the time limit (by reading only needed columns and using vectorized groupby) and produce a valid `.csv` submission.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because the submission is effectively predicting the wrong `article_id` format: `top_*` lists are pre-padded to 10 digits, but then `fmt_aid_list()` tries to `int()` them and re-formats them, which strips leading zeros and produces IDs that don’t match ground truth. I make the smallest fix by ensuring all candidate generators output the same canonical 10-digit `article_id` string exactly once, and I remove the double-formatting bug for the global/7d/28d popular lists. This preserves your core logic (recent-per-customer + three popular lists + the same rank-weighted blend), but makes predictions valid and should move the score upward toward your target. I also keep the sample submission ordering unchanged and still write `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a submission that is “valid CSV” but contains many duplicate article_ids per customer (MAP@12 heavily penalizes lack of unique relevant hits) and some customers getting empty predictions. I make two minimal, metric-aligned fixes without changing your core approach: (1) deduplicate each customer’s blended ranked list while preserving rank order, and (2) backfill any empty blended predictions with the global popular list so every row has 12 items. These changes keep the same candidate sources (recent-per-customer + 7d/28d/global popularity) and the same rank-weighted blending semantics, but should reliably move the score upward toward your target band.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 strongly suggests a submission validity mismatch rather than a weak model, and the most common cause here is `customer_id` formatting: if IDs aren’t exactly the expected 64-hex characters, Kaggle score you as effectively predicting nothing. I make a minimal, metric-preserving fix by forcing `customer_id` to be normalized (lowercased, stripped, and left-padded to 64 chars) consistently for both transactions and sample submission before any grouping/mapping. I also make one small robustness tweak to the empty/backfill logic to ensure no `"nan"` strings leak into blending (without changing candidate sources or the rank-weighted blend). This should move the score up from 0.0 toward your target while keeping your core popularity/recency blending approach intact.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an ID-mismatch against the ground truth rather than weak ranking: you’re zero-padding `customer_id` to 64 chars, but in this competition `customer_id` values are 16-hex characters, so your submission IDs won’t match the evaluation join and you effectively score nothing. I make the smallest change to normalize `customer_id` safely without altering IDs (strip + lowercase only, no zfill), and I keep ordering exactly as in `sample_submission.csv` (don’t sort), which removes another common silent misalignment risk. Everything else (candidate generation, rank-weighted blending, dedup/top-up to 12, output CSV) stays the same so the score should move up toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an evaluation join mismatch: the `customer_id` values in H&M are 64-hex strings, but your current validation only warns (and your earlier attempts mention 16), so submissions can silently score 0 if IDs are altered. I make the smallest fix by normalizing `customer_id` in a metric-safe way (strip + lowercase only; no padding/truncation) and then validating the length against the sample submission (not a hardcoded 16). I also ensure `article_id` is treated as a zero-padded 10-digit string consistently by reading it as string and formatting once, avoiding any accidental float/int coercions that can corrupt IDs. Core logic (recent-per-customer + w7/w28/global popularity + same rank-weighted blend + dedup/top-up) stays identical.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an evaluation join mismatch caused by altering IDs: you lowercase `customer_id`, which can change the hash-like IDs and result in no matches to ground truth. I make the smallest safe fix by normalizing `customer_id` with `strip()` only (no lowercasing/padding), while keeping your candidate generation and blending logic identical. I also strengthen `article_id` handling by reading as string and zero-filling once (already mostly done) and keep the sample submission ordering unchanged. These changes should move the score upward from 0.0 toward your target without changing the recommender/blending core logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an evaluation join mismatch caused by subtly altered `customer_id` values or mis-read dtypes, not with weak ranking. I make the smallest changes to keep IDs byte-identical to the provided files by reading `customer_id` as Python `object` (not pandas `string`) and normalizing with `strip()` only, and I similarly ensure `article_id` is always a 10-char zero-padded string without ever round-tripping through numeric types. I also add one sanity check that `customer_id` values in `transactions_train.csv` are a subset of those in `sample_submission.csv` after normalization (a strong indicator the join work), while keeping your candidate generation, blending, dedup, and top-up logic unchanged. This should move the score up from 0.0 toward your target with minimal risk.'
- What this solution (achieved 0.0212) has done: 'Your 0.0 score is most consistent with the evaluation join failing because `article_id` formatting in the submission doesn’t match the ground-truth type. The H&M metric compares predicted `article_id`s as **strings without leading zeros**, while your pipeline forces **10-digit zero-padded strings**, which can yield zero matches and thus MAP@12 = 0. I keep your exact candidate sources and rank-weighted blending logic, but change the `article_id` canonicalization to the competition-safe format (stringified integer, no leading zeros) and ensure every predicted token uses that same format. This is a minimal, metric-aligned fix that should lift the score upward toward your target without changing the recommender logic.'
- What this solution (achieved 0.01417) has done: 'Your current score (0.0212) is below the target (0.023997), so we should improve ranking slightly without changing the core “recent-per-customer + (7d/28d/global) popularity + rank-weighted blend” logic. The smallest reliable gain for MAP@12 here is to make the “popular” candidate lists recency-weighted (not pure counts), because that typically increases hit-rate in the next-week window while preserving the same candidate sources and blending semantics. Concretely, I replace `value_counts()` for global/7d/28d with a simple exponential time-decay weighted popularity computed from the same transactions (no new features, no new model). I keep your blending, dedup, top-up-to-12, and submission ordering checks unchanged.'
- What this solution (achieved 0.01365) has done: 'We keep your same candidate sources (recent-per-customer + 7d/28d/global time-decayed popularity) and the same rank-weighted blending, but tune the decay half-lives and blend weights slightly toward more recent signals, which typically improves next-week MAP@12 without changing the approach. We also make the ranking deterministic when scores tie by adding a stable secondary sort on `article_id`, reducing noisy small regressions. Finally, we ensure `article_id` canonicalization stays consistent (stringified integer, no leading zeros) and keep the submission ordering identical to `sample_submission.csv`.'
- What this solution (achieved 0.0142) has done: 'Your current score (0.01365) is below the target (0.023997), so we should make a small, metric-aligned improvement without changing your overall “recent-per-customer + (w7/w28/global) time-decayed popularity + rank-weighted blend” approach. The biggest low-risk gain for MAP@12 is usually to emphasize the *most relevant next-week signal*: the customer’s *recent purchases in the last 7–28 days*, rather than all-time recent purchases. I keep your same blending function and candidate sources, but (1) build an additional customer-recency list restricted to the last 28 days and use it as the primary per-customer signal, and (2) slightly rebalance blend weights toward the most recent global lists (w7/w28) while keeping deterministic tie-breaking and the same 12-item top-up logic. This is minimal, stays within your architecture/semantics, and should move the score upward toward the target band.'
- What this solution (achieved 0.01411) has done: 'Your current score (0.0142) is well below the target (0.023997), so we should improve ranking signal while keeping your exact popularity/recency + rank-weighted blending framework. The smallest high-impact adjustment for MAP@12 in this competition is to use *basket co-occurrence* (“customers who bought what you bought also bought…”) as an additional candidate list per customer, because it boosts personalized recall without changing the overall approach. Concretely, we build a lightweight item-to-item co-occurrence table from the last 28 days only (bounded by caps so it runs under the time limit), generate a per-customer “similar items” list from their recent purchases, then blend it as an extra prediction channel with a modest weight and keep your dedup/top-up/ordering checks unchanged. This keeps core semantics (candidate lists + rank-based weighted blend) and should move the score upward toward your target band.'
- What this solution (achieved 0.01412) has done: 'We keep your exact candidate-generation/blending approach, but fix one bug in the co-occurrence builder that can silently degrade personalization: the current code reconstructs `t_dat` for the w28 subset via an index-based lookup into the full `trx`, which is unsafe after filtering/sorting and can attach wrong dates to rows. With correct dates, baskets become accurate, the item-to-item neighbors become more relevant, and MAP@12 should move upward toward your target without changing the overall logic. Additionally, we make the w28 recent-per-customer list deterministic by sorting by both `t_dat` and `article_id` so ties don’t introduce noise across runs. No new models, no new data sources, and the submission format/order checks stay identical.'
- What this solution (achieved 0.01485) has done: 'Your current score (0.01412) is well below the target (0.023997), so we should make a small, low-risk lift without changing your overall “recent-per-customer + (w7/w28/global) time-decayed popularity + item2item + rank-weighted blend” framework. The most direct issue holding MAP@12 back here is that `item2item` candidates are generated but never given enough priority relative to global popularity, and `pred0` uses raw recent items without emphasizing the *latest 7 days* which is closest to the evaluation window. I (1) add a minimal `cust_recent_w7` list (same logic as w28) and use it as the primary per-customer recency signal, and (2) slightly rebalance blend weights toward per-customer channels (recent + cooc) while keeping deterministic tie-breaking, dedup, and the same top-up-to-12 behavior. These are small, metric-aligned changes that preserve your core logic and should move the score upward toward the target band.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import gc



## === cell 1
BASE_DIR_CANDIDATES = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/input",
    "/kaggle/data",
]


def first_existing_path(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


DATA_DIR = first_existing_path(*BASE_DIR_CANDIDATES)
if DATA_DIR is None:
    raise FileNotFoundError("Could not locate Kaggle input data directory.")

transactions_path = first_existing_path(
    os.path.join(DATA_DIR, "transactions_train.csv"),
    os.path.join(
        DATA_DIR,
        "h-and-m-personalized-fashion-recommendations",
        "transactions_train.csv",
    ),
)
sample_sub_path = first_existing_path(
    os.path.join(DATA_DIR, "sample_submission.csv"),
    os.path.join(
        DATA_DIR,
        "h-and-m-personalized-fashion-recommendations",
        "sample_submission.csv",
    ),
)

if transactions_path is None or sample_sub_path is None:
    raise FileNotFoundError(
        f"Missing required files. transactions_path={transactions_path}, sample_sub_path={sample_sub_path}"
    )

print("Using transactions:", transactions_path)
print("Using sample submission:", sample_sub_path)



## === cell 2
dtypes = {
    "t_dat": "object",
    "customer_id": "object",
    "article_id": "object",
}
usecols = ["t_dat", "customer_id", "article_id"]

trx = pd.read_csv(transactions_path, usecols=usecols, dtype=dtypes)
trx["t_dat"] = pd.to_datetime(trx["t_dat"], errors="coerce")

sample_sub = pd.read_csv(sample_sub_path, dtype={"customer_id": "object"})


def norm_cid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def aid_canonical(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.where(~x.isin(["", "nan", "None"]), np.nan)
    x = x.astype("float64")  # safe for numeric-like strings
    x = x.astype("Int64")  # pandas nullable int
    return x.astype(str)


trx["customer_id"] = norm_cid(trx["customer_id"])
sample_sub["customer_id"] = norm_cid(sample_sub["customer_id"])

trx["article_id"] = aid_canonical(trx["article_id"])

sample_sub = sample_sub.reset_index(drop=True)

print(trx.shape, sample_sub.shape)
print(trx.head())
print("sample_sub head:", sample_sub.head())

ss_cids = set(sample_sub["customer_id"].values.tolist())
trx_unique_cids = pd.Index(trx["customer_id"].unique())
in_sample = trx_unique_cids.isin(list(ss_cids)).mean()
print(f"Share of unique trx customer_ids present in sample_submission: {in_sample:.6f}")
if in_sample < 0.95:
    raise ValueError(
        "Most transaction customer_id values are not found in sample_submission after normalization; "
        "this usually indicates an ID parsing/alteration issue that would score ~0."
    )



## === cell 3
last_date = trx["t_dat"].max()
if pd.isna(last_date):
    raise ValueError("Could not parse any dates from t_dat.")

w7_start = last_date - pd.Timedelta(days=7)
w28_start = last_date - pd.Timedelta(days=28)


def top_articles_time_decay(df, topk=200, half_life_days=7.0):
    if df.empty:
        return []
    age_days = (last_date - df["t_dat"]).dt.total_seconds() / (24 * 3600)
    w = np.power(0.5, age_days / float(half_life_days))
    tmp = pd.DataFrame({"article_id": df["article_id"].values, "w": w.values})
    scores = tmp.groupby("article_id", sort=False)["w"].sum()

    scores = scores.reset_index()
    scores = scores.sort_values(
        ["w", "article_id"], ascending=[False, True], kind="mergesort"
    )
    return scores.head(topk)["article_id"].to_list()


top_global = top_articles_time_decay(
    trx[["t_dat", "article_id"]], topk=200, half_life_days=10.0
)

trx_w7 = trx.loc[trx["t_dat"] >= w7_start, ["t_dat", "article_id"]]
top_w7 = top_articles_time_decay(trx_w7, topk=200, half_life_days=3.0)

trx_w28 = trx.loc[trx["t_dat"] >= w28_start, ["t_dat", "article_id"]]
top_w28 = top_articles_time_decay(trx_w28, topk=200, half_life_days=6.0)

print("last_date:", last_date.date())
print("top_global[:5]:", top_global[:5])
print("top_w7[:5]:", top_w7[:5])
print("top_w28[:5]:", top_w28[:5])



## === cell 4
trx_sorted = trx.sort_values(
    ["customer_id", "t_dat", "article_id"],
    ascending=[True, False, True],
    kind="mergesort",
)

cust_recent_all = trx_sorted.groupby("customer_id")["article_id"].apply(
    lambda s: pd.unique(s)[:50]
)

trx_sorted_w7 = trx_sorted.loc[
    trx_sorted["t_dat"] >= w7_start, ["customer_id", "t_dat", "article_id"]
]
cust_recent_w7 = trx_sorted_w7.groupby("customer_id")["article_id"].apply(
    lambda s: pd.unique(s)[:50]
)

trx_sorted_w28 = trx_sorted.loc[
    trx_sorted["t_dat"] >= w28_start, ["customer_id", "t_dat", "article_id"]
]
cust_recent_w28 = trx_sorted_w28.groupby("customer_id")["article_id"].apply(
    lambda s: pd.unique(s)[:50]
)

del trx_sorted
gc.collect()

print("cust_recent_all example:", cust_recent_all.iloc[0][:5])
print("cust_recent_w7 example:", cust_recent_w7.iloc[0][:5])
print("cust_recent_w28 example:", cust_recent_w28.iloc[0][:5])




## === cell 5
def build_item2item_from_w28(
    trx_w28_full: pd.DataFrame,
    top_articles_set: set,
    max_basket_items: int = 12,
    max_neighbors: int = 60,
):
    if trx_w28_full.empty:
        return {}

    df = trx_w28_full.loc[
        trx_w28_full["article_id"].isin(top_articles_set),
        ["customer_id", "t_dat", "article_id"],
    ].copy()
    if df.empty:
        return {}

    df["date"] = df["t_dat"].dt.date
    df = df.sort_values(["customer_id", "date"])
    baskets = df.groupby(["customer_id", "date"])["article_id"].agg(
        lambda s: pd.unique(s)[:max_basket_items]
    )
    if baskets.empty:
        return {}

    pair_counts = {}
    for items in baskets.values:
        items = list(items)
        L = len(items)
        for i in range(L):
            ai = items[i]
            di = pair_counts.get(ai)
            if di is None:
                di = {}
                pair_counts[ai] = di
            for j in range(L):
                if i == j:
                    continue
                aj = items[j]
                di[aj] = di.get(aj, 0) + 1

    item2item = {}
    for a, di in pair_counts.items():
        neigh = sorted(di.items(), key=lambda x: (-x[1], x[0]))[:max_neighbors]
        item2item[a] = [b for b, _ in neigh]
    return item2item


cooc_article_universe = set(top_w28[:200] + top_w7[:200] + top_global[:200])

item2item = build_item2item_from_w28(
    trx_sorted_w28,
    top_articles_set=cooc_article_universe,
    max_basket_items=12,
    max_neighbors=60,
)

del trx_sorted_w28, trx_sorted_w7
gc.collect()

print("Built item2item size:", len(item2item))
if len(item2item) > 0:
    k0 = next(iter(item2item.keys()))
    print("Example item2item:", k0, "->", item2item[k0][:10])



## === cell 6
top_global_fmt = top_global
top_w7_fmt = top_w7
top_w28_fmt = top_w28




## === cell 7
def _dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if not x or x == "nan":
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def cust_blend(dt, W=(1, 1, 1, 1, 1)):
    p0 = "" if pd.isna(dt["prediction0"]) else dt["prediction0"]
    p1 = "" if pd.isna(dt["prediction1"]) else dt["prediction1"]
    p2 = "" if pd.isna(dt["prediction2"]) else dt["prediction2"]
    p3 = "" if pd.isna(dt["prediction3"]) else dt["prediction3"]
    p4 = "" if pd.isna(dt["prediction4"]) else dt["prediction4"]

    recs = [
        str(p0).split(),
        str(p1).split(),
        str(p2).split(),
        str(p3).split(),
        str(p4).split(),
    ]
    scores = {}
    for m, rec in enumerate(recs):
        wm = float(W[m])
        for n, v in enumerate(rec):
            if not v or v == "nan":
                continue
            scores[v] = scores.get(v, 0.0) + wm / (n + 1.0)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))  # deterministic ties
    out = _dedup_keep_order([k for k, _ in ranked])
    return " ".join(out[:12])




## === cell 8
cust_to_recent_w7 = cust_recent_w7.to_dict()
cust_to_recent_w28 = cust_recent_w28.to_dict()
cust_to_recent_all = cust_recent_all.to_dict()


def build_pred0(cid):
    rec = cust_to_recent_w7.get(cid)
    if rec is None or len(rec) == 0:
        rec = cust_to_recent_w28.get(cid)
    if rec is None or len(rec) == 0:
        rec = cust_to_recent_all.get(cid)
    if rec is None or len(rec) == 0:
        return ""
    return " ".join(list(rec)[:50])


def build_pred4(cid):
    base = cust_to_recent_w7.get(cid)
    if base is None or len(base) == 0:
        base = cust_to_recent_w28.get(cid)
    if base is None or len(base) == 0:
        base = cust_to_recent_all.get(cid)
    if base is None or len(base) == 0:
        return ""
    cand = []
    seen = set()
    for a in list(base)[:10]:
        neigh = item2item.get(a)
        if not neigh:
            continue
        for b in neigh:
            if b == a:
                continue
            if b in seen:
                continue
            seen.add(b)
            cand.append(b)
            if len(cand) >= 80:
                break
        if len(cand) >= 80:
            break
    return " ".join(cand)


pred0 = sample_sub["customer_id"].map(build_pred0)
pred4 = sample_sub["customer_id"].map(build_pred4)

pred1 = pd.Series([" ".join(top_w7_fmt[:50])] * len(sample_sub), index=sample_sub.index)
pred2 = pd.Series(
    [" ".join(top_w28_fmt[:50])] * len(sample_sub), index=sample_sub.index
)
pred3 = pd.Series(
    [" ".join(top_global_fmt[:50])] * len(sample_sub), index=sample_sub.index
)

sub0 = pd.DataFrame(
    {
        "customer_id": sample_sub["customer_id"].values,
        "prediction0": pred0.values,
        "prediction1": pred1.values,
        "prediction2": pred2.values,
        "prediction3": pred3.values,
        "prediction4": pred4.values,
    }
)

sub0.head()



## === cell 9
sub0["prediction"] = sub0.apply(cust_blend, W=(1.22, 1.15, 0.88, 0.48, 1.05), axis=1)

global12 = " ".join(top_global_fmt[:12])
pred_tokens = sub0["prediction"].astype(str).str.split()
is_empty = pred_tokens.map(len) == 0
if is_empty.any():
    sub0.loc[is_empty, "prediction"] = global12


def ensure_12(pred_str):
    toks = _dedup_keep_order(str(pred_str).split())
    if len(toks) >= 12:
        return " ".join(toks[:12])
    toks_set = set(toks)
    for a in top_global_fmt:
        if a not in toks_set:
            toks.append(a)
            toks_set.add(a)
        if len(toks) == 12:
            break
    return " ".join(toks[:12])


sub0["prediction"] = sub0["prediction"].map(ensure_12)

submission = sub0[["customer_id", "prediction"]].copy()
submission = submission.reset_index(drop=True)
submission.head()



## === cell 10
if list(submission.columns) != ["customer_id", "prediction"]:
    raise ValueError("Submission has wrong columns.")

if submission["customer_id"].nunique() != len(submission):
    raise ValueError("Duplicate customer_id rows in submission.")

if len(submission) != len(sample_sub):
    raise ValueError(
        f"Submission row count {len(submission)} != sample submission {len(sample_sub)}"
    )

if not submission["customer_id"].equals(sample_sub["customer_id"]):
    raise ValueError(
        "customer_id order/mismatch vs sample_submission (must match exactly)."
    )

expected_cid_len = int(
    pd.Series(sample_sub["customer_id"].astype(str).str.len()).mode().iloc[0]
)
bad_len = submission["customer_id"].astype(str).str.len().ne(expected_cid_len).sum()
if bad_len:
    raise ValueError(
        f"{bad_len} customer_id values have length != expected ({expected_cid_len})."
    )

lens = submission["prediction"].astype(str).str.split().map(len)
if (lens > 12).any():
    raise ValueError("Some rows have more than 12 predictions.")
if (lens < 12).any():
    raise ValueError("Some rows have fewer than 12 predictions after top-up.")

pred_tokens = submission["prediction"].astype(str).str.split()
nonempty = pred_tokens.map(len) > 0
if nonempty.any():
    first_tok = pred_tokens[nonempty].iloc[0][0]
    if not (isinstance(first_tok, str) and first_tok.isdigit()):
        raise ValueError(
            f"Predictions don't look like numeric article_ids. Example token: {first_tok}"
        )

print("All checks passed. Rows:", len(submission))
print("Example customer_id:", submission.iloc[0]["customer_id"])
print("Example prediction:", submission.iloc[0]["prediction"])
print("Expected customer_id length (from sample):", expected_cid_len)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
