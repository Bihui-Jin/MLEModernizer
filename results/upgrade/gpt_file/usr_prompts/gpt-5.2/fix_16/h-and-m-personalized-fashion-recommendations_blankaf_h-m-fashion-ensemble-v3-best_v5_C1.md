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

0.0238254097118143

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your current notebook is an ensembling script that expects several external submission CSVs in `../input/fashions/`, but those files don’t exist in this Kaggle environment, so it fails before producing any `submission.csv`. To keep the core intent (produce predictions) while making minimal, runnable changes, I replace the missing-input ensemble with a standard, fast H&M baseline that generates per-customer recommendations from the training transactions (recent popularity + customer’s recent purchases fallback). This fixes the FileNotFound errors, ensures correct submission formatting and alignment with `sample_submission.csv`, and should yield a reasonable MAP@12 baseline (typically in the right ballpark of your target) without adding heavy models or long training.'
- What this solution (achieved 0.0) has done: 'Your current script is already a valid “recent purchases + global popularity” baseline, so a 0.0 score strongly suggests a submission-format mismatch rather than model quality. I make two minimal, score-relevant fixes: (1) ensure `customer_id` is written as a plain string column (not pandas `string` dtype that can sometimes serialize unexpectedly), and (2) guarantee every row has exactly 12 space-separated `article_id`s by padding with global popular items (some customers can end up with fewer than 12 if `top_global` is too short or overlaps heavily). I also add a couple of lightweight sanity checks on the serialized CSV to catch the kind of issue that can lead to 0.0 even when predictions look fine. Core logic (recent deduped customer history + recent global popularity fallback) remains the same.'
- What this solution (achieved 0.0) has done: 'Your current logic should not score 0.0, so the most likely cause is an ID-format mismatch: H&M `customer_id` must be a 64-hex string, and reading/writing it as pandas `string` can preserve hidden whitespace or NA-like values; if Kaggle can’t match customer_ids, you effectively get 0.0. I make minimal, score-relevant changes to (1) force `customer_id` to a clean, stripped Python `str` consistently in both sample and transactions, (2) build `cust_hist` with those same cleaned IDs so lookups always hit, and (3) add a strict sanity check that all customer_ids are 64-hex before writing. Core recommendation logic (recent unique purchases then global popularity fallback to 12 items) stays the same.'
- What this solution (achieved 0.0) has done: 'Your score of 0.0 strongly indicates Kaggle couldn’t match your `customer_id`s (or predictions) during evaluation, so the smallest score-relevant change is to preserve `customer_id` exactly as provided in `sample_submission.csv` and only use a cleaned copy for lookups against transactions. I keep your core baseline logic (recent unique customer purchases + recent global popularity fallback to 12 items) unchanged, but adjust the ID handling to avoid converting `customer_id` to the literal string `"nan"` and to avoid any accidental whitespace/NA coercion. I also ensure the final submission uses the sample’s original IDs verbatim and add a lightweight check that the output IDs exactly match the sample order/values, which directly prevents the 0.0 failure mode. These changes should move the score up toward (and typically past) your target band without altering the recommendation logic.'
- What this solution (achieved 0.0) has done: 'Your score of 0.0 is most consistent with an ID mismatch (Kaggle can’t match rows to the ground-truth customers), even though your recommender logic is a reasonable baseline. I keep the same “recent unique customer purchases + recent global popularity fallback to 12 items” logic, but make the output `customer_id` *exactly* the values from `sample_submission.csv` (no type conversions, no hidden NA coercion) and ensure the CSV is written with stable quoting/line endings. I also add a strict equality check between the submission `customer_id` column and the sample’s `customer_id` column (not a copied/converted column), because that directly prevents the 0.0 failure mode. These are minimal, score-relevant changes that should move the score up toward your target band without changing the model behavior.'
- What this solution (achieved 0.0) has done: 'Your baseline logic should score well above 0.0, so the most likely cause is that Kaggle is evaluating a malformed submission (often due to quoting/whitespace/BOM issues or a subtle mismatch in the `prediction` strings). I keep your recommender exactly the same, but make the CSV writing more robust (explicit UTF-8, controlled quoting, no extra spaces), and add a final “round-trip” check that the on-disk file matches the in-memory frame byte-for-byte in the critical columns. I also harden the ID key cleaning to avoid any accidental `"nan"`/non-string keys during lookup while still outputting `customer_id` verbatim from `sample_submission.csv`. These minimal changes should move the score from 0.0 up toward (and typically beyond) your target band without changing the recommendation behavior.'
- What this solution (achieved 0.0) has done: 'I fix the KeyError cascade caused by the incorrect merge/rename in the transactions dataframe: after merging `cust_map`, pandas creates `customer_id_x`/`customer_id_y`, so the code referencing `customer_id` and `customer_id_key` breaks. I make the merge explicit with suffixes, select the mapped 64-hex `customer_id` as `customer_id_key`, and keep the submission `customer_id` exactly verbatim from `sample_submission.csv` (to prevent the common 0.0-score ID mismatch). I also remove the unnecessary “short-id” mapping entirely (it’s logically unsafe and can collapse many customers), while keeping the same core recommender logic: per-customer recent unique purchases + recent global popularity fallback to 12 items. Finally, I keep the strict on-disk round-trip submission checks so the notebook always yields a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your logic should score above 0.0, so the most likely cause is an ID mismatch between the `customer_id` values used to build `cust_hist` and the keys used at inference. I make one minimal, score-relevant change: normalize `customer_id_key` in both transactions and sample using the exact same safe string conversion (without introducing `"nan"` or whitespace issues) and avoid any accidental dtype/object inconsistencies by using `.astype("string")` then `.str.strip()` before converting to Python strings. I also tighten the date parsing to avoid `NaT` reducing recent-history coverage, and I keep the output `customer_id` column strictly verbatim from `sample_submission.csv` to prevent evaluation mismatch. Core recommender logic (recent unique customer purchases + global recent popularity fallback to 12) is unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly not model quality but an evaluation-mismatch: many `customer_id_key` values not match between `sample` and `transactions` because `pd.read_csv(..., dtype=object)` can still produce inconsistent string representations (e.g., hidden whitespace/NA coercions), so most customers fall back to global popularity and—worse—some IDs may not match Kaggle’s expected rows. I keep your exact recommender logic (recent unique customer purchases, then global recent popularity padded to 12) but make the customer-key creation deterministic by constructing keys from the raw bytes of the input (no `"nan"` stringification) and normalizing both tables identically. I also make `article_id` formatting robust by reading it as string then zero-padding (avoids any accidental float/int formatting edge cases) while preserving the same ranked lists. Finally, I keep your strict submission round-trip checks to ensure the written CSV matches the sample IDs exactly and every prediction has exactly 12 valid 10-digit article_ids.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with Kaggle not being able to match many (or all) `customer_id` rows at evaluation time due to subtle ID normalization issues, not with the recommender logic itself. I keep your exact recommendation strategy (recent unique customer purchases, padded with recent global popularity to 12) but make one key change: build `customer_id_key` for lookups by *exactly* mirroring how the transactions `customer_id` is represented (strip + lowercase) while still outputting the original `customer_id` verbatim from the sample. I also make `article_id` parsing/formatting fully deterministic by reading it as string and converting via a safe integer cast, which avoids any hidden float/NA edge cases that can cascade into empty/invalid predictions. These are minimal, score-relevant changes aimed at moving the score up toward your target band without changing the core logic.'
- What this solution (achieved 0.0) has done: 'Your current logic should score well above 0.0, so the smallest score-relevant fix is to eliminate any possibility of `customer_id` key mismatches by using the exact `customer_id` strings from `sample_submission.csv` for both lookup and output (no lowercasing/stripping that could desync from transactions). I keep the same core recommender (customer recent unique purchases, padded with recent global popularity to 12) but adjust the ID normalization to be “strip only” and consistent across both tables. I also make `article_id` handling strictly string-to-int with zero-padding at the end to avoid any silent parse issues that could yield empty/invalid predictions. Finally, I keep your strict round-trip checks so the notebook always writes a valid `submission.csv` aligned 1:1 with the sample.'
- What this solution (achieved 0.0) has done: 'Your current baseline should not score 0.0, so the most likely issue is that the `customer_id_key` used for lookup doesn’t match the transactions’ `customer_id` strings (e.g., hidden whitespace/object coercion differences), causing almost everyone to fall back and/or mismatching evaluation rows. I make the keying deterministic by using the exact `customer_id` from `sample_submission.csv` for both lookup and output, and I normalize transaction `customer_id` the same way (strip only) to maximize key matches without altering the recommender logic. I also ensure `customer_id` and `article_id` are read as strings (avoids any edge-case coercions) and keep the same “recent unique purchases then global popularity padded to 12” behavior. These are minimal changes aimed at moving the score up from 0.0 toward your target band by fixing evaluation alignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly caused by a customer_id mismatch at evaluation time, not by the baseline recommender itself. The smallest, most score-relevant change is to avoid generating a separate `customer_id_key` and instead use the exact `customer_id` strings from `sample_submission.csv` as the lookup key *and* the output id, while normalizing transactions `customer_id` in the exact same “strip only” way. I also harden CSV reading/writing to prevent any accidental NA coercion by using `keep_default_na=False` and explicit string dtypes, and I keep your core logic unchanged: recent unique customer purchases, then pad with recent global popularity to 12 items. These fixes should move the score up from 0.0 toward (and typically above) your target band without changing the recommendation strategy.'
- What this solution (achieved 0.0) has done: 'Your baseline recommender logic is fine for a non-zero MAP@12, so the 0.0 score strongly suggests a submission mismatch rather than model quality. I make the smallest score-relevant change: use the exact `customer_id` values from `sample_submission.csv` as the lookup key as well (so transactions are normalized to exactly the same representation), eliminating any possibility that your `customer_id_key` normalization desyncs keys and causes mass fallback/mismatch. I also ensure we never accidentally create `None` keys by using `keep_default_na=False` and a single consistent `strip_only` normalization for both sample and transactions. Everything else (recent unique purchases + recent global popularity padded to 12, same windows and formatting, same CSV checks) stays the same.'

# 9. Code solution

## === cell 0
import os
import csv
import numpy as np
import pandas as pd
import gc



## === cell 1
BASE_DIRS = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/input",
    "/kaggle/data/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for b in BASE_DIRS:
        p = os.path.join(b, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {filename} in any of: {BASE_DIRS}")


transactions_path = find_file("transactions_train.csv")
sample_path = find_file("sample_submission.csv")

transactions_path, sample_path




## === cell 2
def strip_only(series: pd.Series) -> pd.Series:
    s = series.astype("object")
    s = s.where(pd.notna(s), None)
    s = s.map(lambda x: x.strip() if isinstance(x, str) else None)
    s = s.map(lambda x: x if (x is not None and x != "") else None)
    return pd.Series(s, index=series.index, dtype="object")


sample = pd.read_csv(
    sample_path,
    usecols=["customer_id"],
    dtype={"customer_id": "object"},
    keep_default_na=False,
)
sample_customer_id_verbatim = sample["customer_id"].astype("object")
sample["customer_id_key"] = strip_only(sample_customer_id_verbatim)

usecols = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
dtypes = {
    "t_dat": "object",
    "customer_id": "object",
    "article_id": "object",
    "sales_channel_id": "int8",
    "price": "float32",
}
trans = pd.read_csv(
    transactions_path,
    usecols=usecols,
    dtype=dtypes,
    keep_default_na=False,
)

trans["t_dat"] = pd.to_datetime(trans["t_dat"], format="%Y-%m-%d", errors="coerce")

trans["customer_id_key"] = strip_only(trans["customer_id"])

aid = trans["article_id"].astype("object").where(pd.notna(trans["article_id"]), None)
aid = aid.map(lambda x: x.strip() if isinstance(x, str) else None)
aid = aid.map(
    lambda x: "".join(ch for ch in x if ch.isdigit()) if isinstance(x, str) else None
)
trans["article_id"] = pd.to_numeric(aid, errors="coerce").astype("Int64")

trans = trans.loc[
    trans["customer_id_key"].notna()
    & trans["t_dat"].notna()
    & trans["article_id"].notna()
].copy()
trans["article_id"] = trans["article_id"].astype("int64")

sample.shape, trans.shape



## === cell 3
max_date = trans["t_dat"].max()
recent_days = 14
recent_start = max_date - pd.Timedelta(days=recent_days)

recent = trans.loc[trans["t_dat"] >= recent_start, ["article_id"]]

top_global = (
    recent["article_id"].value_counts().head(500).index.astype("int64").tolist()
)

len(top_global), top_global[:5], max_date, recent_start



## === cell 4
trans_sorted = trans.sort_values(["customer_id_key", "t_dat"], ascending=[True, False])

cust_hist = (
    trans_sorted.drop_duplicates(["customer_id_key", "article_id"], keep="first")
    .groupby("customer_id_key")["article_id"]
    .apply(lambda s: s.head(50).astype("int64").tolist())
)

del trans_sorted
gc.collect()

cust_hist.head()



## === cell 5
top_global_str = [f"{int(a):010d}" for a in top_global]


def make_pred(cid_key: str) -> str:
    items = cust_hist.get(cid_key, None)
    rec = []
    seen = set()

    if items is not None:
        for a in items:
            s = f"{int(a):010d}"
            if s not in seen:
                seen.add(s)
                rec.append(s)
            if len(rec) >= 12:
                return " ".join(rec[:12])

    for s in top_global_str:
        if s not in seen:
            seen.add(s)
            rec.append(s)
        if len(rec) >= 12:
            break

    if len(rec) < 12:
        for s in top_global_str:
            if len(rec) >= 12:
                break
            if s not in seen:
                seen.add(s)
                rec.append(s)

    return " ".join(rec[:12])


sample_ids_key = sample["customer_id_key"].tolist()
preds = [make_pred(cidk) for cidk in sample_ids_key]

submission = pd.DataFrame(
    {
        "customer_id": sample_customer_id_verbatim.values,  # must be verbatim for evaluation matching
        "prediction": np.array(preds, dtype=object),
    }
)

submission.head(), submission.shape



## === cell 6
assert submission["customer_id"].isna().sum() == 0
assert submission["prediction"].isna().sum() == 0
assert submission.shape[0] == sample.shape[0]
assert (submission["prediction"].astype(str).str.split().str.len() == 12).all()
assert submission["prediction"].astype(str).str.contains(r"^\d{10}( \d{10}){11}$").all()

assert np.array_equal(
    submission["customer_id"].astype("object").values,
    sample_customer_id_verbatim.values,
)

submission["prediction"] = submission["prediction"].astype(str).str.strip()

submission.to_csv(
    "submission.csv",
    index=False,
    encoding="utf-8",
    lineterminator="\n",
    quoting=csv.QUOTE_MINIMAL,
)

chk = pd.read_csv(
    "submission.csv",
    dtype={"customer_id": "object", "prediction": "object"},
    keep_default_na=False,
)
assert list(chk.columns) == ["customer_id", "prediction"]
assert chk.shape[0] == sample.shape[0]
assert np.array_equal(
    chk["customer_id"].astype("object").values, sample_customer_id_verbatim.values
)
assert (chk["prediction"].astype(str).str.strip().str.split().str.len() == 12).all()
assert (
    chk["prediction"]
    .astype(str)
    .str.strip()
    .str.contains(r"^\d{10}( \d{10}){11}$")
    .all()
)

print("Wrote submission.csv with shape:", chk.shape)
print(chk.head())



## === cell 7
for name in [
    "trans",
    "recent",
    "cust_hist",
    "preds",
    "submission",
    "sample",
    "top_global",
    "top_global_str",
    "chk",
    "sample_ids_key",
    "sample_customer_id_verbatim",
]:
    if name in globals():
        del globals()[name]
gc.collect()
