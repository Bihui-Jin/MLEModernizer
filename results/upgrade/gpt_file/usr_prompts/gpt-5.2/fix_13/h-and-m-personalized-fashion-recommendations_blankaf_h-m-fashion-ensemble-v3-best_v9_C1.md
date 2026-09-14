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

0.0239069984319075

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The current notebook is an ensemble/blending script that depends on several pre-made submission files that don’t exist in this environment, which is why it fails immediately. To keep core intent (produce a submission) with minimal changes, I’m replacing those missing-input reads with a lightweight, fully in-notebook baseline that generates valid predictions from the provided `transactions_train.csv`: a “last-week popular items” list and per-customer recent-history fallback. This runs within the Kaggle time limit by reading only needed columns and only the last ~30 days of transactions. The output is a correctly formatted `submission.csv` with `customer_id,prediction` for all customers in `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is almost certainly from an invalid submission format: H&M requires `customer_id` to be the 64-bit hex string from `sample_submission.csv`, but your code force-casts it to `str`, which turns those hex IDs into the literal text `"0x...."` and no longer matches the expected IDs, yielding effectively zero credit. I keep your core recommender logic intact and make the smallest fix: read and preserve `customer_id` exactly as provided (as a string column), and avoid any transformation that could alter it. I also make the date scan and filtering use consistent string dates to avoid subtle comparison issues while keeping runtime similar. The output remain `submission.csv` with the correct columns and 12-or-fewer article IDs per row.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a “submission not matching customers” issue: you sort `sample_submission` by `customer_id`, which can break the expected row alignment if Kaggle’s evaluator assumes the exact same order as the provided sample file. I make the smallest change: keep `sample_submission.csv` in its original order (no sorting), while preserving `customer_id` as a string exactly as provided. Everything else (your recent-history + last-30-days popularity logic and formatting) stays the same, so runtime and semantics remain stable but the submission should now be properly credited and move the score upward toward the target. I also add a strict check that our output customer_id order exactly matches the sample file to prevent silent 0.0s.'
- What this solution (achieved 0.0) has done: 'Your current pipeline already produces a valid submission, so the 0.0 score is most likely coming from a subtle formatting mismatch: H&M `article_id` predictions must be *10-digit zero-padded strings*, but your `top_pred` and per-customer recs can still include non-padded/duplicate/too-short tokens if anything slips through. I add a strict “sanitize predictions” step to guarantee exactly up to 12 unique, 10-digit numeric article IDs separated by single spaces, and I also ensure the `prediction` column is plain Python `str` (not pandas `string` dtype) to avoid any Kaggle parser edge cases. These are minimal changes that keep your exact core logic (last-30-days popularity + per-customer recent unique items) but should move your score up from 0.0 toward the target. I keep the sample order assertion so we can’t silently generate a misaligned submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with the evaluator not being able to match your `customer_id`s due to pandas `string` dtype / NA semantics and potential hidden whitespace; so the smallest score-moving fix is to preserve IDs exactly as plain Python strings and normalize them safely (strip only). Next, your per-customer “recent” list is built by processing chunks independently, which can miss true recency across chunks; we keep the same core heuristic (recent unique + popularity) but make the “recent” aggregation globally correct by tracking each customer’s latest date per article within the window, then selecting the 12 most recent at the end. Finally, we keep your popularity fallback, strict 10-digit article formatting, and the sample-order assertion to prevent another silent 0.0.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with `article_id` being parsed/written as integers, which strips leading zeros and makes the predicted IDs not match ground-truth IDs (H&M article_ids are 10-digit strings). I keep your exact heuristic (last-30-days per-customer recency + global popularity fallback) but change the transaction read to treat `article_id` as a string and preserve it through counting and recency aggregation. I also ensure popularity counting and sorting are done on the string IDs (still deterministic), and keep your existing strict sanitization and sample-order assertions so we can’t silently produce another invalid submission. These are minimal changes aimed at moving the score up toward the target without changing the approach.'
- What this solution (achieved 0.0) has done: 'Your current logic is a solid “recent-per-customer + global popularity fallback” baseline, so a 0.0 is most likely coming from an ID mismatch rather than model quality. The smallest score-moving fix is to preserve `customer_id` and `article_id` exactly as strings from the CSV reader (avoid any parsing artifacts) and to ensure `top_pred` itself is also sanitized/unique/<=12 just like per-customer predictions. I also make the `max_date` scan robust by parsing to datetime once (still the same max-date logic), which prevents subtle string-comparison issues that could accidentally filter out almost all rows and degrade predictions. These changes keep the exact heuristic/semantics but should move the score up from 0.0 toward your target by ensuring the evaluator can match customers and items.'
- What this solution (achieved 0.0) has done: 'Your current heuristic is fine for a quick baseline, so a 0.0 strongly suggests the evaluator can’t match your `article_id`s (leading zeros lost) or your time-window filter is accidentally emptying the transaction set. I make the smallest safe changes to (1) read `article_id` as an integer and then format it to a strict 10-digit string (this avoids any latent parse/whitespace issues while guaranteeing correct IDs), and (2) compare dates using real datetimes inside the chunk loop (instead of string comparisons) so the last-30-days filter can’t silently drop almost everything. Everything else (recent-per-customer + global popularity fallback, 12 items max, exact sample order) stays the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely caused by the *date-window filter silently dropping almost all rows* due to a boolean mask built from a Series (`tdt`) that no longer aligns after `chunk[...]` filtering, plus doing the datetime parse twice. I keep your exact core heuristic (last-30-days per-customer recency + global popularity fallback) but fix the filtering to be alignment-safe by adding a parsed datetime column and filtering on it. I also make the max-date scan and the window filtering use the same datetime parsing approach to prevent accidental empty windows. These minimal fixes should move the score up from 0.0 toward your target while keeping output format and ID handling intact.'
- What this solution (achieved 0.0) has done: 'Your current logic is a reasonable baseline, so a 0.0 MAP@12 is almost certainly coming from an ID mismatch rather than model quality. The smallest safe fix is to stop reading `article_id` as `int64` (which can irreversibly lose leading zeros) and instead read it as an object/string from the start, then sanitize to strict 10-digit numeric strings. I also make the date window explicitly bounded on both ends (`min_dt <= t_dat <= max_dt`) using one parsed datetime column to avoid any accidental inclusion/exclusion due to parsing quirks. These changes preserve the same “recent-per-customer + global popularity fallback” core heuristic, but should move the score up from 0.0 toward your target by ensuring Kaggle can correctly match predicted `article_id`s.'
- What this solution (achieved 0.0) has done: 'Your current heuristic is fine for a baseline, so a 0.0 almost certainly comes from an evaluator mismatch rather than “bad recommendations.” The smallest score-moving fix is to ensure `customer_id` is preserved exactly as in `sample_submission.csv` (no `.astype(str)` round-trips) and to strictly build the output in the exact same order as the sample via a left merge (this avoids any silent index/order issues). In addition, we keep your exact “last-30-days recency + global popularity fallback” logic, but we make the per-customer “most recent” sort use real datetimes (not string comparisons) to avoid subtle ordering mistakes. Finally, we keep your strict article-id sanitization and write `submission.csv` with the required schema.'
- What this solution (achieved 0.0) has done: 'Your current heuristic is reasonable for a baseline, so a 0.0 is most consistent with the evaluator not matching your predicted `article_id`s (commonly due to leading zeros/format inconsistencies) and/or too few effective candidates. I keep your exact “last-30-days per-customer recency + global popularity fallback” core logic, but make one score-moving, low-risk improvement: compute popularity with a simple recency decay (still the same popularity concept, just weighted by date within the same 30-day window) so the top-12 better matches what’s bought “next week.” I also guarantee `top_pred` is always 12 items (pad deterministically if needed) and make the per-customer list strictly “most recent first” using the same datetime column already computed. These are minimal changes that should move the score up from 0.0 toward your target without changing the overall approach or output schema.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import gc



## === cell 1
DATA_DIRS = [
    "/kaggle/input/h-and-m-personalized-fashion-recommendations",
    "/kaggle/data/h-and-m-personalized-fashion-recommendations",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for d in DATA_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    for d in DATA_DIRS:
        p = os.path.join(d, "h-and-m-personalized-fashion-recommendations", filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in known DATA_DIRS: {DATA_DIRS}"
    )


transactions_path = find_file("transactions_train.csv")
sample_path = find_file("sample_submission.csv")

transactions_path, sample_path



## === cell 2
sample = pd.read_csv(
    sample_path, usecols=["customer_id"], dtype={"customer_id": "object"}
)
sample["customer_id"] = (
    sample["customer_id"]
    .astype("object")
    .map(lambda x: x.strip() if isinstance(x, str) else x)
)
sample.head()



## === cell 3
usecols = ["t_dat", "customer_id", "article_id"]

max_dt = None
for chunk in pd.read_csv(
    transactions_path, usecols=["t_dat"], dtype={"t_dat": "object"}, chunksize=2_000_000
):
    dt = pd.to_datetime(chunk["t_dat"], errors="coerce")
    cmax = dt.max()
    if pd.notna(cmax) and (max_dt is None or cmax > max_dt):
        max_dt = cmax

max_date = max_dt.strftime("%Y-%m-%d")
max_date



## === cell 4
DAYS = 30
max_dt = pd.to_datetime(max_date)
min_dt = max_dt - pd.Timedelta(days=DAYS)

cust_article_last = {}  # customer_id -> dict(article_id_str -> last_datetime)

pop_scores = {}  # article_id_str -> float score

dtypes = {"t_dat": "object", "customer_id": "object", "article_id": "object"}

for chunk in pd.read_csv(
    transactions_path, usecols=usecols, dtype=dtypes, chunksize=2_000_000
):
    chunk["customer_id"] = (
        chunk["customer_id"]
        .astype("object")
        .map(lambda x: x.strip() if isinstance(x, str) else x)
    )

    chunk["_t_dat_dt"] = pd.to_datetime(chunk["t_dat"], errors="coerce")
    chunk = chunk[
        (chunk["_t_dat_dt"] >= min_dt) & (chunk["_t_dat_dt"] <= max_dt)
    ].copy()
    if chunk.empty:
        continue

    chunk["article_id"] = (
        chunk["article_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )

    days_ago = (max_dt - chunk["_t_dat_dt"]).dt.days.astype("int32")
    weights = 1.0 / (1.0 + days_ago.to_numpy(dtype=np.float64))
    aids = chunk["article_id"].to_numpy()

    tmp = pd.DataFrame({"article_id": aids, "_w": weights})
    ws = tmp.groupby("article_id", sort=False)["_w"].sum()
    for aid, sc in ws.items():
        pop_scores[aid] = pop_scores.get(aid, 0.0) + float(sc)

    last_in_chunk = (
        chunk.groupby(["customer_id", "article_id"], sort=False)["_t_dat_dt"]
        .max()
        .reset_index()
    )

    for cid, grp in last_in_chunk.groupby("customer_id", sort=False):
        d = cust_article_last.get(cid)
        if d is None:
            d = {}
            cust_article_last[cid] = d
        for aid, dtv in zip(grp["article_id"].to_numpy(), grp["_t_dat_dt"].to_numpy()):
            aid = str(aid)
            prev = d.get(aid)
            if prev is None or dtv > prev:
                d[aid] = dtv

    del last_in_chunk, tmp, ws, chunk
    gc.collect()

len(pop_scores), len(cust_article_last)




## === cell 5
def _sanitize_pred_tokens(tokens):
    out = []
    seen = set()
    for t in tokens:
        if t is None:
            continue
        t = str(t).strip()
        if not t:
            continue
        if not t.isdigit():
            continue
        t = t.zfill(10)
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= 12:
            break
    return out


top_articles = [
    aid for aid, _ in sorted(pop_scores.items(), key=lambda x: (-x[1], x[0]))
]
top_articles_str = _sanitize_pred_tokens(top_articles)

if len(top_articles_str) < 12:
    pass

top_pred = " ".join(top_articles_str)
top_pred




## === cell 6
def make_pred(cid) -> str:
    d = cust_article_last.get(cid)
    if not d:
        return top_pred

    recent_aids = [
        aid
        for aid, _ in sorted(d.items(), key=lambda x: (x[1], x[0]), reverse=True)[:12]
    ]
    filled = _sanitize_pred_tokens(recent_aids + top_articles_str)
    return " ".join(filled)


sub = sample.copy()
sub["prediction"] = sub["customer_id"].map(make_pred)
sub["prediction"] = sub["prediction"].astype(str)
sub.head()



## === cell 7
sample_ids = sample["customer_id"]

assert sub.shape[0] == sample_ids.shape[0]
assert sub["customer_id"].equals(sample_ids)

assert sub.shape[0] > 0
assert list(sub.columns) == ["customer_id", "prediction"]
assert sub["prediction"].isna().sum() == 0
assert (sub["prediction"].str.split().map(len) <= 12).all()
assert (
    sub["prediction"]
    .str.split()
    .map(lambda xs: all((len(x) == 10 and x.isdigit()) for x in xs))
    .all()
)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
out_path
