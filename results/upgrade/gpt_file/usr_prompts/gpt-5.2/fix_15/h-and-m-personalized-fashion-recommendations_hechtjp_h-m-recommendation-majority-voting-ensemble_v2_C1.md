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

cudf-cu12==25.2.2
cudf-polars-cu12==25.6.0
dask-cudf-cu12==25.2.2
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
libcudf-cu12==25.2.2
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pylibcudf-cu12==25.2.2
seaborn==0.12.2
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

0.0227384268563794

# 6. Current score

0.01416

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The main blocker is that `cudf` fails to import due to an insufficient CUDA driver, so the notebook never reaches submission writing; I remove GPU dependencies and use `pandas` instead. Your current code also references several `../input/...` notebook submissions that are not present in this environment, so I switch to loading the provided `sample_submission.csv` and `transactions_train.csv` and generate five simple rule-based prediction strings to preserve the “5 submissions → ensemble by vote” core logic. I keep the ensembling logic (concatenate predictions, count frequency, take top-12) the same, but make it robust to missing values and ensure the final CSV has exactly `customer_id,prediction`. This run end-to-end and produce `submission.csv` in the working directory.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an invalid/empty submission field rather than a weak model, so the smallest safe improvement is to harden submission correctness and ensure every row has exactly 12 space-separated 10-digit article_ids. I keep your five-rule generation and “vote by frequency” ensembling logic unchanged, but fix two edge cases: (1) `select_topn` currently treats `""` as a token after `str(x).split()` on `nan/empty`, and (2) ties in `Counter.most_common()` can be unstable; we add a deterministic tie-breaker using global popularity so the output is consistent and typically slightly better. Finally, I add a strict post-check that fills/repairs any row not having 12 items with a popularity-based fallback, ensuring Kaggle always scores it.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a submission validity mismatch rather than weak ranking logic, so the smallest score-improving change is to guarantee the `customer_id` format matches Kaggle’s expected 16-hex-character IDs and that predictions are always 12 space-separated 10-digit article IDs. I keep your five-rule generation and frequency-vote ensembling identical, but fix dtype/format handling by reading `customer_id` as plain string (not pandas `"string"` which can introduce `<NA>`), normalizing customer IDs to lowercase, and preserving the exact order from `sample_submission.csv`. Finally, I harden the final submission post-check to ensure every row has exactly 12 valid tokens (digits-only, length 10), falling back deterministically to popularity if not.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 strongly suggests Kaggle is receiving an invalid submission (most commonly: wrong `customer_id` formatting relative to the competition’s expected hashed IDs, even if the CSV “looks” correct locally). I keep your exact 5-rule generation and frequency-vote ensembling unchanged, but stop lowercasing/modifying `customer_id` and instead preserve the exact `customer_id` strings and row order from `sample_submission.csv`. I also ensure joins/lookup keys use the original `customer_id` format by reading transactions with `customer_id` as string without normalization. Finally, I keep your strict “always 12 valid 10-digit article_ids” enforcement and validation so the file remains valid and scores > 0.'
- What this solution (achieved 0.01151) has done: 'Your current 0.0 score is most consistent with Kaggle treating the submission as effectively “no correct matches”, which often happens when `article_id` formatting doesn’t match the competition’s expected IDs. The smallest score-improving change is to stop zero-padding `article_id` tokens to 10 digits and instead emit them exactly as in `transactions_train.csv` (no leading zeros), while keeping your exact five-rule generation and frequency-vote ensembling logic unchanged. I also keep the strict “always 12 tokens” enforcement, but change the validator to accept digit tokens of variable length (since Kaggle accepts the original `article_id` strings). These changes should move the score upward toward your target without altering the overall approach.'
- What this solution (achieved 0.01061) has done: 'Your current approach is a simple “5 rule-based recommenders + frequency-vote ensemble,” and the score gap suggests it’s under-ranking the true next-week items rather than producing invalid submissions. To move the MAP@12 upward with minimal logic change, I keep all five generators and the same voting, but add a tiny, metric-aligned boost: include a per-customer “repurchase” list (top items for that customer across full history) as an additional low-risk signal inside the existing mode 1 and mode 3 lists. I also extend the “recent” window from 7 to 14 days for the recent-top lists (still a simple recency heuristic) and increase candidate pool sizes a bit so the vote has more relevant options while still truncating to 12 outputs. These changes typically improve recall@12 and thus MAP@12 without changing the overall architecture or submission semantics.'
- What this solution (achieved 0.01078) has done: 'You’re currently below the target, so we make the smallest legitimate changes that tend to increase MAP@12 without changing the overall “5 rule-based recommenders + frequency-vote ensemble” core logic. The biggest low-risk gain for H&M usually comes from emphasizing “last items bought” and “repeat buys” more strongly than global popularity, so I only adjust the candidate composition and the vote tie-break to be more customer-centric (without adding new models or changing the ensembling method). Specifically, we (1) add a tiny “weekend” popularity list as an extra fallback signal, (2) add a very recent (last 3 days) per-customer tail to modes 1/3, and (3) slightly change the deterministic tie-break order in `select_topn` to prefer recent/global popularity before lexicographic. These tweaks generally improve recall@12 for true next-week items while keeping the same overall pipeline and producing a valid `submission.csv`.'
- What this solution (achieved 0.0117) has done: 'The timeout is dominated by heavy Pandas groupby/apply with Python lambdas over 31.5M rows, repeated sorting/copying of large frames, and per-customer Python loops plus a `progress_apply` that repeatedly builds `Counter` objects from long strings. I keep the exact same heuristic logic and outputs, but replace the slow groupby/apply lambdas with vectorized/cached operations: compute “last N purchases” via a stable sort + `cumcount`, and compute per-customer top-N frequent items via `groupby.size()` then rank within customer (no Python lambdas). I also avoid copying large slices, minimize intermediate DataFrames, and speed up the final `select_topn` step by parsing tokens and counting with a small dict (equivalent to `Counter`) and pre-binding rank dict lookups. These changes are provably equivalent in semantics (same selection criteria, same tie-breaking), but cut the major Python-level overhead and should fit within 600 seconds.'
- What this solution (achieved 0.01416) has done: 'We’re currently below the target (0.0117 vs 0.02274), so the smallest likely MAP@12 lift without changing your overall “5 rule-based recommenders + vote-by-frequency ensemble” core logic is to (1) fix a bug in `_last_n_purchases_map_with_date` usage (it accidentally drops `t_dat` before sorting in your helper path), and (2) make the voting stage slightly more metric-aligned by weighting customer-specific modes (1/2/3) higher than global-only modes (4/5) while keeping the same exact “count → tie-break → top-12” semantics. This preserves your architecture and heuristics, but nudges the final ranking toward items that are more likely relevant per customer, typically improving MAP@12. I also keep submission correctness checks intact and ensure runtime stays within the limit by avoiding extra heavy groupbys.'
- What this solution (achieved 0.01448) has done: 'We’re currently below the target (0.01416 vs 0.02274; higher is better), so the smallest safe move is to improve ranking quality without changing your overall “5 rule-based recommenders + weighted frequency-vote ensemble” design. I keep your five modes and voting approach intact, but (1) fix a real bug where your per-customer top-24 and top-50 lists are accidentally identical to top-12 due to reusing the same cumcount without offset, and (2) make the voting tie-break slightly more metric-aligned by preferring very-recent popularity as the first deterministic tie-break (still deterministic, still the same count→tie-break→top-12 semantics). These changes are minimal, fully legitimate, and should increase MAP@12 toward the target by improving candidate diversity/recall and resolving more ties in favor of items more likely to be bought next week. The script still runs end-to-end and writes a valid `submission.csv` with exactly 12 space-separated article_ids per customer in sample submission order.'
- What this solution (achieved 0.01388) has done: 'We’re below the target (0.01448 vs 0.02274; higher is better), so the smallest safe improvement is to make your existing vote-ensemble more MAP@12-aligned without changing the 5-mode heuristic generators. I keep all five modes and the same “weighted frequency vote → deterministic tie-break → top-12” core logic, but adjust two low-risk ranking details: (1) slightly upweight the two customer-history-based lists (mode2 and mode3) to improve per-customer relevance, and (2) change the tie-break order to prefer customer-centric recency/frequency ranks before global/popularity (still deterministic). Finally, I make `select_topn_weighted` explicitly deduplicate the final top-12 (defensive; shouldn’t change most rows, but avoids edge-case duplicates hurting MAP@12).'
- What this solution (achieved 0.01415) has done: 'We’re below the target (0.01388 vs 0.02274, higher is better), so we make the smallest changes that tend to increase MAP@12 without changing your 5-mode heuristic generators or the overall “weighted frequency vote → deterministic tie-break → top-12” ensemble. The main issue is that the current vote uses repeated-token multiplication, which effectively discretizes weights and can over-amplify long lists; we keep the same weighting idea but implement it as true weighted counts (same semantics, less noise) and add a tiny, metric-aligned recency boost within the same sorting key. We also slightly rebalance weights to favor the two customer-history modes (1 and 3) a bit more than mode2, since next-week purchases in H&M are typically driven by very recent customer behavior. Finally, we keep the strict “always 12 valid article_id tokens” enforcement and preserve exact `customer_id` order from `sample_submission.csv`.'
- What this solution (achieved 0.01416) has done: 'We’re below the target (0.01415 vs 0.02274), so the smallest likely gain is to improve candidate recall and ranking while keeping your exact “5 heuristic modes → weighted frequency vote → deterministic tie-break → top-12” core logic unchanged. I (1) add one extra low-cost, legitimate signal: “recently trending vs baseline” (lift) as an additional global list used only inside the existing mode4/mode5 candidates and as a late tie-break, and (2) very slightly rebalance mode weights to favor the two most customer-specific modes (1 and 3) without changing any generator structure. These changes typically lift MAP@12 by improving diversity and surfacing items that spike in the final period (common in H&M), while still producing the same valid submission format and staying within runtime constraints.'

# 9. Code solution

## === cell 0
import sys, warnings, time, os, copy, gc, re, random, pickle

warnings.filterwarnings("ignore")

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

try:
    from pandas import json_normalize  # noqa: F401
except Exception:
    from pandas.io.json import json_normalize  # type: ignore # noqa: F401

from pprint import pprint
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()
from collections import Counter



## === cell 1
DEBUG = False



## === cell 2
BASE_PATH = Path("/kaggle/data")
if not BASE_PATH.exists():
    BASE_PATH = Path("/kaggle/input")  # fallback (some Kaggle images use /kaggle/input)

TX_PATH = BASE_PATH / "transactions_train.csv"
SAMPLE_PATH = BASE_PATH / "sample_submission.csv"

assert TX_PATH.exists(), f"transactions_train.csv not found at {TX_PATH}"
assert SAMPLE_PATH.exists(), f"sample_submission.csv not found at {SAMPLE_PATH}"




## === cell 3
def display_df(df, head=3):
    print(f"The shape of df is {df.shape}.\n")
    display(df.head(head))




## === cell 4
dfSample = pd.read_csv(SAMPLE_PATH, dtype={"customer_id": "object"})
dfSample["customer_id"] = dfSample["customer_id"].astype(str).str.strip()
display_df(dfSample, head=3)



## === cell 5
usecols = ["t_dat", "customer_id", "article_id", "sales_channel_id"]
dfTx = pd.read_csv(
    TX_PATH,
    usecols=usecols,
    dtype={
        "t_dat": "object",
        "customer_id": "object",
        "article_id": "int32",
        "sales_channel_id": "int8",
    },
)
dfTx["customer_id"] = dfTx["customer_id"].astype(str).str.strip()
display_df(dfTx, head=3)



## === cell 6
dfTx["t_dat"] = pd.to_datetime(dfTx["t_dat"], errors="coerce")
max_date = dfTx["t_dat"].max()

recent_days = 14
very_recent_days = 3
recent_start = max_date - pd.Timedelta(days=recent_days)
very_recent_start = max_date - pd.Timedelta(days=very_recent_days)

TOPK = 500

global_top = dfTx["article_id"].value_counts().head(TOPK).index.astype("int32").tolist()

_recent_mask = dfTx["t_dat"].values >= np.datetime64(recent_start)
_very_recent_mask = dfTx["t_dat"].values >= np.datetime64(very_recent_start)

recent_top = (
    dfTx.loc[_recent_mask, "article_id"]
    .value_counts()
    .head(TOPK)
    .index.astype("int32")
    .tolist()
)

very_recent_top = (
    dfTx.loc[_very_recent_mask, "article_id"]
    .value_counts()
    .head(TOPK)
    .index.astype("int32")
    .tolist()
)

recent_top_ch1 = (
    dfTx.loc[_recent_mask & (dfTx["sales_channel_id"].values == 1), "article_id"]
    .value_counts()
    .head(TOPK)
    .index.astype("int32")
    .tolist()
)
recent_top_ch2 = (
    dfTx.loc[_recent_mask & (dfTx["sales_channel_id"].values == 2), "article_id"]
    .value_counts()
    .head(TOPK)
    .index.astype("int32")
    .tolist()
)

_weekend_mask = dfTx["t_dat"].dt.dayofweek.values >= 5
weekend_top = (
    dfTx.loc[_weekend_mask, "article_id"]
    .value_counts()
    .head(TOPK)
    .index.astype("int32")
    .tolist()
)

_prev_start = recent_start - pd.Timedelta(days=recent_days)
_prev_mask = (dfTx["t_dat"].values >= np.datetime64(_prev_start)) & (
    dfTx["t_dat"].values < np.datetime64(recent_start)
)

_prev_counts = dfTx.loc[_prev_mask, "article_id"].value_counts()
_recent_counts = dfTx.loc[_recent_mask, "article_id"].value_counts()

_union_ids = pd.Index(_recent_counts.head(TOPK * 3).index).union(
    _prev_counts.head(TOPK * 3).index
)
_recent_u = _recent_counts.reindex(_union_ids, fill_value=0).astype("float32")
_prev_u = _prev_counts.reindex(_union_ids, fill_value=0).astype("float32")

_lift = (_recent_u + 1.0) / (_prev_u + 1.0)
_trending_df = pd.DataFrame(
    {
        "article_id": _union_ids.astype("int32"),
        "lift": _lift.values,
        "rc": _recent_u.values,
    }
)
_trending_df.sort_values(
    ["lift", "rc", "article_id"],
    ascending=[False, False, True],
    kind="mergesort",
    inplace=True,
)
trending_top = _trending_df["article_id"].head(TOPK).tolist()
del _prev_counts, _recent_counts, _union_ids, _recent_u, _prev_u, _lift, _trending_df
gc.collect()


def _last_n_purchases_map_with_date(frame: pd.DataFrame, n: int) -> dict:
    if "t_dat" not in frame.columns:
        frame = frame.copy()
        frame["t_dat"] = pd.Timestamp("1970-01-01")
    srt = frame.sort_values(
        ["customer_id", "t_dat"], kind="mergesort", ignore_index=True
    )
    cnt = srt.groupby("customer_id", sort=False).cumcount(ascending=False)
    sel = srt.loc[cnt < n, ["customer_id", "t_dat", "article_id"]]
    sel = sel.sort_values(["customer_id", "t_dat"], kind="mergesort", ignore_index=True)
    return sel.groupby("customer_id", sort=False)["article_id"].agg(list).to_dict()


last_purchases = _last_n_purchases_map_with_date(
    dfTx[["customer_id", "t_dat", "article_id"]], 12
)
recent_purchases = _last_n_purchases_map_with_date(
    dfTx.loc[_recent_mask, ["customer_id", "t_dat", "article_id"]], 12
)
very_recent_purchases = _last_n_purchases_map_with_date(
    dfTx.loc[_very_recent_mask, ["customer_id", "t_dat", "article_id"]], 12
)

cust_item_cnt = (
    dfTx.groupby(["customer_id", "article_id"], sort=False)
    .size()
    .rename("cnt")
    .reset_index()
)
cust_item_cnt.sort_values(
    ["customer_id", "cnt", "article_id"],
    ascending=[True, False, True],
    kind="mergesort",
    inplace=True,
    ignore_index=True,
)

cust_item_cnt["_rank"] = cust_item_cnt.groupby("customer_id", sort=False).cumcount()

freq_purchases = (
    cust_item_cnt.loc[cust_item_cnt["_rank"] < 12]
    .groupby("customer_id", sort=False)["article_id"]
    .agg(lambda x: x.astype("int32").tolist())
    .to_dict()
)

repurchase_top = (
    cust_item_cnt.loc[cust_item_cnt["_rank"] < 24]
    .groupby("customer_id", sort=False)["article_id"]
    .agg(lambda x: x.astype("int32").tolist())
    .to_dict()
)

cust_top50 = (
    cust_item_cnt.loc[cust_item_cnt["_rank"] < 50]
    .groupby("customer_id", sort=False)["article_id"]
    .agg(lambda x: x.astype("int32").tolist())
    .to_dict()
)

cust_item_cnt.drop(columns=["_rank"], errors="ignore", inplace=True)
del cust_item_cnt
gc.collect()

global_rank = {str(int(a)): i for i, a in enumerate(global_top)}
recent_rank = {str(int(a)): i for i, a in enumerate(recent_top)}
very_recent_rank = {str(int(a)): i for i, a in enumerate(very_recent_top)}
weekend_rank = {str(int(a)): i for i, a in enumerate(weekend_top)}
trending_rank = {str(int(a)): i for i, a in enumerate(trending_top)}

cust_hist_rank = {
    cid: {str(int(a)): i for i, a in enumerate(lst)} for cid, lst in cust_top50.items()
}




## === cell 7
def to_pred_str(article_list):
    seen = set()
    out = []
    for a in article_list:
        s = str(int(a))
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= 12:
            break
    return " ".join(out)


def build_prediction_for_customer(cid, mode):
    if mode == 1:
        vrec = very_recent_purchases.get(cid, [])
        base = recent_purchases.get(cid, [])
        rep = repurchase_top.get(cid, [])
        return to_pred_str(list(vrec) + list(base) + list(rep) + recent_top)
    if mode == 2:
        base = freq_purchases.get(cid, [])
        ctop = cust_top50.get(cid, [])
        return to_pred_str(list(base) + list(ctop) + global_top)
    if mode == 3:
        vrec = very_recent_purchases.get(cid, [])
        base = last_purchases.get(cid, [])
        rep = repurchase_top.get(cid, [])
        return to_pred_str(list(vrec) + list(base) + list(rep) + recent_top)
    if mode == 4:
        return to_pred_str(recent_top_ch1 + trending_top + weekend_top + recent_top)
    if mode == 5:
        return to_pred_str(recent_top_ch2 + trending_top + weekend_top + recent_top)
    raise ValueError("Unknown mode")




## === cell 8
cust_ids = dfSample["customer_id"].astype(str).tolist()
if DEBUG:
    cust_ids = cust_ids[:5000]

preds1 = [build_prediction_for_customer(cid, 1) for cid in tqdm(cust_ids, desc="pred1")]
preds2 = [build_prediction_for_customer(cid, 2) for cid in tqdm(cust_ids, desc="pred2")]
preds3 = [build_prediction_for_customer(cid, 3) for cid in tqdm(cust_ids, desc="pred3")]
preds4 = [build_prediction_for_customer(cid, 4) for cid in tqdm(cust_ids, desc="pred4")]
preds5 = [build_prediction_for_customer(cid, 5) for cid in tqdm(cust_ids, desc="pred5")]

dfSub1 = pd.DataFrame({"customer_id": cust_ids, "prediction": preds1})
dfSub2 = pd.DataFrame({"customer_id": cust_ids, "prediction": preds2})
dfSub3 = pd.DataFrame({"customer_id": cust_ids, "prediction": preds3})
dfSub4 = pd.DataFrame({"customer_id": cust_ids, "prediction": preds4})
dfSub5 = pd.DataFrame({"customer_id": cust_ids, "prediction": preds5})

display_df(dfSub1, head=3)



## === cell 9
dfSub1.columns = ["customer_id", "prediction1"]
dfSub2.columns = ["customer_id", "prediction2"]
dfSub3.columns = ["customer_id", "prediction3"]
dfSub4.columns = ["customer_id", "prediction4"]
dfSub5.columns = ["customer_id", "prediction5"]



## === cell 10
pass



## === cell 11
dfSub = dfSub1.merge(dfSub2, on="customer_id", how="left")
dfSub = dfSub.merge(dfSub3, on="customer_id", how="left")
dfSub = dfSub.merge(dfSub4, on="customer_id", how="left")
dfSub = dfSub.merge(dfSub5, on="customer_id", how="left")
display_df(dfSub, head=3)



## === cell 12
if DEBUG:
    dfSub = dfSub.sample(frac=0.001, random_state=7)



## === cell 13
_rr = recent_rank.get
_vrr = very_recent_rank.get
_gr = global_rank.get
_wr = weekend_rank.get
_tr = trending_rank.get

_MODE_WEIGHTS = {
    "prediction1": 5,  # was 4
    "prediction2": 3,  # keep
    "prediction3": 6,  # was 5
    "prediction4": 1,
    "prediction5": 1,
}


def select_topn_weighted(row, n=12):
    counts = {}

    for col, w in _MODE_WEIGHTS.items():
        x = row.get(col, "")
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        ts = [t for t in str(x).split() if t]
        if not ts:
            continue
        for t in ts:
            counts[t] = counts.get(t, 0) + w

    if not counts:
        return ""

    cid = row.get("customer_id", None)
    ch = cust_hist_rank.get(cid, None)
    if ch is None:
        ch_get = lambda _t, _d=10**9: _d
    else:
        ch_get = ch.get

    items = list(counts.items())

    items.sort(
        key=lambda kv: (
            -kv[1],
            ch_get(kv[0], 10**9),
            _vrr(kv[0], 10**9),
            _rr(kv[0], 10**9),
            _tr(kv[0], 10**9),
            _wr(kv[0], 10**9),
            _gr(kv[0], 10**9),
            kv[0],
        )
    )

    out = []
    seen = set()
    for k, _ in items:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
        if len(out) >= n:
            break
    return " ".join(out)




## === cell 14
for c in ["prediction1", "prediction2", "prediction3", "prediction4", "prediction5"]:
    dfSub[c] = dfSub[c].fillna("")

dfSub["pred_sum"] = (
    (
        dfSub["prediction1"]
        + " "
        + dfSub["prediction2"]
        + " "
        + dfSub["prediction3"]
        + " "
        + dfSub["prediction4"]
        + " "
        + dfSub["prediction5"]
    )
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

dfSub["pred_top12"] = dfSub.progress_apply(select_topn_weighted, axis=1)

display_df(dfSub, head=3)



## === cell 15
idx = 248891
if idx < len(dfSub):
    print(dfSub["prediction1"].iloc[idx])
    print("")
    print(dfSub["prediction2"].iloc[idx])
    print("")
    print(dfSub["prediction3"].iloc[idx])
    print("")
    print(dfSub["prediction4"].iloc[idx])
    print("")
    print(dfSub["prediction5"].iloc[idx])
    print("")
    print(dfSub["pred_sum"].iloc[idx])
    print("")
    print(dfSub["pred_top12"].iloc[idx])



## === cell 16
dfSampleSub = dfSub[["customer_id", "pred_top12"]].copy()
display_df(dfSampleSub, head=3)



## === cell 17
dfSampleSub.columns = ["customer_id", "prediction"]

fallback = to_pred_str(recent_top + trending_top + weekend_top + global_top)

dfSampleSub["customer_id"] = dfSampleSub["customer_id"].astype(str).str.strip()
dfSampleSub["prediction"] = dfSampleSub["prediction"].fillna("").astype(str)

_token_re = re.compile(r"^\d+$")


def enforce_12(pred_str, fallback_str):
    toks0 = [t for t in str(pred_str).split() if t]
    toks = [t for t in toks0 if _token_re.match(t)]
    fb0 = [t for t in str(fallback_str).split() if t]
    fb = [t for t in fb0 if _token_re.match(t)]

    seen = set()
    out = []
    for t in toks + fb:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= 12:
            break
    if len(out) < 12:
        for a in global_top:
            t = str(int(a))
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= 12:
                break
    return " ".join(out[:12])


dfSampleSub["prediction"] = dfSampleSub["prediction"].apply(
    lambda s: enforce_12(s, fallback)
)

dfSampleSub = dfSample[["customer_id"]].merge(dfSampleSub, on="customer_id", how="left")
dfSampleSub["prediction"] = dfSampleSub["prediction"].fillna(enforce_12("", fallback))

dfSampleSub.to_csv("submission.csv", index=False)
print("Saved submission.csv.")



## === cell 18
dfCheck = pd.read_csv(
    "./submission.csv", dtype={"customer_id": "object", "prediction": "object"}
)
dfCheck["customer_id"] = dfCheck["customer_id"].astype(str).str.strip()
dfCheck["prediction"] = dfCheck["prediction"].astype(str)

display_df(dfCheck, head=3)
print(dfCheck.columns.tolist())
print("n_rows:", len(dfCheck))

lens = dfCheck["prediction"].fillna("").astype(str).str.split().str.len()
print("min prediction length:", int(lens.min()))
print("max prediction length:", int(lens.max()))
print("example prediction length:", len(str(dfCheck.loc[0, "prediction"]).split()))

assert dfCheck.shape[0] == dfSample.shape[0], "Row count mismatch vs sample_submission."
assert (
    dfCheck["customer_id"].values
    == dfSample["customer_id"].astype(str).str.strip().values
).all(), "customer_id order mismatch vs sample_submission."
assert (
    lens == 12
).all(), "Not all rows have exactly 12 predictions; Kaggle scoring may fail."
assert (
    dfCheck["prediction"]
    .str.split()
    .explode()
    .map(lambda x: bool(_token_re.match(str(x))))
    .all()
), "Found non-digit tokens in prediction."
