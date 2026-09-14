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

3.11

# 3. Installed packages

cudf-polars-cu12==25.6.0
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
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

0.00822

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Diagnosis: The crash in cell 3 is a Polars `SchemaError` triggered when concatenating `personal_top_<=12` with `global_top_12`. `personal_top_<=12` is a `list[i64]`, but `global_top_12` is currently created as a nested list (`list[list[i64]]`) because of `pl.lit([most_popular_articles])`. Polars cannot find a common supertype between `i64` and `list[i64]` during `concat_list`, so execution fails.

Patch summary: Change the literal construction of `global_top_12` to be a single-level list column (`list[i64]`) by using `pl.lit(most_popular_articles)` instead of wrapping it in an extra list. This preserves the same intended semantics (append global top-12 to each row) and keeps the output column names/format identical for cell 4.

Updated cells: Only cell 3 is modified.

Compatibility notes for cell k+1: `improved_submission` remains a Polars DataFrame with columns `customer_id` and `prediction` (string of space-joined article_ids), so `improved_submission.write_csv("submission.csv")` in cell 4 continues to work unchanged.

Assumptions: `most_popular_articles` is a Python `list[int]` (as produced in cell 1) and `article_id` values fit in Polars `Int64`.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because the submission likely contains invalid `article_id` formatting: the code currently prefixes every id with an extra `"0"`, turning e.g. `108775015` into `"0108775015"` which won’t match the ground-truth `article_id` strings used for scoring. I keep the same popularity-based logic but change only the final formatting step to output 10-digit, zero-padded article ids (the competition’s expected format) without adding an extra leading digit. This should move the score up toward your target with minimal risk and no change to the recommendation logic. The output file remains `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a submission-format mismatch rather than a modeling issue: the join/list logic is fine, but MAP@12 scoring is very sensitive to `article_id` string formatting. I keep the exact same popularity-based approach and only make two minimal, score-relevant robustness fixes: (1) ensure `customer_id` dtypes match between sample submission and transactions (string) to avoid losing many personal histories in the join, and (2) ensure `article_id` is consistently treated as an integer for counting, then formatted to the required 10-digit zero-padded strings at the end. This should move the score upward toward your target without changing the core recommendation logic. The script still run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with `article_id` not matching the ground truth type/format at scoring time, even if the submission CSV structure is valid. I keep your exact popularity-per-user + global-top-12 logic, but make two minimal score-relevant robustness fixes: ensure we never accidentally emit `"null"`/empty predictions, and ensure `article_id` formatting is always exactly 10-digit zero-padded strings derived from integers (no stray whitespace or float casts). I also explicitly deduplicate after concatenation (personal + global) to avoid repeated IDs pushing relevant items out of the top-12, which typically improves MAP@12 without changing the modeling approach. All paths remain unchanged and the script still writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is consistent with a submission that’s valid CSV but whose `prediction` strings don’t match the scorer’s expected `article_id` tokens, often due to null/empty predictions or type/format inconsistencies. I keep your exact “personal top-12 + global top-12 then unique then head(12)” logic, but make two minimal score-relevant robustness changes: enforce `article_id` is always read/counting as `Int64` and ensure we always emit exactly 12 zero-padded 10-digit ids per customer by filling any shortfall with global items after de-duplication. This preserves your core approach and should move the score upward toward the 0.00822 target without changing modeling semantics. The script still runs end-to-end and writes `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a *valid CSV that the evaluator can’t match to ground truth*, usually because the `customer_id` tokens don’t exactly match (H&M uses 64-hex strings, and Polars can silently interpret them oddly if not forced) or because some rows end up with empty/`null` predictions. I keep your exact “personal top-12 + global top-12 → dedupe → head(12)” logic, but make two minimal score-relevant fixes: (1) force `customer_id` to be read/handled as **string** from both files and strip whitespace to guarantee join alignment, and (2) guarantee every customer outputs **exactly 12** article ids by padding from global top-12 after de-duplication. This should move the score upward toward your 0.00822 target without changing the approach. The script still runs end-to-end and writes `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import polars as pl

DATA_DIR = "/kaggle/input/h-and-m-personalized-fashion-recommendations"



## === cell 1
most_popular_articles = (
    pl.read_csv(
        f"{DATA_DIR}/transactions_train.csv",
        columns=["article_id"],
        dtypes={"article_id": pl.Int64},
    )
    .get_column("article_id")
    .value_counts()
    .sort("count", descending=True)
    .head(12)
    .get_column("article_id")
    .to_list()
)



## === cell 2
most_popular_articles_per_user = (
    pl.read_csv(
        f"{DATA_DIR}/transactions_train.csv",
        columns=["customer_id", "article_id"],
        dtypes={"customer_id": pl.Utf8, "article_id": pl.Int64},
    )
    .with_columns(
        pl.col("customer_id")
        .str.strip_chars()
        .alias("customer_id")
    )
    .group_by(["customer_id", "article_id"])
    .agg(pl.count().alias("count"))
    .group_by("customer_id")
    .agg(pl.col("article_id").sort_by("count", descending=True).head(12))
)



## === cell 3
improved_submission = (
    pl.read_csv(
        f"{DATA_DIR}/sample_submission.csv",
        columns=["customer_id"],
        dtypes={"customer_id": pl.Utf8},
    )
    .with_columns(pl.col("customer_id").str.strip_chars().alias("customer_id"))
    .join(most_popular_articles_per_user, on="customer_id", how="left")
    .with_columns(
        [
            pl.col("article_id").fill_null([]).alias("personal_top_<=12"),
            pl.lit(most_popular_articles).alias("global_top_12"),
        ]
    )
    .with_columns(
        [
            pl.concat_list(["personal_top_<=12", "global_top_12"])
            .list.unique(maintain_order=True)
            .alias("deduped"),
        ]
    )
    .with_columns(
        pl.concat_list(["deduped", "global_top_12"])
        .list.unique(maintain_order=True)
        .list.head(12)
        .alias("prediction_list")
    )
    .with_columns(
        pl.when(pl.col("prediction_list").list.len() < 12)
        .then(
            pl.concat_list(["prediction_list", "global_top_12"])
            .list.unique(maintain_order=True)
            .list.head(12)
        )
        .otherwise(pl.col("prediction_list"))
        .alias("prediction_list")
    )
    .select(
        [
            pl.col("customer_id"),
            pl.col("prediction_list")
            .list.eval(
                pl.element()
                .cast(pl.Int64, strict=False)
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.zfill(10)
            )
            .list.join(" ")
            .alias("prediction"),
        ]
    )
)



## === cell 4
improved_submission.write_csv("submission.csv")
