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

- What this solution (achieved 0.0) has done: 'I fix the Polars API issues causing the failures: `value_counts()` now returns a `count` column (not `counts`), and `groupby` is `group_by` in recent Polars. I also make the per-user top-12 aggregation deterministic and compatible by explicitly naming the count column and sorting by it. Finally, I ensure the prediction strings are correctly zero-padded to 10 digits (H&M article_id format) and that `submission.csv` is always written with the required `customer_id,prediction` columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a submission-format mismatch: H&M expects 64-bit integer `article_id`s, and if Polars inferred `article_id` as float anywhere, your `cast(Int64)` can turn them into nulls, producing empty/invalid predictions and thus MAP@12≈0. I make the smallest change that forces correct dtypes at read time (`UInt64`/`Int64` for `article_id`, `Utf8` for `customer_id`) and add a final safety fallback that replaces any empty prediction strings with the global top-12 (so every row has 12 valid 10-digit ids). This preserves your exact core logic (personal top-12 + global backfill) while making the output reliably valid for scoring. The rest stays identical and it still writes `submission.csv` in the required schema.'

# 9. Code solution

## === cell 0
import polars as pl

DATA_DIR = "/kaggle/input/h-and-m-personalized-fashion-recommendations"

TXN_DTYPES = {"customer_id": pl.Utf8, "article_id": pl.UInt64}
SUB_DTYPES = {"customer_id": pl.Utf8}



## === cell 1
most_popular_articles = (
    pl.read_csv(
        f"{DATA_DIR}/transactions_train.csv",
        columns=["article_id"],
        dtypes={"article_id": pl.UInt64},  # keep exact ids
    )
    .get_column("article_id")
    .value_counts()  # columns: ["article_id", "count"]
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
        dtypes=TXN_DTYPES,  # keep exact ids and customer keys
    )
    .group_by(["customer_id", "article_id"])
    .agg(pl.len().alias("cnt"))
    .group_by("customer_id")
    .agg(
        pl.col("article_id")
        .sort_by(pl.col("cnt"), descending=True)
        .head(12)
        .alias("article_id")
    )
)



## === cell 3
improved_submission = (
    pl.read_csv(
        f"{DATA_DIR}/sample_submission.csv",
        columns=["customer_id"],
        dtypes=SUB_DTYPES,  # stable join key dtype
    )
    .join(most_popular_articles_per_user, on="customer_id", how="left")
    .with_columns(
        [
            pl.col("article_id").fill_null([]).alias("personal_top_12"),
            pl.lit(most_popular_articles).alias("global_top_12"),
        ]
    )
    .with_columns(
        pl.col("personal_top_12")
        .list.concat(pl.col("global_top_12"))
        .list.head(12)
        .alias("prediction_list")
    )
    .select(
        [
            pl.col("customer_id"),
            pl.col("prediction_list")
            .list.eval(pl.element().cast(pl.UInt64).cast(pl.Utf8).str.zfill(10))
            .list.join(" ")
            .alias("prediction"),
        ]
    )
    .with_columns(
        pl.when(pl.col("prediction").str.len_chars() == 0)
        .then(pl.lit(" ".join([str(a).zfill(10) for a in most_popular_articles])))
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )
)



## === cell 4
improved_submission.write_csv("submission.csv")
print(improved_submission.head(3))
print("Wrote submission.csv with shape:", improved_submission.shape)
print(
    "Non-empty prediction rows:",
    improved_submission.select((pl.col("prediction").str.len_chars() > 0).sum()).item(),
)
