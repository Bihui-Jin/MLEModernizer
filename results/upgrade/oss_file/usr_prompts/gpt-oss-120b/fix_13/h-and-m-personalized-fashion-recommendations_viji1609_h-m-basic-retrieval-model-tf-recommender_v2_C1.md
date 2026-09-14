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
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.01387

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Wereplace the failing TensorFlow‑based pipeline with a lightweight pandas baseline that predicts the 12 most‑popular articles overall for every customer. This avoids the NumPy import error, guarantees a valid `submission.csv`, and should deliver a MAP@12 score close to the target 0.01387.'
- What this solution (achieved 0.0) has done: 'The fix addresses the merge conflict that created duplicate “prediction” columns, causing a KeyError when trying to fill missing values. We read the sample submission, drop its empty prediction column, merge it with the per‑customer top‑12 predictions, and then fill any missing predictions with the global top‑12 list. Finally we write the corrected DataFrame to `submission.csv`, ensuring the required columns exist.'
- What this solution (achieved 0.0) has done: 'I remove the overly‑restrictive date filter so that the model can use the full transaction history when computing each customer’s most‑frequent articles. Keeping the rest of the pipeline unchanged preserves the original logic while giving a stronger per‑customer signal, which should raise the MAP@12 score toward the target.'
- What this solution (achieved 0.0) has done: 'I add a recent‑purchase weighting: compute each customer’s most‑frequent articles within the last 30 days of the training period (these are more indicative of the next week’s buys) and use those as the per‑customer predictions. Any customers with fewer than 12 recent items be padded with the overall top‑12 articles, preserving the original simple baseline while modestly improving MAP@12 toward the target.'
- What this solution (achieved 0.0) has done: 'The change pads each customer’s recent‑purchase list with the globally most‑popular articles until 12 IDs are present, rather than discarding the popular items when a customer has few recent buys. This keeps the strong signal from recent history while adding likely correct items, which should raise MAP@12 toward the target without altering the overall pipeline. The rest of the script is unchanged except for renumbered cells for clarity.'
- What this solution (achieved 0.0) has done: 'I remove the 30‑day restriction so that each customer’s article frequencies are computed from the full transaction history rather than only the recent slice. This modest change keeps the overall‑top fallback unchanged, guarantees predictions for every customer, and is expected to lift the MAP@12 from 0.0 toward the target 0.01387 without drastically overshooting it.'
- What this solution (achieved 0.0) has done: 'I limit the transaction data used for per‑customer frequencies to the last 30 days of the training period, which usually gives a slightly better MAP@12 while keeping the original logic unchanged. The global top‑12 list stays the same as a fallback, and the merge/fill steps remain identical, ensuring a valid `submission.csv` is created.'
- What this solution (achieved 0.0) has done: 'I switch the per‑customer frequency computation to use the full transaction history (removing the 30‑day restriction) and ensure the `customer_id` column is treated as a string so it matches the sample‑submission IDs. This modest change keeps the original simple baseline while giving it a stronger signal and should move the MAP@12 score closer to the target.'
- What this solution (achieved 0.0) has done: 'I restrict the per‑customer frequency counting to the most recent 30 days of the training period (which usually predicts the next week better) while keeping the global top‑12 fallback unchanged. This small change preserves the original pipeline, guarantees a valid `submission.csv`, and is expected to raise the MAP@12 score toward the target.'
- What this solution (achieved 0.0) has done: 'I switch the per‑customer frequency computation to use the **full transaction history** instead of only the last 30 days. This gives each customer a richer set of past purchases while keeping the same overall‑top fallback, so the predictions become more personalized and the MAP@12 score should move from 0.0 toward the target 0.01387 without altering any other part of the pipeline.'
- What this solution (achieved 0.0) has done: 'I adjust the per‑customer frequency calculation to use only the most recent 30 days of purchases (which usually predicts the next week better) while keeping the global top‑12 fallback unchanged. This small change keeps the overall pipeline intact, guarantees a valid `submission.csv`, and should raise the MAP@12 score from 0 toward the target 0.01387.'

# 9. Code solution

## === cell 0
import pandas as pd
from pathlib import Path

base_path = Path("/kaggle/input/h-and-m-personalized-fashion-recommendations")
if not base_path.exists():
    base_path = Path("../input/h-and-m-personalized-fashion-recommendations")

transactions_path = base_path / "transactions_train.csv"
sample_path = base_path / "sample_submission.csv"



## === cell 1
cols = ["t_dat", "customer_id", "article_id"]
train = pd.read_csv(transactions_path, usecols=cols)

train["article_id"] = train["article_id"].astype(str).str.zfill(10)
train["customer_id"] = train["customer_id"].astype(
    str
)  # match sample_submission format
train["t_dat"] = pd.to_datetime(train["t_dat"])

full_train = train



## === cell 2
overall_top_articles = train["article_id"].value_counts().head(12).index.tolist()
overall_top_str = " ".join(overall_top_articles)



## === cell 3
recent_cutoff = full_train["t_dat"].max() - pd.Timedelta(days=30)
recent_train = full_train[full_train["t_dat"] >= recent_cutoff]

cnts = (
    recent_train.groupby(["customer_id", "article_id"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["customer_id", "cnt"], ascending=[True, False])
)


def build_prediction(ids_series):
    ids = ids_series.head(12).tolist()
    for article in overall_top_articles:
        if len(ids) >= 12:
            break
        if article not in ids:
            ids.append(article)
    return " ".join(ids)


cust_top12 = (
    cnts.groupby("customer_id")["article_id"]
    .apply(build_prediction)
    .reset_index(name="prediction")
)



## === cell 4
sample_sub = pd.read_csv(sample_path, dtype={"customer_id": str, "prediction": str})
sample_sub = sample_sub[["customer_id"]]

submission = sample_sub.merge(cust_top12, on="customer_id", how="left")
submission["prediction"] = submission["prediction"].fillna(overall_top_str)

submission.to_csv("submission.csv", index=False)
