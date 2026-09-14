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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.00544

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Implemented a streamlined pipeline that avoids the earlier merge errors and directly creates a valid submission file. The new steps keep the initial data loading and preprocessing, compute the 12 most popular articles from the recent 21‑day window, assign this same list to every customer, and write the required `customer_id, prediction` CSV. Unused problematic cells are replaced with harmless `pass` statements to ensure smooth execution.'
- What this solution (achieved 0.0) has done: 'I replace the single‑global prediction with a lightweight per‑customer recommendation: for each customer I take their most frequent articles in the last 21 days (up to 12 items) and fall back to the overall top‑12 list when a customer has no recent purchases. This keeps the overall pipeline unchanged while adding useful personalization, which should raise the MAP@12 from 0 toward the target 0.00544. The submission file is still written as `predict.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'The changes expand the personalization window from 21 days to 84 days and add a fallback that uses each customer’s overall most‑frequent items (from the entire training period) before reverting to the global top‑12 list. This modest increase in historical information should raise MAP@12 toward the target without altering the core modeling flow.'
- What this solution (achieved 0.0) has done: 'I keep the existing pipeline but replace the frequency‑based recent‑article ranking with a recency‑based ranking: for each customer I take the most recent distinct articles from the last 84 days (up to 12) and use this list first, falling back to the overall frequency list as before. This small tweak should increase the overlap with the true next‑week purchases and move the MAP@12 score toward the target while leaving the overall architecture unchanged.'
- What this solution (achieved 0.0) has done: 'Implemented small but targeted fixes to improve MAP@12 without altering the core recommendation logic.  
- Adjusted `article_id` handling to use proper zero‑padding (`zfill(10)`) instead of naïvely prefixing a “0”, ensuring IDs match the ground‑truth format.  
- Extended the recent‑history window from 84 days to 120 days to capture more relevant recent purchases, which modestly increases personalization while preserving the overall pipeline.'
- What this solution (achieved 0.0) has done: 'I keep the overall pipeline but improve the per‑customer prediction logic: instead of falling back completely to a single list, I first use the recent‑history items, then fill any remaining slots with the customer’s overall top items, and finally pad with the global top‑12 list while avoiding duplicates. This small change adds useful personalization and should raise the MAP@12 score toward the target without altering the core model or data processing.'
- What this solution (achieved 0.0) has done: 'I slightly expand the recent‑history window from 120 to 180 days so each customer sees more of their latest purchases, and I compute the fallback global top‑12 list from the full training history (instead of the recent slice) to give a stronger overall baseline. These modest tweaks keep the original workflow intact while adding useful signals that should raise the MAP@12 toward the target.'
- What this solution (achieved 0.0) has done: 'Implemented a modest but targeted tweak: narrowed the recent‑history window from 180 days to the last 30 days. This keeps the core recommendation flow unchanged while focusing on the most recent purchases, which are more indicative of the next‑week buying behavior and should move the MAP@12 score closer to the target without over‑hauling the model.'
- What this solution (achieved 0.0) has done: 'I extend the recent‑history window from 30 days to 84 days, which gives each customer a richer, more relevant set of recent purchases while keeping the rest of the pipeline unchanged. This small change should add useful items to the predictions and move the MAP@12 score toward the target 0.00544.'
- What this solution (achieved 0.0) has done: 'I increase the recency window used for the personalized part of the recommendation from 84 days to 120 days. This modest change keeps the core pipeline unchanged but gives each customer more recent purchase information, which should raise the MAP@12 score from 0 toward the target 0.00544. No other logic is altered, and the script still writes a valid `predict.csv` file.'
- What this solution (achieved 0.0) has done: 'I narrow the recent‑history window from 120 days to 30 days so the personalized part of the recommendation focuses on the most recent purchases, which are more predictive of the next‑week buying behaviour. This tiny tweak keeps the entire pipeline unchanged while adding a modest signal that should raise the MAP@12 from 0 toward the target 0.00544. The only code change is the `pd.Timedelta("30 days")` replacement in the filtering step.'
- What this solution (achieved 0.0) has done: 'I increase the recency window from 30 days to 84 days so that the per‑customer recent‑purchase list contains more items, improving personalization while keeping the rest of the pipeline unchanged. This small change is expected to raise MAP@12 from 0 toward the target 0.00544.'
- What this solution (achieved 0.0) has done: 'I increase the recency window from 84 days to 180 days so the personalized recent‑purchase list contains more relevant items, which should provide a modest lift in MAP@12 toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'I adjust the recency window used for the personalized part of the recommendation from 180 days to a shorter 30‑day window. A tighter window focuses on the most recent purchases, which are more indicative of the next‑week buying behaviour and should lift the MAP@12 score toward the target while leaving the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")



## === cell 1
general_path = "../input/h-and-m-personalized-fashion-recommendations/"



## === cell 2
articles = pd.read_csv(general_path + "articles.csv")
customers = pd.read_csv(general_path + "customers.csv")
sample_submission = pd.read_csv(general_path + "sample_submission.csv")
transactions_train = pd.read_csv(general_path + "transactions_train.csv")



## === cell 3
transactions_train.info()



## === cell 4
transactions_train.sample(5)



## === cell 5
transactions_train["article_id"] = (
    transactions_train["article_id"].astype(str).str.zfill(10)
)



## === cell 6
articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)



## === cell 7
transactions_train["t_dat"] = pd.to_datetime(
    transactions_train["t_dat"], format="%Y-%m-%d"
)



## === cell 8
maximum_history_date = transactions_train["t_dat"].max()



## === cell 9
print(f"We have {len(customers)} unique customers")
print(f'And {len(customers["postal_code"].unique())} unique locations')



## === cell 10
customer_per_location = customers.pivot_table(
    index="postal_code", aggfunc={"customer_id": "count"}
)



## === cell 11
customer_per_location.sort_values(by=["customer_id"])



## === cell 12
customers["location"] = "other"
customers.loc[
    customers["postal_code"]
    == "2c29ae653a9282cce4151bd87643c907644e09541abc28ae87dea0d1f6603b1c",
    "location",
] = "big_city"



## === cell 13
transactions_train["price"].hist(bins=30, figsize=(16, 5))
plt.title("Prices")
plt.show()



## === cell 14
articles.nunique()



## === cell 15
articles.sample(1)



## === cell 16
articles["index_name"].unique()



## === cell 17
customers["age_group"] = "<20"
customers.loc[customers["age"] > 20, "age_group"] = "20-45"
customers.loc[customers["age"] > 45, "age_group"] = ">45"




## === cell 18
def aggregate_to_list(row):
    return [i for i in row]




## === cell 19
transactions_train = pd.merge(
    transactions_train,
    articles[["article_id", "index_name"]],
    on="article_id",
    how="left",
)



## === cell 20
transactions_train = pd.merge(
    transactions_train,
    customers[["customer_id", "age_group"]],
    on="customer_id",
    how="left",
)



## === cell 21
sample_submission = sample_submission[["customer_id"]].copy()



## === cell 22
transactions_full = transactions_train.copy()
transactions_train = transactions_train[
    transactions_train["t_dat"]
    > transactions_train["t_dat"].max() - pd.Timedelta("30 days")
]



## === cell 23
top_articles = (
    transactions_full.groupby("article_id")
    .size()
    .sort_values(ascending=False)
    .head(12)
    .index.tolist()
)
top_prediction_str = " ".join(top_articles)



## === cell 24
recent_articles = (
    transactions_train.sort_values(["customer_id", "t_dat"], ascending=[True, False])
    .groupby("customer_id")["article_id"]
    .apply(lambda x: x.drop_duplicates().head(12).tolist())
    .to_dict()
)
overall_counts = (
    transactions_full.groupby(["customer_id", "article_id"])
    .size()
    .reset_index(name="cnt")
)
overall_counts = overall_counts.sort_values(
    ["customer_id", "cnt"], ascending=[True, False]
)
overall_top_articles = (
    overall_counts.groupby("customer_id")["article_id"]
    .apply(lambda x: x.head(12).tolist())
    .to_dict()
)




## === cell 25
def make_prediction(cid):
    recent = recent_articles.get(cid, [])
    if recent is None:
        recent = []
    prediction = list(recent)  # copy
    if len(prediction) < 12:
        overall = overall_top_articles.get(cid, [])
        for art in overall:
            if art not in prediction:
                prediction.append(art)
            if len(prediction) == 12:
                break
    if len(prediction) < 12:
        for art in top_articles:
            if art not in prediction:
                prediction.append(art)
            if len(prediction) == 12:
                break
    return " ".join(prediction)




## === cell 26
sample_submission["prediction"] = sample_submission["customer_id"].apply(
    make_prediction
)
sample_submission.to_csv("predict.csv", index=False)



## === cell 27
pass



## === cell 28
pass



## === cell 29
pass



## === cell 30
pass



## === cell 31
pass



## === cell 32
pass



## === cell 33
pass



## === cell 34
pass



## === cell 35
pass



## === cell 36
pass



## === cell 37
pass



## === cell 38
pass



## === cell 39
pass



## === cell 40
pass
