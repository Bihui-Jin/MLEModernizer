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

0.02249

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fixes address the import error, ensure the transaction dates are sorted for slicing, and replace the fragmented per‑age‑bin prediction logic with a straightforward baseline that predicts the 12 most popular articles for every customer. This guarantees a valid submission containing all required `customer_id`s while keeping the original data‑handling steps intact.'
- What this solution (achieved 0.00296) has done: 'The fix adds the missing imports for **cudf**, **pandas**, **matplotlib**, and **seaborn**, allowing all cells to run without NameError. No core modeling logic is changed; the baseline that predicts the 12 most frequent articles remains intact, and a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
DEBUG = False
PATH_INPUT = r"../input/h-and-m-personalized-fashion-recommendations/"

import cudf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




## === cell 1
def display_df(df, head=3):
    print(f"The shape of df is {df.shape}.\n")
    display(df.head(head))




## === cell 2
dfArticles = cudf.read_csv(
    PATH_INPUT + "articles.csv",
    usecols=["article_id", "product_group_name", "perceived_colour_master_name"],
)
display_df(dfArticles, head=3)




## === cell 3
dfCustomers = cudf.read_csv(
    PATH_INPUT + "customers.csv", usecols=["customer_id", "age"]
)
display_df(dfCustomers, head=3)




## === cell 4
dfCustomers = dfCustomers.to_pandas()
listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
dfCustomers["age_bins"] = pd.cut(dfCustomers["age"], listBin)
display_df(dfCustomers, head=3)




## === cell 5
x = dfCustomers[dfCustomers["age_bins"].isnull()].shape[0]
print(f"{x} customer_id do not have age information.\n")




## === cell 6
dfTransactions = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"article_id": "int32", "t_dat": "string", "customer_id": "string"},
)
dfTransactions["t_dat"] = cudf.to_datetime(dfTransactions["t_dat"])
dfTransactions.set_index("t_dat", inplace=True)
dfTransactions = dfTransactions.sort_index()
dfTransactions = dfTransactions.to_pandas()
display_df(dfTransactions, head=3)




## === cell 7
max_date = dfTransactions.index.max()
min_date = max_date - pd.Timedelta(days=30)  # inclusive 31‑day window
dfRecent = dfTransactions.loc[min_date:max_date]

if dfRecent.empty:
    print("Recent window empty – falling back to full transactions.")
    dfRecent = dfTransactions

display_df(dfRecent, head=3)




## === cell 8
dfRecent = dfRecent.merge(
    dfCustomers[["customer_id", "age_bins"]], on="customer_id", how="inner"
)
display_df(dfRecent, head=3)




## === cell 9
dfRecent_detail = dfRecent.copy()

dfFull = dfTransactions.merge(
    dfCustomers[["customer_id", "age_bins"]], on="customer_id", how="inner"
)
dfFull_detail = dfFull.copy()

dfFull_counts = (
    dfFull_detail.groupby(["age_bins", "article_id"]).size().reset_index(name="counts")
)

listUniBins = dfFull_counts["age_bins"].unique().tolist()
dict100 = {}
for uniBin in listUniBins:
    dfTemp = dfFull_counts[dfFull_counts["age_bins"] == uniBin]
    dfTemp = dfTemp.sort_values(by="counts", ascending=False)
    dict100[uniBin] = dfTemp.head(100)["article_id"].values.tolist()
df100 = pd.DataFrame([dict100]).T.rename(columns={0: "top100"})




## === cell 10
for index in df100.index:
    df100[index] = [
        len(set(df100.at[index, "top100"]) & set(df100.at[x, "top100"])) / 100
        for x in df100.index
    ]
df100 = df100.drop(columns="top100")
plt.figure(figsize=(10, 6))
sns.heatmap(df100, annot=True, cbar=False)




## === cell 11
N = 12

df_full_global = (
    dfTransactions.groupby("article_id")
    .size()
    .reset_index(name="counts")
    .sort_values("counts", ascending=False)
    .head(N)
)
global_top12 = df_full_global["article_id"].astype(str).tolist()

dict_top12 = {}
for age_bin, articles in dict100.items():
    top12 = [str(a) for a in articles[:N]]
    dict_top12[age_bin] = top12

cust_top = (
    dfRecent_detail.groupby("customer_id")["article_id"]
    .apply(lambda x: x.value_counts().head(N).index.tolist())
    .to_dict()
)

sub = pd.read_csv(PATH_INPUT + "sample_submission.csv", dtype={"customer_id": "string"})
sub = sub.merge(dfCustomers[["customer_id", "age_bins"]], on="customer_id", how="left")


def make_prediction(row):
    """Return a space‑separated list of up to N article IDs.
    Starts with the customer’s own recent purchases, then fills
    missing slots with age‑bin popularity or the global popularity."""
    cust_id = row["customer_id"]
    pred_list = []

    if cust_id in cust_top:
        pred_list = [str(a) for a in cust_top[cust_id]]

    if len(pred_list) < N:
        bin_val = row["age_bins"]
        source = global_top12
        if not pd.isna(bin_val) and bin_val in dict_top12:
            source = dict_top12[bin_val]

        for a in source:
            if a not in pred_list:
                pred_list.append(a)
            if len(pred_list) == N:
                break

    return " ".join(pred_list[:N])


sub["prediction"] = sub.apply(make_prediction, axis=1)

sub.to_csv("submission.csv", index=False)
print(f"Submission file saved with {sub.shape[0]} rows.")




## === cell 12
dfCheck = pd.read_csv("submission.csv")
display_df(dfCheck, head=3)




## === cell 13
print("All done. The file submission.csv is ready for upload.")
