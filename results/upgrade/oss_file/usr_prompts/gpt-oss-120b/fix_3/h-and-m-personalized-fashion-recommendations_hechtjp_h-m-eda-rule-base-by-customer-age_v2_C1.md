# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The fixes address the import error, ensure the transaction dates are sorted for slicing, and replace the fragmented per‑age‑bin prediction logic with a straightforward baseline that predicts the 12 most popular articles for every customer. This guarantees a valid submission containing all required `customer_id`s while keeping the original data‑handling steps intact.'

# 9. Code solution

## === cell 0
DEBUG = False
PATH_INPUT = r"../input/h-and-m-personalized-fashion-recommendations/"




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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3999615787.py in <cell line: 0>()
----> 1 dfArticles = cudf.read_csv(
      2     PATH_INPUT + "articles.csv",
      3     usecols=["article_id", "product_group_name", "perceived_colour_master_name"],
      4 )
      5 display_df(dfArticles, head=3)

NameError: name 'cudf' is not defined

## === cell 3
dfCustomers = cudf.read_csv(
    PATH_INPUT + "customers.csv", usecols=["customer_id", "age"]
)
display_df(dfCustomers, head=3)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2248824303.py in <cell line: 0>()
----> 1 dfCustomers = cudf.read_csv(
      2     PATH_INPUT + "customers.csv", usecols=["customer_id", "age"]
      3 )
      4 display_df(dfCustomers, head=3)
      5 

NameError: name 'cudf' is not defined

## === cell 4
dfCustomers = dfCustomers.to_pandas()
listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
dfCustomers["age_bins"] = pd.cut(dfCustomers["age"], listBin)
display_df(dfCustomers, head=3)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2468536183.py in <cell line: 0>()
----> 1 dfCustomers = dfCustomers.to_pandas()
      2 listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
      3 dfCustomers["age_bins"] = pd.cut(dfCustomers["age"], listBin)
      4 display_df(dfCustomers, head=3)
      5 

NameError: name 'dfCustomers' is not defined

## === cell 5
x = dfCustomers[dfCustomers["age_bins"].isnull()].shape[0]
print(f"{x} customer_id do not have age information.\n")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3066383129.py in <cell line: 0>()
----> 1 x = dfCustomers[dfCustomers["age_bins"].isnull()].shape[0]
      2 print(f"{x} customer_id do not have age information.\n")
      3 

NameError: name 'dfCustomers' is not defined

## === cell 6
dfTransactions = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"article_id": "int32", "t_dat": "string", "customer_id": "string"},
)
dfTransactions["t_dat"] = cudf.to_datetime(dfTransactions["t_dat"])
dfTransactions.set_index("t_dat", inplace=True)
dfTransactions = dfTransactions.sort_index()
display_df(dfTransactions, head=3)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/514896048.py in <cell line: 0>()
----> 1 dfTransactions = cudf.read_csv(
      2     PATH_INPUT + "transactions_train.csv",
      3     usecols=["t_dat", "customer_id", "article_id"],
      4     dtype={"article_id": "int32", "t_dat": "string", "customer_id": "string"},
      5 )

NameError: name 'cudf' is not defined

## === cell 7
try:
    dfRecent = dfTransactions.loc["2020-09-01":"2020-09-21"]
except KeyError:
    dfRecent = dfTransactions
display_df(dfRecent, head=3)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/748944469.py in <cell line: 0>()
      2 # fall back to the whole dataframe to keep the notebook running.
      3 try:
----> 4     dfRecent = dfTransactions.loc["2020-09-01":"2020-09-21"]
      5 except KeyError:
      6     dfRecent = dfTransactions

NameError: name 'dfTransactions' is not defined

## === cell 8
dfRecent = dfRecent.to_pandas()
dfRecent = dfRecent.merge(
    dfCustomers[["customer_id", "age_bins"]], on="customer_id", how="inner"
)
display_df(dfRecent, head=3)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/931555914.py in <cell line: 0>()
----> 1 dfRecent = dfRecent.to_pandas()
      2 dfRecent = dfRecent.merge(
      3     dfCustomers[["customer_id", "age_bins"]], on="customer_id", how="inner"
      4 )
      5 display_df(dfRecent, head=3)

NameError: name 'dfRecent' is not defined

## === cell 9
dfRecent = (
    dfRecent.groupby(["age_bins", "article_id"])
    .count()
    .reset_index()
    .rename(columns={"customer_id": "counts"})
)
listUniBins = dfRecent["age_bins"].unique().tolist()
dict100 = {}
for uniBin in listUniBins:
    dfTemp = dfRecent[dfRecent["age_bins"] == uniBin]
    dfTemp = dfTemp.sort_values(by="counts", ascending=False)
    dict100[uniBin] = dfTemp.head(100)["article_id"].values.tolist()
df100 = pd.DataFrame([dict100]).T.rename(columns={0: "top100"})



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/605498440.py in <cell line: 0>()
      1 dfRecent = (
----> 2     dfRecent.groupby(["age_bins", "article_id"])
      3     .count()
      4     .reset_index()
      5     .rename(columns={"customer_id": "counts"})

NameError: name 'dfRecent' is not defined

## === cell 10
for index in df100.index:
    df100[index] = [
        len(set(df100.at[index, "top100"]) & set(df100.at[x, "top100"])) / 100
        for x in df100.index
    ]
df100 = df100.drop(columns="top100")
plt.figure(figsize=(10, 6))
sns.heatmap(df100, annot=True, cbar=False)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2019000375.py in <cell line: 0>()
----> 1 for index in df100.index:
      2     df100[index] = [
      3         len(set(df100.at[index, "top100"]) & set(df100.at[x, "top100"])) / 100
      4         for x in df100.index
      5     ]

NameError: name 'df100' is not defined

## === cell 11
N = 12
df_trans = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["article_id"],
    dtype={"article_id": "string"},
)
top_articles = df_trans["article_id"].value_counts().head(N).index.to_pandas().tolist()
top_pred_str = " ".join(top_articles)

sub = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv", dtype={"customer_id": "string"}
)
sub["prediction"] = top_pred_str
sub.to_csv("submission.csv", index=False)
print(f"Submission file saved with {sub.shape[0]} rows.")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1836887739.py in <cell line: 0>()
      1 # Baseline: predict the 12 most frequent articles (preserving leading zeros)
      2 N = 12
----> 3 df_trans = cudf.read_csv(
      4     PATH_INPUT + "transactions_train.csv",
      5     usecols=["article_id"],

NameError: name 'cudf' is not defined

## === cell 12
dfCheck = cudf.read_csv("submission.csv")
display_df(dfCheck, head=3)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2229139862.py in <cell line: 0>()
----> 1 dfCheck = cudf.read_csv("submission.csv")
      2 display_df(dfCheck, head=3)
      3 

NameError: name 'cudf' is not defined

## === cell 13
print("All done. The file submission.csv is ready for upload.")
