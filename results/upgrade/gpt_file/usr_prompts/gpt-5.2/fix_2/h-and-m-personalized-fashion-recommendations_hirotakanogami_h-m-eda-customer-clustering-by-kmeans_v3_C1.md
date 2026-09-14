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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.02246

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys, warnings, time, os, copy, gc, re, random, pickle

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

from collections import Counter
from datetime import datetime, timedelta

import cudf

from sklearn.cluster import KMeans
from sklearn import preprocessing

from pandas import json_normalize




## === cell 1
class Clustering_HandM:
    def customers_preprocessing(self, customers, dropcol=["postal_code"], **kwargs):
        customers = customers.drop(dropcol, axis=1)
        customers_col = list(customers.columns)

        if "fashion_news_frequency" in customers_col:
            customers["fashion_news_frequency"] = customers[
                "fashion_news_frequency"
            ].replace("NONE", "None")
            customers["fashion_news_frequency"] = customers[
                "fashion_news_frequency"
            ].replace({np.nan: 0, "None": 0, "Monthly": 1, "Regularly": 2})

        if "club_member_status" in customers_col:
            customers["club_member_status"] = customers["club_member_status"].replace(
                {np.nan: 0, "PRE-CREATE": 1, "ACTIVE": 2, "LEFT CLUB": -1}
            )

        if "age" in customers_col:
            customers["age"] = customers["age"].fillna(-1)

        if "FN" in customers_col:
            customers["FN"] = customers["FN"].fillna(0)

        if "Active" in customers_col:
            customers["Active"] = customers["Active"].fillna(0)

            print(f"###NULL DESCRIPTION###\n{customers.isnull().sum()}")

        return customers

    def clustering(
        self, df, predcol, usecol, normmethod="StandardScaler", clusters=12, DEBUG=False
    ):
        X = np.array(df[usecol])

        if normmethod == "StandardScaler":
            nm = preprocessing.StandardScaler()
            X = nm.fit_transform(X)
        elif normmethod == "minMax":
            nm = preprocessing.MinMaxScaler()
            X = nm.fit_transform(X)
        print(f"NormarlizationMethod:{normmethod}")

        km = KMeans(n_clusters=clusters, random_state=2022)
        km.fit(X)
        print("Distortion: %.2f" % km.inertia_)

        pred = km.labels_
        df_pred = pd.DataFrame(pred, columns=["pred"])
        df_pred = pd.concat([df, df_pred], axis=1)

        df_norm = pd.DataFrame(X, columns=usecol)
        print(df_norm.describe())

        if DEBUG:
            df_norm = pd.concat([df[predcol], df_norm], axis=1)
            return df_pred, df_norm
        else:
            return df_pred




## === cell 2
DEBUG = False

PATH_INPUT = r"../input/h-and-m-personalized-fashion-recommendations/"

if not os.path.exists(PATH_INPUT):
    PATH_INPUT = r"/kaggle/input/h-and-m-personalized-fashion-recommendations/"

print("Using PATH_INPUT:", PATH_INPUT)



## === cell 3
customers = pd.read_csv(PATH_INPUT + "customers.csv")

clst = Clustering_HandM()
customers = clst.customers_preprocessing(customers)

usecol = ["club_member_status", "fashion_news_frequency", "age", "FN", "Active"]
predcol = ["customer_id"]

dfCustomers = clst.clustering(customers, predcol=predcol, usecol=usecol)
dfCustomers.head()



## === cell 4
listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
dfCustomers["age_bins"] = pd.cut(dfCustomers["age"], listBin)
dfCustomers["age_bins"].value_counts(dropna=False).head()



## === cell 5
_ = pd.crosstab(dfCustomers["pred"], dfCustomers["age_bins"])
dfCustomers = dfCustomers.drop(["age_bins"], axis=1)



## === cell 6
dfTransactions = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"article_id": "int32", "t_dat": "string", "customer_id": "string"},
)
dfTransactions["t_dat"] = cudf.to_datetime(dfTransactions["t_dat"])
dfTransactions = dfTransactions.set_index("t_dat")

dfTransactions = dfTransactions.sort_index()
dfTransactions.head()



## === cell 7
dfRecent = dfTransactions.loc["2020-09-01":"2020-09-21"]
dfRecent.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1916296057.py in <cell line: 0>()
      1 # Bugfix: with sorted datetime index, partial label slicing works.
----> 2 dfRecent = dfTransactions.loc["2020-09-01":"2020-09-21"]
      3 dfRecent.head()
      4 

/usr/local/lib/python3.11/dist-packages/cudf/core/dataframe.py in __getitem__(self, arg)
    138             if not isinstance(arg, tuple):
    139                 arg = (arg, slice(None))
--> 140             return self._getitem_tuple_arg(arg)
    141 
    142     def __setitem__(self, key, value):

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/dataframe.py in _getitem_tuple_arg(self, arg)
    275         else:
    276             if isinstance(arg[0], slice):
--> 277                 out = _get_label_range_or_mask(
    278                     columns_df.index, arg[0].start, arg[0].stop, arg[0].step
    279                 )

/usr/local/lib/python3.11/dist-packages/cudf/core/indexed_frame.py in _get_label_range_or_mask(index, start, stop, step)
    201                 return slice(start_loc, stop_loc)
    202             else:
--> 203                 raise KeyError(
    204                     "Value based partial slicing on non-monotonic "
    205                     "DatetimeIndexes with non-existing keys is not allowed.",

KeyError: 'Value based partial slicing on non-monotonic DatetimeIndexes with non-existing keys is not allowed.'

## === cell 8
dfRecent_pd = dfRecent.to_pandas()
dfRecent_pd = dfRecent_pd.merge(
    dfCustomers[["customer_id", "pred"]], on="customer_id", how="inner"
)
dfRecent_pd.head()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1805035195.py in <cell line: 0>()
      1 # Merge recent transactions with customer cluster labels for any analysis
----> 2 dfRecent_pd = dfRecent.to_pandas()
      3 dfRecent_pd = dfRecent_pd.merge(
      4     dfCustomers[["customer_id", "pred"]], on="customer_id", how="inner"
      5 )

NameError: name 'dfRecent' is not defined

## === cell 9
dfRecentAgg = (
    dfRecent_pd.groupby(["pred", "article_id"])["customer_id"]
    .count()
    .reset_index()
    .rename(columns={"customer_id": "counts"})
)
listUniBins_recent = dfRecentAgg["pred"].unique().tolist()

dict100 = {}
for uniBin in listUniBins_recent:
    dfTemp = dfRecentAgg[dfRecentAgg["pred"] == uniBin].sort_values(
        by="counts", ascending=False
    )
    dict100[uniBin] = dfTemp.head(100)["article_id"].values.tolist()

df100 = pd.DataFrame([dict100]).T.rename(columns={0: "top100"})
df100.head()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3654465740.py in <cell line: 0>()
      1 # Build top-100 per cluster from recent period (kept for continuity; not essential for final submission)
      2 dfRecentAgg = (
----> 3     dfRecent_pd.groupby(["pred", "article_id"])["customer_id"]
      4     .count()
      5     .reset_index()

NameError: name 'dfRecent_pd' is not defined

## === cell 10
if df100.shape[0] > 1:
    dfOverlap = df100.copy()
    for index in dfOverlap.index:
        dfOverlap[index] = [
            len(set(dfOverlap.at[index, "top100"]) & set(dfOverlap.at[x, "top100"]))
            / 100
            for x in dfOverlap.index
        ]
    dfOverlap = dfOverlap.drop(columns="top100")
    plt.figure(figsize=(10, 6))
    sns.heatmap(dfOverlap, annot=False, cbar=False)
    plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3628711035.py in <cell line: 0>()
      1 # Heatmap overlap (exploratory). Keep but make it robust if some bins are missing.
----> 2 if df100.shape[0] > 1:
      3     dfOverlap = df100.copy()
      4     for index in dfOverlap.index:
      5         dfOverlap[index] = [

NameError: name 'df100' is not defined

## === cell 11
N = 12
listUniBins = sorted(dfCustomers["pred"].dropna().unique().tolist())
print("Num clusters in dfCustomers:", len(listUniBins))
print("Clusters:", listUniBins[:20], "..." if len(listUniBins) > 20 else "")



## === cell 12
df_all = cudf.read_csv(
    PATH_INPUT + "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"article_id": "int32", "t_dat": "string", "customer_id": "string"},
)
df_all["t_dat"] = cudf.to_datetime(df_all["t_dat"])

sub_all = cudf.read_csv(
    PATH_INPUT + "sample_submission.csv",
    usecols=["customer_id"],
    dtype={"customer_id": "string"},
)
numCustomers = int(sub_all.shape[0])

dfCustomers_cu = cudf.from_pandas(dfCustomers[["customer_id", "pred", "age"]])

for uniBin in listUniBins:
    dfCustomersTemp = dfCustomers_cu[dfCustomers_cu["pred"] == uniBin][
        ["customer_id", "age"]
    ]

    df = df_all.merge(dfCustomersTemp, on="customer_id", how="inner")
    print(f"The shape of scope transaction for cluster {uniBin} is {df.shape}.")

    df["customer_id"] = df["customer_id"].str[-16:].str.hex_to_int().astype("int64")
    last_ts = df["t_dat"].max()

    tmp = df[["t_dat"]].copy().to_pandas()
    tmp["dow"] = tmp["t_dat"].dt.dayofweek
    tmp["ldbw"] = tmp["t_dat"] - pd.TimedeltaIndex(tmp["dow"] - 1, unit="D")
    tmp.loc[tmp["dow"] >= 2, "ldbw"] = tmp.loc[
        tmp["dow"] >= 2, "ldbw"
    ] + pd.TimedeltaIndex(np.ones(len(tmp.loc[tmp["dow"] >= 2])) * 7, unit="D")
    df["ldbw"] = tmp["ldbw"].values

    weekly_sales = (
        df.drop("customer_id", axis=1)
        .groupby(["ldbw", "article_id"])
        .count()
        .reset_index()
    )
    weekly_sales = weekly_sales.rename(columns={"t_dat": "count"})

    df = df.merge(weekly_sales, on=["ldbw", "article_id"], how="left")

    weekly_sales = weekly_sales.reset_index().set_index("article_id")
    df = df.merge(
        weekly_sales.loc[weekly_sales["ldbw"] == last_ts, ["count"]],
        on="article_id",
        suffixes=("", "_targ"),
    )
    df["count_targ"].fillna(0, inplace=True)
    del weekly_sales

    df["quotient"] = df["count_targ"] / df["count"]

    target_sales = (
        df.drop("customer_id", axis=1).groupby("article_id")["quotient"].sum()
    )
    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = ["0" + str(article_id) for article_id in general_pred]
    general_pred_str = " ".join(general_pred)
    del target_sales

    tmp = df.copy().to_pandas()
    tmp["x"] = ((last_ts.to_pandas() - tmp["t_dat"]) / np.timedelta64(1, "D")).astype(
        int
    )
    tmp["dummy_1"] = 1
    tmp["x"] = tmp[["x", "dummy_1"]].max(axis=1)

    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    tmp["y"] = a / np.sqrt(tmp["x"]) + b * np.exp(-c * tmp["x"]) - d
    tmp["dummy_0"] = 0
    tmp["y"] = tmp[["y", "dummy_0"]].max(axis=1)
    tmp["value"] = tmp["quotient"] * tmp["y"]

    tmp = tmp.groupby(["customer_id", "article_id"]).agg({"value": "sum"}).reset_index()
    tmp = tmp.loc[tmp["value"] > 0]
    tmp["rank"] = tmp.groupby("customer_id")["value"].rank("dense", ascending=False)
    tmp = tmp.loc[tmp["rank"] <= 12]

    purchase_df = tmp.sort_values(
        ["customer_id", "value"], ascending=False
    ).reset_index(drop=True)
    purchase_df["prediction"] = "0" + purchase_df["article_id"].astype(str) + " "
    purchase_df = (
        purchase_df.groupby("customer_id").agg({"prediction": sum}).reset_index()
    )
    purchase_df["prediction"] = purchase_df["prediction"].str.strip()
    purchase_df = cudf.DataFrame(purchase_df)

    sub = sub_all.merge(dfCustomersTemp, on="customer_id", how="inner")
    sub["customer_id2"] = sub["customer_id"].str[-16:].str.hex_to_int().astype("int64")

    sub = sub.merge(
        purchase_df,
        left_on="customer_id2",
        right_on="customer_id",
        how="left",
        suffixes=("", "_ignored"),
    )

    sub = sub.to_pandas()
    sub["prediction"] = sub["prediction"].fillna(general_pred_str)
    sub["prediction"] = (sub["prediction"] + " " + general_pred_str).str.strip()
    sub["prediction"] = sub["prediction"].str[:131]
    sub = sub[["customer_id", "prediction"]]
    sub.to_csv(f"submission_{uniBin}.csv", index=False)
    print(f"Saved prediction for cluster {uniBin}. The shape is {sub.shape}.")
    print("-" * 50)

print("Finished per-cluster files.\n" + "=" * 50)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1106264172.py in <cell line: 0>()
     69 
     70     tmp = df.copy().to_pandas()
---> 71     tmp["x"] = ((last_ts.to_pandas() - tmp["t_dat"]) / np.timedelta64(1, "D")).astype(
     72         int
     73     )

AttributeError: 'numpy.datetime64' object has no attribute 'to_pandas'

## === cell 13
dfSub_list = []
for i, uniBin in enumerate(listUniBins):
    dfTemp = pd.read_csv(f"submission_{uniBin}.csv")
    dfSub_list.append(dfTemp)

dfSub = pd.concat(dfSub_list, axis=0, ignore_index=True)

sample_sub = pd.read_csv(PATH_INPUT + "sample_submission.csv", usecols=["customer_id"])
dfSub = sample_sub.merge(dfSub, on="customer_id", how="left")

if dfSub["prediction"].isna().any():
    df_freq = cudf.read_csv(
        PATH_INPUT + "transactions_train.csv",
        usecols=["article_id"],
        dtype={"article_id": "int32"},
    )
    top_global = (
        df_freq["article_id"].value_counts().head(12).index.to_pandas().tolist()
    )
    top_global = ["0" + str(a) for a in top_global]
    top_global_str = " ".join(top_global)
    dfSub["prediction"] = dfSub["prediction"].fillna(top_global_str)

assert (
    dfSub.shape[0] == sample_sub.shape[0]
), f"Row count mismatch: {dfSub.shape[0]} vs {sample_sub.shape[0]}"
assert dfSub["customer_id"].isna().sum() == 0
assert dfSub["prediction"].isna().sum() == 0

dfSub.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", dfSub.shape)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1925288210.py in <cell line: 0>()
      2 dfSub_list = []
      3 for i, uniBin in enumerate(listUniBins):
----> 4     dfTemp = pd.read_csv(f"submission_{uniBin}.csv")
      5     dfSub_list.append(dfTemp)
      6 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission_0.csv'

## === cell 14
dfCheck = pd.read_csv("./submission.csv")
print(dfCheck.head())
print(dfCheck.columns.tolist())
print("Rows:", dfCheck.shape[0], "Unique customers:", dfCheck["customer_id"].nunique())
print("Max prediction length:", dfCheck["prediction"].map(len).max())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2216200773.py in <cell line: 0>()
      1 # Final sanity check
----> 2 dfCheck = pd.read_csv("./submission.csv")
      3 print(dfCheck.head())
      4 print(dfCheck.columns.tolist())
      5 print("Rows:", dfCheck.shape[0], "Unique customers:", dfCheck["customer_id"].nunique())

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './submission.csv'
