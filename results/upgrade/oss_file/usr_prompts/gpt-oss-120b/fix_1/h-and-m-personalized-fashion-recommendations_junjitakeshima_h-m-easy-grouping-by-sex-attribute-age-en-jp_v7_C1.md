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

geopandas==0.14.4
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
pillow==11.3.0
seaborn==0.12.2
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

0.00775

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from datetime import datetime, date, timedelta

from collections import Counter, defaultdict
from PIL import Image
from pathlib import Path

path = Path("/kaggle/input/h-and-m-personalized-fashion-recommendations/")


## === cell 1
transactions_df = pd.read_csv(path / "transactions_train.csv", dtype = {'article_id': str})
articles_df = pd.read_csv(path / "articles.csv", dtype = {'article_id': str})
customers_df = pd.read_csv(path / "customers.csv")
submission = pd.read_csv(path / "sample_submission.csv")


## === cell 2
transactions_df["t_dat"] = pd.to_datetime(transactions_df['t_dat'])
transactions_df


## === cell 3
customers_df


## === cell 4
customers_df.isnull().sum()


## === cell 5
customers_df["age"].plot.hist(bins=50)


## === cell 6
age_id = 0
age_group = pd.DataFrame(columns =["age","age_id"])
age=16

for i in range(53) :
    if age < 30 :
        temp_group = pd.DataFrame({"age":[age, age+1], "age_id":[age_id, age_id]})
        age_group = age_group.append(temp_group)
        age += 2
        age_id += 1
    elif  age < 60 :
        temp_group = pd.DataFrame({"age":[age, age+1, age+2, age+3, age+4],"age_id":[age_id, age_id, age_id, age_id, age_id]})
        age_group = age_group.append(temp_group)
        age += 5
        age_id += 1
    else:
        temp_group = pd.DataFrame({"age":[age], "age_id":[age_id]})
        age_group = age_group.append(temp_group)
        age += 1


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1723353563.py in <cell line: 0>()
      6     if age < 30 :
      7         temp_group = pd.DataFrame({"age":[age, age+1], "age_id":[age_id, age_id]})
----> 8         age_group = age_group.append(temp_group)
      9         age += 2
     10         age_id += 1

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 7
age_group


## === cell 8
customers_df = pd.merge(customers_df, age_group, on="age", how = "left")
customers_df = customers_df.drop(["FN", "Active", "club_member_status", "fashion_news_frequency", "postal_code"], axis=1)
customers_df


## === cell 9
articles_df.head(5)


## === cell 10
print(articles_df["index_group_name"].unique())
print(articles_df["index_group_no"].unique())


## === cell 11
sex_category = articles_df[["index_group_no", "index_group_name"]].reset_index()
display(sex_category["index_group_name"].value_counts())


## === cell 12
sex_category_list = sex_category["index_group_name"].value_counts().index.to_list()
plt.figure(figsize=(5, 5))
plt.rcParams["font.size"] = 12
plt.pie(sex_category["index_group_name"].value_counts().sort_values(ascending=False), 
        labels = sex_category_list, startangle = 90, counterclock=False, autopct="%1.1f%%")
plt.show


## === cell 13
del sex_category_list


## === cell 14
articles_category_df = pd.DataFrame(articles_df[["article_id", "index_group_no"]])
articles_category_df.columns = ["article_id", "sex_attribute"]
articles_category_df


## === cell 15
transactions_df = pd.merge(transactions_df, articles_category_df, on = "article_id", how = "left")
transactions_df


## === cell 16
cust_sex = transactions_df[["customer_id", "sex_attribute", "article_id"]].groupby(["customer_id","sex_attribute"]).count().unstack()
cust_sex.columns = ["Woman", "Young", "Man", "Have-kids", "Sports-person"]
cust_sex


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2258658811.py in <cell line: 0>()
      1 cust_sex = transactions_df[["customer_id", "sex_attribute", "article_id"]].groupby(["customer_id","sex_attribute"]).count().unstack()
----> 2 cust_sex.columns = ["Woman", "Young", "Man", "Have-kids", "Sports-person"]
      3 cust_sex

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __setattr__(self, name, value)
   6311         try:
   6312             object.__getattribute__(self, name)
-> 6313             return object.__setattr__(self, name, value)
   6314         except AttributeError:
   6315             pass

properties.pyx in pandas._libs.properties.AxisProperty.__set__()

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _set_axis(self, axis, labels)
    812         """
    813         labels = ensure_index(labels)
--> 814         self._mgr.set_axis(axis, labels)
    815         self._clear_item_cache()
    816 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in set_axis(self, axis, new_labels)
    236     def set_axis(self, axis: AxisInt, new_labels: Index) -> None:
    237         # Caller is responsible for ensuring we have an Index object.
--> 238         self._validate_set_axis(axis, new_labels)
    239         self.axes[axis] = new_labels
    240 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/base.py in _validate_set_axis(self, axis, new_labels)
     96 
     97         elif new_len != old_len:
---> 98             raise ValueError(
     99                 f"Length mismatch: Expected axis has {old_len} elements, new "
    100                 f"values have {new_len} elements"

ValueError: Length mismatch: Expected axis has 0 elements, new values have 5 elements

## === cell 17
%%time
cust_sex["attribute"] = cust_sex.apply(lambda x : list(x[x == x.max()].index), axis=1)
cust_sex


## === cell 18
cust_sex1 = pd.DataFrame(cust_sex[["attribute"]]).reset_index()
cust_sex1["attribute"] = cust_sex1["attribute"].apply(",".join).astype(str)
del cust_sex
cust_sex1


## === cell 19
print(cust_sex1.attribute.unique())


## === cell 20
cust_sex1.loc[~((cust_sex1["attribute"] == "Woman") |
               (cust_sex1["attribute"] == "Young")  |
               (cust_sex1["attribute"] == "Man")    |
               (cust_sex1["attribute"] == "Have-kids") |
               (cust_sex1["attribute"] == "Sports-person")), "attribute"] = "Woman"
cust_sex1


## === cell 21
print(cust_sex1.attribute.unique())


## === cell 22
temp = cust_sex1["attribute"].value_counts().index.to_list()
plt.figure(figsize=(5, 5))
plt.rcParams["font.size"] = 12
plt.pie(cust_sex1["attribute"].value_counts().sort_values(ascending=False), 
        labels = temp, startangle = 90, counterclock=False, autopct="%1.1f%%")
plt.show


## === cell 23
print(cust_sex1["attribute"].value_counts().sort_values(ascending=False))


## === cell 24
customers_df = pd.merge(customers_df, cust_sex1, on ="customer_id", how="left")
customers_df


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
MergeError                                Traceback (most recent call last)
/tmp/ipykernel_11/2987360737.py in <cell line: 0>()
----> 1 customers_df = pd.merge(customers_df, cust_sex1, on ="customer_id", how="left")
      2 customers_df

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    782                 f"{_right.columns.nlevels} on the right)"
    783             )
--> 784             raise MergeError(msg)
    785 
    786         self.left_on, self.right_on = self._validate_left_right_on(left_on, right_on)

MergeError: Not allowed to merge between different levels. (1 levels on the left, 2 on the right)

## === cell 25
customers_df.isnull().sum()


## === cell 26
customers_df["attribute"].fillna("Woman", inplace = True)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'attribute'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1548564198.py in <cell line: 0>()
----> 1 customers_df["attribute"].fillna("Woman", inplace = True)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'attribute'

## === cell 27
age_mean = customers_df[["age", "attribute"]].groupby("attribute").mean().round().reset_index()
age_mean.columns = ["attribute", "age_mean"]
age_mean


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3575015157.py in <cell line: 0>()
----> 1 age_mean = customers_df[["age", "attribute"]].groupby("attribute").mean().round().reset_index()
      2 age_mean.columns = ["attribute", "age_mean"]
      3 age_mean

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['attribute'] not in index"

## === cell 28
customers_df = pd.merge(customers_df, age_mean, on = "attribute", how ="left")
customers_df.loc[(customers_df["age"].isnull()), "age"] = customers_df["age_mean"]
customers_df = customers_df.drop(["age_mean", "age_id"], axis =1)
customers_df = pd.merge(customers_df, age_group, on="age", how="left")
customers_df.isnull().sum()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1415590870.py in <cell line: 0>()
----> 1 customers_df = pd.merge(customers_df, age_mean, on = "attribute", how ="left")
      2 customers_df.loc[(customers_df["age"].isnull()), "age"] = customers_df["age_mean"]
      3 customers_df = customers_df.drop(["age_mean", "age_id"], axis =1)
      4 customers_df = pd.merge(customers_df, age_group, on="age", how="left")
      5 customers_df.isnull().sum()

NameError: name 'age_mean' is not defined

## === cell 29
transactions_df = pd.merge(transactions_df, customers_df, on ="customer_id", how ="left")
transactions_df.isnull().sum()


## === cell 30
del cust_sex1


## === cell 31
transactions_df = transactions_df.loc[transactions_df.t_dat >= pd.to_datetime('2020-09-01')]
transactions_df


## === cell 32
transactions_df.article_id = ' ' + transactions_df.article_id.astype('str')
temp = transactions_df.groupby(['age_id','attribute','article_id'])['customer_id'].agg('count').reset_index()
temp.columns = ['age_id','attribute','article_id','count']
transactions_df = transactions_df.merge(temp, on=['age_id','attribute','article_id'], how='left')
transactions_df


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2705953135.py in <cell line: 0>()
      1 transactions_df.article_id = ' ' + transactions_df.article_id.astype('str')
----> 2 temp = transactions_df.groupby(['age_id','attribute','article_id'])['customer_id'].agg('count').reset_index()
      3 temp.columns = ['age_id','attribute','article_id','count']
      4 transactions_df = transactions_df.merge(temp, on=['age_id','attribute','article_id'], how='left')
      5 transactions_df

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'attribute'

## === cell 33
transactions_df = transactions_df.sort_values(['count','t_dat'],ascending=False)
transactions_df = transactions_df.drop_duplicates(['age_id','attribute','article_id'])
transactions_df


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3830155628.py in <cell line: 0>()
----> 1 transactions_df = transactions_df.sort_values(['count','t_dat'],ascending=False)
      2 transactions_df = transactions_df.drop_duplicates(['age_id','attribute','article_id'])
      3 transactions_df

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in sort_values(self, by, axis, ascending, inplace, kind, na_position, ignore_index, key)
   7170             )
   7171         if len(by) > 1:
-> 7172             keys = [self._get_label_or_level_values(x, axis=axis) for x in by]
   7173 
   7174             # need to rewrap columns in Series to apply key function

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in <listcomp>(.0)
   7170             )
   7171         if len(by) > 1:
-> 7172             keys = [self._get_label_or_level_values(x, axis=axis) for x in by]
   7173 
   7174             # need to rewrap columns in Series to apply key function

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1909             values = self.axes[axis].get_level_values(key)._values
   1910         else:
-> 1911             raise KeyError(key)
   1912 
   1913         # Check for duplicates

KeyError: 'count'

## === cell 34
def show_images(article_ids, cols=1, rows=-1):
    if isinstance(article_ids, int) or isinstance(article_ids, str):
        article_ids = [article_ids]
    article_count = len(article_ids)
    if rows < 0: rows = (article_count // cols) + 1
    plt.figure(figsize=(3 + 3.5 * cols, 3 + 5 * rows))
    for i in range(article_count):
        article_id = ("0" + str(article_ids[i]))[-10:]
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        plt.title(article_id)
        try:
            image = Image.open(f"/kaggle/input/h-and-m-personalized-fashion-recommendations/images/{article_id[:3]}/{article_id}.jpg")
            plt.imshow(image)
        except:
            pass


## === cell 35
show_images("0915526001")


## === cell 36
recommend_sex_age = pd.DataFrame(transactions_df.groupby(["age_id", "attribute"]).article_id.sum().reset_index())
recommend_sex_age["len"] = recommend_sex_age["article_id"].apply(lambda x : len(x))
recommend_sex_age


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2487538585.py in <cell line: 0>()
----> 1 recommend_sex_age = pd.DataFrame(transactions_df.groupby(["age_id", "attribute"]).article_id.sum().reset_index())
      2 recommend_sex_age["len"] = recommend_sex_age["article_id"].apply(lambda x : len(x))
      3 recommend_sex_age

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'attribute'

## === cell 37
recommend_sex_age["article_id"] = recommend_sex_age["article_id"].str.strip()
recommend_sex_age["article_id"] = recommend_sex_age["article_id"].str[:131]
recommend_sex_age = recommend_sex_age.drop(["len"], axis =1)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/882048515.py in <cell line: 0>()
----> 1 recommend_sex_age["article_id"] = recommend_sex_age["article_id"].str.strip()
      2 recommend_sex_age["article_id"] = recommend_sex_age["article_id"].str[:131]
      3 recommend_sex_age = recommend_sex_age.drop(["len"], axis =1)

NameError: name 'recommend_sex_age' is not defined

## === cell 38
submission = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
submission = submission[['customer_id']]
submission = pd.merge(submission, customers_df, on = "customer_id", how = "left")
submission = pd.merge(submission, recommend_sex_age, on = ["age_id", "attribute"], how="left")
submission = submission.drop(["age", "age_id", "attribute"], axis =1)
submission.columns = ("customer_id", "prediction")
submission.to_csv("submission.csv",index=False)
submission


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4155878727.py in <cell line: 0>()
      2 submission = submission[['customer_id']]
      3 submission = pd.merge(submission, customers_df, on = "customer_id", how = "left")
----> 4 submission = pd.merge(submission, recommend_sex_age, on = ["age_id", "attribute"], how="left")
      5 submission = submission.drop(["age", "age_id", "attribute"], axis =1)
      6 submission.columns = ("customer_id", "prediction")

NameError: name 'recommend_sex_age' is not defined
