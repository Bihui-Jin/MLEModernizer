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

0.0227384268563794

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import sys, warnings, time, os, copy, gc, re, random, pickle, cudf
warnings.filterwarnings('ignore')
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
from pandas.io.json import json_normalize
from pprint import pprint
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()
from collections import Counter

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
CUDARuntimeError                          Traceback (most recent call last)
/tmp/ipykernel_11/558540708.py in <cell line: 0>()
      1 # === General ===
----> 2 import sys, warnings, time, os, copy, gc, re, random, pickle, cudf
      3 warnings.filterwarnings('ignore')
      4 from IPython.display import display
      5 import matplotlib.pyplot as plt

/usr/local/lib/python3.11/dist-packages/cudf/__init__.py in <module>
     18 
     19 _setup_numba()
---> 20 validate_setup()
     21 
     22 import cupy

/usr/local/lib/python3.11/dist-packages/cudf/utils/gpu_utils.py in validate_setup()
     53     except CUDARuntimeError as e:
     54         if e.status in notify_caller_errors:
---> 55             raise e
     56         # If there is no GPU detected, set `gpus_count` to -1
     57         gpus_count = -1

/usr/local/lib/python3.11/dist-packages/cudf/utils/gpu_utils.py in validate_setup()
     50 
     51     try:
---> 52         gpus_count = getDeviceCount()
     53     except CUDARuntimeError as e:
     54         if e.status in notify_caller_errors:

/usr/local/lib/python3.11/dist-packages/rmm/_cuda/gpu.py in getDeviceCount()
    100     status, count = runtime.cudaGetDeviceCount()
    101     if status != runtime.cudaError_t.cudaSuccess:
--> 102         raise CUDARuntimeError(status)
    103     return count
    104 

CUDARuntimeError: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version

## === cell 3
DEBUG = False

## === cell 5
def display_df(df, head=3):
    print(f'The shape of df is {df.shape}.\n')
    display(df.head(head))

## === cell 6
dfSub1 = cudf.read_csv('../input/h-m-eda-rule-base-by-customer-age/submission.csv')
display_df(dfSub1, head=3)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2306085524.py in <cell line: 0>()
----> 1 dfSub1 = cudf.read_csv('../input/h-m-eda-rule-base-by-customer-age/submission.csv')
      2 display_df(dfSub1, head=3)

NameError: name 'cudf' is not defined

## === cell 7
dfSub2 = cudf.read_csv('../input/h-m-framework-for-partitioned-validation/submission.csv')
display_df(dfSub2, head=3)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/512577641.py in <cell line: 0>()
----> 1 dfSub2 = cudf.read_csv('../input/h-m-framework-for-partitioned-validation/submission.csv')
      2 display_df(dfSub2, head=3)

NameError: name 'cudf' is not defined

## === cell 8
dfSub3 = cudf.read_csv('../input/trending/submission.csv')
display_df(dfSub3, head=3)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3015330997.py in <cell line: 0>()
----> 1 dfSub3 = cudf.read_csv('../input/trending/submission.csv')
      2 display_df(dfSub3, head=3)

NameError: name 'cudf' is not defined

## === cell 9
dfSub4 = cudf.read_csv('../input/lb-0-0236-ensemble-gives-you-bronze-medal/submission.csv')
display_df(dfSub4, head=3)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1207614549.py in <cell line: 0>()
----> 1 dfSub4 = cudf.read_csv('../input/lb-0-0236-ensemble-gives-you-bronze-medal/submission.csv')
      2 display_df(dfSub4, head=3)

NameError: name 'cudf' is not defined

## === cell 10
dfSub5 = cudf.read_csv('../input/h-m-ensembling-with-lstm-be7226/submission.csv')
display_df(dfSub5, head=3)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/119004004.py in <cell line: 0>()
----> 1 dfSub5 = cudf.read_csv('../input/h-m-ensembling-with-lstm-be7226/submission.csv')
      2 display_df(dfSub5, head=3)

NameError: name 'cudf' is not defined

## === cell 11
dfSub1.columns = ['customer_id', 'prediction1']
dfSub2.columns = ['customer_id', 'prediction2']
dfSub3.columns = ['customer_id', 'prediction3']
dfSub4.columns = ['customer_id', 'prediction4']
dfSub5.columns = ['customer_id', 'prediction5']

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2443291276.py in <cell line: 0>()
----> 1 dfSub1.columns = ['customer_id', 'prediction1']
      2 dfSub2.columns = ['customer_id', 'prediction2']
      3 dfSub3.columns = ['customer_id', 'prediction3']
      4 dfSub4.columns = ['customer_id', 'prediction4']
      5 dfSub5.columns = ['customer_id', 'prediction5']

NameError: name 'dfSub1' is not defined

## === cell 13
dfSub = dfSub1.merge(dfSub2, on='customer_id', how='left')
dfSub = dfSub.merge(dfSub3, on='customer_id', how='left')
dfSub = dfSub.merge(dfSub4, on='customer_id', how='left')
dfSub = dfSub.merge(dfSub5, on='customer_id', how='left')
display_df(dfSub, head=3)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2931154861.py in <cell line: 0>()
----> 1 dfSub = dfSub1.merge(dfSub2, on='customer_id', how='left')
      2 dfSub = dfSub.merge(dfSub3, on='customer_id', how='left')
      3 dfSub = dfSub.merge(dfSub4, on='customer_id', how='left')
      4 dfSub = dfSub.merge(dfSub5, on='customer_id', how='left')
      5 display_df(dfSub, head=3)

NameError: name 'dfSub1' is not defined

## === cell 14
if DEBUG:
    dfSub = dfSub.sample(frac=0.001, random_state=7)

## === cell 15
def select_topn(x, n=12):
    listX = str(x).split()
    c = Counter(listX)
    values, counts = zip(*c.most_common(n))
    listY = ' '.join(values)
    return listY    

## === cell 16
dfSub['pred_sum'] = dfSub['prediction1'] + ' ' + dfSub['prediction2'] + ' ' + dfSub['prediction3'] + ' ' + dfSub['prediction4'] + ' ' + dfSub['prediction5']

dfSub = dfSub.to_pandas()

dfSub['pred_top12'] = dfSub['pred_sum'].progress_apply(select_topn)

display_df(dfSub, head=3)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1571752386.py in <cell line: 0>()
----> 1 dfSub['pred_sum'] = dfSub['prediction1'] + ' ' + dfSub['prediction2'] + ' ' + dfSub['prediction3'] + ' ' + dfSub['prediction4'] + ' ' + dfSub['prediction5']
      2 
      3 dfSub = dfSub.to_pandas()
      4 
      5 dfSub['pred_top12'] = dfSub['pred_sum'].progress_apply(select_topn)

NameError: name 'dfSub' is not defined

## === cell 17
print(dfSub['prediction1'][248891])
print('')
print(dfSub['prediction2'][248891])
print('')
print(dfSub['prediction3'][248891])
print('')
print(dfSub['prediction4'][248891])
print('')
print(dfSub['prediction5'][248891])
print('')
print(dfSub['pred_sum'][248891])
print('')
print(dfSub['pred_top12'][248891])

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/370546065.py in <cell line: 0>()
----> 1 print(dfSub['prediction1'][248891])
      2 print('')
      3 print(dfSub['prediction2'][248891])
      4 print('')
      5 print(dfSub['prediction3'][248891])

NameError: name 'dfSub' is not defined

## === cell 18
dfSampleSub = dfSub[['customer_id', 'pred_top12']]
display_df(dfSampleSub, head=3)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3369362045.py in <cell line: 0>()
----> 1 dfSampleSub = dfSub[['customer_id', 'pred_top12']]
      2 display_df(dfSampleSub, head=3)

NameError: name 'dfSub' is not defined

## === cell 19
dfSampleSub.columns = ['customer_id', 'prediction']

dfSampleSub.to_csv(f'submission.csv', index=False)
print(f'Saved submission.csv.')

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1449887108.py in <cell line: 0>()
----> 1 dfSampleSub.columns = ['customer_id', 'prediction']
      2 
      3 dfSampleSub.to_csv(f'submission.csv', index=False)
      4 print(f'Saved submission.csv.')

NameError: name 'dfSampleSub' is not defined

## === cell 20
dfCheck = cudf.read_csv('./submission.csv')
display_df(dfCheck, head=3)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2738376280.py in <cell line: 0>()
----> 1 dfCheck = cudf.read_csv('./submission.csv')
      2 display_df(dfCheck, head=3)

NameError: name 'cudf' is not defined
