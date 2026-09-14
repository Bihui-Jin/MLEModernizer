# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import pandas as pd

from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

import warnings
warnings.filterwarnings("ignore")




train_x= pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test_x= pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
train_x.dropna(inplace=True)
train_x.head()


## === cell 1
train_x.describe()


## === cell 2
import re
import numpy as np

def number_words(text):
    text=re.sub(r'[^\w\s]','',text)
    text.strip()    
    text_list=text.split()
    return len(text_list)
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
df=train_x.assign(jaccard_score=np.nan)
df["jaccard_score"]= [jaccard(df.at[i,"selected_text"],df.at[i,"text"]) for i in df.index]
df["St_words_number"]= [number_words(df.at[i,"selected_text"]) for i in df.index]
df["text_word_number"]= [number_words(df.at[i,"text"]) for i in df.index]
df["diff_number_words"]= df["text_word_number"]-df["St_words_number"]
df.head()


## === cell 3
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.figure(figsize=(12,6))

sns.kdeplot(data=df['text_word_number'], shade=True)
sns.kdeplot(data=df['St_words_number'], shade=True)


## === cell 4
plt.figure(figsize=(12,6))
df_neutral=df[df['sentiment']=='neutral']
plt.figure(figsize=(12,6))
sns.distplot(df_neutral['jaccard_score'],kde=False)


## === cell 5
plt.figure(figsize=(12,6))
df_positive=df[df['sentiment']=='positive']
sns.kdeplot(data=df_positive['jaccard_score'],label="positive",shade=True)


## === cell 6
plt.figure(figsize=(12,6))
df_negative=df[df['sentiment']=='negative']
sns.kdeplot(data=df_negative['jaccard_score'],label="negative",color='red',shade=True)


## === cell 7
k = df[df["text_word_number"] <= 3]

k.groupby("sentiment")["jaccard_score"].mean()


## === cell 8
k=df[df['text_word_number']<=2]
k.groupby('sentiment').mean()['jaccard_score']


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m_agg_py_fallback[0;34m(self, how, values, ndim, alt)[0m
[1;32m   1941[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1942[0;31m             [0mres_values[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_grouper[0m[0;34m.[0m[0magg_series[0m[0;34m([0m[0mser[0m[0;34m,[0m [0malt[0m[0;34m,[0m [0mpreserve_dtype[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1943[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36magg_series[0;34m(self, obj, func, preserve_dtype)[0m
[1;32m    863[0m [0;34m[0m[0m
[0;32m--> 864[0;31m         [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_aggregate_series_pure_python[0m[0;34m([0m[0mobj[0m[0;34m,[0m [0mfunc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    865[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36m_aggregate_series_pure_python[0;34m(self, obj, func)[0m
[1;32m    884[0m         [0;32mfor[0m [0mi[0m[0;34m,[0m [0mgroup[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0msplitter[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 885[0;31m             [0mres[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mgroup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    886[0m             [0mres[0m [0;34m=[0m [0mextract_result[0m[0;34m([0m[0mres[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m<lambda>[0;34m(x)[0m
[1;32m   2453[0m                 [0;34m"mean"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2454[0;31m                 [0malt[0m[0;34m=[0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mSeries[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2455[0m                 [0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/series.py[0m in [0;36mmean[0;34m(self, axis, skipna, numeric_only, **kwargs)[0m
[1;32m   6548[0m     ):
[0;32m-> 6549[0;31m         [0;32mreturn[0m [0mNDFrame[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mself[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m,[0m [0mnumeric_only[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6550[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36mmean[0;34m(self, axis, skipna, numeric_only, **kwargs)[0m
[1;32m  12419[0m     ) -> Series | float:
[0;32m> 12420[0;31m         return self._stat_function(
[0m[1;32m  12421[0m             [0;34m"mean"[0m[0;34m,[0m [0mnanops[0m[0;34m.[0m[0mnanmean[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m,[0m [0mnumeric_only[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m_stat_function[0;34m(self, name, func, axis, skipna, numeric_only, **kwargs)[0m
[1;32m  12376[0m [0;34m[0m[0m
[0;32m> 12377[0;31m         return self._reduce(
[0m[1;32m  12378[0m             [0mfunc[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0mname[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/series.py[0m in [0;36m_reduce[0;34m(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)[0m
[1;32m   6456[0m                 )
[0;32m-> 6457[0;31m             [0;32mreturn[0m [0mop[0m[0;34m([0m[0mdelegate[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6458[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36mf[0;34m(values, axis, skipna, **kwds)[0m
[1;32m    146[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 147[0;31m                 [0mresult[0m [0;34m=[0m [0malt[0m[0;34m([0m[0mvalues[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    148[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36mnew_func[0;34m(values, axis, skipna, mask, **kwargs)[0m
[1;32m    403[0m [0;34m[0m[0m
[0;32m--> 404[0;31m         [0mresult[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mvalues[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0mmask[0m[0;34m=[0m[0mmask[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    405[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36mnanmean[0;34m(values, axis, skipna, mask)[0m
[1;32m    719[0m     [0mthe_sum[0m [0;34m=[0m [0mvalues[0m[0;34m.[0m[0msum[0m[0;34m([0m[0maxis[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype_sum[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 720[0;31m     [0mthe_sum[0m [0;34m=[0m [0m_ensure_numeric[0m[0;34m([0m[0mthe_sum[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    721[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36m_ensure_numeric[0;34m(x)[0m
[1;32m   1700[0m             [0;31m# GH#44008, GH#36703 avoid casting e.g. strings to numeric[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1701[0;31m             [0;32mraise[0m [0mTypeError[0m[0;34m([0m[0;34mf"Could not convert string '{x}' to numeric"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1702[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Could not convert string '0167c461c60af1aa8543a8734230b63e4a49b4cc97afa4ae07fbc22efc4d6ecaa33c8aa7876c66dad18c11a2bf6dd07105c9bc879a25b0cdeaa93be48f74c4f71c44e6dbd2ce66a3b2a0365bb9044322834196cadfb9c95f65f89b26804c67a9d739cbcd234a6c8c359ff15d5ccfd4c6184863433cb03f2ad2d4deeb4302ea061b1db6f6bd82c00404648e1c9bd1d86c69e08f8b684d59b77080c1c8bcfff76285178f7eb64213f65406771b845b3841e2147763f0460d611dc02977bde0ae95d29a4ca6ad5df0caf6f69e50733b921ccd0bde99f08494b93e52800b1cafbadd63ec1c0bed851fdeebda414f566959d384eeb1fdcdcdeb258b89fdde1449917e4ed52c4a60c947090baaf786894b22175e950e706e150b70cfedf94a53a663e7c6021ea46b2b89c0da2b190ba0475aee7fdcf1e95b1c9810abc7072385a6b7ea870375e4c90e2f13043e97f08b57e1317db125d6a52b69d7e10c6ea9affd001e03457069c690e2a2a796ba1be0bae541437764181f409702aacdcb65ec08321ce18078049b72f7e258097a5eb0e374b2c6ca216383e2708719ccbbbe6803f7b732cd6641bed4306d9b852edc3769bc361e245e1559c3c50f9485357290d8d909d39112c89d051d9f3be484af5ef2cf68b15b3f3b0809086deb67c5777182cef9f25996c2368bf982f686d1880523261f7ac5ca5e8698d535f225329e7d96930b5a21be327a15d669' to numeric

The above exception was the direct cause of the following exception:

[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1638589351.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mk[0m[0;34m=[0m[0mdf[0m[0;34m[[0m[0mdf[0m[0;34m[[0m[0;34m'text_word_number'[0m[0;34m][0m[0;34m<=[0m[0;36m2[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mk[0m[0;34m.[0m[0mgroupby[0m[0;34m([0m[0;34m'sentiment'[0m[0;34m)[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0;34m'jaccard_score'[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36mmean[0;34m(self, numeric_only, engine, engine_kwargs)[0m
[1;32m   2450[0m             )
[1;32m   2451[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2452[0;31m             result = self._cython_agg_general(
[0m[1;32m   2453[0m                 [0;34m"mean"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2454[0m                 [0malt[0m[0;34m=[0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mSeries[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m_cython_agg_general[0;34m(self, how, alt, numeric_only, min_count, **kwargs)[0m
[1;32m   1996[0m             [0;32mreturn[0m [0mresult[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1997[0m [0;34m[0m[0m
[0;32m-> 1998[0;31m         [0mnew_mgr[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mgrouped_reduce[0m[0;34m([0m[0marray_func[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1999[0m         [0mres[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_wrap_agged_manager[0m[0;34m([0m[0mnew_mgr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2000[0m         [0;32mif[0m [0mhow[0m [0;32min[0m [0;34m[[0m[0;34m"idxmin"[0m[0;34m,[0m [0;34m"idxmax"[0m[0;34m][0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36mgrouped_reduce[0;34m(self, func)[0m
[1;32m   1467[0m                 [0;31m#  while others do not.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1468[0m                 [0;32mfor[0m [0msb[0m [0;32min[0m [0mblk[0m[0;34m.[0m[0m_split[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1469[0;31m                     [0mapplied[0m [0;34m=[0m [0msb[0m[0;34m.[0m[0mapply[0m[0;34m([0m[0mfunc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1470[0m                     [0mresult_blocks[0m [0;34m=[0m [0mextend_blocks[0m[0;34m([0m[0mapplied[0m[0;34m,[0m [0mresult_blocks[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1471[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py[0m in [0;36mapply[0;34m(self, func, **kwargs)[0m
[1;32m    391[0m         [0mone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    392[0m         """
[0;32m--> 393[0;31m         [0mresult[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mvalues[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    394[0m [0;34m[0m[0m
[1;32m    395[0m         [0mresult[0m [0;34m=[0m [0mmaybe_coerce_values[0m[0;34m([0m[0mresult[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36marray_func[0;34m(values)[0m
[1;32m   1993[0m [0;34m[0m[0m
[1;32m   1994[0m             [0;32massert[0m [0malt[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1995[0;31m             [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_agg_py_fallback[0m[0;34m([0m[0mhow[0m[0;34m,[0m [0mvalues[0m[0;34m,[0m [0mndim[0m[0;34m=[0m[0mdata[0m[0;34m.[0m[0mndim[0m[0;34m,[0m [0malt[0m[0;34m=[0m[0malt[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1996[0m             [0;32mreturn[0m [0mresult[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1997[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m_agg_py_fallback[0;34m(self, how, values, ndim, alt)[0m
[1;32m   1944[0m             [0mmsg[0m [0;34m=[0m [0;34mf"agg function failed [how->{how},dtype->{ser.dtype}]"[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1945[0m             [0;31m# preserve the kind of exception that raised[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1946[0;31m             [0;32mraise[0m [0mtype[0m[0;34m([0m[0merr[0m[0;34m)[0m[0;34m([0m[0mmsg[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1947[0m [0;34m[0m[0m
[1;32m   1948[0m         [0;32mif[0m [0mser[0m[0;34m.[0m[0mdtype[0m [0;34m==[0m [0mobject[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: agg function failed [how->mean,dtype->object]

## === cell 9
import string
df['text']=df['text'].str.replace('[^\w\s]','')
df['selected_text']=df['selected_text'].str.replace('[^\w\s]','')
k=df[df['text_word_number']<=2][df['jaccard_score']<1]
k
