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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.9

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.65537

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
k=df[df['text_word_number']<=3]
k.groupby('sentiment').mean()['jaccard_score']


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _agg_py_fallback(self, how, values, ndim, alt)
   1941         try:
-> 1942             res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
   1943         except Exception as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in agg_series(self, obj, func, preserve_dtype)
    863 
--> 864         result = self._aggregate_series_pure_python(obj, func)
    865 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _aggregate_series_pure_python(self, obj, func)
    884         for i, group in enumerate(splitter):
--> 885             res = func(group)
    886             res = extract_result(res)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in <lambda>(x)
   2453                 "mean",
-> 2454                 alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
   2455                 numeric_only=numeric_only,

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in mean(self, axis, skipna, numeric_only, **kwargs)
   6548     ):
-> 6549         return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
   6550 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in mean(self, axis, skipna, numeric_only, **kwargs)
  12419     ) -> Series | float:
> 12420         return self._stat_function(
  12421             "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
  12376 
> 12377         return self._reduce(
  12378             func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
   6456                 )
-> 6457             return op(delegate, skipna=skipna, **kwds)
   6458 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in f(values, axis, skipna, **kwds)
    146             else:
--> 147                 result = alt(values, axis=axis, skipna=skipna, **kwds)
    148 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in new_func(values, axis, skipna, mask, **kwargs)
    403 
--> 404         result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
    405 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in nanmean(values, axis, skipna, mask)
    719     the_sum = values.sum(axis, dtype=dtype_sum)
--> 720     the_sum = _ensure_numeric(the_sum)
    721 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in _ensure_numeric(x)
   1700             # GH#44008, GH#36703 avoid casting e.g. strings to numeric
-> 1701             raise TypeError(f"Could not convert string '{x}' to numeric")
   1702         try:

TypeError: Could not convert string 'd92eb7fb320167c461c617ee870d970af1aa8543342b10863fa8734230b626ba81090a26475e440a5dcaf3b6713e4a49b4cc82c1f4f77897afa4ae07fbc22efc4dd6567a34256ecaa33c8ac16832facb3d1d0de7055863f5f7ee498145e798a7876c66dad18c11a2bf6dd07105c9a40e2e9476046a00d709bf51b259192be4cf8a34ea69d9e8a6bc879a25b00161b55ae167a8fd1f5bc48e16fcba98cb7c67dea830e774a5cdeaa93be48f74c4f71c752814dbb059e6f3eeff7b8fcb4aac0688b31fc26f34b79a4944e6dbd2ce211077a9e94c8908e55c9063631ab141a7861269cf6f36463766a3b2a036e9f4bb31d6f7c173a4f31dec7920ea5bb9044322d878cf0cb3834196cadf12cdafc3e33fbabb87b51a35c1eb9ffc04b84a90b9c95f65f89b26804c673c066cba4aa9d739cbcd46d631c295c148cfebc9cd2cc739a1f18b75e863234a6c8c35f3f729c0149ff15d5ccfd4c6184863c8ac0b9545ebe4b0856b6a258849a8433cb03f2ad2d4deeb4302ea061b1db6f6bd82c0265d9276028882fdfd34f398ee6a603f2395f6760404648e1cfd4146baf71149e3a28d9bd1d86c69993ed46c7ee08f8b684d59b77080c10f29b07b6ba757e7758d491fcc6675ed3f64b744c8bcfff7624ba599559885178f7eb64213f65406197aa929a2b034d03e50771b845b38f2fbf39ab541e214776336cda58aab9be6abb89a40d44632203c718f6359f0460d611d6ecab08a388dea6d2dcdc1970069c45572972d36c02977bde08d484d4f898c06e11a4d47171c09515655c50f42b5a5f202618984c25aa359d23b3db7ff86f909582cc97395e74d2d321760698223075a13df2f43861c70315ba0e19a478d4efe80c21e0f82d1532224ae95d29a4ca6ad5df0caf6f69e50736dc072d0753b921ccd0bedeeaa97561e8e9c0d23de99f08494b79ebb8e91c4ab28566eb93e52800bbbeaac58feeb70f907401cafbadd63ec1c0bed85a8a5a1024d1fdeebda416681f6db3e30634daa756c340185da1e31e6cd4f6c8eef92734f566959d384eeb1fdcda42f6b93be02e7843b761be7769340cdeb258b89fdde1449916ee1c4d4db7e4ed52c4a60c947090baaf786894b22175e950ef4f0b3fbe1706e150b70cfedf94a534f15595ce8f7b3e7059951910843dcf393e46a24872bbb79408d31b4cc44a663e7c6021ea46b2b89c0da2b190b428baa3d14a0475aee7fdcf1e95b1c2622bd3bb604e52d731e9810abc70748a4dafe66c6039022142385a6b7ea870375e4c9de6e157ccf0e2f13043e0da5b1c60597f08b57e18defc498d587c1b65b36983fd7ffdc4c7e112f861febeb1f94317db125d6ab0484a31aa52b69d7e10e75c788380c6ea9affd001e0345707556e8c6ad9fda61503d69c690e2a20acf4e4fd9e8ea30cd585c0d87f191c162ce503d8da6a35478a796ba1be0a2056ff40abae5414377e8c940f9e35f2c569776f26e00d39608005e1d8a64181f4097e4ad1b35d2986afda3ac739630337f02aacdcb653e2e986a68ec08321ce1363b081e11c10c15d3395fa6f3300e12c60311f1076b526c6e8078049b72f7e258097a5eb0e374b290f6bee725c3d40fcfffc6ca216383e2708719ccbbbe6803f71948c92d6c1adf0ebca0e3678f6badd3d6d88c73b732cd66415386764f06bed4306d9b852edc37691aa63342fb0a799eb87e49cd042b83472e1fc800bc361e245e443f6d754f46f6f3b275e738f6af0b2a9009ce2f3de5f8836e28d618869c2614f7b5de1559c3c50f8ad203fa429485357290f254f7b8ffd8d909d3919dd04e1a9f12c89d051d69b5639159cd5f10b26f9f3be484afc0193894ca5ef2cf68b15b3f3b0809c9b769fbca67b47fcb79f564af8976646f9a697274f9de64b1496406a27de5d6992ece2553d377c0a5f2d4e83e2d6e96cb8554fed64ecb2e361ca13d4ac0d18d0bfee2de889ebebd27b79b086deb67c516344ee21c7c55adcb6a777182cef9f25996c236839a62cfdaa8b4edc9d43c9cd969e38bf982f68653b213be68bcec89dc94d18805232629d1c331239b17f7572e1f7ac5ca5e8698d535f2337490d6a2831e0978b363a5bb931c25329e7d96930b5a21bec07e69852317fc6384ed327a15d669d7ec3333b76c11dc1f12' to numeric

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1477231320.py in <cell line: 0>()
      1 k=df[df['text_word_number']<=3]
----> 2 k.groupby('sentiment').mean()['jaccard_score']

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in mean(self, numeric_only, engine, engine_kwargs)
   2450             )
   2451         else:
-> 2452             result = self._cython_agg_general(
   2453                 "mean",
   2454                 alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
   1996             return result
   1997 
-> 1998         new_mgr = data.grouped_reduce(array_func)
   1999         res = self._wrap_agged_manager(new_mgr)
   2000         if how in ["idxmin", "idxmax"]:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in grouped_reduce(self, func)
   1467                 #  while others do not.
   1468                 for sb in blk._split():
-> 1469                     applied = sb.apply(func)
   1470                     result_blocks = extend_blocks(applied, result_blocks)
   1471             else:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in apply(self, func, **kwargs)
    391         one
    392         """
--> 393         result = func(self.values, **kwargs)
    394 
    395         result = maybe_coerce_values(result)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in array_func(values)
   1993 
   1994             assert alt is not None
-> 1995             result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
   1996             return result
   1997 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _agg_py_fallback(self, how, values, ndim, alt)
   1944             msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
   1945             # preserve the kind of exception that raised
-> 1946             raise type(err)(msg) from err
   1947 
   1948         if ser.dtype == object:

TypeError: agg function failed [how->mean,dtype->object]

## === cell 8
k=df[df['text_word_number']<=2]
k.groupby('sentiment').mean()['jaccard_score']


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _agg_py_fallback(self, how, values, ndim, alt)
   1941         try:
-> 1942             res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
   1943         except Exception as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in agg_series(self, obj, func, preserve_dtype)
    863 
--> 864         result = self._aggregate_series_pure_python(obj, func)
    865 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _aggregate_series_pure_python(self, obj, func)
    884         for i, group in enumerate(splitter):
--> 885             res = func(group)
    886             res = extract_result(res)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in <lambda>(x)
   2453                 "mean",
-> 2454                 alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
   2455                 numeric_only=numeric_only,

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in mean(self, axis, skipna, numeric_only, **kwargs)
   6548     ):
-> 6549         return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
   6550 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in mean(self, axis, skipna, numeric_only, **kwargs)
  12419     ) -> Series | float:
> 12420         return self._stat_function(
  12421             "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
  12376 
> 12377         return self._reduce(
  12378             func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
   6456                 )
-> 6457             return op(delegate, skipna=skipna, **kwds)
   6458 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in f(values, axis, skipna, **kwds)
    146             else:
--> 147                 result = alt(values, axis=axis, skipna=skipna, **kwds)
    148 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in new_func(values, axis, skipna, mask, **kwargs)
    403 
--> 404         result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
    405 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in nanmean(values, axis, skipna, mask)
    719     the_sum = values.sum(axis, dtype=dtype_sum)
--> 720     the_sum = _ensure_numeric(the_sum)
    721 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in _ensure_numeric(x)
   1700             # GH#44008, GH#36703 avoid casting e.g. strings to numeric
-> 1701             raise TypeError(f"Could not convert string '{x}' to numeric")
   1702         try:

TypeError: Could not convert string '0167c461c60af1aa8543a8734230b63e4a49b4cc97afa4ae07fbc22efc4d6ecaa33c8aa7876c66dad18c11a2bf6dd07105c9bc879a25b0cdeaa93be48f74c4f71c44e6dbd2ce66a3b2a0365bb9044322834196cadfb9c95f65f89b26804c67a9d739cbcd234a6c8c359ff15d5ccfd4c6184863433cb03f2ad2d4deeb4302ea061b1db6f6bd82c00404648e1c9bd1d86c69e08f8b684d59b77080c1c8bcfff76285178f7eb64213f65406771b845b3841e2147763f0460d611dc02977bde0ae95d29a4ca6ad5df0caf6f69e50733b921ccd0bde99f08494b93e52800b1cafbadd63ec1c0bed851fdeebda414f566959d384eeb1fdcdcdeb258b89fdde1449917e4ed52c4a60c947090baaf786894b22175e950e706e150b70cfedf94a53a663e7c6021ea46b2b89c0da2b190ba0475aee7fdcf1e95b1c9810abc7072385a6b7ea870375e4c90e2f13043e97f08b57e1317db125d6a52b69d7e10c6ea9affd001e03457069c690e2a2a796ba1be0bae541437764181f409702aacdcb65ec08321ce18078049b72f7e258097a5eb0e374b2c6ca216383e2708719ccbbbe6803f7b732cd6641bed4306d9b852edc3769bc361e245e1559c3c50f9485357290d8d909d39112c89d051d9f3be484af5ef2cf68b15b3f3b0809086deb67c5777182cef9f25996c2368bf982f686d1880523261f7ac5ca5e8698d535f225329e7d96930b5a21be327a15d669' to numeric

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1638589351.py in <cell line: 0>()
      1 k=df[df['text_word_number']<=2]
----> 2 k.groupby('sentiment').mean()['jaccard_score']

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in mean(self, numeric_only, engine, engine_kwargs)
   2450             )
   2451         else:
-> 2452             result = self._cython_agg_general(
   2453                 "mean",
   2454                 alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
   1996             return result
   1997 
-> 1998         new_mgr = data.grouped_reduce(array_func)
   1999         res = self._wrap_agged_manager(new_mgr)
   2000         if how in ["idxmin", "idxmax"]:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in grouped_reduce(self, func)
   1467                 #  while others do not.
   1468                 for sb in blk._split():
-> 1469                     applied = sb.apply(func)
   1470                     result_blocks = extend_blocks(applied, result_blocks)
   1471             else:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in apply(self, func, **kwargs)
    391         one
    392         """
--> 393         result = func(self.values, **kwargs)
    394 
    395         result = maybe_coerce_values(result)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in array_func(values)
   1993 
   1994             assert alt is not None
-> 1995             result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
   1996             return result
   1997 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _agg_py_fallback(self, how, values, ndim, alt)
   1944             msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
   1945             # preserve the kind of exception that raised
-> 1946             raise type(err)(msg) from err
   1947 
   1948         if ser.dtype == object:

TypeError: agg function failed [how->mean,dtype->object]

## === cell 9
import string
df['text']=df['text'].str.replace('[^\w\s]','')
df['selected_text']=df['selected_text'].str.replace('[^\w\s]','')
k=df[df['text_word_number']<=2][df['jaccard_score']<1]
k


## === cell 10
df_train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
df_submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')


## === cell 11
df_train['Num_words_text'] = df_train['text'].apply(lambda x:len(str(x).split()))
df_train = df_train[df_train['Num_words_text']>3]


## === cell 12
def save_model(output_dir, nlp, new_model_name):
    ''' This Function Saves model to 
    given output directory'''
    
    output_dir = f'./tse-spacy-model/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


## === cell 13
def get_model_out_path(sentiment):
    '''
    Returns Model output path
    '''
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    return model_out_path


## === cell 14
def get_training_data(sentiment):
    '''
    Returns Trainong data in the format needed to train spacy NER
    '''
    train_data = []
    for index, row in df_train.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))
    return train_data


## === cell 15
def train(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    ""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()


        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,  # batch of texts
                            annotations,  # batch of annotations
                            drop=0.5,   # dropout - make it harder to memorise data
                            losses=losses, 
                            )
            print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')


## === cell 16
sentiment = 'positive'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=3, model=None)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2499446429.py in <cell line: 0>()
      4 model_path = get_model_out_path(sentiment)
      5 # For DEmo Purposes I have taken 3 iterations you can train the model as you want
----> 6 train(train_data, model_path, n_iter=3, model=None)

/tmp/ipykernel_11/2610718720.py in train(train_data, output_dir, n_iter, model)
     13     if "ner" not in nlp.pipe_names:
     14         ner = nlp.create_pipe("ner")
---> 15         nlp.add_pipe(ner, last=True)
     16     # otherwise, get it so we can add labels
     17     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7f49acaa6180> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 17
sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=3, model=None)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4010580326.py in <cell line: 0>()
      4 model_path = get_model_out_path(sentiment)
      5 
----> 6 train(train_data, model_path, n_iter=3, model=None)

/tmp/ipykernel_11/2610718720.py in train(train_data, output_dir, n_iter, model)
     13     if "ner" not in nlp.pipe_names:
     14         ner = nlp.create_pipe("ner")
---> 15         nlp.add_pipe(ner, last=True)
     16     # otherwise, get it so we can add labels
     17     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7f49b15fee30> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 18
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text


## === cell 19
selected_texts = []
MODELS_BASE_PATH = './tse-spacy-model/models/'

if MODELS_BASE_PATH is not None:
    print("Loading Models  from ", MODELS_BASE_PATH)
    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
        
    for index, row in df_test.iterrows():
        text = row.text
        output_str = ""
        if row.sentiment == 'neutral' or len(text.split()) <= 2:
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))
        
df_test['selected_text'] = selected_texts


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/1402838012.py in <cell line: 0>()
      4 if MODELS_BASE_PATH is not None:
      5     print("Loading Models  from ", MODELS_BASE_PATH)
----> 6     model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
      7     model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
      8 

/usr/local/lib/python3.11/dist-packages/spacy/__init__.py in load(name, vocab, disable, enable, exclude, config)
     50     RETURNS (Language): The loaded nlp object.
     51     """
---> 52     return util.load_model(
     53         name,
     54         vocab=vocab,

/usr/local/lib/python3.11/dist-packages/spacy/util.py in load_model(name, vocab, disable, enable, exclude, config)
    482     if name in OLD_MODEL_SHORTCUTS:
    483         raise IOError(Errors.E941.format(name=name, full=OLD_MODEL_SHORTCUTS[name]))  # type: ignore[index]
--> 484     raise IOError(Errors.E050.format(name=name))
    485 
    486 

OSError: [E050] Can't find model './tse-spacy-model/models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 20
df_submission['selected_text'] = df_test['selected_text']
df_submission.to_csv("submission.csv", index=False)
display(df_submission.head(10))


## --- ERROR in cell 20, traceback:
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

KeyError: 'selected_text'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3876337172.py in <cell line: 0>()
----> 1 df_submission['selected_text'] = df_test['selected_text']
      2 df_submission.to_csv("submission.csv", index=False)
      3 display(df_submission.head(10))

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

KeyError: 'selected_text'
