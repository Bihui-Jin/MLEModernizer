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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.51789

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
The fix ensures the notebook runs without missing‑file errors and creates a proper submission file. We stop trying to load a non‑existent learner, keep the original `Phrase` column for test data, and build the submission DataFrame with the required `PhraseId` and `Sentiment` columns instead of deleting them. Finally, the CSV is written to the working directory with the correct column names.

```


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/9376997.py", line 1
    The fix ensures the notebook runs without missing‑file errors and creates a proper submission file. We stop trying to load a non‑existent learner, keep the original `Phrase` column for test data, and build the submission DataFrame with the required `PhraseId` and `Sentiment` columns instead of deleting them. Finally, the CSV is written to the working directory with the correct column names.
                                                     ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 1
from fastai.text.all import * 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch import optim

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
!unzip '/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip'
!unzip '/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip'



## === cell 3
df = pd.read_csv('train.tsv', sep="\t")
df_test = pd.read_csv('test.tsv', sep="\t")   # keep original column names (Phrase, PhraseId, SentenceId)



## === cell 4
df = df[:10000]
df.head()



## === cell 5
df_test.head()



## === cell 6
dls = TextDataLoaders.from_df(df, text_col='Phrase', label_col='Sentiment')



## === cell 7
awd_learner = text_classifier_learner(dls, AWD_LSTM, metrics=accuracy)



## === cell 8
awd_learner.fine_tune(100, cbs=[SaveModelCallback, EarlyStoppingCallback(patience=10, min_delta=0.01)])



## === cell 9
awd_learner.export('export_lab7.pkl')



## === cell 10
vocab_size = len(dls.train.vocab[0])



## === cell 11
def to_multi_hot(arr):
  mh = [0]*vocab_size
  for i in range(len(arr)):
    mh[arr[i]] = 1
  return mh

def batch_mh(big_arr):
  big_arr = big_arr.tolist()
  result=[]
  for x in big_arr:
    result += [to_multi_hot(x)]
  result = torch.Tensor(result)
  result = to_device(result)
  return result



## === cell 12
model = nn.Sequential(
    Lambda(batch_mh),
    nn.Linear(vocab_size, 30),
    nn.ReLU(),
    nn.Linear(30,5)  # note that output doesn't have a softmax layer.
)



## === cell 13
bow_learner = Learner(dls=dls, model=model, 
                opt_func=SGD, 
                loss_func=CrossEntropyLossFlat(), 
                metrics=accuracy)
bow_learner.summary()



## === cell 14
bow_learner.fit(100, cbs=[SaveModelCallback(), EarlyStoppingCallback(), ReduceLROnPlateau()])



## === cell 15
bow_learner.export('bag_of_words.pkl')



## === cell 16
review_size = 100
embedding_size = 10
hidden_layer_size = 20



## === cell 17
import torch.nn.functional as F
class FirstHundred(Module):
    def forward(self, tns):
        padded_tns = F.pad(tns, pad=(0, review_size - tns.shape[1], 0, 0), value=1)
        padded_tns = padded_tns[:, :review_size]
        padded_tns = to_device(padded_tns)
        return padded_tns

class PrintShape(Module):
    def forward(self, arr):
        print(arr.size())
        return arr



## === cell 18
model = nn.Sequential(
    FirstHundred(),
    nn.Embedding(vocab_size, embedding_size),
    nn.Flatten(),
    nn.Linear(embedding_size * review_size, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size,5) 
)



## === cell 19
emb_learner = Learner(dls=dls, model=model,
                loss_func=CrossEntropyLossFlat(), 
                      opt_func=SGD,
                metrics=accuracy)
emb_learner.summary()



## === cell 20
emb_learner.fit(100, cbs=[SaveModelCallback(), EarlyStoppingCallback(patience=20)])



## === cell 21
emb_learner.export('embeddings.pkl')



## === cell 23
test_dl = bow_learner.dls.test_dl(df_test)



## --- ERROR in cell 23, traceback:
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

KeyError: 'text'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3847417721.py in <cell line: 0>()
----> 1 test_dl = bow_learner.dls.test_dl(df_test)
      2 

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in test_dl(self, test_items, rm_type_tfms, with_labels, **kwargs)
    531     test_ds = test_set(self.valid_ds, test_items, rm_tfms=rm_type_tfms, with_labels=with_labels
    532                       ) if isinstance(self.valid_ds, (Datasets, TfmdLists)) else test_items
--> 533     return self.valid.new(test_ds, **kwargs)

/usr/local/lib/python3.11/dist-packages/fastai/text/data.py in new(self, dataset, **kwargs)
    218         if 'val_res' in kwargs and kwargs['val_res'] is not None: res = kwargs['val_res']
    219         else: res = self.res if dataset is None else None
--> 220         return super().new(dataset=dataset, res=res, **kwargs)
    221 
    222 # %% ../../nbs/31_text.data.ipynb 62

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in new(self, dataset, cls, **kwargs)
     99         **kwargs
    100     ):
--> 101         res = super().new(dataset, cls, do_setup=False, **kwargs)
    102         if not hasattr(self, '_n_inp') or not hasattr(self, '_types'):
    103             try:

/usr/local/lib/python3.11/dist-packages/fastai/data/load.py in new(self, dataset, cls, **kwargs)
    148             o = getattr(self, n)
    149             if not isinstance(o, MethodType): cur_kwargs[n] = o
--> 150         return cls(**merge(cur_kwargs, kwargs))
    151 
    152     @property

/usr/local/lib/python3.11/dist-packages/fastai/text/data.py in __init__(self, dataset, sort_func, res, **kwargs)
    191         self.sort_func = _default_sort if sort_func is None else sort_func
    192         if res is None and self.sort_func == _default_sort: res = _get_lengths(dataset)
--> 193         self.res = [self.sort_func(self.do_item(i)) for i in range_of(self.dataset)] if res is None else res
    194         if len(self.res) > 0: self.idx_max = np.argmax(self.res)
    195 

/usr/local/lib/python3.11/dist-packages/fastai/text/data.py in <listcomp>(.0)
    191         self.sort_func = _default_sort if sort_func is None else sort_func
    192         if res is None and self.sort_func == _default_sort: res = _get_lengths(dataset)
--> 193         self.res = [self.sort_func(self.do_item(i)) for i in range_of(self.dataset)] if res is None else res
    194         if len(self.res) > 0: self.idx_max = np.argmax(self.res)
    195 

/usr/local/lib/python3.11/dist-packages/fastai/data/load.py in do_item(self, s)
    168     def prebatched(self): return self.bs is None
    169     def do_item(self, s):
--> 170         try: return self.after_item(self.create_item(s))
    171         except SkipItemException: return None
    172     def chunkify(self, b): return b if self.prebatched else chunked(b, self.bs, self.drop_last)

/usr/local/lib/python3.11/dist-packages/fastai/data/load.py in create_item(self, s)
    175     def retain(self, res, b):  return retain_types(res, b[0] if is_listy(b) else b)
    176     def create_item(self, s):
--> 177         if self.indexed: return self.dataset[s or 0]
    178         elif s is None:  return next(self.it)
    179         else: raise IndexError("Cannot index an iterable dataset numerically - must use `None`.")

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in __getitem__(self, it)
    452 
    453     def __getitem__(self, it):
--> 454         res = tuple([tl[it] for tl in self.tls])
    455         return res if is_indexer(it) else list(zip(*res))
    456 

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in <listcomp>(.0)
    452 
    453     def __getitem__(self, it):
--> 454         res = tuple([tl[it] for tl in self.tls])
    455         return res if is_indexer(it) else list(zip(*res))
    456 

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in __getitem__(self, idx)
    411         res = super().__getitem__(idx)
    412         if self._after_item is None: return res
--> 413         return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)
    414 
    415 # %% ../../nbs/03_data.core.ipynb 54

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in _after_item(self, o)
    371             raise
    372     def subset(self, i): return self._new(self._get(self.splits[i]), split_idx=i)
--> 373     def _after_item(self, o): return self.tfms(o)
    374     def __repr__(self): return f"{self.__class__.__name__}: {self.items}\ntfms - {self.tfms.fs}"
    375     def __iter__(self): return (self[i] for i in range(len(self)))

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in __call__(self, o)
    246         self.fs = self.fs.sorted(key='order')
    247 
--> 248     def __call__(self, o): return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
    249     def __repr__(self): return f"Pipeline: {' -> '.join([f.name for f in self.fs if f.name != 'noop'])}"
    250     def __getitem__(self,i): return self.fs[i]

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in compose_tfms(x, tfms, is_enc, reverse, **kwargs)
    195     for f in tfms:
    196         if not is_enc: f = f.decode
--> 197         x = f(x, **kwargs)
    198     return x
    199 

/usr/local/lib/python3.11/dist-packages/fastai/data/transforms.py in __call__(self, o, **kwargs)
    218 
    219     def __call__(self, o, **kwargs):
--> 220         if len(self.cols) == 1: return self._do_one(o, self.cols[0])
    221         return L(self._do_one(o, c) for c in self.cols)
    222 

/usr/local/lib/python3.11/dist-packages/fastai/data/transforms.py in _do_one(self, r, c)
    212 
    213     def _do_one(self, r, c):
--> 214         o = r[c] if isinstance(c, int) or not c in getattr(r, '_fields', []) else getattr(r, c)
    215         if len(self.pref)==0 and len(self.suff)==0 and self.label_delim is None: return o
    216         if self.label_delim is None: return f'{self.pref}{o}{self.suff}'

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'text'

## === cell 24
bow_preds,bow_probs = bow_learner.get_preds(dl=test_dl)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2579298862.py in <cell line: 0>()
----> 1 bow_preds,bow_probs = bow_learner.get_preds(dl=test_dl)
      2 

NameError: name 'test_dl' is not defined

## === cell 25
bow_list_preds = []
for row in bow_preds:
    bow_list_preds.append(torch.argmax(row).item())



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3785279155.py in <cell line: 0>()
      1 bow_list_preds = []
----> 2 for row in bow_preds:
      3     bow_list_preds.append(torch.argmax(row).item())
      4 

NameError: name 'bow_preds' is not defined

## === cell 26
submission = pd.DataFrame({
    'PhraseId': df_test['PhraseId'],
    'Sentiment': bow_list_preds
})



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2805931851.py in <cell line: 0>()
      1 # Build submission DataFrame with required columns
----> 2 submission = pd.DataFrame({
      3     'PhraseId': df_test['PhraseId'],
      4     'Sentiment': bow_list_preds
      5 })

/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py in __init__(self, data, index, columns, dtype, copy)
    590 def __init__(self:pd.DataFrame, data=None, index=None, columns=None, dtype=None, copy=None):
    591     if data is not None and isinstance(data, Tensor): data = to_np(data)
--> 592     self._old_init(data, index=index, columns=columns, dtype=dtype, copy=copy)
    593 
    594 # %% ../nbs/00_torch_core.ipynb 153

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    688                     f"length {len(index)}"
    689                 )
--> 690                 raise ValueError(msg)
    691         else:
    692             index = default_index(lengths[0])

ValueError: array length 0 does not match index length 46818

## === cell 27
submission.to_csv('submission.csv', index=False)
```

## --- ERROR in cell 27, traceback:
  File "/tmp/ipykernel_55/1146646358.py", line 2
    ```
    ^
SyntaxError: invalid syntax
