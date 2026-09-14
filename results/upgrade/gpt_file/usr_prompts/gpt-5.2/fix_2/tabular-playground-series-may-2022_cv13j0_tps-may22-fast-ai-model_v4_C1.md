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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

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
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.95965

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, warnings

warnings.filterwarnings("ignore")



## === cell 1
import numpy as np
import pandas as pd

from fastai.tabular.all import *

pd.options.display.float_format = "{:,.5f}".format



## === cell 2
for dirname, _, filenames in os.walk(
    "/kaggle/input/tabular-playground-series-may-2022"
):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 3
trn_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
tst_data = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)




## === cell 4
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df




## === cell 6
def count_chars(df, field):
    """
    Extract per-position character codes and number of unique characters from a string field.
    """
    for i in range(10):
        df[f"ch_{i}"] = df[field].str.get(i).apply(ord) - ord("A")
    df["unique_characters"] = df[field].apply(lambda s: len(set(s)))
    return df


trn_data = count_chars(trn_data, "f_27")
tst_data = count_chars(tst_data, "f_27")



## === cell 7
continuous_feat_base = [
    "f_00",
    "f_01",
    "f_02",
    "f_03",
    "f_04",
    "f_05",
    "f_06",
    "f_19",
    "f_20",
    "f_21",
    "f_22",
    "f_23",
    "f_24",
    "f_25",
    "f_26",
    "f_28",
]


def stat_features(df, cols=continuous_feat_base):
    """
    Calculate aggregated features across selected continuous columns.
    """
    df["f_sum"] = df[cols].sum(axis=1)
    df["f_min"] = df[cols].min(axis=1)
    df["f_max"] = df[cols].max(axis=1)
    df["f_std"] = df[cols].std(axis=1)
    df["f_mad"] = df[cols].mad(axis=1)
    df["f_mean"] = df[cols].mean(axis=1)
    df["f_kurt"] = df[cols].kurt(axis=1)

    df["f_prod"] = df[cols].prod(axis=1)
    df["f_range"] = df[cols].max(axis=1) - df[cols].min(axis=1)
    df["f_count_pos"] = df[df[cols].gt(0)].count(axis=1)
    df["f_count_neg"] = df[df[cols].lt(0)].count(axis=1)
    return df


trn_data = stat_features(trn_data, continuous_feat_base)
tst_data = stat_features(tst_data, continuous_feat_base)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1237627814.py in <cell line: 0>()
     39 
     40 
---> 41 trn_data = stat_features(trn_data, continuous_feat_base)
     42 tst_data = stat_features(tst_data, continuous_feat_base)
     43 

/tmp/ipykernel_11/1237627814.py in stat_features(df, cols)
     28     df["f_std"] = df[cols].std(axis=1)
     29     # NOTE: DataFrame.mad exists in pandas 2.2; keep as-is.
---> 30     df["f_mad"] = df[cols].mad(axis=1)
     31     df["f_mean"] = df[cols].mean(axis=1)
     32     df["f_kurt"] = df[cols].kurt(axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'mad'

## === cell 8
continuous_feat = [
    "unique_characters",
    "f_06",
    "ch_7",
    "ch_0",
    "ch_8",
    "f_std",
    "f_range",
    "f_24",
    "f_min",
    "f_21",
    "ch_2",
    "f_03",
    "f_sum",
    "f_05",
    "f_count_neg",
    "f_22",
    "f_02",
    "ch_3",
    "f_26",
    "f_00",
    "ch_6",
    "f_23",
    "f_mean",
    "f_count_pos",
    "ch_9",
    "f_prod",
    "f_kurt",
    "ch_4",
    "f_mad",
    "f_max",
    "f_25",
    "f_04",
    "f_20",
    "f_19",
    "f_01",
    "f_28",
    "ch_1",
    "ch_5",
    "f_07",
    "f_08",
    "f_09",
    "f_10",
    "f_11",
    "f_12",
    "f_13",
    "f_14",
    "f_15",
    "f_16",
    "f_17",
    "f_18",
    "f_29",
    "f_30",
]
categorical_feat = []  # avoid embeddings as originally intended



## === cell 9
data_processing = [
    FillMissing,
    Categorify,
    Normalize,
]

batch_size = 1024
valid_pct = 0.10

set_seed(42, reproducible=True)

data = TabularDataLoaders.from_df(
    df=trn_data,
    path=".",
    procs=data_processing,
    cat_names=categorical_feat,
    cont_names=continuous_feat,
    valid_pct=valid_pct,
    bs=batch_size,
    y_block=CategoryBlock,
    y_names="target",
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/897964140.py in <cell line: 0>()
     11 set_seed(42, reproducible=True)
     12 
---> 13 data = TabularDataLoaders.from_df(
     14     df=trn_data,
     15     path=".",

/usr/local/lib/python3.11/dist-packages/fastai/tabular/data.py in from_df(cls, df, path, procs, cat_names, cont_names, y_names, y_block, valid_idx, **kwargs)
     32         if cont_names is None: cont_names = list(set(df)-set(L(cat_names))-set(L(y_names)))
     33         splits = RandomSplitter()(df) if valid_idx is None else IndexSplitter(valid_idx)(df)
---> 34         to = TabularPandas(df, procs, cat_names, cont_names, y_names, splits=splits, y_block=y_block)
     35         return to.dataloaders(path=path, **kwargs)
     36 

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in __init__(self, df, procs, cat_names, cont_names, y_names, y_block, splits, do_setup, device, inplace, reduce_memory)
    168         self.cat_names,self.cont_names,self.procs = L(cat_names),L(cont_names),Pipeline(procs)
    169         self.split = len(df) if splits is None else len(splits[0])
--> 170         if do_setup: self.setup()
    171 
    172     def new(self, df, inplace=False):

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in setup(self)
    179     def decode_row(self, row): return self.new(pd.DataFrame(row).T).decode().items.iloc[0]
    180     def show(self, max_n=10, **kwargs): display_df(self.new(self.all_cols[:max_n]).decode().items)
--> 181     def setup(self): self.procs.setup(self)
    182     def process(self): self.procs(self)
    183     def loc(self): return self.items.loc

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in setup(self, items, train_setup)
    238         tfms = self.fs[:]
    239         self.fs.clear()
--> 240         for t in tfms: self.add(t,items, train_setup)
    241 
    242     def add(self,ts, items=None, train_setup=False):

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in add(self, ts, items, train_setup)
    242     def add(self,ts, items=None, train_setup=False):
    243         if not is_listy(ts): ts=[ts]
--> 244         for t in ts: t.setup(items, train_setup)
    245         self.fs+=ts
    246         self.fs = self.fs.sorted(key='order')

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in setup(self, items, train_setup)
    224     "Base class to write a non-lazy tabular processor for dataframes"
    225     def setup(self, items=None, train_setup=False): #TODO: properly deal with train_setup
--> 226         super().setup(getattr(items,'train',items), train_setup=False)
    227         # Procs are called as soon as data is available
    228         return self(items.items if isinstance(items,Datasets) else items)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in setup(self, items, train_setup)
    117         train_setup = train_setup if self.train_setup is None else self.train_setup
    118         items = getattr(items, 'train', items) if train_setup else items
--> 119         try: return self.setups(items)
    120         except (AttributeError, NotFoundLookupError): return None
    121 

/usr/local/lib/python3.11/dist-packages/plum/function.py in __call__(self, _, *args, **kw_args)
    507 
    508     def __call__(self, _, *args, **kw_args):
--> 509         return self._f(self._instance, *args, **kw_args)
    510 
    511     def invoke(self, *types):

    [... skipping hidden 1 frame]

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in setups(self, to)
    302 
    303     def setups(self, to):
--> 304         missing = pd.isnull(to.conts).any()
    305         store_attr(but='to', na_dict={n:self.fill_strategy(to[n], self.fill_vals[n])
    306                             for n in missing[missing].keys()})

/usr/local/lib/python3.11/dist-packages/fastai/tabular/core.py in f(o)
    208 def _add_prop(cls, nm):
    209     @property
--> 210     def f(o): return o[list(getattr(o,nm+'_names'))]
    211     @f.setter
    212     def fset(o, v): o[getattr(o,nm+'_names')] = v

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __getitem__(self, k)
     93     def __init__(self, items): self.items = items
     94     def __len__(self): return len(self.items)
---> 95     def __getitem__(self, k): return self.items[list(k) if isinstance(k,CollBase) else k]
     96     def __setitem__(self, k, v): self.items[list(k) if isinstance(k,CollBase) else k] = v
     97     def __delitem__(self, i): del(self.items[i])

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

KeyError: "['f_range', 'f_count_neg', 'f_mean', 'f_count_pos', 'f_prod', 'f_kurt', 'f_mad'] not in index"

## === cell 10
layers_definition = [256, 128, 64, 64, 16]
emb_size = None
my_config = tabular_config(y_range=(0, 1))

learn = tabular_learner(
    dls=data,
    layers=layers_definition,
    emb_szs=emb_size,
    metrics=[accuracy],
    config=my_config,
).to_fp16()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2033216573.py in <cell line: 0>()
      5 
      6 learn = tabular_learner(
----> 7     dls=data,
      8     layers=layers_definition,
      9     emb_szs=emb_size,

NameError: name 'data' is not defined

## === cell 11
learn.fit_one_cycle(1)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3656084488.py in <cell line: 0>()
      1 # Training (kept identical to original)
----> 2 learn.fit_one_cycle(1)
      3 

NameError: name 'learn' is not defined

## === cell 12
_ = learn.lr_find()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2871704498.py in <cell line: 0>()
      1 # Original notebook ran lr_find interactively; keep it but don't require UI.
----> 2 _ = learn.lr_find()
      3 

NameError: name 'learn' is not defined

## === cell 13
lr = 0.00120
learn.fit_one_cycle(3, lr_max=lr)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2904993436.py in <cell line: 0>()
      1 lr = 0.00120
----> 2 learn.fit_one_cycle(3, lr_max=lr)
      3 

NameError: name 'learn' is not defined

## === cell 14
learn.fine_tune(5, base_lr=lr, freeze_epochs=3)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3067121701.py in <cell line: 0>()
----> 1 learn.fine_tune(5, base_lr=lr, freeze_epochs=3)
      2 

NameError: name 'learn' is not defined

## === cell 15
dl = learn.dls.test_dl(tst_data)

probs, _ = learn.get_preds(dl=dl)  # shape: (n,2)

pos_idx = (
    int(np.where(learn.dls.vocab == "1")[0][0]) if hasattr(learn.dls, "vocab") else 1
)
test_pred = probs[:, pos_idx].cpu().numpy()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4203320778.py in <cell line: 0>()
      1 # Inference on test
----> 2 dl = learn.dls.test_dl(tst_data)
      3 
      4 # FIX: get_preds returns probabilities per class; for AUC submission we need P(class==1)
      5 probs, _ = learn.get_preds(dl=dl)  # shape: (n,2)

NameError: name 'learn' is not defined

## === cell 16
sub["target"] = test_pred.astype(np.float64)
sub.to_csv("submission_fastai.csv", index=False)

print(sub.head())
print("Wrote:", os.path.abspath("submission_fastai.csv"), "rows:", len(sub))

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2113904044.py in <cell line: 0>()
      1 # FIX: Submission must be probabilities, not argmax hard labels.
----> 2 sub["target"] = test_pred.astype(np.float64)
      3 sub.to_csv("submission_fastai.csv", index=False)
      4 
      5 print(sub.head())

NameError: name 'test_pred' is not defined
