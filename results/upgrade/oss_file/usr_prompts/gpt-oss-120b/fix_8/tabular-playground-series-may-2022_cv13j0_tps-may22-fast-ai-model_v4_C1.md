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

None

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings, os, torch

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from fastai.tabular.all import (
    TabularDataLoaders,
    FillMissing,
    Categorify,
    Normalize,
    tabular_learner,
    tabular_config,
    accuracy,
    RocAuc,
    CategoryBlock,
)

pd.options.display.float_format = "{:,.5f}".format
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 25)




## === cell 1
DATA_PATH = "/kaggle/input/tabular-playground-series-may-2022"
train_path = os.path.join(DATA_PATH, "train.csv")
test_path = os.path.join(DATA_PATH, "test.csv")
sample_path = os.path.join(DATA_PATH, "sample_submission.csv")

trn_data = pd.read_csv(train_path)
tst_data = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)




## === cell 2
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith("int"):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min) and (
                    c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min) and (
                    c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df


trn_data = reduce_memory_usage(trn_data, verbose=False)
tst_data = reduce_memory_usage(tst_data, verbose=False)




## === cell 3
def count_chars(df, field):
    for i in range(10):
        df[f"ch_{i}"] = (
            df[field]
            .astype(str)
            .str.get(i)
            .apply(lambda x: ord(x) - ord("A") if pd.notnull(x) else np.nan)
        )
    df["unique_characters"] = (
        df[field].astype(str).apply(lambda s: len(set(s)) if pd.notnull(s) else np.nan)
    )
    return df


def stat_features(df, cols):
    df["f_sum"] = df[cols].sum(axis=1)
    df["f_min"] = df[cols].min(axis=1)
    df["f_max"] = df[cols].max(axis=1)
    df["f_std"] = df[cols].std(axis=1)
    df["f_mad"] = df[cols].apply(lambda row: np.mean(np.abs(row - row.mean())), axis=1)
    df["f_mean"] = df[cols].mean(axis=1)
    df["f_kurt"] = df[cols].kurt(axis=1)
    df["f_prod"] = df[cols].prod(axis=1)
    df["f_range"] = df["f_max"] - df["f_min"]
    df["f_count_pos"] = (df[cols] > 0).sum(axis=1)
    df["f_count_neg"] = (df[cols] < 0).sum(axis=1)
    return df




## === cell 4
trn_data = count_chars(trn_data, "f_27")
tst_data = count_chars(tst_data, "f_27")

orig_continuous = [
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

trn_data = stat_features(trn_data, orig_continuous)
tst_data = stat_features(tst_data, orig_continuous)

trn_data = trn_data.fillna(0)
tst_data = tst_data.fillna(0)




## === cell 5
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

categorical_feat = []  # keep empty to avoid embeddings




## === cell 6
data_processing = [FillMissing, Categorify, Normalize]




## === cell 7
from fastai.losses import CrossEntropyLossFlat

valid_pct = 0.10
batch_size = 1024

dls = TabularDataLoaders.from_df(
    df=trn_data,
    path=".",
    procs=data_processing,
    cat_names=categorical_feat,
    cont_names=continuous_feat,
    y_names="target",
    y_block=CategoryBlock,
    valid_pct=valid_pct,
    bs=batch_size,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

layers_definition = [256, 128, 64, 64, 16]
my_config = tabular_config()  # default config; no y_range for classification

learn = tabular_learner(
    dls,
    layers=layers_definition,
    emb_szs=None,  # no embeddings
    metrics=[RocAuc()],  # <-- include ROC‑AUC metric
    config=my_config,
    loss_func=CrossEntropyLossFlat(),
)




## === cell 8
learn.fit_one_cycle(10, 1e-3)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1721811186.py in <cell line: 0>()
      1 # Increase training cycles modestly to give the model more opportunity to improve AUC
----> 2 learn.fit_one_cycle(10, 1e-3)
      3 
      4 

/usr/local/lib/python3.11/dist-packages/fastai/callback/schedule.py in fit_one_cycle(self, n_epoch, lr_max, div, div_final, pct_start, wd, moms, cbs, reset_opt, start_epoch)
    119     scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
    120               'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
--> 121     self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)
    122 
    123 # %% ../../nbs/14_callback.schedule.ipynb 50

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in fit(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)
    270             self.opt.set_hypers(lr=self.lr if lr is None else lr)
    271             self.n_epoch = n_epoch
--> 272             self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
    273 
    274     def _end_cleanup(self): self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_fit(self)
    259         for epoch in range(self.n_epoch):
    260             self.epoch=epoch
--> 261             self._with_events(self._do_epoch, 'epoch', CancelEpochException)
    262 
    263     def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch(self)
    254     def _do_epoch(self):
    255         self._do_epoch_train()
--> 256         self._do_epoch_validate()
    257 
    258     def _do_fit(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_validate(self, ds_idx, dl)
    250         if dl is None: dl = self.dls[ds_idx]
    251         self.dl = dl
--> 252         with torch.no_grad(): self._with_events(self.all_batches, 'validate', CancelValidException)
    253 
    254     def _do_epoch(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
--> 209         self(f'after_{event_type}');  final()
    210 
    211     def all_batches(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in __call__(self, event_name)
    178 
    179     def ordered_cbs(self, event): return [cb for cb in self.cbs.sorted('order') if hasattr(cb, event)]
--> 180     def __call__(self, event_name): L(event_name).map(self._call_one)
    181 
    182     def _call_one(self, event_name):

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in map(self, f, *args, **kwargs)
    166     def range(cls, a, b=None, step=None): return cls(range_of(a, b=b, step=step))
    167 
--> 168     def map(self, f, *args, **kwargs): return self._new(map_ex(self, f, *args, gen=False, **kwargs))
    169     def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))
    170     def argfirst(self, f, negate=False):

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in map_ex(iterable, f, gen, *args, **kwargs)
    949     res = map(g, iterable)
    950     if gen: return res
--> 951     return list(res)
    952 
    953 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __call__(self, *args, **kwargs)
    934             if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
    935         fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
--> 936         return self.func(*fargs, **kwargs)
    937 
    938 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _call_one(self, event_name)
    182     def _call_one(self, event_name):
    183         if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
--> 184         for cb in self.cbs.sorted('order'): cb(event_name)
    185 
    186     def _bn_bias_state(self, with_bias): return norm_bias_params(self.model, with_bias).map(self.opt.state)

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
---> 64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)
     65         if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
     66         return res

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     60         res = None
     61         if self.run and _run:
---> 62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
     64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in after_validate(self)
    587     def before_validate(self): self._valid_mets.map(Self.reset())
    588     def after_train   (self): self.log += self._train_mets.map(_maybe_item)
--> 589     def after_validate(self): self.log += self._valid_mets.map(_maybe_item)
    590     def after_cancel_train(self):    self.cancel_train = True
    591     def after_cancel_validate(self): self.cancel_valid = True

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in map(self, f, *args, **kwargs)
    166     def range(cls, a, b=None, step=None): return cls(range_of(a, b=b, step=step))
    167 
--> 168     def map(self, f, *args, **kwargs): return self._new(map_ex(self, f, *args, gen=False, **kwargs))
    169     def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))
    170     def argfirst(self, f, negate=False):

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in map_ex(iterable, f, gen, *args, **kwargs)
    949     res = map(g, iterable)
    950     if gen: return res
--> 951     return list(res)
    952 
    953 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __call__(self, *args, **kwargs)
    934             if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
    935         fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
--> 936         return self.func(*fargs, **kwargs)
    937 
    938 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _maybe_item(t)
    541 # %% ../nbs/13a_learner.ipynb 134
    542 def _maybe_item(t):
--> 543     t = t.value
    544     try: return t.item()
    545     except: return t

/usr/local/lib/python3.11/dist-packages/fastai/metrics.py in value(self)
     71         preds,targs = torch.cat(self.preds),torch.cat(self.targs)
     72         if self.to_np: preds,targs = preds.numpy(),targs.numpy()
---> 73         return self.func(targs, preds, **self.kwargs) if self.invert_args else self.func(preds, targs, **self.kwargs)
     74 
     75     @property

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_ranking.py in roc_auc_score(y_true, y_score, average, sample_weight, max_fpr, multi_class, labels)
    549     y_type = type_of_target(y_true, input_name="y_true")
    550     y_true = check_array(y_true, ensure_2d=False, dtype=None)
--> 551     y_score = check_array(y_score, ensure_2d=False)
    552 
    553     if y_type == "multiclass" or (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Exception occured in `Recorder` when calling event `after_validate`:
	Input contains NaN.

## === cell 9
import torch.nn.functional as F

test_dl = learn.dls.test_dl(tst_data)
logits, _ = learn.get_preds(dl=test_dl)
prob_class1 = F.softmax(logits, dim=1)[:, 1].cpu().numpy()




## === cell 10
sub["target"] = prob_class1
submission_path = "submission_fastai.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 11
sub.head()
