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

# 3. Data file paths

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

# 4. Code solution

## === cell 0

from fastai.text.all import * 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch import optim

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
!unzip '/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip'
!unzip '/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip'


## === cell 2
df = pd.read_csv('train.tsv', sep="\t")
df_test = pd.read_csv('test.tsv', sep="\t")
df_test = df_test.rename({'Phrase': 'text'}, axis='columns')


## === cell 3
df = df[:10000]
df.head()


## === cell 4
df_test.head()


## === cell 5
dls = TextDataLoaders.from_df(df, text_col='Phrase', label_col='Sentiment')


## === cell 6
awd_learner = text_classifier_learner(dls, AWD_LSTM, metrics=accuracy)


## === cell 7
awd_learner.fine_tune(100, cbs=[SaveModelCallback, EarlyStoppingCallback(patience=10, min_delta=0.01)])


## === cell 8
awd_learner.export('export_lab7.pkl')


## === cell 9
vocab_size = len(dls.train.vocab[0])


## === cell 10
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


## === cell 11
model = nn.Sequential(
    Lambda(batch_mh),
    nn.Linear(vocab_size, 30),
    nn.ReLU(),
    nn.Linear(30,5)  # note that output doesn't have a softmax layer.
)


## === cell 12
bow_learner = Learner(dls=dls, model=model, 
                opt_func=SGD, 
                loss_func=CrossEntropyLossFlat(), 
                metrics=accuracy)
bow_learner.summary()


## === cell 13
bow_learner.fit(100, cbs=[SaveModelCallback(), EarlyStoppingCallback, ReduceLROnPlateau])


## === cell 14
bow_learner.export('bag_of_words.pkl')


## === cell 15
review_size = 100
embedding_size = 10
hidden_layer_size = 20


## === cell 16
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


## === cell 17
model = nn.Sequential(
    FirstHundred(),
    nn.Embedding(vocab_size, embedding_size),
    nn.Flatten(),
    nn.Linear(embedding_size * review_size, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size,5) 
)


## === cell 18
emb_learner = Learner(dls=dls, model=model,
                loss_func=CrossEntropyLossFlat(), 
                      opt_func=SGD,
                metrics=accuracy)
emb_learner.summary()


## === cell 19
emb_learner.fit(100, cbs=[SaveModelCallback(), EarlyStoppingCallback(patience=20)])


## === cell 20
emb_learner.export('embeddings.pkl')


## === cell 21
import os

_local_pkl = "bag_of_words.pkl"
_kaggle_pkl = "../input/bagofwords/bag_of_words.pkl"

bow_learner = load_learner(_local_pkl if os.path.exists(_local_pkl) else _kaggle_pkl)


## === cell 22
test_dl = bow_learner.dls.test_dl(df_test)


## === cell 23
defaults.device = torch.device("cpu")
bow_learner.to("cpu")

test_dl = bow_learner.dls.test_dl(df_test)

bow_preds, bow_probs = bow_learner.get_preds(dl=test_dl)


## --- ERROR in cell 23, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/11441470.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      7[0m [0mtest_dl[0m [0;34m=[0m [0mbow_learner[0m[0;34m.[0m[0mdls[0m[0;34m.[0m[0mtest_dl[0m[0;34m([0m[0mdf_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m [0;34m[0m[0m
[0;32m----> 9[0;31m [0mbow_preds[0m[0;34m,[0m [0mbow_probs[0m [0;34m=[0m [0mbow_learner[0m[0;34m.[0m[0mget_preds[0m[0;34m([0m[0mdl[0m[0;34m=[0m[0mtest_dl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mget_preds[0;34m(self, ds_idx, dl, with_input, with_decoded, with_loss, act, inner, reorder, cbs, **kwargs)[0m
[1;32m    314[0m         [0;32mif[0m [0mwith_loss[0m[0;34m:[0m [0mctx_mgrs[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mloss_not_reduced[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    315[0m         [0;32mwith[0m [0mContextManagers[0m[0;34m([0m[0mctx_mgrs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 316[0;31m             [0mself[0m[0;34m.[0m[0m_do_epoch_validate[0m[0;34m([0m[0mdl[0m[0;34m=[0m[0mdl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    317[0m             [0;32mif[0m [0mact[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mact[0m [0;34m=[0m [0mgetcallable[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mloss_func[0m[0;34m,[0m [0;34m'activation'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    318[0m             [0mres[0m [0;34m=[0m [0mcb[0m[0;34m.[0m[0mall_tensors[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_epoch_validate[0;34m(self, ds_idx, dl)[0m
[1;32m    250[0m         [0;32mif[0m [0mdl[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mdl[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdls[0m[0;34m[[0m[0mds_idx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    251[0m         [0mself[0m[0;34m.[0m[0mdl[0m [0;34m=[0m [0mdl[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 252[0;31m         [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0mno_grad[0m[0;34m([0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mall_batches[0m[0;34m,[0m [0;34m'validate'[0m[0;34m,[0m [0mCancelValidException[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    253[0m [0;34m[0m[0m
[1;32m    254[0m     [0;32mdef[0m [0m_do_epoch[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mall_batches[0;34m(self)[0m
[1;32m    211[0m     [0;32mdef[0m [0mall_batches[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    212[0m         [0mself[0m[0;34m.[0m[0mn_iter[0m [0;34m=[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 213[0;31m         [0;32mfor[0m [0mo[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdl[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mone_batch[0m[0;34m([0m[0;34m*[0m[0mo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    214[0m [0;34m[0m[0m
[1;32m    215[0m     [0;32mdef[0m [0m_backward[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mloss_grad[0m[0;34m.[0m[0mbackward[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mone_batch[0;34m(self, i, b)[0m
[1;32m    241[0m         [0mb[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_set_device[0m[0;34m([0m[0mb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    242[0m         [0mself[0m[0;34m.[0m[0m_split[0m[0;34m([0m[0mb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 243[0;31m         [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_do_one_batch[0m[0;34m,[0m [0;34m'batch'[0m[0;34m,[0m [0mCancelBatchException[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    244[0m [0;34m[0m[0m
[1;32m    245[0m     [0;32mdef[0m [0m_do_epoch_train[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_one_batch[0;34m(self)[0m
[1;32m    222[0m [0;34m[0m[0m
[1;32m    223[0m     [0;32mdef[0m [0m_do_one_batch[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 224[0;31m         [0mself[0m[0;34m.[0m[0mpred[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mmodel[0m[0;34m([0m[0;34m*[0m[0mself[0m[0;34m.[0m[0mxb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    225[0m         [0mself[0m[0;34m([0m[0;34m'after_pred'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    226[0m         [0;32mif[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0myb[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    248[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    249[0m         [0;32mfor[0m [0mmodule[0m [0;32min[0m [0mself[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 250[0;31m             [0minput[0m [0;34m=[0m [0mmodule[0m[0;34m([0m[0minput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    251[0m         [0;32mreturn[0m [0minput[0m[0;34m[0m[0;34m[0m[0m
[1;32m    252[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    123[0m [0;34m[0m[0m
[1;32m    124[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m:[0m [0mTensor[0m[0;34m)[0m [0;34m->[0m [0mTensor[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 125[0;31m         [0;32mreturn[0m [0mF[0m[0;34m.[0m[0mlinear[0m[0;34m([0m[0minput[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mweight[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mbias[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    126[0m [0;34m[0m[0m
[1;32m    127[0m     [0;32mdef[0m [0mextra_repr[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0mstr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)

## === cell 24
    
bow_list_preds = []
for row in bow_preds:
    bow_list_preds.append(torch.argmax(row).item())
