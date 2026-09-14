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
bow_learner = load_learner('../input/bagofwords/bag_of_words.pkl')


## --- ERROR in cell 21, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2272306387.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;31m# awd_learner = load_learner('../input/export/export_lab7.pkl')[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;31m# emb_learner = load_learner('../input/embeddings/embeddings.pkl')[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0mbow_learner[0m [0;34m=[0m [0mload_learner[0m[0;34m([0m[0;34m'../input/bagofwords/bag_of_words.pkl'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mload_learner[0;34m(fname, cpu, pickle_module)[0m
[1;32m    455[0m         [0mwarn[0m[0;34m([0m[0;34m"load_learner` uses Python's insecure pickle module, which can execute malicious arbitrary code when loading. Only load files you trust.\nIf you only need to load model weights and optimizer state, use the safe `Learner.load` instead."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    456[0m         [0mload_kwargs[0m [0;34m=[0m [0;34m{[0m[0;34m"weights_only"[0m[0;34m:[0m [0;32mFalse[0m[0;34m}[0m [0;32mif[0m [0mismin_torch[0m[0;34m([0m[0;34m"2.6"[0m[0;34m)[0m [0;32melse[0m [0;34m{[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 457[0;31m         [0mres[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mfname[0m[0;34m,[0m [0mmap_location[0m[0;34m=[0m[0mmap_loc[0m[0;34m,[0m [0mpickle_module[0m[0;34m=[0m[0mpickle_module[0m[0;34m,[0m [0;34m**[0m[0mload_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    458[0m     [0;32mexcept[0m [0mImportError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    459[0m         [0;32mif[0m [0many[0m[0;34m([0m[0mo[0m [0;32min[0m [0mstr[0m[0;34m([0m[0me[0m[0;34m)[0m [0;32mfor[0m [0mo[0m [0;32min[0m [0;34m([0m[0;34m"fastcore.transform"[0m[0;34m,[0m[0;34m"fastcore.dispatch"[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/serialization.py[0m in [0;36mload[0;34m(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)[0m
[1;32m   1423[0m         [0mpickle_load_args[0m[0;34m[[0m[0;34m"encoding"[0m[0;34m][0m [0;34m=[0m [0;34m"utf-8"[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1424[0m [0;34m[0m[0m
[0;32m-> 1425[0;31m     [0;32mwith[0m [0m_open_file_like[0m[0;34m([0m[0mf[0m[0;34m,[0m [0;34m"rb"[0m[0;34m)[0m [0;32mas[0m [0mopened_file[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1426[0m         [0;32mif[0m [0m_is_zipfile[0m[0;34m([0m[0mopened_file[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1427[0m             [0;31m# The zipfile reader is going to advance the current file position.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/serialization.py[0m in [0;36m_open_file_like[0;34m(name_or_buffer, mode)[0m
[1;32m    749[0m [0;32mdef[0m [0m_open_file_like[0m[0;34m([0m[0mname_or_buffer[0m[0;34m,[0m [0mmode[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    750[0m     [0;32mif[0m [0m_is_path[0m[0;34m([0m[0mname_or_buffer[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 751[0;31m         [0;32mreturn[0m [0m_open_file[0m[0;34m([0m[0mname_or_buffer[0m[0;34m,[0m [0mmode[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    752[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    753[0m         [0;32mif[0m [0;34m"w"[0m [0;32min[0m [0mmode[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/serialization.py[0m in [0;36m__init__[0;34m(self, name, mode)[0m
[1;32m    730[0m [0;32mclass[0m [0m_open_file[0m[0;34m([0m[0m_opener[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    731[0m     [0;32mdef[0m [0m__init__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mname[0m[0;34m,[0m [0mmode[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 732[0;31m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mopen[0m[0;34m([0m[0mname[0m[0;34m,[0m [0mmode[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    733[0m [0;34m[0m[0m
[1;32m    734[0m     [0;32mdef[0m [0m__exit__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../input/bagofwords/bag_of_words.pkl'

## === cell 22
test_dl = bow_learner.dls.test_dl(df_test)
