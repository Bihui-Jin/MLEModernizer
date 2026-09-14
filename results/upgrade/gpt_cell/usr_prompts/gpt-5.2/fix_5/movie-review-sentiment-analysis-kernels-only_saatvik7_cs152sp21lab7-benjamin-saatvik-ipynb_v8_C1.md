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
bow_learner = bow_learner.to("cpu")

bow_preds, bow_probs = bow_learner.get_preds(dl=test_dl)


## --- ERROR in cell 23, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3611639137.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m [0mbow_learner[0m [0;34m=[0m [0mbow_learner[0m[0;34m.[0m[0mto[0m[0;34m([0m[0;34m"cpu"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[0;32m----> 5[0;31m [0mbow_preds[0m[0;34m,[0m [0mbow_probs[0m [0;34m=[0m [0mbow_learner[0m[0;34m.[0m[0mget_preds[0m[0;34m([0m[0mdl[0m[0;34m=[0m[0mtest_dl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m__getattr__[0;34m(self, name)[0m
[1;32m   1926[0m             [0;32mif[0m [0mname[0m [0;32min[0m [0mmodules[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1927[0m                 [0;32mreturn[0m [0mmodules[0m[0;34m[[0m[0mname[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1928[0;31m         raise AttributeError(
[0m[1;32m   1929[0m             [0;34mf"'{type(self).__name__}' object has no attribute '{name}'"[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1930[0m         )

[0;31mAttributeError[0m: 'Sequential' object has no attribute 'get_preds'

## === cell 24
    
bow_list_preds = []
for row in bow_preds:
    bow_list_preds.append(torch.argmax(row).item())
