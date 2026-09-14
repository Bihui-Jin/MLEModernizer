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
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2272306387.py in <cell line: 0>()
      2 # awd_learner = load_learner('../input/export/export_lab7.pkl')
      3 # emb_learner = load_learner('../input/embeddings/embeddings.pkl')
----> 4 bow_learner = load_learner('../input/bagofwords/bag_of_words.pkl')

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in load_learner(fname, cpu, pickle_module)
    455         warn("load_learner` uses Python's insecure pickle module, which can execute malicious arbitrary code when loading. Only load files you trust.\nIf you only need to load model weights and optimizer state, use the safe `Learner.load` instead.")
    456         load_kwargs = {"weights_only": False} if ismin_torch("2.6") else {}
--> 457         res = torch.load(fname, map_location=map_loc, pickle_module=pickle_module, **load_kwargs)
    458     except ImportError as e:
    459         if any(o in str(e) for o in ("fastcore.transform","fastcore.dispatch")):

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../input/bagofwords/bag_of_words.pkl'

## === cell 22
test_dl = bow_learner.dls.test_dl(df_test)


## === cell 23
bow_preds,bow_probs = bow_learner.get_preds(dl=test_dl)


## === cell 24
    
bow_list_preds = []
for row in bow_preds:
    bow_list_preds.append(torch.argmax(row).item())
    


## === cell 25
del df_test['SentenceId']
del df_test['text']


## === cell 26


df_test.to_csv('submission.csv', index=False)




## --- ERROR in outputing the csv:
Invalid submission: Submission must have a `Sentiment` column
