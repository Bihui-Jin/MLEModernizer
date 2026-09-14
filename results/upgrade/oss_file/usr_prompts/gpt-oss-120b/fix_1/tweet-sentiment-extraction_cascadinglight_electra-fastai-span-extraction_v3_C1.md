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

3.8

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tokenizers==0.21.2
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3

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

0.7037971019744873

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
from transformers import AutoModelForQuestionAnswering, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import os
from itertools import compress

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3001200960.py in <cell line: 0>()
      1 from transformers import AutoModelForQuestionAnswering, AutoModel, AutoConfig, get_linear_schedule_with_warmup
----> 2 from transformers.optimization import AdamW
      3 import torch
      4 import torch.nn as nn
      5 import torch.nn.functional as F

ImportError: cannot import name 'AdamW' from 'transformers.optimization' (/usr/local/lib/python3.11/dist-packages/transformers/optimization.py)

## === cell 3
from torch.utils.data import DataLoader
from functools import partial
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from fastai.core import *
from fastai.text import *

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/4169179328.py in <cell line: 0>()
      4 from sklearn.model_selection import train_test_split
      5 from tqdm import tqdm
----> 6 from fastai.core import *
      7 from fastai.text import *

ModuleNotFoundError: No module named 'fastai.core'

## === cell 4
file_dir, electra_dir = [Path(f'/kaggle/input/{i}') for i in ['tweet-sentiment-extraction', 'electrabase']]

train_df = pd.read_csv(file_dir/'train.csv')
train_df['text'] = train_df['text'].apply(lambda x: str(x))
train_df['sentiment'] = train_df['sentiment'].apply(lambda x: str(x))
train_df['selected_text'] = train_df['selected_text'].apply(lambda x: str(x))

test_df = pd.read_csv(file_dir/'test.csv')
test_df['text'] = test_df['text'].apply(lambda x: str(x))
test_df['sentiment'] = test_df['sentiment'].apply(lambda x: str(x))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2031258192.py in <cell line: 0>()
----> 1 file_dir, electra_dir = [Path(f'/kaggle/input/{i}') for i in ['tweet-sentiment-extraction', 'electrabase']]
      2 
      3 train_df = pd.read_csv(file_dir/'train.csv')
      4 train_df['text'] = train_df['text'].apply(lambda x: str(x))
      5 train_df['sentiment'] = train_df['sentiment'].apply(lambda x: str(x))

/tmp/ipykernel_11/2031258192.py in <listcomp>(.0)
----> 1 file_dir, electra_dir = [Path(f'/kaggle/input/{i}') for i in ['tweet-sentiment-extraction', 'electrabase']]
      2 
      3 train_df = pd.read_csv(file_dir/'train.csv')
      4 train_df['text'] = train_df['text'].apply(lambda x: str(x))
      5 train_df['sentiment'] = train_df['sentiment'].apply(lambda x: str(x))

NameError: name 'Path' is not defined

## === cell 5
max_len = 128
bs = 64
tokenizer = BertWordPieceTokenizer(str(electra_dir/'vocab.txt'), lowercase=True)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/752719816.py in <cell line: 0>()
      1 max_len = 128
      2 bs = 64
----> 3 tokenizer = BertWordPieceTokenizer(str(electra_dir/'vocab.txt'), lowercase=True)

NameError: name 'electra_dir' is not defined

## === cell 6
def preprocess(sentiment, tweet, selected, tokenizer, max_len):
    _input = tokenizer.encode(sentiment, tweet)
    _span = tokenizer.encode(selected, add_special_tokens=False)
    
    len_span = len(_span.ids)
    start_idx = None
    end_idx = None
    
    for ind in (i for i, e in enumerate(_input.ids) if e == _span.ids[0]):
        if _input.ids[ind: ind + len_span] == _span.ids:
            start_idx = ind
            end_idx = ind + len_span - 1
            break
    
    if not start_idx:
        idx0 = tweet.find(selected)
        idx1 = idx0 + len(selected)
        
        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1):
                char_targets[ct] = 1
                
        tweet_offsets = list(compress(_input.offsets, _input.type_ids))[0:-1]
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
                
        start_idx, end_idx = target_idx[0] +3 , target_idx[-1] + 3
        
    _input.start_target = start_idx
    _input.end_target = end_idx
    _input.tweet = tweet
    _input.sentiment = sentiment
    _input.selected = selected
    
    _input.pad(max_len)
    
    return _input

## === cell 7
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss    

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε:float=0.1, reduction='mean'):
        super().__init__()
        self.ε,self.reduction = ε,reduction
    
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return torch.lerp(nll, loss/c, self.ε) 

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1999233828.py in <cell line: 0>()
      2     return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss
      3 
----> 4 class LabelSmoothingCrossEntropy(nn.Module):
      5     def __init__(self, ε:float=0.1, reduction='mean'):
      6         super().__init__()

NameError: name 'nn' is not defined

## === cell 8
class TweetDataset(Dataset):
    def __init__(self, dataset, test = None):
        self.df = dataset
        self.test = test
        
    def __getitem__(self, idx):
        if not self.test:
            sentiment, tweet, selected = (self.df[col][idx] for col in ['sentiment', 'text', 'selected_text'])
            _input = preprocess(sentiment, tweet, selected, tokenizer, max_len)
            
            yb = [torch.tensor(_input.start_target), torch.tensor(_input.end_target)]
            
        else:
            _input = tokenizer.encode(self.df.sentiment[idx], self.df.text[idx])
            _input.pad(max_len)
            
            yb = 0

        xb = [torch.LongTensor(_input.ids),
              torch.LongTensor(_input.attention_mask),
              torch.LongTensor(_input.type_ids),
              np.array(_input.offsets)]

        return xb, yb     

    def __len__(self):
        return len(self.df)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/836159846.py in <cell line: 0>()
----> 1 class TweetDataset(Dataset):
      2     def __init__(self, dataset, test = None):
      3         self.df = dataset
      4         self.test = test
      5 

NameError: name 'Dataset' is not defined

## === cell 9
pt_model = AutoModel.from_pretrained(electra_dir)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1989437146.py in <cell line: 0>()
----> 1 pt_model = AutoModel.from_pretrained(electra_dir)

NameError: name 'electra_dir' is not defined

## === cell 11
class SpanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = pt_model
        self.drop_out = nn.Dropout(0.1)
        self.qa_outputs = nn.Linear(768 * 2, 2)

    def forward(self, input_ids, attention_mask, token_type_ids, offsets = None):
        
        _, hidden_states = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.qa_outputs(out)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1314304388.py in <cell line: 0>()
----> 1 class SpanModel(nn.Module):
      2     def __init__(self):
      3         super().__init__()
      4         self.model = pt_model
      5         self.drop_out = nn.Dropout(0.1)

NameError: name 'nn' is not defined

## === cell 12
class CELoss(Module):
    def __init__(self, loss_fn = nn.CrossEntropyLoss()): 
        self.loss_fn = loss_fn
        
    def forward(self, inputs, start_targets, end_targets):
        start_logits, end_logits = inputs # assumes tuple input
        
        logits = torch.cat([start_logits, end_logits]).contiguous()
        
        targets = torch.cat([start_targets, end_targets]).contiguous()
        
        return self.loss_fn(logits, targets)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2707406920.py in <cell line: 0>()
----> 1 class CELoss(Module):
      2     def __init__(self, loss_fn = nn.CrossEntropyLoss()):
      3         self.loss_fn = loss_fn
      4 
      5     def forward(self, inputs, start_targets, end_targets):

NameError: name 'Module' is not defined

## === cell 13
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

## === cell 14

class JaccardScore(Callback):
    "Stores predictions and targets to perform calculations on epoch end."
    def __init__(self, valid_ds): 
        self.valid_ds = valid_ds
        self.context_text = valid_ds.df.text
        self.answer_text = valid_ds.df.selected_text
        
    def on_epoch_begin(self, **kwargs):
        self.jaccard_scores = []  
        self.valid_ds_idx = 0
        
        
    def on_batch_end(self, last_input:Tensor, last_output:Tensor, last_target:Tensor, **kwargs):
              
        input_ids = last_input[0]
        attention_masks = last_input[1].bool()
        token_type_ids = last_input[2].bool()
        offsets = last_input[3]

        start_logits, end_logits = last_output
        
        for i in range(len(input_ids)):
            
            _offsets = offsets[i]
            start_idx, end_idx = torch.argmax(start_logits[i]), torch.argmax(end_logits[i])
            _answer_text = self.answer_text[self.valid_ds_idx]
            original_start, original_end = _offsets[start_idx][0], _offsets[end_idx][1]
            pred_span = self.context_text[self.valid_ds_idx][original_start : original_end]
                
            score = jaccard(pred_span, _answer_text)
            self.jaccard_scores.append(score)

            self.valid_ds_idx += 1
            
    def on_epoch_end(self, last_metrics, **kwargs):        
        res = np.mean(self.jaccard_scores)
        return add_metrics(last_metrics, res)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/896189653.py in <cell line: 0>()
      2 # https://forums.fast.ai/t/how-to-set-shuffle-false-of-train-and-val/33730
      3 
----> 4 class JaccardScore(Callback):
      5     "Stores predictions and targets to perform calculations on epoch end."
      6     def __init__(self, valid_ds):

NameError: name 'Callback' is not defined

## === cell 15
model = SpanModel()
tr_df, val_df = train_test_split(train_df, test_size = 0.2, random_state = 42)
tr_df, val_df  = [df.reset_index(drop=True) for df in [tr_df, val_df]]
train_ds, valid_ds = [TweetDataset(i) for i in [tr_df, val_df]]

test_ds = TweetDataset(test_df, test = True)
loss_fn = partial(CELoss, LabelSmoothingCrossEntropy())

data = DataBunch.create(train_ds, valid_ds, test_ds, path=".", bs = bs)
learner = Learner(data, model, loss_func = loss_fn(), path = electra_dir/'models', model_dir=f".")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/500280463.py in <cell line: 0>()
----> 1 model = SpanModel()
      2 tr_df, val_df = train_test_split(train_df, test_size = 0.2, random_state = 42)
      3 tr_df, val_df  = [df.reset_index(drop=True) for df in [tr_df, val_df]]
      4 train_ds, valid_ds = [TweetDataset(i) for i in [tr_df, val_df]]
      5 

NameError: name 'SpanModel' is not defined

## === cell 16
preds = []
test_df_idx = 0

with torch.no_grad():
    
    for xb,yb in tqdm(learner.data.test_dl):
        model0 = learner.load(f'electra_fold_es_0').model.eval()
        start_logits0, end_logits0 = to_cpu(model0(*xb))
        start_logits0, end_logits0 = start_logits0.float(), end_logits0.float()
        
        model1 = learner.load(f'electra_fold_es_1').model.eval()
        start_logits1, end_logits1 = to_cpu(model1(*xb))
        start_logits1, end_logits1 = start_logits1.float(), end_logits1.float()
        
        model2 = learner.load(f'electra_fold_es_2').model.eval()
        start_logits2, end_logits2 = to_cpu(model2(*xb))
        start_logits2, end_logits2 = start_logits2.float(), end_logits2.float()
        
        model3 = learner.load(f'electra_fold_es_3').model.eval()
        start_logits3, end_logits3 = to_cpu(model3(*xb))
        start_logits3, end_logits3 = start_logits3.float(), end_logits3.float()
        
        model4 = learner.load(f'electra_fold_es_4').model.eval()
        start_logits4, end_logits4 = to_cpu(model4(*xb))
        start_logits4, end_logits4 = start_logits4.float(), end_logits4.float()
        
        input_ids = to_cpu(xb[0])
        attention_masks = to_cpu(xb[1].bool())
        token_type_ids = to_cpu(xb[2].bool())
        offsets = to_cpu(xb[3])
        
        start_logits = (start_logits0 + start_logits1 + start_logits2 + start_logits3 + start_logits4) / 5
        end_logits = (end_logits0 + end_logits1 + end_logits2 + end_logits3 + end_logits4) / 5
        
        for i in range(len(input_ids)):
            
            _offsets = offsets[i]
            start_idx, end_idx = torch.argmax(start_logits[i]), torch.argmax(end_logits[i])
            original_start, original_end = _offsets[start_idx][0], _offsets[end_idx][1]
            pred_span = test_ds.df.text[test_df_idx][original_start : original_end]
            preds.append(pred_span)
            test_df_idx += 1

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3813698809.py in <cell line: 0>()
      2 test_df_idx = 0
      3 
----> 4 with torch.no_grad():
      5 
      6     for xb,yb in tqdm(learner.data.test_dl):

NameError: name 'torch' is not defined

## === cell 17
test_df['selected_text'] = preds
test_df['selected_text'] = test_df.apply(lambda o: o['text'] if len(o['text']) < 3 else o['selected_text'], 1)
test_df['selected_text'] = test_df.apply(lambda o: o['text'] if o['sentiment'] == 'neutral' else o['selected_text'], 1)
subdf = test_df[['textID', 'selected_text']]
subdf.to_csv("submission.csv", index=False)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/978900310.py in <cell line: 0>()
----> 1 test_df['selected_text'] = preds
      2 test_df['selected_text'] = test_df.apply(lambda o: o['text'] if len(o['text']) < 3 else o['selected_text'], 1)
      3 test_df['selected_text'] = test_df.apply(lambda o: o['text'] if o['sentiment'] == 'neutral' else o['selected_text'], 1)
      4 subdf = test_df[['textID', 'selected_text']]
      5 subdf.to_csv("submission.csv", index=False)

NameError: name 'test_df' is not defined
