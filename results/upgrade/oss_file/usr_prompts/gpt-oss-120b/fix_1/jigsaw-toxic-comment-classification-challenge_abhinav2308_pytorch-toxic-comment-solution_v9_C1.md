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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
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
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.50776

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%load_ext autoreload
%autoreload 2

%matplotlib inline
import numpy as np 
import pandas as pd 
import torch
import torchtext
from torchtext import data
import spacy
import os
import re


os.environ['OMP_NUM_THREADS'] = '4'
my_tok = spacy.load('en')
my_stopwords = spacy.lang.en.stop_words.STOP_WORDS
my_stopwords.update(['wikipedia','article','articles','im','page'])

def spacy_tok(x):
    x= re.sub(r'[^a-zA-Z\s]','',x)
    x= re.sub(r'[\n]',' ',x)
    return [tok.text for tok in my_tok.tokenizer(x)]


TEXT = data.Field(lower=True, tokenize=spacy_tok,eos_token='EOS',stop_words=my_stopwords,include_lengths=True)
LABEL = data.Field(sequential=False, 
                         use_vocab=False, 
                         pad_token=None, 
                            unk_token=None)

dataFields = [("id", None),
                 ("comment_text", TEXT), ("toxic", LABEL),
                 ("severe_toxic", LABEL), ("threat", LABEL),
                 ("obscene", LABEL), ("insult", LABEL),
                 ("identity_hate", LABEL)]

dataset= data.TabularDataset(path='../input/train.csv', 
                                            format='csv',
                                            fields=dataFields, 
                                            skip_header=True)


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2780978632.py in <cell line: 0>()
      6 import pandas as pd
      7 import torch
----> 8 import torchtext
      9 from torchtext import data
     10 import spacy

ModuleNotFoundError: No module named 'torchtext'

## === cell 1
train,val= dataset.split()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3738507542.py in <cell line: 0>()
----> 1 train,val= dataset.split()

NameError: name 'dataset' is not defined

## === cell 2
TEXT.build_vocab(train,vectors='fasttext.simple.300d')


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1054575484.py in <cell line: 0>()
----> 1 TEXT.build_vocab(train,vectors='fasttext.simple.300d')

NameError: name 'TEXT' is not defined

## === cell 3
traindl, valdl = torchtext.data.BucketIterator.splits(datasets=(train, val),
                                            batch_sizes=(128,1024),
                                            sort_key=lambda x: len(x.comment_text),
                                            device=torch.device('cuda:0'),
                                            sort_within_batch=True
                                                     )


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/293598444.py in <cell line: 0>()
----> 1 traindl, valdl = torchtext.data.BucketIterator.splits(datasets=(train, val),
      2                                             batch_sizes=(128,1024),
      3                                             sort_key=lambda x: len(x.comment_text),
      4                                             device=torch.device('cuda:0'),
      5                                             sort_within_batch=True

NameError: name 'torchtext' is not defined

## === cell 4
vectors= train.fields['comment_text'].vocab.vectors.cuda()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/756203368.py in <cell line: 0>()
----> 1 vectors= train.fields['comment_text'].vocab.vectors.cuda()

NameError: name 'train' is not defined

## === cell 5
class BatchGenerator:
    def __init__(self, dl):
        self.dl = dl
        self.yFields= ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        self.x= 'comment_text'
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x)
            y = torch.transpose( torch.stack([getattr(batch, y) for y in self.yFields]),0,1)
            yield (X,y)


## === cell 6
import torch
import torch.nn as nn
import torch.nn.functional as F


## === cell 7
class MyModel(nn.Module):
    def __init__(self,op_size,n_tokens,pretrained_vectors,nl=2,bidirectional=True,emb_sz=300,n_hiddenUnits=100):
        super(MyModel, self).__init__()
        self.n_hidden= n_hiddenUnits
        self.embeddings= nn.Embedding(n_tokens,emb_sz)
        self.embeddings.weight.data.copy_(pretrained_vectors)
        self.embeddings.weight.requires_grad = False
        self.rnn= nn.LSTM(emb_sz,n_hiddenUnits,num_layers=2,bidirectional=True,dropout=0.2)
        self.lArr=[]
        if bidirectional:
            n_hiddenUnits= 2* n_hiddenUnits
        self.bn1 = nn.BatchNorm1d(num_features=n_hiddenUnits)
        for i in range(nl):
            self.lArr.append(nn.Linear(n_hiddenUnits,n_hiddenUnits))
        self.lArr= nn.ModuleList(self.lArr)
        self.l1= nn.Linear(n_hiddenUnits,op_size)
        
    def forward(self,data,lengths):
        torch.cuda.empty_cache()
        bs= data.shape[1]
        self.h= self.init_hidden(bs)
        embedded= self.embeddings(data)
        embedded= nn.Dropout()(embedded)
        rnn_out, self.h = self.rnn(embedded, (self.h,self.h))
        ipForLinearLayer= rnn_out[-1]
        for linearlayer in self.lArr:
            outp= linearlayer(ipForLinearLayer)
            ipForLinearLayer= self.bn1(F.relu(outp))
            ipForLinearLayer= nn.Dropout(p=0.6)(ipForLinearLayer)
        outp = self.l1(ipForLinearLayer)
        del embedded;del rnn_out;del self.h;
        torch.cuda.empty_cache()
        return outp
        
    def init_hidden(self, batch_size):
        return torch.zeros((4,batch_size,self.n_hidden),device="cuda:0")


## === cell 8
def getValidationLoss(valdl,model,loss_func):
    model.eval()
    runningLoss=0
    valid_batch_it = BatchGenerator(valdl)
    with torch.no_grad():
        for i,obj in enumerate(valid_batch_it):
            obj= ( (obj[0][0].cuda(),obj[0][1].cuda()),obj[1] )
            preds = model(obj[0][0],obj[0][1])
            loss = loss_func(preds,obj[1].float())
            runningLoss+= loss.item()
        return runningLoss/len(valid_batch_it)


## === cell 9
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
def oneEpoch(lr):
    train_batch_it = BatchGenerator(traindl)
    opt = optim.Adam(model.parameters(),lr)
    runningLoss= 0
    for i,obj in enumerate(train_batch_it):
        obj= ( (obj[0][0].cuda(),obj[0][1].cuda()),obj[1] )
        model.train()
        opt.zero_grad()
        preds = model(obj[0][0],obj[0][1])
        loss = loss_func(preds,obj[1].float())
        runningLoss+= loss.item()
        loss.backward()
        opt.step()
        del obj
    runningLoss= runningLoss/len(train_batch_it)
    valLoss= getValidationLoss(valdl,model,loss_func)
    torch.cuda.empty_cache()
    return runningLoss,valLoss


## === cell 10
epochs= 3
trainLossArr=[]
valLossArr=[]
model= MyModel(6,len(TEXT.vocab),vectors,3)
loss_func= torch.nn.BCEWithLogitsLoss()
model = model.cuda()
for i in range(epochs):
    %time tLoss,vLoss= oneEpoch(1e-4)
    trainLossArr.append(tLoss)
    valLossArr.append(vLoss)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/794084501.py in <cell line: 0>()
      2 trainLossArr=[]
      3 valLossArr=[]
----> 4 model= MyModel(6,len(TEXT.vocab),vectors,3)
      5 loss_func= torch.nn.BCEWithLogitsLoss()
      6 model = model.cuda()

NameError: name 'TEXT' is not defined

## === cell 11
import matplotlib.pyplot as plt 
plt.plot(trainLossArr,color='b')
plt.plot(valLossArr,color='g')
plt.show()


## === cell 12
torch.save(model.state_dict(), "myFirstModel1")


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1477201615.py in <cell line: 0>()
----> 1 torch.save(model.state_dict(), "myFirstModel1")

NameError: name 'model' is not defined

## === cell 13
dataFields = [("id", None),
                 ("comment_text", TEXT)
             ]

testDataset= data.TabularDataset(path='../input/test.csv', 
                                            format='csv',
                                            fields=dataFields, 
                                            skip_header=True)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3836766929.py in <cell line: 0>()
      1 dataFields = [("id", None),
----> 2                  ("comment_text", TEXT)
      3              ]
      4 
      5 testDataset= data.TabularDataset(path='../input/test.csv', 

NameError: name 'TEXT' is not defined

## === cell 14
test_iter1 = torchtext.data.Iterator(testDataset, batch_size=32, device=torch.device('cuda:0'), sort=False, sort_within_batch=False, repeat=False,shuffle=False)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/263992133.py in <cell line: 0>()
----> 1 test_iter1 = torchtext.data.Iterator(testDataset, batch_size=32, device=torch.device('cuda:0'), sort=False, sort_within_batch=False, repeat=False,shuffle=False)

NameError: name 'torchtext' is not defined

## === cell 15
testDF= pd.read_csv("../input/test.csv")


## === cell 16
myPreds=[]
with torch.no_grad():
    model.eval()
    for obj in test_iter1:
        torch.cuda.empty_cache()
        pred= model(obj.comment_text[0],obj.comment_text[1])
        pred= torch.sigmoid(pred)
        myPreds.append(pred.cpu().numpy())
        del pred;del obj;
        torch.cuda.empty_cache()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2034034023.py in <cell line: 0>()
      1 myPreds=[]
      2 with torch.no_grad():
----> 3     model.eval()
      4     for obj in test_iter1:
      5 #         print(torch.transpose(obj.comment_text[0],0,1)[:10].shape)

NameError: name 'model' is not defined

## === cell 17
myPreds= np.vstack(myPreds)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2914851537.py in <cell line: 0>()
----> 1 myPreds= np.vstack(myPreds)

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate

## === cell 18
for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
    testDF[col] = myPreds[:, i]


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3373024045.py in <cell line: 0>()
      1 for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
----> 2     testDF[col] = myPreds[:, i]

TypeError: list indices must be integers or slices, not tuple

## === cell 19
testDF.drop("comment_text", axis=1).to_csv("submission.csv", index=False)


## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'obscene', 'severe_toxic', 'threat', 'insult', 'identity_hate', 'toxic'}
