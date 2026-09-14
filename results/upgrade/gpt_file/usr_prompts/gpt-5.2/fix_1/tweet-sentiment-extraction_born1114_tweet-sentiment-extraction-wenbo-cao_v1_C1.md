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

0.7109977006912231

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import tokenizers
import transformers
import string
from tqdm.autonotebook import tqdm
from transformers import BertTokenizer,BertConfig,RobertaConfig
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
data_path = r"/kaggle/input/tweet-sentiment-extraction/"
train_data = pd.read_csv(os.path.join(data_path,'train.csv'))
train_data.dropna(inplace=True)
test_data = pd.read_csv(os.path.join(data_path,'test.csv'))

## === cell 2
train_data['text'] = train_data['text'].apply(lambda x:x.strip())
test_data['text'] = test_data['text'].apply(lambda x:x.strip())

## === cell 3
model_path = r"/kaggle/input/robertabase/"
cf = transformers.RobertaConfig.from_json_file(os.path.join(model_path,'config.json'))
MAX_LEN = 192
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 8
EPOCHS = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{model_path}/vocab.json", 
    merges_file=f"{model_path}/merges.txt", 
    lowercase=True,
    add_prefix_space=True)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4139938802.py in <cell line: 0>()
      1 model_path = r"/kaggle/input/robertabase/"
----> 2 cf = transformers.RobertaConfig.from_json_file(os.path.join(model_path,'config.json'))
      3 MAX_LEN = 192
      4 TRAIN_BATCH_SIZE = 64
      5 VALID_BATCH_SIZE = 8

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in from_json_file(cls, json_file)
    791 
    792         """
--> 793         config_dict = cls._dict_from_json_file(json_file)
    794         return cls(**config_dict)
    795 

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _dict_from_json_file(cls, json_file)
    796     @classmethod
    797     def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
--> 798         with open(json_file, encoding="utf-8") as reader:
    799             text = reader.read()
    800         return json.loads(text)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/robertabase/config.json'

## === cell 5
class Tweet_Dataset:
    def __init__(self, raw_text, sentiment, selected_text):
        self.raw_text = raw_text
        self.sentiment = sentiment
        self.selected_text = selected_text
        
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
        
        
    def __len__(self):
        return len(self.raw_text)
    
    def __getitem__(self,idx):
        dataset = self.preprocess_text(self.raw_text[idx],self.sentiment[idx],self.selected_text[idx])
        return dataset
        
        
    def preprocess_text(self, tweet_text, sent, s_text):
        
        tweet_text = " ".join(str(tweet_text).split())
        s_text = " ".join(str(s_text).split())
        s_len = len(s_text)
        
        
        idx_start, idx_end = None, None
        for idx in (i for i,e in enumerate(tweet_text) if e == s_text[0]): 
            if tweet_text[idx:idx+s_len] == s_text:
                idx_start = idx
                idx_end = idx + s_len
                break      
        target_position_list = [0] * len(tweet_text)
        if idx_start!=None and idx_end!=None:
            for char_idx in range(idx_start, idx_end):
                target_position_list[char_idx] = 1 # [0,0,0,1,1,1,1,0,0,0]
                
        
        
        encode_tweet = self.tokenizer.encode(tweet_text)
        input_ids_ori, input_offsets = encode_tweet.ids, encode_tweet.offsets
        target_ids = []
        for e, (o1, o2) in enumerate(input_offsets):
            if sum(target_position_list[o1:o2]) >0:
                target_ids.append(e)
        tar_st = target_ids[0]
        tar_end = target_ids[-1]
        
        
        sentiment_map = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
        
        input_ids = [0] + [sentiment_map[sent]] + [2] + [2] + input_ids_ori + [2]
        input_mask = [0] * (len(input_ids))
        input_type_ids = [0] * (len(input_ids))
        input_offsets = [(0,0)] * 4 + input_offsets + [(0,0)]
        tar_st += 4
        tar_end += 4
        
        
        
        padding_len = self.max_len - len(input_ids)
        if padding_len >0:
            input_ids = input_ids + [0] * padding_len
            input_mask = input_mask + [0] * padding_len
            input_type_ids = input_type_ids +  [0] * padding_len
            input_offsets = input_offsets + [(0, 0)] * padding_len
            
        else:
            pass
        
        return {
            "ids":torch.tensor(input_ids , dtype=torch.long),
            "mask":torch.tensor(input_mask , dtype=torch.long),
            "token_type_ids":torch.tensor(input_type_ids , dtype=torch.long),
            "target_start":torch.tensor(tar_st, dtype=torch.long),
            "target_end":torch.tensor(tar_end, dtype=torch.long),
            "tweet" : tweet_text,
            "sentiment" : sent,
            "selected_text": s_text,
            "offsets":torch.tensor(input_offsets,dtype=torch.long)
            
        }
    
def create_data_loader(dataset,use_gpu=True):
    tr_df, val_df = train_test_split(dataset,test_size=0.1,stratify=dataset['sentiment'])

    tr_data = Tweet_Dataset(raw_text=tr_df['text'].values,
                             sentiment=tr_df['sentiment'].values,
                             selected_text=tr_df['selected_text'].values)
    val_data = Tweet_Dataset(raw_text=val_df['text'].values,
                             sentiment=val_df['sentiment'].values,
                             selected_text=val_df['selected_text'].values)

    tr_ = torch.utils.data.DataLoader(tr_data,batch_size=TRAIN_BATCH_SIZE,pin_memory =use_gpu,shuffle=True)
    val_ = torch.utils.data.DataLoader(val_data,batch_size=VALID_BATCH_SIZE,pin_memory =use_gpu,shuffle=True)
    return tr_,val_


## === cell 6
class HighWay_Model(nn.Module):
    def __init__(self,input_size, gate_bias = -1):
        super().__init__()
        self.normal_layer = nn.Linear(input_size, input_size)
        self.gate_layer = nn.Linear(input_size,input_size)
        self.gate_layer.bias.data.fill_(gate_bias)
        
    def forward(self,x):
        norm_x = F.relu(self.normal_layer(x))
        gate_x = torch.softmax(self.gate_layer(x), dim=0)
        gate_norm = torch.mul(norm_x,gate_x)
        gate_input = torch.mul((1-gate_x),x)
        return torch.add(gate_norm,gate_input)
    
    



class roBerta_Model(transformers.BertPreTrainedModel):
    def __init__(self):
        super().__init__(cf)
        self.bert = transformers.RobertaModel.from_pretrained(model_path,config=cf)
        self.dropout = nn.Dropout(0.3)
       
        self.fc1 = nn.Linear(768,2)
        torch.nn.init.xavier_normal_(self.fc1.weight)

        
    def forward(self,ids,mask,token_type_ids):
        seq_output, pooled_output = self.bert(ids, attention_mask = mask, token_type_ids=token_type_ids)
      
        
        h_vec = self.dropout(self.fc1(seq_output))
        
        st_logits,end_logits = h_vec.split(1,dim=-1)
        
        return st_logits.squeeze() , end_logits.squeeze()
    
    '''def cnn_decoder(self,x):
        # x -> [batch,num_token, seq_len]
        x = x.unsqueeze(1)
        convx = F.leaky_relu(self.cnn(x)).squeeze(3)
        pool_x = F.avg_pool1d(convx, convx.shape[2]).squeeze(2) 
        #cat = self.dropout(torch.cat(pool_x,dim=1))
        return self.highway(pool_x)
    
    def cnn_decoder2(self,x):
        # x -> [batch,num_token, seq_len]
        x = x.unsqueeze(1)
        convx = F.leaky_relu(self.cnn2(x)).squeeze(3)
        pool_x = F.avg_pool1d(convx, convx.shape[2]).squeeze(2)
        #cat = self.dropout(torch.cat(pool_x,dim=1))
        return self.highway2(pool_x)
    '''
    
    

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
model = roBerta_Model().to(device)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/779107441.py in <cell line: 0>()
----> 1 model = roBerta_Model().to(device)

/tmp/ipykernel_11/3504726508.py in __init__(self)
     20 class roBerta_Model(transformers.BertPreTrainedModel):
     21     def __init__(self):
---> 22         super().__init__(cf)
     23         self.bert = transformers.RobertaModel.from_pretrained(model_path,config=cf)
     24         self.dropout = nn.Dropout(0.3)

NameError: name 'cf' is not defined

## === cell 8

def loss_fn(o1, o2, t1, t2):
    loss_fct = nn.CrossEntropyLoss()
    l1 = loss_fct (o1, t1)
    l2 = loss_fct (o2, t2)
    return l1+l2

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def train_fn(data_loader, model, opt, device, scheduler):
    model.train()
    
    losses = AverageMeter()
    
    tk0 = tqdm(data_loader,total=len(data_loader))
    for b, data in enumerate(tk0):
        ids = data['ids'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        mask = data['mask'].to(device)
        target_start = data['target_start'].to(device)
        target_end = data['target_end'].to(device)
        
        opt.zero_grad()
        o1,o2 = model(ids = ids,
                        mask = mask,
                        token_type_ids = token_type_ids
                       )
        
        loss = loss_fn(o1,o2,target_start,target_end)
        loss.backward()
        opt.step()
        losses.update(loss.item(),ids.size(0))
        tk0.set_postfix(loss=losses.avg)
        
def eval_fn(data_loader, model, device,test=False):
    model.eval()
    losses = AverageMeter()
    jac_vals = AverageMeter()
    fin_selected_text = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for b,data in enumerate(tk0):
        ids = data['ids'].to(device,dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        
        target_start = data['target_start'].to(device, dtype=torch.long)
        target_end = data['target_end'].to(device, dtype=torch.long)
        
        tweet = data['tweet']
        sentiment = data['sentiment']
        selected_text = data['selected_text'] 
        
        offsets = data['offsets'].numpy()
        
        o1, o2 = model(
            ids = ids,
            mask = mask,
            token_type_ids = token_type_ids
        )
        loss = loss_fn(o1,o2,target_start,target_end)
        o_start = torch.softmax(o1,dim=1).cpu().detach().numpy()
        o_end = torch.softmax(o2,dim=1).cpu().detach().numpy()
        
        jacc_scores = []
        batch_target_outputs = []
        
        for idx, batch_tweet in enumerate(tweet):
            batch_selected_text = selected_text[idx]
            batch_sentiment = sentiment[idx]
            sent_val = sentiment[idx]
            offset_var = offsets[idx]
            
            
            idx_st = np.argmax(o_start[idx,:])
            idx_end = np.argmax(o_end[idx,:])
            
            if idx_st>idx_end:
                idx_end = idx_st
                
            filter_output = ""
            if sent_val == 'neutral' or len(batch_tweet.split()) <2:
                filter_output = batch_tweet
            else:
                for ix in range(idx_st,idx_end+1):
                    filter_output += batch_tweet[offset_var[ix][0]:offset_var[ix][1]]
                    if (ix+1) < len(offset_var) and offset_var[ix][1] < offset_var[ix+1][0]:
                        filter_output += " "
            jac = jaccard(batch_selected_text.strip(),filter_output.strip())
            jacc_scores.append(jac)
            batch_target_outputs.append(filter_output.strip())
            
            
        jac_vals.update(np.mean(jacc_scores),ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jac_vals.avg)
        
        fin_selected_text += batch_target_outputs
        
    return jac_vals.avg,fin_selected_text       
        

## === cell 9
no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
opt_params = [
    {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":0.0},
    {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0}
]

num_train_steps = int((train_data.shape[0]*0.1)/TRAIN_BATCH_SIZE * EPOCHS)
optimizer = transformers.AdamW(opt_params,lr=3e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=1)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1548091081.py in <cell line: 0>()
      1 no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
      2 opt_params = [
----> 3     {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":0.0},
      4     {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0}
      5 ]

NameError: name 'model' is not defined

## === cell 12
order_dict_param = torch.load(r"/kaggle/input/trained-roberta-model/tweet_model_roBERTa_base.pt")
for name, param in model.named_parameters():
    if name in order_dict_param.keys():
        param.data.copy_(order_dict_param[name])

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/156519350.py in <cell line: 0>()
----> 1 order_dict_param = torch.load(r"/kaggle/input/trained-roberta-model/tweet_model_roBERTa_base.pt")
      2 for name, param in model.named_parameters():
      3     if name in order_dict_param.keys():
      4         param.data.copy_(order_dict_param[name])

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/trained-roberta-model/tweet_model_roBERTa_base.pt'

## === cell 13
ts_data = Tweet_Dataset(raw_text = test_data['text'].values, 
          sentiment=test_data['sentiment'].values, 
          selected_text = test_data['text'].values)

ts_dataloader = torch.utils.data.DataLoader(ts_data, batch_size=16, pin_memory=False,shuffle=False)
jacc,fin_output = eval_fn(ts_dataloader,model,device)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/817929420.py in <cell line: 0>()
----> 1 ts_data = Tweet_Dataset(raw_text = test_data['text'].values, 
      2           sentiment=test_data['sentiment'].values,
      3           selected_text = test_data['text'].values)
      4 
      5 ts_dataloader = torch.utils.data.DataLoader(ts_data, batch_size=16, pin_memory=False,shuffle=False)

/tmp/ipykernel_11/3214983811.py in __init__(self, raw_text, sentiment, selected_text)
      5         self.selected_text = selected_text
      6 
----> 7         self.tokenizer = TOKENIZER
      8         self.max_len = MAX_LEN
      9 

NameError: name 'TOKENIZER' is not defined

## === cell 14
def post_process(selected):
    return " ".join(set(selected.lower().split()))
os.listdir(data_path)
sub = pd.read_csv(os.path.join(data_path,'sample_submission.csv'))
sub['selected_text'] = fin_output
sub['selected_text'] = sub['selected_text'].apply(lambda x:post_process(x))
sub.to_csv("submission.csv",index=False)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/661559473.py in <cell line: 0>()
      3 os.listdir(data_path)
      4 sub = pd.read_csv(os.path.join(data_path,'sample_submission.csv'))
----> 5 sub['selected_text'] = fin_output
      6 sub['selected_text'] = sub['selected_text'].apply(lambda x:post_process(x))
      7 sub.to_csv("submission.csv",index=False)

NameError: name 'fin_output' is not defined
