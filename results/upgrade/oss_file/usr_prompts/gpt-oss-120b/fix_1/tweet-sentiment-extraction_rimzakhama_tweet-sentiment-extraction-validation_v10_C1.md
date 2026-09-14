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

0.6118680238723755

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
from tqdm import tqdm
import string
import csv

import transformers
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup



class config:
    MAX_LEN = 141
    TRAIN_BATCH_SIZE = 40 # 8 #??
    VALID_BATCH_SIZE = 16 #4 #??
    EPOCHS = 2
    BERT_PATH = "../input/bertbaseuncased/"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "../input/tweet-sentiment-extraction/train.csv"

    TOKENIZER = tokenizers.BertWordPieceTokenizer(
        os.path.join(BERT_PATH, 'vocab.txt'),
        lowercase=True
    )

    
    

class BERTBaseUncased(nn.Module): #The parent class is nn.Module
    def __init__(self):
        super(BERTBaseUncased, self).__init__() #A call to super here ensures that both variables exist in the inherited class.
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.l0 = nn.Linear(768, 2) #output layer : fully_connected layer
    
    def forward(self, ids, mask, token_type_ids):  #setiment can be the parameters here
        ''' See how to use sentiment in this part '''
        sequence_output, pooled_output = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        
        logits = self.l0(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1) # dim=-1 to seperate columns

        start_logits = start_logits.squeeze(-1) #it says where my selected_text is beginning
        end_logits = end_logits.squeeze(-1) #it says where my selected_text is ending, 1 : axis=1

        
        return start_logits, end_logits
    

    

device = torch.device("cuda") #??
model = BERTBaseUncased()
model.to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('../input/kernel2c74baab28/model.bin'))

model.eval()

    
''
class TweetDataset:
    ''' initialization function '''
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN #max_len from config.py
        self.tokenizer = config.TOKENIZER #TOLENIZER from config.py

    ''' fct : length of the dataset '''
    def __len__(self): 
        return(len(self.tweet))

    def __getitem__(self, item): #item is the index value

        tweet = str(self.tweet[item]) # converting to string just to ckech that nothing breaks
        tweet = " ".join(tweet.split())

        selected_text = str(self.selected_text[item]) # converting to string just to ckech that nothing breaks
        selected_text = " ".join(selected_text.split())

        ''' Prepare what we need to train our model '''
        len_sel_text = len (selected_text)

        idx0 = -1 
        idx1 = -1
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind+len_sel_text] == selected_text:
                idx0 = ind
                idx1 = ind + len_sel_text - 1 #I added '-1' because len takes index+1
                break
        
        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1 :
            for j in range(idx0, idx1 + 1):
                if tweet[j] != ' ':  #if not space
                    char_targets[j] = 1

        tok_tweet = self.tokenizer.encode(tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[1:-1] #because always the fisrt token is [CLS] and the last token is [SEP]

        targets = [0] * (len(tok_tweet_tokens)-2) # -2 to remove [CLS] and [SEP] tokens
        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1:offset2]) > 0: #??
                targets[j] = 1

        targets = [0] + targets + [0] #cls, sep
        targets_start = [0] * len(targets)
        targets_end = [0] *len(targets)

        non_zero = np.nonzero(targets)[0] 
        
        if len(non_zero) > 0: #if there are some values
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1 

        ''' Attention mask: a sequence of 1s and 0s, with 1s for all input tokens (actual words) and 0s
        for all padding tokens.'''
        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * len(tok_tweet_ids) #in case of two segments(texts), we need to differentiate them. We don't need it in our case.

        ''' 
        The convention in BERT is:
        (a) For sequence pairs:
        tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1 

        (b) For single sequences:
        tokens:   [CLS] the dog is hairy . [SEP]
        type_ids: 0     0   0   0  0     0 0

        '''
        
        padding_len = self.max_len - len(tok_tweet_ids)
        ''' pad everything you need: (all the outputs sould have the same length as max_len): '''
        ids = tok_tweet_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        targets = targets + [0] * padding_len
        targets_start = targets_start + [0] * padding_len
        targets_end = targets_end + [0] * padding_len


        sentiment = [1, 0, 0] #neutral
        if self.sentiment[item] == 'positive':
            sentiment = [0, 0, 1] #one_hot encoding
        if self.sentiment[item] == 'negative':
            sentiment = [0, 1, 0]

        return{
            'ids': torch.tensor(ids, dtype=torch.long), #convert tp pytorch tensor
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            'padding_len': torch.tensor(padding_len, dtype=torch.long),
            'tweet_tokens': ' '.join(tok_tweet_tokens),
            'orig_tweet': self.tweet[item],
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'orig_sentiment': self.sentiment[item],
            'orig_selected_text': self.selected_text[item]
           
        }   


df_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
df_test.loc[:, 'selected_text'] = df_test.text.values #just to make sur the dataset works without any change there
    
    
test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values
)
    
valid_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle = False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
)    
    
final_col = [] # to submit
fin_output_start = []
fin_output_end = []
fin_padding_lens = []
fin_tweet_tokens = []
fin_orig_sentiment = []
fin_orig_selected_text = []
fin_orig_tweet =[]


with torch.no_grad():
    for bi, d in enumerate(valid_data_loader): # bi : batch index, d : dataset
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        tweet_tokens =  d['tweet_tokens']
        padding_len = d['padding_len']
        orig_sentiment = d['orig_sentiment']
        orig_selected_text = d['orig_selected_text']
        orig_tweet = d['orig_tweet']
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]


        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long) #long is 64-bit integer
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float) #targets_start
        targets_end = targets_end.to(device, dtype=torch.float) #targets_end


        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        fin_output_start.append(torch.sigmoid(o1).cpu().detach().numpy()) #??
        fin_output_end.append(torch.sigmoid(o2).cpu().detach().numpy())
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist()) #1 value so we can use extend

        fin_tweet_tokens.extend(tweet_tokens) #it is just a list of strings
        fin_orig_sentiment.extend(orig_sentiment)
        fin_orig_selected_text.extend(orig_selected_text)
        fin_orig_tweet.extend(orig_tweet)

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    threshold = 0.2 #?? i its related to sigmoid??
    jaccards =[]
    for j in range(len(fin_tweet_tokens)):
        target_string = fin_orig_selected_text[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_lens[j]
        original_tweet = fin_orig_tweet[j]
        sentiment = fin_orig_sentiment[j]

        if padding_len > 0:
            mask_start = fin_output_start[j, :][:-padding_len] >= threshold #-padding_len??
            mask_end = fin_output_end[j, :][:-padding_len] >= threshold
        else:
            mask_start = fin_output_start[j, :] >= threshold
            mask_end = fin_output_end[j, :] >= threshold

        mask = [0] * len(mask_start)
        idx_start = np.nonzero(mask_start)[0]
        idx_end = np.nonzero(mask_end)[0]

        if len(idx_start) > 0:
            idx_start = idx_start[0]
            if len(idx_end) > 0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
        else:
            idx_start = 0
            idx_end = 0

        for mj in range(idx_start, idx_end + 1):
            mask[mj] = 1

        output_tokens = [x for p, x in enumerate(tweet_tokens.split()) if mask[p] == 1]
        output_tokens = [x for x in output_tokens if x not in ('[CLS]', '[SEP]')]

        final_output = ''

        for ot in output_tokens:
            if ot.startswith('##'):
                final_output = final_output + ot[2:]
            elif len(ot) == 1 and ot in string.punctuation:
                final_output = final_output + ot
            else:
                final_output = final_output + ' ' + ot
        final_output = final_output.strip()
        if sentiment == 'neutral' or len(original_tweet.split()) < 4: #??
            final_output = original_tweet

        final_col.append(final_output) # to submit 

        
submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
submission.selected_text = final_col
submission.to_csv('submission.csv',index=False)

print(submission.head())



    


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/511787184.py in <cell line: 0>()
     10 import transformers
     11 from sklearn import model_selection
---> 12 from transformers import AdamW
     13 from transformers import get_linear_schedule_with_warmup
     14 

ImportError: cannot import name 'AdamW' from 'transformers' (/usr/local/lib/python3.11/dist-packages/transformers/__init__.py)
