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

emoji==2.15.0
geopandas==0.14.4
nltk==3.9.2
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

0.7031523585319519

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install ../input/fairseq-and-fastbpe/sacrebleu-1.4.9-py3-none-any.whl
!pip install ../input/fairseq-and-fastbpe/fairseq-0.9.0-cp37-cp37m-linux_x86_64.whl
!pip install ../input/fairseq-and-fastbpe/fastBPE-0.1.0-cp37-cp37m-linux_x86_64.whl

## === cell 1
import numpy as np
import pandas as pd
import os
import argparse
import warnings
import random
import torch 
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import tokenizers
from transformers import RobertaModel, RobertaConfig
warnings.filterwarnings('ignore')
seed=18

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re

tokenizer = TweetTokenizer()

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace("ca n't", "can't").replace("ai n't", "ain't")
    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ", " 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")
    normTweet = normTweet.replace(" p . m .", "  p.m.") .replace(" p . m ", " p.m ").replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")

    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)
    
    return " ".join(normTweet.split())

## === cell 3
normalizeTweet(' I`d have responded, if I were going')

## === cell 4
base_path='../input/bertweet-dataset'
config = RobertaConfig.from_pretrained(
    os.path.join(base_path,"BERTweet_base_transformers/config.json"), output_hidden_states=True)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertweet-dataset/BERTweet_base_transformers/config.json'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertweet-dataset/BERTweet_base_transformers/config.json'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_10/212821342.py in <cell line: 0>()
      1 base_path='../input/bertweet-dataset'
----> 2 config = RobertaConfig.from_pretrained(
      3     os.path.join(base_path,"BERTweet_base_transformers/config.json"), output_hidden_states=True)

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, **kwargs)
    566         cls._set_token_in_kwargs(kwargs, token)
    567 
--> 568         config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    569         if cls.base_config_key and cls.base_config_key in config_dict:
    570             config_dict = config_dict[cls.base_config_key]

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    688             except Exception:
    689                 # For any other exception, we throw a generic error.
--> 690                 raise OSError(
    691                     f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
    692                     " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"

OSError: Can't load the configuration of '../input/bertweet-dataset/BERTweet_base_transformers/config.json'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '../input/bertweet-dataset/BERTweet_base_transformers/config.json' is the correct path to a directory containing a config.json file

## === cell 5
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
args = argparse.Namespace(bpe_codes= os.path.join(base_path,"BERTweet_base_transformers/bpe.codes"))
bpe = fastBPE(args)
vocab = Dictionary()
vocab.add_from_file(os.path.join(base_path,"BERTweet_base_transformers/dict.txt"))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_10/3004030251.py in <cell line: 0>()
----> 1 from fairseq.data.encoders.fastbpe import fastBPE
      2 from fairseq.data import Dictionary
      3 args = argparse.Namespace(bpe_codes= os.path.join(base_path,"BERTweet_base_transformers/bpe.codes"))
      4 bpe = fastBPE(args)
      5 vocab = Dictionary()

ModuleNotFoundError: No module named 'fairseq'

## === cell 6
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, bpe, vocab, max_len=96):
        self.df = df
        self.labeled = 'selected_text' in df
        self.bpe = bpe
        self.vocab = vocab
        self.max_len=max_len
        
    def __getitem__(self, index):
        data={}
        row=self.df.iloc[index]
        ids, masks, tweets_encoded = self.get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweets_encoded'] = tweets_encoded
        data['tweet'] = row.text
        if self.labeled:
            data['selected_tweet'] = row.selected_text
            start_idx, end_idx = self.get_target_idx(row, tweets_encoded)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx
        return data
    
    def __len__(self):
        return len(self.df)
    
    def get_input_data(self, row):
        normalized_tweets = normalizeTweet(row.text)
        normalized_tweets = " " + " ".join(normalized_tweets.split())
        tweets_encoded = self.bpe.encode(normalized_tweets)
        encoding_ids = self.vocab.encode_line(tweets_encoded, append_eos=False, add_if_not_exist=False).long().tolist()
        sentiment_id = self.vocab.encode_line(self.bpe.encode(row.sentiment), append_eos=False, add_if_not_exist=False).long().tolist()
        ids = [0]+sentiment_id+[2,2]+encoding_ids+[2]
        
        pad_len = self.max_len-len(ids)
        if pad_len>0:
            ids += [1] * pad_len
        ids = torch.tensor(ids)
        masks = torch.where(ids!=1, torch.tensor(1), torch.tensor(0))
        
        return ids, masks, tweets_encoded
    
    def get_target_idx(self, row, tweets_encoded):
        normalized_selected_tweets = normalizeTweet(row.selected_text)
        normalized_selected_tweets = ' '+' '.join(normalized_selected_tweets.split())
        normalized_tweets = normalizeTweet(row.text)
        normalized_tweets = " " + " ".join(normalized_tweets.split())
        
        len_st = len(normalized_selected_tweets) - 1
        idx0 = None
        idx1 = None
        for ind in (i for i, e in enumerate(normalized_tweets) if e == normalized_selected_tweets[1]):
            if " " + normalized_tweets[ind: ind+len_st] == normalized_selected_tweets:
                idx0 = ind
                idx1 = ind+len_st-1
                break
        if idx0==None and len(normalized_selected_tweets.split())>1:
            normalized_selected_tweets_1=' '+' '.join(normalized_selected_tweets.split()[1:])
            len_st_1 = len(normalized_selected_tweets_1) - 1
            for ind in (i for i, e in enumerate(normalized_tweets) if e == normalized_selected_tweets_1[1]):
                if " " + normalized_tweets[ind: ind+len_st_1] == normalized_selected_tweets_1:
                    idx0 = ind
                    idx1 = ind+len_st_1-1
                    break
        if idx0==None and len(normalized_selected_tweets.split())>1:
            normalized_selected_tweets_2=' '+' '.join(normalized_selected_tweets.split()[:-1])
            len_st_2 = len(normalized_selected_tweets_2) - 1
            for ind in (i for i, e in enumerate(normalized_tweets) if e == normalized_selected_tweets_2[1]):
                if " " + normalized_tweets[ind: ind+len_st_2] == normalized_selected_tweets_2:
                    idx0 = ind
                    idx1 = ind+len_st_2-1
                    break
        if idx0==None and len(normalized_selected_tweets.split())>1:
            normalized_selected_tweets_3=' '+' '.join(normalized_selected_tweets_2.split()[:-1])
            len_st_3 = len(normalized_selected_tweets_3) - 1
            for ind in (i for i, e in enumerate(normalized_tweets) if e == normalized_selected_tweets_3[1]):
                if " " + normalized_tweets[ind: ind+len_st_3] == normalized_selected_tweets_3:
                    idx0 = ind
                    idx1 = ind+len_st_3-1
                    break 
        sum_tot=-1
        flag = 0
        if idx0 != None and idx1 != None:
            for i, token in enumerate(tweets_encoded.split()):
                if '@@' not in token:
                    sum_tot += len(token)+1
                else:
                    sum_tot += len(token)-2
                if sum_tot>=idx0 and flag==0:
                    start_idx = i
                    flag = 1
                if sum_tot>=idx1:
                    end_idx = i
                    break
        if idx0==None or idx1==None:
            start_idx=0
            end_idx=0
        return start_idx+4, end_idx+4

## === cell 7
def get_train_val_loaders(df, train_idx, val_idx, batch_size=32):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df, bpe, vocab), 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, bpe, vocab), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict

## === cell 8
import torch.nn as nn
import torch.optim as optim

class BERTweetModel(nn.Module):
    def __init__(self, conf):
        super(BERTweetModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(os.path.join(base_path,"BERTweet_base_transformers/model.bin"),config=conf)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(conf.hidden_size*4,2)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias,0)
        
    def forward(self, input_ids, attention_mask):
        a, b, h = self.roberta(input_ids, attention_mask)
        x = torch.cat([h[-1],h[-2],h[-3], h[-4]],dim=-1)
        x = self.fc(self.dropout(x))
        start_logits, end_logits = x.split(1, -1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

## === cell 9
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)    
    total_loss = start_loss + end_loss
    return total_loss

## === cell 10
def get_selected_text(tweets_encoded, start_idx, end_idx):
    selected_text = ""
    for i, token in enumerate(tweets_encoded.split()[start_idx-4:end_idx-3]):
            token=' '+token
            selected_text+=token
    selected_text=re.sub('@@ ', '', selected_text)
    selected_text=re.sub('@@', '', selected_text)
    return selected_text
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def compute_jaccard_score(tweets_encoded, start_idx, end_idx, start_logits, end_logits):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    length = len(tweets_encoded.split())
    if start_pred<4:
        start_pred=4
    if end_pred>3+length:
        end_pred=3+length
    if start_pred > end_pred:
        start_pred=4
        end_pred=3+length
        pred = get_selected_text(tweets_encoded, start_pred, end_pred).strip()
    else:
        pred = get_selected_text(tweets_encoded, start_pred, end_pred).strip()    
    true = get_selected_text(tweets_encoded, start_idx, end_idx).strip()
    
    return jaccard(true, pred)

## === cell 11
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    if torch.cuda.is_available():
        model.cuda()
    
    loss_check=1000
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0
            count=0
            for data in (dataloaders_dict[phase]):
                if count%100==0:
                    print(count)
                count+=1
                ids = data['ids']
                masks = data['masks']
                tweets_encoded = data['tweets_encoded']
                
                selected_tweet = data['selected_tweet']
                start_idx = data['start_idx']
                end_idx = data['end_idx']

                if torch.cuda.is_available():
                  ids=ids.cuda()
                  masks=masks.cuda()
                  start_idx=start_idx.cuda()
                  end_idx=end_idx.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    start_logits, end_logits = model(ids, masks)

                    loss = criterion(start_logits, end_logits, start_idx, end_idx)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(ids)
                    
                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
                    
                    for i in range(len(ids)): 
                        jaccard_score = compute_jaccard_score(
                            tweets_encoded[i],
                            start_idx[i],
                            end_idx[i],
                            start_logits[i], 
                            end_logits[i])
                        epoch_jaccard += jaccard_score
                    
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)
            
            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
                epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard))
        if epoch_loss<loss_check:
            loss_check=epoch_loss
            print("Saving model")
            torch.save(model.state_dict(), filename)
        elif epoch>1:
            print('Training stopping')
            break

## === cell 12
num_epochs = 10
batch_size = 32
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)

## === cell 13
def run(fold):
    train_df = pd.read_csv('tweet-sentiment-extraction/train.csv').dropna().reset_index(drop=True)
    train_df['text'] = train_df['text'].astype(str)
    train_df['selected_text'] = train_df['selected_text'].astype(str)

    (train_idx, val_idx) = list(skf.split(train_df, train_df.sentiment))[fold]
    print(f'Fold: {fold}')
    model = BERTweetModel(conf=config)
    torch.save(model.state_dict(), f'drive/My Drive/Kaggle/check.pth')
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    criterion = loss_fn    
    dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)
    print('starting training')
    train_model(
        model, 
        dataloaders_dict,
        criterion, 
        optimizer, 
        num_epochs,
        f'drive/My Drive/Kaggle/roberta_fold{fold}.pth')

## === cell 14
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df, bpe, vocab), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)    
    return loader

## === cell 15
base_path='../input/bertweet-dataset'
config = RobertaConfig.from_pretrained(
    os.path.join(base_path,"BERTweet_base_transformers/config.json"), output_hidden_states=True)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertweet-dataset/BERTweet_base_transformers/config.json'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertweet-dataset/BERTweet_base_transformers/config.json'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_10/212821342.py in <cell line: 0>()
      1 base_path='../input/bertweet-dataset'
----> 2 config = RobertaConfig.from_pretrained(
      3     os.path.join(base_path,"BERTweet_base_transformers/config.json"), output_hidden_states=True)

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, **kwargs)
    566         cls._set_token_in_kwargs(kwargs, token)
    567 
--> 568         config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    569         if cls.base_config_key and cls.base_config_key in config_dict:
    570             config_dict = config_dict[cls.base_config_key]

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    688             except Exception:
    689                 # For any other exception, we throw a generic error.
--> 690                 raise OSError(
    691                     f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
    692                     " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"

OSError: Can't load the configuration of '../input/bertweet-dataset/BERTweet_base_transformers/config.json'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '../input/bertweet-dataset/BERTweet_base_transformers/config.json' is the correct path to a directory containing a config.json file

## === cell 16
def postprocessing(pred, tweet):
    pred_wo_spaces=''.join(pred.split())
    length = len(pred_wo_spaces)
    flag=0
    if tweet[-1]=='@':
        return tweet
    else:
        for index, value in enumerate(tweet):
            count=0
            letter=pred_wo_spaces[count]
            if value==letter:
                start_idx=index
                end_idx=index
                count+=1
                end_idx+=1
                while True:
                    if tweet[end_idx]==' ':
                        end_idx+=1
                    elif tweet[end_idx]=='!' and pred_wo_spaces[count]!='!':
                        end_idx+=1
                    elif tweet[end_idx]=='.' and pred_wo_spaces[count]!='.':
                        end_idx+=1
                    elif tweet[end_idx]=='*' and pred_wo_spaces[count]!='*':
                        end_idx+=1
                    elif tweet[end_idx]=='-' and pred_wo_spaces[count]!='-':
                        end_idx+=1
                    elif tweet[end_idx]=='?' and pred_wo_spaces[count]!='?':
                        end_idx+=1
                    elif tweet[end_idx]==pred_wo_spaces[count]:
                        end_idx+=1
                        count+=1
                    else:
                        break
                    if count==length:
                        flag=1
                        break
                if flag==1:
                    break
        if flag==1:
            return tweet[start_idx:end_idx]
        elif 'HTTPURL' in pred.split():
            return tweet
        elif '@USER' in pred.split():
            return tweet
        else:
            print('No match found')
            print(pred)
            print(tweet)
            return tweet

## === cell 17
test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
test_df['text'] = test_df['text'].astype(str)
test_loader = get_test_loader(test_df)
predictions = []
models = []
model = BERTweetModel(conf=config)
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(f'../input/mosh1-data-orig/roberta_fold3.pth', map_location=torch.device('cpu')))
model.eval()
models.append(model)
count=0
for data in test_loader:
    print(count)
    count+=1
    ids = data['ids']
    masks = data['masks']
    tweets_encoded = data['tweets_encoded']
    tweet = data['tweet']
    
    if torch.cuda.is_available():
        ids=ids.cuda()
        masks=masks.cuda()

    start_logits = []
    end_logits = []
    for model in models:
        with torch.no_grad():
            output = model(ids, masks)
            start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
            end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())
    
    start_logits = np.mean(start_logits, axis=0)
    end_logits = np.mean(end_logits, axis=0)
    for i in range(len(ids)):    
        start_pred = np.argmax(start_logits[i])
        end_pred = np.argmax(end_logits[i])
        length = len(tweets_encoded[i].split())
        if start_pred<4:
            start_pred=4
        if end_pred>3+length:
            end_pred=3+length
        if start_pred > end_pred:
            start_pred=4
            end_pred=3+length
            pred = get_selected_text(tweets_encoded[i], start_pred, end_pred).strip()
        else:
            pred = get_selected_text(tweets_encoded[i], start_pred, end_pred).strip()    
        
        try:
            pred=postprocessing(pred, tweet[i].strip())
        except:
            pred=tweet[i]

        predictions.append(pred)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1795405970.py in <cell line: 0>()
      1 test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
      2 test_df['text'] = test_df['text'].astype(str)
----> 3 test_loader = get_test_loader(test_df)
      4 predictions = []
      5 models = []

/tmp/ipykernel_10/4010696731.py in get_test_loader(df, batch_size)
      1 def get_test_loader(df, batch_size=32):
      2     loader = torch.utils.data.DataLoader(
----> 3         TweetDataset(df, bpe, vocab),
      4         batch_size=batch_size,
      5         shuffle=False,

NameError: name 'bpe' is not defined

## === cell 18
sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
sub_df['selected_text'] = predictions
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
sub_df.to_csv('submission.csv', index=False)
sub_df.head()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4117557064.py in <cell line: 0>()
      1 sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
----> 2 sub_df['selected_text'] = predictions
      3 sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
      4 sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
      5 sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)

NameError: name 'predictions' is not defined
